use crate::error::AppError;
use crate::media::ffmpeg;
use crate::media::mask::{dilate_mask, load_mask, solidify_mask};
use crate::media::restoration::model::TemporalSettings;
use crate::media::restoration::temporal::CandidateFrame;
use crate::media::restoration::{artifact, fallback, temporal};
use crate::project::model::{
    BoundingBox, Project, RemovalConfig, RemovalMode, TrackingFrame, TrackingStatus,
};
use image::{imageops, GrayImage, Rgb, RgbImage};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn render_project<F>(
    directory: &Path,
    project: &Project,
    config: &RemovalConfig,
    cancel: &AtomicBool,
    mut on_progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(u64, u64),
{
    let tracking = project.tracking.as_ref().ok_or_else(|| {
        AppError::InvalidRequest("Analyze the track before rendering.".to_string())
    })?;
    if tracking.frames.len() != project.video.frame_count as usize {
        return Err(AppError::InvalidRequest(
            "Tracking data does not cover the complete video.".to_string(),
        ));
    }
    if tracking.frames.iter().any(|frame| {
        matches!(
            frame.status,
            TrackingStatus::NeedReview | TrackingStatus::AutoWeak
        )
    }) {
        return Err(AppError::InvalidRequest(
            "There are unresolved tracking frames. Review them before rendering.".to_string(),
        ));
    }
    if matches!(config.mode, RemovalMode::Replacement) && config.replacement_path.is_none() {
        return Err(AppError::InvalidRequest(
            "Choose a replacement PNG before rendering.".to_string(),
        ));
    }
    let mask_path = project
        .watermark
        .templates
        .as_ref()
        .and_then(|templates| templates.mask.as_ref())
        .map(|path| directory.join(path))
        .ok_or_else(|| {
            AppError::InvalidRequest(
                "Save the watermark anchor and mask before rendering.".to_string(),
            )
        })?;
    // Expand the glyph-derived mask slightly so anti-aliased watermark strokes
    // are covered without falling back to a full rectangular blur.
    let mask = dilate_mask(&solidify_mask(&load_mask(&mask_path)?, 64, 1), 3);
    let replacement = if matches!(config.mode, RemovalMode::Replacement) {
        Some(load_replacement(config)?)
    } else {
        None
    };
    if matches!(
        config.mode,
        RemovalMode::TemporalRestore | RemovalMode::AutoBest
    ) {
        return render_temporal_project(directory, project, config, &mask, cancel, on_progress);
    }
    let raw_video = directory.join("cache").join("render-video.mp4");
    let final_temp = directory.join("cache").join("render-final.mp4");
    let output = directory.join(match config.mode {
        RemovalMode::Replacement => "output-replacement.mp4",
        RemovalMode::Blur => "output-blur.mp4",
        RemovalMode::Inpaint => "output-inpaint.mp4",
        RemovalMode::TemporalRestore => "output-temporal.mp4",
        RemovalMode::AutoBest => "output-auto-best.mp4",
    });
    let _ = std::fs::remove_file(&raw_video);
    let _ = std::fs::remove_file(&final_temp);
    let mut decoder = ffmpeg::open_raw_decoder(
        Path::new(&project.source.path),
        project.video.width,
        project.video.height,
    )?;
    let mut encoder = ffmpeg::open_raw_encoder(
        project.video.width,
        project.video.height,
        project.video.fps,
        &raw_video,
    )?;
    for index in 0..project.video.frame_count {
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::OperationCancelled);
        }
        let bytes = decoder.next_frame()?.ok_or_else(|| {
            AppError::FfmpegFailed("Video ended before the expected frame count.".to_string())
        })?;
        let mut frame = RgbImage::from_raw(project.video.width, project.video.height, bytes)
            .ok_or_else(|| AppError::FfmpegFailed("Invalid raw frame dimensions.".to_string()))?;
        let tracking_frame = &tracking.frames[index as usize];
        if !matches!(tracking_frame.status, TrackingStatus::Occluded) {
            process_frame(
                &mut frame,
                tracking_frame,
                &mask,
                config,
                replacement.as_ref(),
            )?;
        }
        encoder.write_frame(frame.as_raw())?;
        on_progress(index + 1, project.video.frame_count);
    }
    decoder.finish()?;
    encoder.finish()?;
    mux_audio(&raw_video, Path::new(&project.source.path), &final_temp)?;
    std::fs::rename(&final_temp, &output).or_else(|_| {
        if output.exists() {
            std::fs::remove_file(&output)?;
        }
        std::fs::rename(&final_temp, &output)
    })?;
    Ok(output)
}

struct BufferedFrame {
    index: u64,
    image: RgbImage,
}

fn render_temporal_project<F>(
    directory: &Path,
    project: &Project,
    config: &RemovalConfig,
    mask: &GrayImage,
    cancel: &AtomicBool,
    mut on_progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(u64, u64),
{
    let _requested_strategy =
        crate::media::restoration::model::RestorationStrategy::from_mode(config.mode);
    let raw_video = directory.join("cache").join("render-video.mp4");
    let final_temp = directory.join("cache").join("render-final.mp4");
    let output = directory.join(match config.mode {
        RemovalMode::TemporalRestore => "output-temporal.mp4",
        RemovalMode::AutoBest => "output-auto-best.mp4",
        _ => unreachable!("temporal renderer received an explicit non-temporal mode"),
    });
    let mut decoder = ffmpeg::open_raw_decoder(
        Path::new(&project.source.path),
        project.video.width,
        project.video.height,
    )?;
    let mut encoder = ffmpeg::open_raw_encoder(
        project.video.width,
        project.video.height,
        project.video.fps,
        &raw_video,
    )?;
    let before = config.temporal_window_before.clamp(1, 32) as u64;
    let after = config.temporal_window_after.clamp(1, 32) as u64;
    let max_candidates = config.max_temporal_candidates.clamp(2, 16) as usize;
    let settings = TemporalSettings {
        max_candidates,
        alignment_radius: 6,
        roi_padding: config.restoration_roi_padding.clamp(8, 96) as i32,
        artifact_threshold: config.artifact_threshold,
    };
    let mut buffer = VecDeque::new();
    let mut history = VecDeque::with_capacity(before as usize);
    let mut temporal_successes = 0u64;
    let mut temporal_fallbacks = 0u64;
    let mut inpaint_fallbacks = 0u64;
    let mut blur_fallbacks = 0u64;
    let mut next_frame = 0u64;
    let mut current = 0u64;
    fill_temporal_buffer(
        &mut decoder,
        &mut buffer,
        &mut next_frame,
        (after + 1).min(project.video.frame_count),
        project.video.width,
        project.video.height,
    )?;

    while current < project.video.frame_count {
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::OperationCancelled);
        }
        let desired_end = (current + after).min(project.video.frame_count.saturating_sub(1));
        fill_temporal_buffer(
            &mut decoder,
            &mut buffer,
            &mut next_frame,
            desired_end + 1,
            project.video.width,
            project.video.height,
        )?;
        let target = buffer.front().ok_or_else(|| {
            AppError::FfmpegFailed("Temporal frame buffer was unexpectedly empty.".to_string())
        })?;
        if target.index != current {
            return Err(AppError::FfmpegFailed(
                "Temporal frame buffer lost ordering.".to_string(),
            ));
        }
        let mut frame = target.image.clone();
        let tracking_frame = &project
            .tracking
            .as_ref()
            .expect("validated tracking above")
            .frames[current as usize];
        if tracking_frame.status != TrackingStatus::Occluded {
            if let Some((x0, y0, x1, y1)) = mask_bounds(
                frame.width(),
                frame.height(),
                &tracking_frame.bbox,
                config.mask_padding,
            ) {
                let mask_region = prepare_mask_region(mask, x1 - x0 + 1, y1 - y0 + 1, config);
                let tracking_frames = &project
                    .tracking
                    .as_ref()
                    .expect("validated tracking above")
                    .frames;
                let mut candidate_frames = history
                    .iter()
                    .chain(buffer.iter().skip(1))
                    .map(|item| CandidateFrame {
                        frame: item.index,
                        image: &item.image,
                        tracking: &tracking_frames[item.index as usize],
                    })
                    .collect::<Vec<_>>();
                candidate_frames.sort_by_key(|item| item.frame.abs_diff(current));
                let temporal_result = temporal::restore_frame(
                    &mut frame,
                    tracking_frame,
                    &mask_region,
                    x0,
                    y0,
                    &candidate_frames,
                    settings,
                );
                let _quality_snapshot = (
                    temporal_result.strategy,
                    temporal_result.artifact_score,
                    temporal_result.temporal_consistency_score,
                    temporal_result.valid_pixel_ratio,
                    temporal_result.fallback_used,
                );
                if temporal_result.success {
                    temporal_successes += 1;
                } else {
                    temporal_fallbacks += 1;
                    let source_frame = frame.clone();
                    for fallback_mode in fallback::fallback_modes(config.fallback_policy) {
                        match fallback_mode {
                            RemovalMode::Inpaint => {
                                inpaint_fallbacks += 1;
                                let mut inpainted = frame.clone();
                                apply_inpaint(&mut inpainted, &mask_region, x0, y0)?;
                                let score = artifact::spatial_artifact_score(
                                    &source_frame,
                                    &inpainted,
                                    &mask_region,
                                    x0,
                                    y0,
                                );
                                frame = inpainted;
                                if score <= settings.artifact_threshold {
                                    break;
                                }
                            }
                            RemovalMode::Blur => {
                                blur_fallbacks += 1;
                                apply_blur(&mut frame, &mask_region, x0, y0)?;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                // Keep untouched source frames for the configured look-behind window.
                history.push_back(BufferedFrame {
                    index: current,
                    image: target.image.clone(),
                });
                while history.len() > before as usize {
                    history.pop_front();
                }
            }
        }
        encoder.write_frame(frame.as_raw())?;
        on_progress(current + 1, project.video.frame_count);
        buffer.pop_front();
        current += 1;
    }
    decoder.finish()?;
    encoder.finish()?;
    eprintln!(
        "[restoration] mode={:?} temporal_successes={} temporal_fallbacks={} inpaint_fallbacks={} blur_fallbacks={}",
        config.mode,
        temporal_successes,
        temporal_fallbacks,
        inpaint_fallbacks,
        blur_fallbacks
    );
    mux_audio(&raw_video, Path::new(&project.source.path), &final_temp)?;
    std::fs::rename(&final_temp, &output).or_else(|_| {
        if output.exists() {
            std::fs::remove_file(&output)?;
        }
        std::fs::rename(&final_temp, &output)
    })?;
    Ok(output)
}

fn fill_temporal_buffer(
    decoder: &mut ffmpeg::RawVideoDecoder,
    buffer: &mut VecDeque<BufferedFrame>,
    next_frame: &mut u64,
    target_len: u64,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    while *next_frame < target_len {
        let bytes = decoder.next_frame()?.ok_or_else(|| {
            AppError::FfmpegFailed("Video ended before the expected frame count.".to_string())
        })?;
        let image = RgbImage::from_raw(width, height, bytes)
            .ok_or_else(|| AppError::FfmpegFailed("Invalid raw frame dimensions.".to_string()))?;
        buffer.push_back(BufferedFrame {
            index: *next_frame,
            image,
        });
        *next_frame += 1;
    }
    Ok(())
}

fn prepare_mask_region(
    mask: &GrayImage,
    width: i32,
    height: i32,
    config: &RemovalConfig,
) -> GrayImage {
    let mut region = imageops::resize(
        mask,
        width.max(1) as u32,
        height.max(1) as u32,
        imageops::FilterType::Triangle,
    );
    let feather = config.feather_radius.max(1) as f32;
    region = imageops::blur(&region, feather);
    region
}

fn load_replacement(config: &RemovalConfig) -> Result<RgbImage, AppError> {
    let path = config.replacement_path.as_ref().ok_or_else(|| {
        AppError::InvalidRequest("Choose a replacement PNG before rendering.".to_string())
    })?;
    let path = Path::new(path);
    if !path.is_file()
        || path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("png"))
            != Some(true)
    {
        return Err(AppError::InvalidRequest(
            "Replacement must be an existing PNG file.".to_string(),
        ));
    }
    image::open(path)
        .map(|image| image.to_rgb8())
        .map_err(|error| AppError::Io(format!("Unable to load replacement PNG: {error}")))
}

fn process_frame(
    frame: &mut RgbImage,
    tracking_frame: &TrackingFrame,
    mask: &GrayImage,
    config: &RemovalConfig,
    replacement: Option<&RgbImage>,
) -> Result<(), AppError> {
    if matches!(tracking_frame.status, TrackingStatus::Occluded) {
        return Ok(());
    }
    let bbox = &tracking_frame.bbox;
    let Some((x0, y0, x1, y1)) =
        mask_bounds(frame.width(), frame.height(), bbox, config.mask_padding)
    else {
        return Ok(());
    };
    let mut mask_region = imageops::resize(
        mask,
        (x1 - x0 + 1) as u32,
        (y1 - y0 + 1) as u32,
        imageops::FilterType::Triangle,
    );
    if config.feather_radius > 0 {
        mask_region = imageops::blur(&mask_region, config.feather_radius as f32);
    }
    match config.mode {
        RemovalMode::Replacement => apply_replacement(
            frame,
            bbox,
            &mask_region,
            x0,
            y0,
            replacement.ok_or_else(|| {
                AppError::InvalidRequest("Replacement PNG is missing.".to_string())
            })?,
            config,
        ),
        RemovalMode::Blur => apply_blur(frame, &mask_region, x0, y0),
        RemovalMode::Inpaint => apply_inpaint(frame, &mask_region, x0, y0),
        RemovalMode::TemporalRestore | RemovalMode::AutoBest => Err(AppError::InvalidRequest(
            "Temporal modes must use the temporal render pipeline.".to_string(),
        )),
    }
}

fn mask_bounds(
    frame_width: u32,
    frame_height: u32,
    bbox: &BoundingBox,
    padding: u32,
) -> Option<(i32, i32, i32, i32)> {
    if frame_width == 0
        || frame_height == 0
        || ![bbox.x, bbox.y, bbox.width, bbox.height]
            .into_iter()
            .all(f64::is_finite)
        || bbox.width <= 0.0
        || bbox.height <= 0.0
    {
        return None;
    }
    let padding = padding.min(i32::MAX as u32) as i32;
    let x0 = (bbox.x.floor() as i32 - padding).max(0);
    let y0 = (bbox.y.floor() as i32 - padding).max(0);
    let x1 = ((bbox.x + bbox.width).ceil() as i32 + padding).min(frame_width as i32 - 1);
    let y1 = ((bbox.y + bbox.height).ceil() as i32 + padding).min(frame_height as i32 - 1);
    (x1 >= x0 && y1 >= y0).then_some((x0, y0, x1, y1))
}

fn apply_replacement(
    frame: &mut RgbImage,
    bbox: &BoundingBox,
    mask: &GrayImage,
    x0: i32,
    y0: i32,
    replacement: &RgbImage,
    config: &RemovalConfig,
) -> Result<(), AppError> {
    let width = (bbox.width * config.replacement_scale.max(0.01))
        .round()
        .max(1.0) as u32;
    let height = (bbox.height * config.replacement_scale.max(0.01))
        .round()
        .max(1.0) as u32;
    let replacement = imageops::resize(replacement, width, height, imageops::FilterType::Lanczos3);
    let origin_x = bbox.x + config.replacement_offset_x;
    let origin_y = bbox.y + config.replacement_offset_y;
    for y in 0..replacement.height() {
        for x in 0..replacement.width() {
            let dx = (origin_x.round() as i32 + x as i32).clamp(0, frame.width() as i32 - 1);
            let dy = (origin_y.round() as i32 + y as i32).clamp(0, frame.height() as i32 - 1);
            let mx = (dx - x0).clamp(0, mask.width() as i32 - 1) as u32;
            let my = (dy - y0).clamp(0, mask.height() as i32 - 1) as u32;
            let alpha = f32::from(mask.get_pixel(mx, my)[0]) / 255.0
                * config.replacement_opacity.clamp(0.0, 1.0) as f32;
            blend(
                frame.get_pixel_mut(dx as u32, dy as u32),
                replacement.get_pixel(x, y),
                alpha,
            );
        }
    }
    Ok(())
}

fn apply_blur(frame: &mut RgbImage, mask: &GrayImage, x0: i32, y0: i32) -> Result<(), AppError> {
    // Blur only the tracked ROI; blurring the whole 1080x1920 frame per frame is unnecessary.
    let margin = 8i32;
    let roi_x0 = (x0 - margin).max(0);
    let roi_y0 = (y0 - margin).max(0);
    let roi_x1 = (x0 + mask.width() as i32 + margin).min(frame.width() as i32);
    let roi_y1 = (y0 + mask.height() as i32 + margin).min(frame.height() as i32);
    if roi_x1 <= roi_x0 || roi_y1 <= roi_y0 {
        return Ok(());
    }
    let roi_width = (roi_x1 - roi_x0) as u32;
    let roi_height = (roi_y1 - roi_y0) as u32;
    let source =
        imageops::crop_imm(frame, roi_x0 as u32, roi_y0 as u32, roi_width, roi_height).to_image();
    // The watermark is semi-transparent and comparatively large at source
    // resolution; a small radius only softens glyph edges while leaving the
    // lettering readable. Keep the blur local, but use enough spread to
    // suppress the glyph structure without affecting the surrounding frame.
    let blurred = imageops::blur(&source, 12.0);
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let alpha = f32::from(mask.get_pixel(x, y)[0]) / 255.0;
            let dx = x0 + x as i32;
            let dy = y0 + y as i32;
            if dx >= 0 && dy >= 0 && (dx as u32) < frame.width() && (dy as u32) < frame.height() {
                blend(
                    frame.get_pixel_mut(dx as u32, dy as u32),
                    blurred.get_pixel((dx - roi_x0) as u32, (dy - roi_y0) as u32),
                    alpha,
                );
            }
        }
    }
    Ok(())
}

fn apply_inpaint(frame: &mut RgbImage, mask: &GrayImage, x0: i32, y0: i32) -> Result<(), AppError> {
    let mut current = frame.clone();
    for _ in 0..10 {
        let previous = current.clone();
        for y in 0..mask.height() {
            for x in 0..mask.width() {
                let alpha = f32::from(mask.get_pixel(x, y)[0]) / 255.0;
                if alpha < 0.05 {
                    continue;
                }
                let dx = x0 + x as i32;
                let dy = y0 + y as i32;
                if dx < 1
                    || dy < 1
                    || dx as u32 + 1 >= frame.width()
                    || dy as u32 + 1 >= frame.height()
                {
                    continue;
                }
                let neighbors = [
                    previous.get_pixel(dx as u32 - 1, dy as u32),
                    previous.get_pixel(dx as u32 + 1, dy as u32),
                    previous.get_pixel(dx as u32, dy as u32 - 1),
                    previous.get_pixel(dx as u32, dy as u32 + 1),
                ];
                let average = Rgb([
                    (neighbors.iter().map(|p| p[0] as u32).sum::<u32>() / 4) as u8,
                    (neighbors.iter().map(|p| p[1] as u32).sum::<u32>() / 4) as u8,
                    (neighbors.iter().map(|p| p[2] as u32).sum::<u32>() / 4) as u8,
                ]);
                blend(current.get_pixel_mut(dx as u32, dy as u32), &average, alpha);
            }
        }
    }
    *frame = current;
    Ok(())
}

fn blend(destination: &mut Rgb<u8>, source: &Rgb<u8>, alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    for channel in 0..3 {
        destination[channel] = ((f32::from(destination[channel]) * (1.0 - alpha))
            + (f32::from(source[channel]) * alpha))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

fn mux_audio(video: &Path, source: &Path, output: &Path) -> Result<(), AppError> {
    let result = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(video)
        .args(["-i"])
        .arg(source)
        .args([
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
        ])
        .arg(output)
        .output()
        .map_err(|error| AppError::FfmpegFailed(error.to_string()))?;
    if result.status.success() {
        return Ok(());
    }
    let fallback = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(video)
        .args(["-i"])
        .arg(source)
        .args([
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ])
        .arg(output)
        .output()
        .map_err(|error| AppError::FfmpegFailed(error.to_string()))?;
    if fallback.status.success() {
        Ok(())
    } else {
        Err(AppError::FfmpegFailed(
            String::from_utf8_lossy(&fallback.stderr)
                .chars()
                .take(500)
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::mask_bounds;
    use crate::project::model::BoundingBox;

    #[test]
    fn mask_bounds_cover_the_full_bbox_and_padding() {
        let bbox = BoundingBox {
            x: 729.7565,
            y: 305.6702,
            width: 256.9653,
            height: 74.4828,
        };

        assert_eq!(
            mask_bounds(1080, 1920, &bbox, 4),
            Some((725, 301, 991, 385))
        );
    }

    #[test]
    fn mask_bounds_clip_to_frame_edges() {
        let bbox = BoundingBox {
            x: 0.5,
            y: 0.5,
            width: 8.0,
            height: 8.0,
        };

        assert_eq!(mask_bounds(10, 10, &bbox, 4), Some((0, 0, 9, 9)));
        assert_eq!(mask_bounds(10, 10, &bbox, 0), Some((0, 0, 9, 9)));
    }

}
