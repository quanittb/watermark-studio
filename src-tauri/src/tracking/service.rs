use crate::error::AppError;
use crate::media::ffmpeg;
use crate::project::model::{
    BoundingBox, ManualAnchor, ProblemRange, Project, TrackingConfig, TrackingData, TrackingFrame,
    TrackingSource, TrackingStatus,
};
use crate::project::service;
use crate::tracking::bidirectional::{fuse_tracks, initial_anchor, make_seed};
use crate::tracking::matcher::TemplateBank;
use crate::tracking::tracker::track_direction;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

pub fn analyze_project<F>(
    app_data_dir: &Path,
    project_id: &str,
    cancel: &AtomicBool,
    mut progress: F,
) -> Result<Project, AppError>
where
    F: FnMut(&str, u64, u64) -> bool + Send + 'static,
{
    let project = service::load_project(app_data_dir, project_id)?;
    let config = normalized_config(
        project
            .tracking
            .as_ref()
            .map(|tracking| tracking.config.clone())
            .unwrap_or_default(),
    );
    let directory = service::project_directory(app_data_dir, project_id)?;
    let (frames, bank, analysis_width, analysis_height) = prepare(&project, &directory, &config)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(AppError::OperationCancelled);
    }
    let anchors = normalized_anchors(&project)?;
    let initial = anchors
        .first()
        .expect("normalized_anchors always returns one anchor");
    let initial_seed = make_seed(
        initial,
        project.video.width,
        project.video.height,
        analysis_width,
        analysis_height,
        project.watermark.template_padding,
    );
    // Always keep a complete forward/backward baseline so agreement is available
    // outside manually constrained segments as well.
    let baseline_forward = track_direction(
        &frames,
        &bank,
        &initial_seed,
        (frames.len() - 1) as u64,
        1,
        &config,
        project.video.fps,
        |current| {
            progress("tracking_forward", current, frames.len() as u64)
                && !cancel.load(Ordering::Relaxed)
        },
    );
    let baseline_backward = track_direction(
        &frames,
        &bank,
        &initial_seed,
        0,
        -1,
        &config,
        project.video.fps,
        |current| {
            progress("tracking_backward", current, frames.len() as u64)
                && !cancel.load(Ordering::Relaxed)
        },
    );
    let baseline = fuse_tracks(
        &baseline_forward,
        &baseline_backward,
        project.video.width,
        project.video.height,
        analysis_width,
        analysis_height,
        project.video.fps,
        project.watermark.template_padding,
        &config,
    );
    let mut tracked = baseline.into_iter().map(Some).collect::<Vec<_>>();
    if cancel.load(Ordering::Relaxed) {
        return Err(AppError::OperationCancelled);
    }
    for pair in anchors.windows(2) {
        let left = &pair[0];
        let right = &pair[1];
        let left_seed = make_seed(
            left,
            project.video.width,
            project.video.height,
            analysis_width,
            analysis_height,
            project.watermark.template_padding,
        );
        let right_seed = make_seed(
            right,
            project.video.width,
            project.video.height,
            analysis_width,
            analysis_height,
            project.watermark.template_padding,
        );
        let forward = track_direction(
            &frames,
            &bank,
            &left_seed,
            right.frame,
            1,
            &config,
            project.video.fps,
            |current| {
                progress("tracking_forward", current, frames.len() as u64)
                    && !cancel.load(Ordering::Relaxed)
            },
        );
        let backward = track_direction(
            &frames,
            &bank,
            &right_seed,
            left.frame,
            -1,
            &config,
            project.video.fps,
            |current| {
                progress("tracking_backward", current, frames.len() as u64)
                    && !cancel.load(Ordering::Relaxed)
            },
        );
        let fused = fuse_tracks(
            &forward,
            &backward,
            project.video.width,
            project.video.height,
            analysis_width,
            analysis_height,
            project.video.fps,
            project.watermark.template_padding,
            &config,
        );
        merge_range(&mut tracked, &fused, left.frame, right.frame);
        if cancel.load(Ordering::Relaxed) {
            return Err(AppError::OperationCancelled);
        }
    }

    let mut output = tracked
        .into_iter()
        .enumerate()
        .map(|(index, frame)| frame.unwrap_or_else(|| fallback(index as u64, &project)))
        .collect::<Vec<_>>();
    apply_manual_anchors(&mut output, &project);
    apply_motion_sanity(&mut output, config.max_frame_displacement);
    mark_leading_occluded_range(&mut output, &anchors);
    apply_manual_anchors(&mut output, &project);
    save_tracking(project, directory, config, output)
}

pub fn retrack_project<F>(
    app_data_dir: &Path,
    project_id: &str,
    frame: u64,
    cancel: &AtomicBool,
    mut progress: F,
) -> Result<Project, AppError>
where
    F: FnMut(&str, u64, u64) -> bool + Send + 'static,
{
    let mut project = service::load_project(app_data_dir, project_id)?;
    if frame >= project.video.frame_count {
        return Err(AppError::InvalidRequest(
            "Frame is outside the video.".to_string(),
        ));
    }
    if project.tracking.is_none() {
        return analyze_project(app_data_dir, project_id, cancel, progress);
    }
    let config = normalized_config(
        project
            .tracking
            .as_ref()
            .map(|value| value.config.clone())
            .unwrap_or_default(),
    );
    let directory = service::project_directory(app_data_dir, project_id)?;
    let (frames, bank, analysis_width, analysis_height) = prepare(&project, &directory, &config)?;
    let anchors = normalized_anchors(&project)?;
    let (left, right) = tracking_segment_for_frame(&anchors, frame);
    let end_frame = right
        .as_ref()
        .map(|anchor| anchor.frame)
        .unwrap_or((frames.len() - 1) as u64);
    let left_seed = make_seed(
        &left,
        project.video.width,
        project.video.height,
        analysis_width,
        analysis_height,
        project.watermark.template_padding,
    );
    let forward = track_direction(
        &frames,
        &bank,
        &left_seed,
        end_frame,
        1,
        &config,
        project.video.fps,
        |current| {
            progress("tracking_forward", current, frames.len() as u64)
                && !cancel.load(Ordering::Relaxed)
        },
    );
    let backward = right.as_ref().map(|anchor| {
        let seed = make_seed(
            anchor,
            project.video.width,
            project.video.height,
            analysis_width,
            analysis_height,
            project.watermark.template_padding,
        );
        track_direction(
            &frames,
            &bank,
            &seed,
            left.frame,
            -1,
            &config,
            project.video.fps,
            |current| {
                progress("tracking_backward", current, frames.len() as u64)
                    && !cancel.load(Ordering::Relaxed)
            },
        )
    });
    if cancel.load(Ordering::Relaxed) {
        return Err(AppError::OperationCancelled);
    }
    let fused = fuse_tracks(
        &forward,
        backward.as_deref().unwrap_or(&[]),
        project.video.width,
        project.video.height,
        analysis_width,
        analysis_height,
        project.video.fps,
        project.watermark.template_padding,
        &config,
    );
    let mut result = project.tracking.take().expect("tracking exists");
    merge_existing_range(&mut result.frames, &fused, left.frame, end_frame);
    apply_manual_anchors(&mut result.frames, &project);
    apply_motion_sanity(&mut result.frames, config.max_frame_displacement);
    mark_leading_occluded_range(&mut result.frames, &anchors);
    apply_manual_anchors(&mut result.frames, &project);
    result.problem_ranges = group_problem_ranges(&result.frames);
    project.tracking = Some(result);
    service::save_project_atomic(&directory, &project)?;
    Ok(project)
}

pub fn group_problem_ranges(frames: &[TrackingFrame]) -> Vec<ProblemRange> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (index, frame) in frames.iter().enumerate() {
        let problem = matches!(
            frame.status,
            TrackingStatus::NeedReview
                | TrackingStatus::AutoWeak
                // Interpolation is a geometric estimate only. It must remain
                // review-gated because a moving watermark can follow a curved
                // path between two anchors.
                | TrackingStatus::Interpolated
        );
        if problem && start.is_none() {
            start = Some(index);
        }
        if (!problem || index + 1 == frames.len()) && start.is_some() {
            let begin = start.take().expect("range start exists");
            let end = if problem && index + 1 == frames.len() {
                index
            } else {
                index - 1
            };
            let worst = (begin..=end)
                .min_by(|a, b| frames[*a].confidence.total_cmp(&frames[*b].confidence))
                .unwrap_or(begin);
            ranges.push(ProblemRange {
                start_frame: begin as u64,
                end_frame: end as u64,
                worst_frame: worst as u64,
                min_confidence: frames[worst].confidence,
            });
        }
    }
    ranges
}

pub fn interpolate_between_anchors(
    frames: &mut [TrackingFrame],
    left: &ManualAnchor,
    right: &ManualAnchor,
    start: u64,
    end: u64,
) -> Result<(), AppError> {
    if right.frame <= left.frame || start < left.frame || end > right.frame || start > end {
        return Err(AppError::InvalidRequest(
            "Invalid interpolation range.".to_string(),
        ));
    }
    let span = (right.frame - left.frame) as f64;
    for frame_number in start..=end {
        if frame_number == left.frame || frame_number == right.frame {
            continue;
        }
        if let Some(frame) = frames.get_mut(frame_number as usize) {
            let ratio = (frame_number - left.frame) as f64 / span;
            frame.bbox = BoundingBox {
                x: left.bbox.x + (right.bbox.x - left.bbox.x) * ratio,
                y: left.bbox.y + (right.bbox.y - left.bbox.y) * ratio,
                width: left.bbox.width + (right.bbox.width - left.bbox.width) * ratio,
                height: left.bbox.height + (right.bbox.height - left.bbox.height) * ratio,
            };
            // Do not present an interpolated position as image-validated
            // confidence. The bbox is retained for inspection, but the frame
            // must be manually reviewed before it can be rendered.
            frame.confidence = 0.0;
            frame.status = TrackingStatus::Interpolated;
            frame.source = TrackingSource::Interpolated;
            frame.locked = false;
            frame.scores.motion_smoothness = Some(1.0);
        }
    }
    Ok(())
}

fn prepare(
    project: &Project,
    directory: &Path,
    config: &TrackingConfig,
) -> Result<(Vec<crate::media::ffmpeg::GrayFrame>, TemplateBank, u32, u32), AppError> {
    let (analysis_width, analysis_height) = ffmpeg::analysis_dimensions(
        project.video.width,
        project.video.height,
        config.analysis_long_edge,
    );
    let frames = ffmpeg::read_analysis_frames(
        Path::new(&project.source.path),
        project.video.width,
        project.video.height,
        config.analysis_long_edge,
    )?;
    if frames.len() != project.video.frame_count as usize {
        return Err(AppError::FfmpegFailed(format!(
            "Decoded {} frames but metadata reports {}.",
            frames.len(),
            project.video.frame_count
        )));
    }
    let anchor = project.watermark.anchor.as_ref().ok_or_else(|| {
        AppError::InvalidRequest("Save a watermark anchor before tracking.".to_string())
    })?;
    let padded = crate::tracking::bidirectional::expand_bbox(
        &anchor.bbox,
        f64::from(project.watermark.template_padding),
    );
    let template_width = (padded.width * f64::from(analysis_width) / f64::from(project.video.width))
        .round()
        .max(8.0) as u32;
    let template_height = (padded.height * f64::from(analysis_height)
        / f64::from(project.video.height))
    .round()
    .max(8.0) as u32;
    let template_rel = project
        .watermark
        .templates
        .as_ref()
        .ok_or_else(|| {
            AppError::InvalidRequest("Save a watermark anchor before tracking.".to_string())
        })?
        .original
        .clone();
    let bank = TemplateBank::from_path(
        &directory.join(template_rel),
        template_width,
        template_height,
    )?;
    Ok((frames, bank, analysis_width, analysis_height))
}

fn normalized_config(mut config: TrackingConfig) -> TrackingConfig {
    // Projects created by the earlier prototype used permissive thresholds and
    // would otherwise keep accepting ambiguous matches after an upgrade.
    if config.accept_threshold < 0.60 || config.weak_threshold < 0.50 {
        let defaults = TrackingConfig::default();
        config.accept_threshold = defaults.accept_threshold;
        config.weak_threshold = defaults.weak_threshold;
        config.global_search_threshold = defaults.global_search_threshold;
        config.max_frame_displacement = defaults.max_frame_displacement;
    }
    config
}

fn normalized_anchors(project: &Project) -> Result<Vec<ManualAnchor>, AppError> {
    let anchor = project.watermark.anchor.as_ref().ok_or_else(|| {
        AppError::InvalidRequest("Save a watermark anchor before tracking.".to_string())
    })?;
    let mut anchors = project.anchors.clone();
    if !anchors.iter().any(|item| item.frame == anchor.frame) {
        anchors.push(initial_anchor(
            anchor.frame,
            anchor.timestamp_seconds,
            anchor.bbox.clone(),
        ));
    }
    anchors.sort_by_key(|item| item.frame);
    anchors.dedup_by_key(|item| item.frame);
    Ok(anchors)
}

fn tracking_segment_for_frame(
    anchors: &[ManualAnchor],
    frame: u64,
) -> (ManualAnchor, Option<ManualAnchor>) {
    let current_index = anchors.iter().position(|anchor| anchor.frame == frame);
    if let Some(index) = current_index {
        if index > 0 {
            return (anchors[index - 1].clone(), Some(anchors[index].clone()));
        }
        return (anchors[index].clone(), anchors.get(index + 1).cloned());
    }

    let left_index = anchors
        .iter()
        .rposition(|anchor| anchor.frame < frame)
        .unwrap_or(0);
    (
        anchors[left_index].clone(),
        anchors.get(left_index + 1).cloned(),
    )
}

fn merge_range(
    target: &mut [Option<TrackingFrame>],
    source: &[TrackingFrame],
    start: u64,
    end: u64,
) {
    for index in start..=end.min(source.len().saturating_sub(1) as u64) {
        target[index as usize] = Some(source[index as usize].clone());
    }
}

fn merge_existing_range(
    target: &mut [TrackingFrame],
    source: &[TrackingFrame],
    start: u64,
    end: u64,
) {
    for index in start..=end.min(source.len().saturating_sub(1) as u64) {
        target[index as usize] = source[index as usize].clone();
    }
}

fn apply_manual_anchors(frames: &mut [TrackingFrame], project: &Project) {
    for anchor in &project.anchors {
        if let Some(frame) = frames.get_mut(anchor.frame as usize) {
            frame.bbox = anchor.bbox.clone();
            frame.timestamp_seconds = anchor.timestamp_seconds;
            frame.confidence = 1.0;
            frame.status = TrackingStatus::Manual;
            frame.source = TrackingSource::Manual;
            frame.locked = anchor.locked;
        }
    }
}

fn apply_motion_sanity(frames: &mut [TrackingFrame], max_displacement: f64) {
    if frames.len() < 3 || !max_displacement.is_finite() || max_displacement <= 0.0 {
        return;
    }
    for index in 1..frames.len() - 1 {
        if frames[index].locked {
            continue;
        }
        let previous = center(&frames[index - 1].bbox);
        let current = center(&frames[index].bbox);
        let next = center(&frames[index + 1].bbox);
        let incoming = distance(previous, current);
        let outgoing = distance(current, next);
        let surrounding = distance(previous, next);
        if incoming > max_displacement
            && outgoing > max_displacement
            && surrounding < max_displacement * 1.5
        {
            frames[index].confidence = frames[index].confidence.min(0.25);
            frames[index].status = TrackingStatus::NeedReview;
            frames[index].scores.motion_smoothness = Some(0.0);
        }
    }
}

fn mark_leading_occluded_range(frames: &mut [TrackingFrame], anchors: &[ManualAnchor]) {
    let Some(first_anchor) = anchors.first() else {
        return;
    };
    let end = (first_anchor.frame as usize).min(frames.len());
    if end == 0
        || !frames[..end].iter().all(|frame| {
            matches!(
                frame.status,
                TrackingStatus::AutoWeak | TrackingStatus::NeedReview
            )
        })
    {
        return;
    }

    // A fully covered leading section cannot be safely corrected from pixels.
    // Preserve its measured bbox for timeline inspection, but make rendering a
    // strict no-op and keep the section out of the unresolved problem list.
    for frame in &mut frames[..end] {
        frame.status = TrackingStatus::Occluded;
        frame.source = TrackingSource::Occluded;
        frame.confidence = 0.0;
        frame.locked = false;
        frame.scores.motion_smoothness = Some(0.0);
    }
}

fn center(bbox: &BoundingBox) -> (f64, f64) {
    (bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn save_tracking(
    mut project: Project,
    directory: PathBuf,
    config: TrackingConfig,
    mut frames: Vec<TrackingFrame>,
) -> Result<Project, AppError> {
    for (index, frame) in frames.iter_mut().enumerate() {
        frame.frame = index as u64;
        frame.timestamp_seconds = index as f64 / project.video.fps.max(0.000_001);
    }
    let problem_ranges = group_problem_ranges(&frames);
    project.tracking = Some(TrackingData {
        config,
        frames,
        problem_ranges,
        analyzed_at: Some(format_timestamp()),
    });
    service::save_project_atomic(&directory, &project)?;
    Ok(project)
}

fn fallback(frame: u64, project: &Project) -> TrackingFrame {
    TrackingFrame {
        frame,
        timestamp_seconds: frame as f64 / project.video.fps.max(0.000_001),
        bbox: BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
        },
        confidence: 0.0,
        status: TrackingStatus::NeedReview,
        source: TrackingSource::Interpolated,
        locked: false,
        scores: crate::project::model::TrackingScores {
            template: 0.0,
            highpass: 0.0,
            edge: 0.0,
            motion: 0.0,
            position: 0.0,
            size: 0.0,
            optical_flow: None,
            forward_backward: None,
            motion_smoothness: None,
            match_margin: None,
        },
    }
}

fn format_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::{group_problem_ranges, mark_leading_occluded_range};
    use crate::project::model::{
        AnchorType, BoundingBox, ManualAnchor, TrackingFrame, TrackingScores, TrackingSource,
        TrackingStatus,
    };

    fn frame(index: u64, status: TrackingStatus, confidence: f64) -> TrackingFrame {
        TrackingFrame {
            frame: index,
            timestamp_seconds: index as f64,
            bbox: BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 20.0,
                height: 10.0,
            },
            confidence,
            status,
            source: TrackingSource::Forward,
            locked: false,
            scores: TrackingScores {
                template: confidence,
                highpass: confidence,
                edge: confidence,
                motion: confidence,
                position: confidence,
                size: 1.0,
                optical_flow: None,
                forward_backward: None,
                motion_smoothness: None,
                match_margin: None,
            },
        }
    }

    #[test]
    fn groups_adjacent_weak_frames_and_selects_worst_frame() {
        let frames = vec![
            frame(0, TrackingStatus::AutoGood, 0.9),
            frame(1, TrackingStatus::AutoWeak, 0.5),
            frame(2, TrackingStatus::NeedReview, 0.2),
            frame(3, TrackingStatus::AutoGood, 0.9),
            frame(4, TrackingStatus::NeedReview, 0.3),
        ];
        let ranges = group_problem_ranges(&frames);
        assert_eq!(ranges.len(), 2);
        assert_eq!(
            (
                ranges[0].start_frame,
                ranges[0].end_frame,
                ranges[0].worst_frame
            ),
            (1, 2, 2)
        );
        assert_eq!(
            (
                ranges[1].start_frame,
                ranges[1].end_frame,
                ranges[1].worst_frame
            ),
            (4, 4, 4)
        );
    }

    #[test]
    fn treats_interpolated_frames_as_review_required() {
        let frames = vec![
            frame(0, TrackingStatus::Manual, 1.0),
            frame(1, TrackingStatus::Interpolated, 0.0),
            frame(2, TrackingStatus::AutoGood, 0.9),
        ];

        let ranges = group_problem_ranges(&frames);

        assert_eq!(ranges.len(), 1);
        assert_eq!(
            (
                ranges[0].start_frame,
                ranges[0].end_frame,
                ranges[0].worst_frame
            ),
            (1, 1, 1)
        );
    }

    #[test]
    fn interpolates_bbox_between_manual_anchors() {
        let left = crate::project::model::ManualAnchor {
            frame: 0,
            timestamp_seconds: 0.0,
            bbox: BoundingBox {
                x: 0.0,
                y: 10.0,
                width: 20.0,
                height: 10.0,
            },
            anchor_type: crate::project::model::AnchorType::Manual,
            locked: true,
        };
        let right = crate::project::model::ManualAnchor {
            frame: 10,
            timestamp_seconds: 1.0,
            bbox: BoundingBox {
                x: 100.0,
                y: 30.0,
                width: 40.0,
                height: 20.0,
            },
            anchor_type: crate::project::model::AnchorType::Manual,
            locked: true,
        };
        let mut frames = (0..=10)
            .map(|index| frame(index, TrackingStatus::NeedReview, 0.1))
            .collect::<Vec<_>>();
        super::interpolate_between_anchors(&mut frames, &left, &right, 1, 9)
            .expect("range should interpolate");
        assert_eq!(frames[5].bbox.x, 50.0);
        assert_eq!(frames[5].status, TrackingStatus::Interpolated);
    }

    #[test]
    fn marks_an_unresolved_leading_range_as_occluded() {
        let mut frames = vec![
            frame(0, TrackingStatus::NeedReview, 0.2),
            frame(1, TrackingStatus::AutoWeak, 0.4),
            frame(2, TrackingStatus::NeedReview, 0.3),
            frame(3, TrackingStatus::Manual, 1.0),
        ];
        let anchors = vec![ManualAnchor {
            frame: 3,
            timestamp_seconds: 3.0,
            bbox: frames[3].bbox.clone(),
            anchor_type: AnchorType::Manual,
            locked: true,
        }];

        mark_leading_occluded_range(&mut frames, &anchors);

        assert!(frames[..3]
            .iter()
            .all(|item| item.status == TrackingStatus::Occluded));
        assert!(frames[..3]
            .iter()
            .all(|item| item.source == TrackingSource::Occluded));
        assert_eq!(frames[3].status, TrackingStatus::Manual);
        assert!(group_problem_ranges(&frames).is_empty());
    }

    #[test]
    fn retracking_a_manual_anchor_uses_the_previous_anchor_as_left_bound() {
        let anchors = vec![
            ManualAnchor {
                frame: 10,
                timestamp_seconds: 10.0,
                bbox: BoundingBox {
                    x: 10.0,
                    y: 10.0,
                    width: 20.0,
                    height: 10.0,
                },
                anchor_type: AnchorType::Initial,
                locked: true,
            },
            ManualAnchor {
                frame: 20,
                timestamp_seconds: 20.0,
                bbox: BoundingBox {
                    x: 20.0,
                    y: 20.0,
                    width: 20.0,
                    height: 10.0,
                },
                anchor_type: AnchorType::Manual,
                locked: true,
            },
        ];

        let (left, right) = super::tracking_segment_for_frame(&anchors, 20);

        assert_eq!(left.frame, 10);
        assert_eq!(
            right
                .expect("manual anchor should have a right bound")
                .frame,
            20
        );
    }
}
