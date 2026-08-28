use super::alignment::estimate_translation;
use super::artifact::{
    accept_temporal_patch, artifact_score, spatial_artifact_score, temporal_consistency,
};
use super::model::{RestorationResult, RestorationStrategy, TemporalSettings};
use crate::project::model::{BoundingBox, TrackingFrame, TrackingStatus};
use image::{GrayImage, Rgb, RgbImage};

pub struct CandidateFrame<'a> {
    pub frame: u64,
    pub image: &'a RgbImage,
    pub tracking: &'a TrackingFrame,
}

#[derive(Clone, Copy)]
struct PixelSample {
    pixel: Rgb<u8>,
    alignment_error: f64,
    frame_distance: u64,
}

/// Reconstructs the tracked mask from nearby clean frames. Candidate pixels
/// are aligned locally, rejected if still inside a candidate watermark box,
/// and combined with a per-pixel consensus representative to resist ghosting.
pub fn restore_frame(
    target: &mut RgbImage,
    target_tracking: &TrackingFrame,
    mask: &GrayImage,
    x0: i32,
    y0: i32,
    candidates: &[CandidateFrame<'_>],
    settings: TemporalSettings,
) -> RestorationResult {
    let candidates = candidates
        .iter()
        .filter(|candidate| {
            !matches!(
                candidate.tracking.status,
                TrackingStatus::Occluded | TrackingStatus::AutoWeak | TrackingStatus::NeedReview
            )
        })
        .take(settings.max_candidates.max(1))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return RestorationResult::failed(RestorationStrategy::TemporalRestore);
    }

    let mut aligned_candidates = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let bounds = bbox_bounds(&target_tracking.bbox, settings.roi_padding);
        let translation = estimate_translation(
            target,
            candidate.image,
            bounds.0,
            bounds.1,
            bounds.2,
            bounds.3,
            settings.alignment_radius,
        );
        aligned_candidates.push((candidate, translation));
    }
    let best_alignment_error = aligned_candidates
        .iter()
        .map(|(_, translation)| translation.error)
        .fold(f64::INFINITY, f64::min);
    // A locally incorrect alignment can still have a low pixel spread when
    // the scene is repetitive. Reject those outliers before reconstruction.
    let aligned_candidates = aligned_candidates
        .into_iter()
        .filter(|(_, translation)| translation.error <= (best_alignment_error + 0.08).min(0.24))
        .collect::<Vec<_>>();
    if aligned_candidates.len() < 2 {
        return RestorationResult::failed(RestorationStrategy::TemporalRestore);
    }
    let alignment_error = aligned_candidates
        .iter()
        .map(|(_, translation)| translation.error)
        .sum::<f64>()
        / aligned_candidates.len() as f64;

    let mut updates = Vec::new();
    let mut masked_pixels = 0usize;
    let mut valid_pixels = 0usize;
    let mut total_spread = 0.0;
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            let alpha = f64::from(mask.get_pixel(x, y)[0]) / 255.0;
            if alpha < 0.05 {
                continue;
            }
            masked_pixels += 1;
            let tx = x0 + x as i32;
            let ty = y0 + y as i32;
            let mut samples = Vec::<PixelSample>::new();
            for (candidate, translation) in &aligned_candidates {
                let cx = tx + translation.dx;
                let cy = ty + translation.dy;
                if !in_bounds(candidate.image, cx, cy)
                    // Keep a conservative margin around the candidate bbox:
                    // antialiased watermark pixels can extend a few pixels
                    // past the tracked rectangle and would otherwise leak
                    // back into the reconstruction.
                    || inside_bbox(cx, cy, &candidate.tracking.bbox, 8.0)
                {
                    continue;
                }
                samples.push(PixelSample {
                    pixel: *candidate.image.get_pixel(cx as u32, cy as u32),
                    alignment_error: translation.error,
                    frame_distance: candidate.frame.abs_diff(target_tracking.frame),
                });
            }
            if samples.len() < 2 {
                continue;
            }
            // Select an actual candidate pixel from the largest consensus
            // rather than averaging channels from different frames. This
            // preserves texture and avoids colored ghost edges on motion.
            let representative = representative_pixel(&samples);
            let spread = samples
                .iter()
                .map(|sample| color_distance(&sample.pixel, &representative))
                .sum::<f64>()
                / samples.len() as f64;
            total_spread += spread;
            valid_pixels += 1;
            updates.push((tx, ty, representative, alpha));
        }
    }

    let valid_ratio = if masked_pixels == 0 {
        0.0
    } else {
        valid_pixels as f64 / masked_pixels as f64
    };
    let spread = if valid_pixels == 0 {
        1.0
    } else {
        (total_spread / valid_pixels as f64).clamp(0.0, 1.0)
    };
    let provisional_artifact = artifact_score(alignment_error, spread);
    let original = target.clone();
    let provisional_success = accept_temporal_patch(
        valid_ratio,
        provisional_artifact,
        settings.artifact_threshold,
    );
    if provisional_success {
        for (x, y, pixel, alpha) in updates {
            if in_bounds(target, x, y) {
                blend(
                    target.get_pixel_mut(x as u32, y as u32),
                    &pixel,
                    alpha as f32,
                );
            }
        }
    }
    let residual_artifact = if provisional_success {
        // Score against a small inactive context ring as well as the tracked
        // ROI. A full-coverage restoration mask otherwise has no interior
        // boundary, allowing a flat/dark temporal block to pass when the
        // tracked background is also low-luminance.
        let (score_mask, score_x0, score_y0) = mask_with_context(mask, x0, y0, 16);
        spatial_artifact_score(&original, target, &score_mask, score_x0, score_y0)
    } else {
        1.0
    };
    let artifact = provisional_artifact.max(residual_artifact);
    let success = provisional_success
        && accept_temporal_patch(valid_ratio, artifact, settings.artifact_threshold);
    if !success {
        *target = original;
    }
    RestorationResult {
        strategy: RestorationStrategy::TemporalRestore,
        success,
        artifact_score: artifact,
        temporal_consistency_score: temporal_consistency(valid_ratio, artifact),
        valid_pixel_ratio: valid_ratio,
        fallback_used: false,
    }
}

fn mask_with_context(mask: &GrayImage, x0: i32, y0: i32, margin: u32) -> (GrayImage, i32, i32) {
    let left = margin.min(x0.max(0) as u32);
    let top = margin.min(y0.max(0) as u32);
    let right = margin;
    let bottom = margin;
    let mut expanded = GrayImage::new(
        mask.width().saturating_add(left).saturating_add(right),
        mask.height().saturating_add(top).saturating_add(bottom),
    );
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            expanded.put_pixel(x + left, y + top, *mask.get_pixel(x, y));
        }
    }
    (expanded, x0 - left as i32, y0 - top as i32)
}

fn bbox_bounds(bbox: &BoundingBox, margin: i32) -> (i32, i32, u32, u32) {
    (
        (bbox.x.floor() as i32 - margin).max(0),
        (bbox.y.floor() as i32 - margin).max(0),
        (bbox.width.ceil() as u32).saturating_add((margin * 2) as u32),
        (bbox.height.ceil() as u32).saturating_add((margin * 2) as u32),
    )
}

fn inside_bbox(x: i32, y: i32, bbox: &BoundingBox, padding: f64) -> bool {
    x as f64 >= bbox.x - padding
        && x as f64 <= bbox.x + bbox.width + padding
        && y as f64 >= bbox.y - padding
        && y as f64 <= bbox.y + bbox.height + padding
}

fn in_bounds(image: &RgbImage, x: i32, y: i32) -> bool {
    x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height()
}

fn color_distance(a: &Rgb<u8>, b: &Rgb<u8>) -> f64 {
    (f64::from(a[0].abs_diff(b[0]))
        + f64::from(a[1].abs_diff(b[1]))
        + f64::from(a[2].abs_diff(b[2])))
        / (3.0 * 255.0)
}

fn representative_pixel(samples: &[PixelSample]) -> Rgb<u8> {
    samples
        .iter()
        .min_by(|left, right| {
            let left_score = sample_consensus_score(left, samples);
            let right_score = sample_consensus_score(right, samples);
            left_score
                .partial_cmp(&right_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|sample| sample.pixel)
        .unwrap_or(Rgb([0, 0, 0]))
}

fn sample_consensus_score(sample: &PixelSample, samples: &[PixelSample]) -> f64 {
    let consensus = samples
        .iter()
        .map(|other| color_distance(&sample.pixel, &other.pixel))
        .sum::<f64>();
    // Alignment quality is the primary tie-breaker. A small distance bias
    // prefers nearby frames without overriding a stronger pixel consensus.
    consensus + sample.alignment_error * 0.15 + (sample.frame_distance as f64 / 32.0) * 0.01
}

fn blend(destination: &mut Rgb<u8>, source: &Rgb<u8>, alpha: f32) {
    for channel in 0..3 {
        destination[channel] = (f32::from(destination[channel]) * (1.0 - alpha)
            + f32::from(source[channel]) * alpha)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::{mask_with_context, restore_frame, CandidateFrame};
    use crate::project::model::{
        BoundingBox, TrackingFrame, TrackingScores, TrackingSource, TrackingStatus,
    };
    use image::{GrayImage, Luma, Rgb, RgbImage};

    fn tracking(frame: u64, bbox: BoundingBox) -> TrackingFrame {
        TrackingFrame {
            frame,
            timestamp_seconds: frame as f64,
            bbox,
            confidence: 1.0,
            status: TrackingStatus::Interpolated,
            source: TrackingSource::Interpolated,
            locked: false,
            scores: TrackingScores {
                template: 1.0,
                highpass: 1.0,
                edge: 1.0,
                motion: 1.0,
                position: 1.0,
                size: 1.0,
                optical_flow: Some(1.0),
                forward_backward: Some(1.0),
                motion_smoothness: Some(1.0),
                match_margin: Some(1.0),
            },
        }
    }

    #[test]
    fn reconstructs_masked_pixels_from_a_clean_neighbor() {
        let mut target = RgbImage::from_pixel(24, 24, Rgb([100, 100, 100]));
        for y in 12..16 {
            for x in 12..16 {
                target.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        let candidate = RgbImage::from_pixel(24, 24, Rgb([100, 100, 100]));
        let target_tracking = tracking(
            1,
            BoundingBox {
                x: 12.0,
                y: 12.0,
                width: 4.0,
                height: 4.0,
            },
        );
        let candidate_tracking = tracking(
            0,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        let candidate_tracking_2 = tracking(
            2,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        let mut mask = GrayImage::new(4, 4);
        for pixel in mask.pixels_mut() {
            *pixel = Luma([255]);
        }
        let result = restore_frame(
            &mut target,
            &target_tracking,
            &mask,
            12,
            12,
            &[
                CandidateFrame {
                    frame: 0,
                    image: &candidate,
                    tracking: &candidate_tracking,
                },
                CandidateFrame {
                    frame: 2,
                    image: &candidate,
                    tracking: &candidate_tracking_2,
                },
            ],
            super::super::model::TemporalSettings {
                max_candidates: 2,
                alignment_radius: 0,
                roi_padding: 2,
                artifact_threshold: 0.25,
            },
        );

        assert!(result.success, "{result:?}");
        assert_eq!(target.get_pixel(14, 14), &Rgb([100, 100, 100]));
    }

    #[test]
    fn temporal_artifact_scoring_mask_includes_only_an_inactive_context_ring() {
        let mask = GrayImage::from_pixel(4, 3, Luma([255]));
        let (expanded, x0, y0) = mask_with_context(&mask, 8, 9, 2);

        assert_eq!((expanded.width(), expanded.height()), (8, 7));
        assert_eq!((x0, y0), (6, 7));
        assert_eq!(expanded.get_pixel(0, 0), &Luma([0]));
        assert_eq!(expanded.get_pixel(2, 2), &Luma([255]));
        assert_eq!(expanded.get_pixel(7, 6), &Luma([0]));
    }

    #[test]
    fn rejects_unusable_tracking_candidates() {
        let mut target = RgbImage::from_pixel(16, 16, Rgb([100, 100, 100]));
        let candidate_a = RgbImage::from_pixel(16, 16, Rgb([100, 100, 100]));
        let candidate_b = candidate_a.clone();
        let target_tracking = tracking(
            1,
            BoundingBox {
                x: 6.0,
                y: 6.0,
                width: 4.0,
                height: 4.0,
            },
        );
        let mut tracking_a = tracking(
            0,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        tracking_a.status = TrackingStatus::NeedReview;
        let mut tracking_b = tracking_a.clone();
        tracking_b.frame = 2;
        tracking_b.status = TrackingStatus::AutoWeak;
        let mask = GrayImage::from_pixel(4, 4, Luma([255]));

        let result = restore_frame(
            &mut target,
            &target_tracking,
            &mask,
            6,
            6,
            &[
                CandidateFrame {
                    frame: 0,
                    image: &candidate_a,
                    tracking: &tracking_a,
                },
                CandidateFrame {
                    frame: 2,
                    image: &candidate_b,
                    tracking: &tracking_b,
                },
            ],
            super::super::model::TemporalSettings {
                max_candidates: 2,
                alignment_radius: 0,
                roi_padding: 2,
                artifact_threshold: 0.25,
            },
        );

        assert!(!result.success);
    }

    #[test]
    fn consensus_selects_an_actual_majority_pixel() {
        let samples = [
            super::PixelSample {
                pixel: Rgb([100, 100, 100]),
                alignment_error: 0.1,
                frame_distance: 1,
            },
            super::PixelSample {
                pixel: Rgb([100, 100, 100]),
                alignment_error: 0.2,
                frame_distance: 2,
            },
            super::PixelSample {
                pixel: Rgb([220, 220, 220]),
                alignment_error: 0.0,
                frame_distance: 1,
            },
        ];

        assert_eq!(super::representative_pixel(&samples), Rgb([100, 100, 100]));
    }
}
