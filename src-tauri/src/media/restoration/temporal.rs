use super::alignment::{estimate_translation, AlignmentRegion};
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

pub struct StabilizationInput<'a> {
    pub restored: &'a mut RgbImage,
    pub source: &'a RgbImage,
    pub previous_processed: &'a RgbImage,
    pub target_bbox: &'a BoundingBox,
    pub previous_bbox: &'a BoundingBox,
    pub mask: &'a GrayImage,
    pub x0: i32,
    pub y0: i32,
    pub settings: TemporalSettings,
}

/// Selects usable neighboring frames in deterministic distance order. The
/// caller supplies only the configured before/after window; this layer owns
/// the safety filtering and candidate budget so every restoration entry point
/// applies the same rules.
pub fn select_candidate_frames<'a>(
    target_frame: u64,
    candidates: &[CandidateFrame<'a>],
    settings: TemporalSettings,
) -> Vec<CandidateFrame<'a>> {
    let mut selected = candidates
        .iter()
        .filter(|candidate| {
            !matches!(
                candidate.tracking.status,
                TrackingStatus::Occluded | TrackingStatus::AutoWeak | TrackingStatus::NeedReview
            )
        })
        .map(|candidate| CandidateFrame {
            frame: candidate.frame,
            image: candidate.image,
            tracking: candidate.tracking,
        })
        .collect::<Vec<_>>();
    selected.sort_by_key(|candidate| candidate.frame.abs_diff(target_frame));
    selected.truncate(settings.max_candidates.max(1));
    selected
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
    let candidates = select_candidate_frames(target_tracking.frame, candidates, settings);
    if candidates.is_empty() {
        return RestorationResult::failed(RestorationStrategy::TemporalRestore);
    }

    let mut aligned_candidates = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let bounds = bbox_bounds(&target_tracking.bbox, settings.roi_padding);
        let translation = estimate_translation(
            target,
            candidate.image,
            AlignmentRegion {
                target_bbox: &target_tracking.bbox,
                candidate_bbox: &candidate.tracking.bbox,
                x0: bounds.0,
                y0: bounds.1,
                width: bounds.2,
                height: bounds.3,
                radius: settings.alignment_radius,
            },
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
    let mask_padding_x = (target_tracking.bbox.x.floor() as i32 - x0).max(0);
    let mask_padding_y = (target_tracking.bbox.y.floor() as i32 - y0).max(0);
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
                    // Reject only pixels covered by the candidate's glyph
                    // mask. Rejecting its full bbox discards clean pixels in
                    // the whitespace between letters and leaves too little
                    // temporal coverage for a wide moving watermark.
                    || inside_candidate_mask(
                        cx,
                        cy,
                        &candidate.tracking.bbox,
                        mask,
                        mask_padding_x,
                        mask_padding_y,
                    )
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
            // Do not synthesize a pixel from candidates that disagree too
            // strongly. A missing consensus is safer than a ghosted blend;
            // the caller can then use its configured spatial fallback.
            if spread > 0.30 {
                continue;
            }
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

/// Applies a conservative patch-only temporal stabilizer using the previous
/// processed frame. It is deliberately gated by local alignment and color
/// agreement so ordinary scene motion is not smoothed across the whole frame.
/// Only pixels covered by the current restoration mask can change.
pub fn stabilize_frame(input: StabilizationInput<'_>) -> bool {
    if input.mask.width() == 0 || input.mask.height() == 0 {
        return false;
    }
    let bounds = bbox_bounds(input.target_bbox, input.settings.roi_padding.max(12));
    let translation = estimate_translation(
        input.source,
        input.previous_processed,
        AlignmentRegion {
            target_bbox: input.target_bbox,
            candidate_bbox: input.previous_bbox,
            x0: bounds.0,
            y0: bounds.1,
            width: bounds.2,
            height: bounds.3,
            radius: input.settings.alignment_radius,
        },
    );
    const MAX_ALIGNMENT_ERROR: f64 = 0.18;
    if translation.error > MAX_ALIGNMENT_ERROR {
        return false;
    }
    let weight = (0.18 * (1.0 - translation.error / MAX_ALIGNMENT_ERROR)).clamp(0.04, 0.18);
    let mut changed = false;
    for y in 0..input.mask.height() {
        for x in 0..input.mask.width() {
            let alpha = f32::from(input.mask.get_pixel(x, y)[0]) / 255.0;
            if alpha < 0.35 {
                continue;
            }
            let current_x = input.x0 + x as i32;
            let current_y = input.y0 + y as i32;
            let previous_x = current_x + translation.dx;
            let previous_y = current_y + translation.dy;
            if !in_bounds(input.restored, current_x, current_y)
                || !in_bounds(input.source, current_x, current_y)
                || !in_bounds(input.previous_processed, previous_x, previous_y)
            {
                continue;
            }
            let previous_pixel = input
                .previous_processed
                .get_pixel(previous_x as u32, previous_y as u32);
            let source_pixel = input.source.get_pixel(current_x as u32, current_y as u32);
            // A large disagreement usually means a cut, occlusion, or an
            // incorrect local alignment. Keep the current restoration then.
            if color_distance(previous_pixel, source_pixel) > 0.35 {
                continue;
            }
            blend(
                input
                    .restored
                    .get_pixel_mut(current_x as u32, current_y as u32),
                previous_pixel,
                weight as f32 * alpha,
            );
            changed = true;
        }
    }
    changed
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

fn inside_candidate_mask(
    x: i32,
    y: i32,
    bbox: &BoundingBox,
    mask: &GrayImage,
    padding_x: i32,
    padding_y: i32,
) -> bool {
    if mask.width() == 0
        || mask.height() == 0
        || !inside_bbox(x, y, bbox, f64::from(padding_x.max(padding_y)))
    {
        return false;
    }
    let left = bbox.x.floor() as i32 - padding_x;
    let top = bbox.y.floor() as i32 - padding_y;
    let right = (bbox.x + bbox.width).ceil() as i32 + padding_x;
    let bottom = (bbox.y + bbox.height).ceil() as i32 + padding_y;
    if x < left || x > right || y < top || y > bottom {
        return false;
    }
    let region_width = (right - left + 1).max(1) as u32;
    let region_height = (bottom - top + 1).max(1) as u32;
    let mask_x = if region_width <= 1 {
        0
    } else {
        (((x - left) as f64 / f64::from(region_width - 1))
            * f64::from(mask.width().saturating_sub(1)))
        .round() as u32
    };
    let mask_y = if region_height <= 1 {
        0
    } else {
        (((y - top) as f64 / f64::from(region_height - 1))
            * f64::from(mask.height().saturating_sub(1)))
        .round() as u32
    };
    mask.get_pixel(mask_x.min(mask.width() - 1), mask_y.min(mask.height() - 1))[0] >= 13
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
    use super::{
        inside_candidate_mask, mask_with_context, restore_frame, select_candidate_frames,
        stabilize_frame, CandidateFrame, StabilizationInput,
    };
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
    fn reconstructs_a_moving_watermark_over_a_synthetic_gradient() {
        let mut clean = RgbImage::new(48, 48);
        for y in 0..48 {
            for x in 0..48 {
                let base = 35_u8.saturating_add(((x * 5 + y * 3) % 160) as u8);
                clean.put_pixel(
                    x,
                    y,
                    Rgb([base, base.saturating_add(12), base.saturating_add(24)]),
                );
            }
        }

        let target_bbox = BoundingBox {
            x: 18.0,
            y: 16.0,
            width: 12.0,
            height: 8.0,
        };
        let mut target = clean.clone();
        for y in 16..24 {
            for x in 18..30 {
                let pixel = clean.get_pixel(x, y);
                target.put_pixel(x, y, Rgb([pixel[0] / 2, pixel[1] / 2, pixel[2] / 2]));
            }
        }

        let mut candidate_before = clean.clone();
        for y in 0..4 {
            for x in 0..8 {
                let pixel = clean.get_pixel(x, y);
                candidate_before.put_pixel(x, y, Rgb([pixel[0] / 2, pixel[1] / 2, pixel[2] / 2]));
            }
        }
        let mut candidate_after = clean.clone();
        for y in 36..40 {
            for x in 36..44 {
                let pixel = clean.get_pixel(x, y);
                candidate_after.put_pixel(x, y, Rgb([pixel[0] / 2, pixel[1] / 2, pixel[2] / 2]));
            }
        }

        let target_tracking = tracking(10, target_bbox);
        let candidate_before_tracking = tracking(
            9,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 8.0,
                height: 4.0,
            },
        );
        let candidate_after_tracking = tracking(
            11,
            BoundingBox {
                x: 36.0,
                y: 36.0,
                width: 8.0,
                height: 4.0,
            },
        );
        let mask = GrayImage::from_pixel(12, 8, Luma([255]));
        let outside_before = *target.get_pixel(0, 0);

        let result = restore_frame(
            &mut target,
            &target_tracking,
            &mask,
            18,
            16,
            &[
                CandidateFrame {
                    frame: 9,
                    image: &candidate_before,
                    tracking: &candidate_before_tracking,
                },
                CandidateFrame {
                    frame: 11,
                    image: &candidate_after,
                    tracking: &candidate_after_tracking,
                },
            ],
            super::super::model::TemporalSettings {
                max_candidates: 2,
                alignment_radius: 0,
                roi_padding: 4,
                artifact_threshold: 0.25,
            },
        );

        assert!(result.success, "{result:?}");
        assert_eq!(target.get_pixel(24, 20), clean.get_pixel(24, 20));
        assert_eq!(*target.get_pixel(0, 0), outside_before);
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

    #[test]
    fn candidate_selection_is_sorted_bounded_and_safe() {
        let image = RgbImage::from_pixel(8, 8, Rgb([1, 1, 1]));
        let mut weak = tracking(
            3,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        weak.status = TrackingStatus::AutoWeak;
        let good8 = tracking(
            8,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        let good6 = tracking(
            6,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        let good5 = tracking(
            5,
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },
        );
        let candidates = [
            CandidateFrame {
                frame: 8,
                image: &image,
                tracking: &good8,
            },
            CandidateFrame {
                frame: 3,
                image: &image,
                tracking: &weak,
            },
            CandidateFrame {
                frame: 6,
                image: &image,
                tracking: &good6,
            },
            CandidateFrame {
                frame: 5,
                image: &image,
                tracking: &good5,
            },
        ];
        let selected = select_candidate_frames(
            5,
            &candidates,
            super::super::model::TemporalSettings {
                max_candidates: 2,
                alignment_radius: 0,
                roi_padding: 2,
                artifact_threshold: 0.25,
            },
        );
        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.frame)
                .collect::<Vec<_>>(),
            vec![5, 6]
        );
    }

    #[test]
    fn candidate_eligibility_rejects_glyphs_not_the_full_bbox() {
        let mut mask = GrayImage::new(12, 8);
        mask.put_pixel(5, 3, Luma([255]));
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 8.0,
            height: 4.0,
        };

        assert!(inside_candidate_mask(13, 21, &bbox, &mask, 2, 2));
        assert!(!inside_candidate_mask(11, 21, &bbox, &mask, 2, 2));
        assert!(!inside_candidate_mask(30, 30, &bbox, &mask, 2, 2));
    }

    #[test]
    fn stabilization_changes_only_the_masked_patch() {
        let mut restored = RgbImage::from_pixel(24, 24, Rgb([100, 100, 100]));
        let source = restored.clone();
        let previous = RgbImage::from_pixel(24, 24, Rgb([140, 140, 140]));
        let mask = GrayImage::from_pixel(4, 4, Luma([255]));
        let bbox = BoundingBox {
            x: 10.0,
            y: 10.0,
            width: 4.0,
            height: 4.0,
        };
        let changed = stabilize_frame(StabilizationInput {
            restored: &mut restored,
            source: &source,
            previous_processed: &previous,
            target_bbox: &bbox,
            previous_bbox: &bbox,
            mask: &mask,
            x0: 10,
            y0: 10,
            settings: super::super::model::TemporalSettings {
                max_candidates: 2,
                alignment_radius: 0,
                roi_padding: 2,
                artifact_threshold: 0.25,
            },
        });
        assert!(changed);
        assert!(restored.get_pixel(11, 11)[0] > 100);
        assert_eq!(restored.get_pixel(0, 0), &Rgb([100, 100, 100]));
    }
}
