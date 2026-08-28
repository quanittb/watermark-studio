use crate::media::ffmpeg::GrayFrame;
use crate::project::model::{BoundingBox, TrackingConfig, TrackingFrame, TrackingSource};
use crate::tracking::confidence::{fuse_scores, status_for};
use crate::tracking::matcher::{find_best, frame_features, TemplateBank};
use crate::tracking::motion::{estimate_pyramidal_lk, MotionState};

#[allow(clippy::too_many_arguments)]
pub fn track_direction<F>(
    frames: &[GrayFrame],
    bank: &TemplateBank,
    seed: &TrackingFrame,
    end_frame: u64,
    direction: i32,
    config: &TrackingConfig,
    fps: f64,
    mut on_progress: F,
) -> Vec<Option<TrackingFrame>>
where
    F: FnMut(u64) -> bool,
{
    let mut results = vec![None; frames.len()];
    if seed.frame as usize >= frames.len() || direction == 0 {
        return results;
    }
    results[seed.frame as usize] = Some(seed.clone());
    let mut previous = seed.clone();
    let mut motion = MotionState::default();
    let mut frame_number = seed.frame as i64;
    while frame_number != end_frame as i64 {
        let next_number = frame_number + i64::from(direction);
        if next_number < 0 || next_number >= frames.len() as i64 {
            break;
        }
        let previous_frame = &frames[frame_number as usize];
        let current_frame = &frames[next_number as usize];
        let flow = estimate_pyramidal_lk(
            previous_frame,
            current_frame,
            &previous.bbox,
            config.optical_flow_radius,
        );
        if let Some((dx, dy, _)) = flow {
            motion.velocity_x = motion.velocity_x * 0.65 + dx * 0.35;
            motion.velocity_y = motion.velocity_y * 0.65 + dy * 0.35;
        }
        // Optical flow is measured in traversal direction, including backward runs.
        let predicted = BoundingBox {
            x: previous.bbox.x + motion.velocity_x,
            y: previous.bbox.y + motion.velocity_y,
            width: previous.bbox.width,
            height: previous.bbox.height,
        };
        let features = frame_features(current_frame);
        let local = find_best(
            current_frame,
            &features,
            bank,
            &predicted,
            config.local_search_radius.max(8),
            false,
        );
        let mut candidate = local;
        if candidate
            .as_ref()
            .map(|(_, scores)| scores.template * 0.35 + scores.highpass * 0.4 + scores.edge * 0.25)
            .unwrap_or(0.0)
            < config.global_search_threshold
        {
            if let Some(global) = find_best(
                current_frame,
                &features,
                bank,
                &predicted,
                config.local_search_radius,
                true,
            ) {
                let global_score =
                    global.1.template * 0.35 + global.1.highpass * 0.4 + global.1.edge * 0.25;
                let local_score = candidate
                    .as_ref()
                    .map(|(_, scores)| {
                        scores.template * 0.35 + scores.highpass * 0.4 + scores.edge * 0.25
                    })
                    .unwrap_or(0.0);
                if global_score > local_score {
                    candidate = Some(global);
                }
            }
        }
        let next = candidate.map(|(bbox, mut scores)| {
            scores.optical_flow = flow.map(|(_, _, confidence)| confidence);
            let expected_x = previous.bbox.x + motion.velocity_x;
            let expected_y = previous.bbox.y + motion.velocity_y;
            let jump =
                ((bbox.x - previous.bbox.x).powi(2) + (bbox.y - previous.bbox.y).powi(2)).sqrt();
            let prediction_error =
                ((bbox.x - expected_x).abs() + (bbox.y - expected_y).abs()) / 24.0;
            scores.motion_smoothness = Some((1.0 - prediction_error).clamp(0.0, 1.0));
            let mut confidence = fuse_scores(&scores);
            // A weak match is evidence for review, not a new tracking state. Keep
            // its measured bbox for the UI, but do not let it move the trajectory.
            // Penalize large jumps here as well so a global false positive cannot
            // become the next frame's prediction.
            if jump > 24.0 {
                confidence = (confidence - ((jump - 24.0) / 72.0).clamp(0.0, 0.25)).max(0.0);
            }
            if scores.match_margin.unwrap_or(0.0) < 0.03 {
                confidence = confidence.min(config.weak_threshold - 0.01);
            }
            TrackingFrame {
                frame: next_number as u64,
                timestamp_seconds: next_number as f64 / fps.max(0.000_001),
                bbox,
                confidence,
                status: status_for(confidence, config),
                source: if direction > 0 {
                    TrackingSource::Forward
                } else {
                    TrackingSource::Backward
                },
                locked: false,
                scores,
            }
        });
        if let Some(frame) = next {
            // Only AUTO_GOOD frames are allowed to update the state used for the
            // final output without independent reverse-track validation. A
            // locally coherent image match may still advance the internal
            // trajectory; bidirectional fusion decides whether it is safe to
            // certify or must remain review-gated.
            if can_advance_trajectory(&frame, config) {
                previous = frame.clone();
            }
            results[next_number as usize] = Some(frame);
        }
        if !on_progress(next_number as u64) {
            break;
        }
        frame_number = next_number;
    }
    results
}

fn can_advance_trajectory(frame: &TrackingFrame, config: &TrackingConfig) -> bool {
    let image_score =
        frame.scores.template * 0.35 + frame.scores.highpass * 0.40 + frame.scores.edge * 0.25;
    image_score >= config.weak_threshold
        && frame.scores.match_margin.unwrap_or(0.0) >= 0.03
        && frame.scores.optical_flow.unwrap_or(0.0) >= 0.35
        && frame.scores.motion_smoothness.unwrap_or(0.0) >= 0.45
}

#[cfg(test)]
mod tests {
    use super::can_advance_trajectory;
    use crate::project::model::{
        BoundingBox, TrackingConfig, TrackingFrame, TrackingScores, TrackingSource, TrackingStatus,
    };

    fn frame() -> TrackingFrame {
        TrackingFrame {
            frame: 1,
            timestamp_seconds: 0.0,
            bbox: BoundingBox {
                x: 10.0,
                y: 10.0,
                width: 20.0,
                height: 10.0,
            },
            confidence: 0.58,
            status: TrackingStatus::AutoWeak,
            source: TrackingSource::Forward,
            locked: false,
            scores: TrackingScores {
                template: 0.7,
                highpass: 0.7,
                edge: 0.7,
                motion: 0.8,
                position: 0.8,
                size: 1.0,
                optical_flow: Some(0.8),
                forward_backward: None,
                motion_smoothness: Some(0.8),
                match_margin: Some(0.08),
            },
        }
    }

    #[test]
    fn coherent_weak_match_can_advance_internal_trajectory() {
        assert!(can_advance_trajectory(&frame(), &TrackingConfig::default()));

        let mut ambiguous = frame();
        ambiguous.scores.match_margin = Some(0.01);
        assert!(!can_advance_trajectory(
            &ambiguous,
            &TrackingConfig::default()
        ));
    }
}
