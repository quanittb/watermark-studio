use crate::project::model::{TrackingScores, TrackingStatus};

pub fn fuse_scores(scores: &TrackingScores) -> f64 {
    let optical = scores.optical_flow.unwrap_or(0.5);
    let agreement = scores.forward_backward.unwrap_or(0.5);
    let smoothness = scores.motion_smoothness.unwrap_or(scores.motion);
    let margin = scores.match_margin.unwrap_or(0.0);
    (scores.template * 0.25
        + scores.highpass * 0.27
        + scores.edge * 0.20
        + scores.motion * 0.06
        + scores.position * 0.05
        + scores.size * 0.04
        + optical * 0.04
        + agreement * 0.04
        + smoothness * 0.03
        + margin * 0.02)
        .clamp(0.0, 1.0)
}

pub fn status_for(
    confidence: f64,
    config: &crate::project::model::TrackingConfig,
) -> TrackingStatus {
    if confidence >= config.accept_threshold {
        TrackingStatus::AutoGood
    } else if confidence >= config.weak_threshold {
        TrackingStatus::AutoWeak
    } else {
        TrackingStatus::NeedReview
    }
}
