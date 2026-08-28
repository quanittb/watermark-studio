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

/// AUTO_GOOD is reserved for a match that is supported by independent image
/// channels and a tight forward/backward trajectory agreement. A high weighted
/// average alone can hide one catastrophically weak signal.
pub fn validated_status(
    scores: &TrackingScores,
    confidence: f64,
    config: &crate::project::model::TrackingConfig,
) -> TrackingStatus {
    let independently_supported = scores.template >= 0.60
        && scores.highpass >= 0.58
        && scores.edge >= 0.52
        && scores.match_margin.unwrap_or(0.0) >= 0.03
        && scores.optical_flow.unwrap_or(0.0) >= 0.40
        && scores.motion_smoothness.unwrap_or(0.0) >= 0.50
        && scores.forward_backward.unwrap_or(0.0) >= 0.70;

    if independently_supported && confidence >= config.accept_threshold {
        TrackingStatus::AutoGood
    } else if confidence >= config.weak_threshold {
        TrackingStatus::AutoWeak
    } else {
        TrackingStatus::NeedReview
    }
}

#[cfg(test)]
mod tests {
    use super::validated_status;
    use crate::project::model::{TrackingConfig, TrackingScores, TrackingStatus};

    fn scores() -> TrackingScores {
        TrackingScores {
            template: 0.82,
            highpass: 0.80,
            edge: 0.74,
            motion: 0.9,
            position: 0.9,
            size: 1.0,
            optical_flow: Some(0.8),
            forward_backward: Some(0.9),
            motion_smoothness: Some(0.85),
            match_margin: Some(0.08),
        }
    }

    #[test]
    fn auto_good_requires_independent_validation_signals() {
        let config = TrackingConfig::default();
        assert_eq!(
            validated_status(&scores(), 0.9, &config),
            TrackingStatus::AutoGood
        );

        let mut ambiguous = scores();
        ambiguous.match_margin = Some(0.01);
        assert_eq!(
            validated_status(&ambiguous, 0.9, &config),
            TrackingStatus::AutoWeak
        );

        let mut disagreed = scores();
        disagreed.forward_backward = Some(0.4);
        assert_eq!(
            validated_status(&disagreed, 0.9, &config),
            TrackingStatus::AutoWeak
        );
    }
}
