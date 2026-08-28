use crate::project::model::{FallbackPolicy, RemovalMode};

pub fn cascade(policy: FallbackPolicy, temporal_success: bool) -> RemovalMode {
    if temporal_success {
        return RemovalMode::TemporalRestore;
    }
    match policy {
        FallbackPolicy::TemporalInpaintBlur | FallbackPolicy::InpaintBlur => RemovalMode::Inpaint,
        FallbackPolicy::BlurOnly => RemovalMode::Blur,
    }
}

/// Returns the deterministic fallback order after temporal restoration fails.
/// The caller evaluates each method in order and can continue to the next one
/// when the previous result has a poor artifact score.
pub fn fallback_modes(policy: FallbackPolicy) -> Vec<RemovalMode> {
    let first = cascade(policy, false);
    match first {
        RemovalMode::Inpaint => vec![first, RemovalMode::Blur],
        RemovalMode::Blur => vec![first],
        _ => vec![RemovalMode::Blur],
    }
}

#[cfg(test)]
mod tests {
    use super::{cascade, fallback_modes};
    use crate::project::model::{FallbackPolicy, RemovalMode};

    #[test]
    fn temporal_failure_uses_configured_fallback() {
        assert_eq!(
            cascade(FallbackPolicy::TemporalInpaintBlur, false),
            RemovalMode::Inpaint
        );
        assert_eq!(cascade(FallbackPolicy::BlurOnly, false), RemovalMode::Blur);
        assert_eq!(
            cascade(FallbackPolicy::BlurOnly, true),
            RemovalMode::TemporalRestore
        );
        assert_eq!(
            fallback_modes(FallbackPolicy::TemporalInpaintBlur),
            vec![RemovalMode::Inpaint, RemovalMode::Blur]
        );
    }
}
