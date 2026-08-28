use crate::project::model::RemovalMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestorationStrategy {
    Blur,
    Replacement,
    InpaintSpatial,
    TemporalRestore,
    AutoBest,
}

impl RestorationStrategy {
    pub fn from_mode(mode: RemovalMode) -> Self {
        match mode {
            RemovalMode::Blur => Self::Blur,
            RemovalMode::Replacement => Self::Replacement,
            RemovalMode::Inpaint => Self::InpaintSpatial,
            RemovalMode::TemporalRestore => Self::TemporalRestore,
            RemovalMode::AutoBest => Self::AutoBest,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalSettings {
    pub max_candidates: usize,
    pub alignment_radius: i32,
    pub roi_padding: i32,
    pub artifact_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RestorationResult {
    pub strategy: RestorationStrategy,
    pub success: bool,
    pub artifact_score: f64,
    pub temporal_consistency_score: f64,
    pub valid_pixel_ratio: f64,
    pub fallback_used: bool,
}

impl RestorationResult {
    pub fn failed(strategy: RestorationStrategy) -> Self {
        Self {
            strategy,
            success: false,
            artifact_score: 1.0,
            temporal_consistency_score: 0.0,
            valid_pixel_ratio: 0.0,
            fallback_used: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RestorationStrategy;
    use crate::project::model::RemovalMode;

    #[test]
    fn strategies_map_to_explicit_output_modes() {
        assert_eq!(
            RestorationStrategy::from_mode(RemovalMode::Blur),
            RestorationStrategy::Blur
        );
        assert_eq!(
            RestorationStrategy::from_mode(RemovalMode::Replacement),
            RestorationStrategy::Replacement
        );
        assert_eq!(
            RestorationStrategy::from_mode(RemovalMode::Inpaint),
            RestorationStrategy::InpaintSpatial
        );
        assert_eq!(
            RestorationStrategy::from_mode(RemovalMode::TemporalRestore),
            RestorationStrategy::TemporalRestore
        );
        assert_eq!(
            RestorationStrategy::from_mode(RemovalMode::AutoBest),
            RestorationStrategy::AutoBest
        );
    }
}
