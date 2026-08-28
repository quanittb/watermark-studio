use crate::project::model::RemovalMode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrategyCandidate {
    pub mode: RemovalMode,
    pub score: f64,
    pub success: bool,
}

/// Selects the lowest-scoring successful candidate with a stable preference
/// order. The preference only breaks ties, keeping TemporalRestore preferred
/// when two methods are effectively indistinguishable.
/// Adds a small hysteresis penalty to a strategy switch. AutoBest can then
/// keep one method over adjacent frames when scores are nearly identical,
/// reducing mode-switch flicker without overriding a clearly better method.
pub fn select_best_with_preference(
    candidates: &[StrategyCandidate],
    previous_mode: Option<RemovalMode>,
) -> Option<RemovalMode> {
    candidates
        .iter()
        .filter(|candidate| candidate.success && candidate.score.is_finite())
        .min_by(|left, right| {
            effective_score(**left, previous_mode)
                .partial_cmp(&effective_score(**right, previous_mode))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| priority(left.mode).cmp(&priority(right.mode)))
        })
        .map(|candidate| candidate.mode)
}

fn effective_score(candidate: StrategyCandidate, previous_mode: Option<RemovalMode>) -> f64 {
    const SWITCH_PENALTY: f64 = 0.02;
    candidate.score
        + if previous_mode.is_some_and(|mode| mode != candidate.mode) {
            SWITCH_PENALTY
        } else {
            0.0
        }
}

fn priority(mode: RemovalMode) -> u8 {
    match mode {
        RemovalMode::TemporalRestore => 0,
        RemovalMode::Inpaint => 1,
        RemovalMode::Blur => 2,
        RemovalMode::Replacement => 3,
        RemovalMode::AutoBest => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::{select_best_with_preference, StrategyCandidate};
    use crate::project::model::RemovalMode;

    #[test]
    fn chooses_the_lowest_successful_score() {
        let candidates = [
            StrategyCandidate {
                mode: RemovalMode::TemporalRestore,
                score: 0.22,
                success: true,
            },
            StrategyCandidate {
                mode: RemovalMode::Inpaint,
                score: 0.14,
                success: true,
            },
            StrategyCandidate {
                mode: RemovalMode::Blur,
                score: 0.31,
                success: true,
            },
        ];

        assert_eq!(
            select_best_with_preference(&candidates, None),
            Some(RemovalMode::Inpaint)
        );
    }

    #[test]
    fn uses_stable_temporal_preference_for_equal_scores() {
        let candidates = [
            StrategyCandidate {
                mode: RemovalMode::Blur,
                score: 0.2,
                success: true,
            },
            StrategyCandidate {
                mode: RemovalMode::TemporalRestore,
                score: 0.2,
                success: true,
            },
        ];

        assert_eq!(
            select_best_with_preference(&candidates, None),
            Some(RemovalMode::TemporalRestore)
        );
    }

    #[test]
    fn ignores_failed_candidates() {
        let candidates = [StrategyCandidate {
            mode: RemovalMode::TemporalRestore,
            score: 0.01,
            success: false,
        }];

        assert_eq!(select_best_with_preference(&candidates, None), None);
    }

    #[test]
    fn keeps_the_previous_method_when_scores_are_nearly_equal() {
        let candidates = [
            StrategyCandidate {
                mode: RemovalMode::TemporalRestore,
                score: 0.10,
                success: true,
            },
            StrategyCandidate {
                mode: RemovalMode::Inpaint,
                score: 0.11,
                success: true,
            },
        ];

        assert_eq!(
            select_best_with_preference(&candidates, Some(RemovalMode::Inpaint)),
            Some(RemovalMode::Inpaint)
        );
    }
}
