# Task status checkpoints

## Safety note — current local sample state (2026-08-28)

- The previously validated render outputs were produced before the current project JSON was last saved.
- Current project `7d4d4da4-8b57-42d1-a786-62b80fcaa758` has `OCCLUDED=150`, `MANUAL=5`, `NEED_REVIEW=748`, `AUTO_WEAK=1`, and 4 problem ranges (`151–291`, `293–479`, `481–652`, `654–902`).
- Rendering this project is intentionally blocked by the existing unresolved-frame safety gate. No tracking data was overwritten or force-accepted.
- Other local project folders were inspected but use a different frame-100 anchor and different templates, so they are not valid snapshots for this sample.

## Goal Set 1 — Tracking and manual enrollment

- Tracking: `COMPLETE` — project case covers 904 frames; unresolved problem ranges remain 0.
- Manual correction: `COMPLETE` — manual anchors are preserved and locked.
- Masking: `COMPLETE` — persisted template/mask and render-time derived mask remain available.
- Existing render modes: `COMPLETE` — Blur, Replacement, and Inpaint paths remain selectable.

## Goal Set 2 — Restoration quality

- G Restoration Architecture: `COMPLETE` — restoration layer, model, alignment, artifact, temporal, and fallback modules exist.
- H Temporal Restoration Core: `PARTIAL` — neighboring-frame selection, lookbehind/lookahead buffering, local translation alignment, median reconstruction, and conservative rejection are implemented; the real sample still has difficult scenes with visible residue.
- I Artifact Reduction / Anti-Flicker: `PARTIAL` — alignment outlier filtering, post-composite artifact checks, glyph-residual heuristic, and deterministic fallback reduce bad patches; visual review still found residual blur in some ranges.
- J AutoBest / Fallback Cascade: `PARTIAL` — AutoBest now evaluates successful Temporal, Inpaint, and Blur candidates on each frame according to the configured fallback policy, then selects deterministically with strategy hysteresis; real-case visual proof is still pending.
- K Inpaint Quality Improvement: `PARTIAL` — inpaint remains functional and can be evaluated before Blur fallback, but the real sample does not yet prove a consistent visual improvement over the previous MVP in every range.
- L UI Integration: `COMPLETE` — Temporal Restore, Auto Best, restoration settings, ROI padding, fallback policy, and progress behavior are integrated without redesigning the UI.
- M Tests / Validation: `PARTIAL` — unit tests, Rust/TypeScript quality gates, and real 904-frame Temporal/AutoBest renders pass container/decode checks; visual quality acceptance is intentionally not marked complete.

## Latest real-case validation

- Project: `7d4d4da4-8b57-42d1-a786-62b80fcaa758`
- Video: `clip_test.mp4`, 1080×1920, 904 frames, `7232/241` FPS.
- Audio: AAC preserved in both rendered outputs.
- Outputs: `output-temporal.mp4` and `output-auto-best.mp4` render and decode successfully.
- Current quality caveat: faint watermark residue remains in difficult textured/background ranges; do not treat this checkpoint as production-quality completion.

## Restoration refinement checkpoint — 2026-08-28

- Mask refinement: enclosed zero regions in the edge-derived glyph mask are filled before render-time dilation, preventing readable glyph interiors from being left untouched.
- Temporal refinement: `AUTO_WEAK`, `NEED_REVIEW`, and `OCCLUDED` frames are excluded as temporal candidates; per-pixel reconstruction now selects an actual consensus candidate pixel instead of combining channels from different frames.
- Inpaint refinement: the fixed ten-pass full-frame neighbor loop was replaced with a local breadth-first boundary fill that reaches the center of wide masks without feeding original watermark pixels back into later passes.
- Tests added for wide-mask inpaint coverage, unusable temporal candidates, and deterministic majority-pixel selection.
- Validation: `cargo fmt --all -- --check`, `cargo test` (26 passed), `cargo clippy --all-targets --all-features -- -D warnings`, and `npm run build` pass.
- Visual/904-frame validation remains pending until the same sample project has a clean, confirmed tracking state; existing outputs must not be treated as validation of this refinement checkpoint.

## AutoBest selection checkpoint — 2026-08-28

- Added `restoration/selector.rs` with deterministic quality-score selection and a small previous-strategy hysteresis penalty to reduce adjacent-frame mode switching.
- `AutoBest` now evaluates TemporalRestore, configured spatial fallbacks, and chooses the lowest-scoring successful result instead of aliasing the TemporalRestore path.
- Existing explicit modes and fallback policies remain unchanged; Replacement remains an explicit deliberate mode.
- Added selector tests for lowest score, stable tie preference, failed candidates, and near-equal previous-method preference.
- Validation: `cargo fmt --all -- --check`, `cargo test` (30 passed), `cargo clippy --all-targets --all-features -- -D warnings`, and `npm run build` pass.
