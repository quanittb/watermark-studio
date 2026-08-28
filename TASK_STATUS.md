# Task status checkpoints

## Safety note — current local sample state (2026-08-28, supersedes earlier note)

- Project `7d4d4da4-8b57-42d1-a786-62b80fcaa758` was backed up before tracking interpolation at `project.json.pre-interpolation-20260828.bak`.
- Current tracking state is `OCCLUDED=150`, `MANUAL=5`, `INTERPOLATED=749`, `NEED_REVIEW=0`, `AUTO_WEAK=0`, with `problemRanges=0`.
- No frame was force-accepted as `MANUAL`; manual anchors remain locked and interpolated ranges retain their existing status semantics.
- The renders below were produced after this clean tracking state was confirmed.

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

## Explicit Inpaint render checkpoint — 2026-08-28

- The current project was rendered through the Tauri UI with `RemovalMode=INPAINT`; the app completed successfully and wrote `output-inpaint.mp4`.
- Decode validation: H.264 video, `1080×1920`, `904` frames, `7232/241` FPS, AAC 48 kHz stereo, duration `30.125s`.
- Visual QA used the same tracked ROI on frames `170, 292, 350, 530, 710`. The new explicit path is functional and the local BFS fill avoids the old fixed-pass implementation, but frame 292 still contains readable `Learna AI` residue on textured hair; seam/texture differences remain in some ranges.
- Acceptance result: explicit Inpaint is validated as runnable, but K (consistent visual improvement over the previous MVP) and M (visual quality acceptance) remain `PARTIAL`. Temporal and AutoBest also remain `PARTIAL` because difficult ranges are not yet residue-free.
- QA artifacts are kept outside the repository under `C:\Users\quant\AppData\Local\Temp\watermark-studio-qa-20260828`.

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

## Artifact scoring and Temporal render checkpoint — 2026-08-28

- Artifact analysis now densifies only sparse analysis masks before scoring, so enclosed glyph interiors contribute to residual detection without expanding the render-time mask or changing persisted tracking data.
- Validation: `cargo fmt --all -- --check`, `cargo test` (31 passed), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build`, and `npm run build` pass.
- TemporalRestore was rendered through the Tauri UI after the clean tracking state was confirmed. The app reported `Render complete` for `output-temporal.mp4`; decode validation passed with H.264 `1080×1920`, `904` frames, `7232/241` FPS, AAC 48 kHz stereo, and duration `30.125s`.
- Visual QA reviewed frames `170, 292, 350, 530, 710` against the source and spatial outputs. Temporal/AutoBest are cleaner in several ranges, but frame 292 still shows readable `Learna AI` residue on textured hair and some seam/texture differences remain.
- Acceptance result: artifact scoring and the Temporal render are operational, but H, I, J, K, and M remain `PARTIAL` because visual residue is not consistently eliminated. No tracking data was changed during this checkpoint.
