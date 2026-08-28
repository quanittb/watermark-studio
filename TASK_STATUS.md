# Task status checkpoints

## Progress lifecycle checkpoint — 2026-08-28

- Cleared stale operation progress when tracking, interpolation, manual correction, occlusion marking, or rendering finishes or fails, so a canceled/failed operation cannot appear to keep running.
- TypeScript/Vite production build passes after the UI change; tracking data remains untouched and review gating is unchanged.
- Rebuilt and restarted the Tauri executable; one responsive window remains and the saved review-gated project reloads successfully.
- Changed the sidebar tracking action to `Review` once tracking exists, so it jumps to the next unresolved range; the top action is now explicitly labeled `Re-analyze track` to prevent accidental full retracking during review.

## Review-safe UI verification checkpoint — 2026-08-28

- Restarted the debug app after the previous unresponsive instance; only one responsive `watermark-studio` process/window remains.
- The saved project reloads with frame `125` visibly labeled `NEED_REVIEW`, `3 problem range(s)`, and the Render action disabled until review is complete.
- An accidental full tracking run was canceled during verification; the project file timestamp and tracking totals remained unchanged: `INTERPOLATED=745`, `MANUAL=10`, `OCCLUDED=142`, `NEED_REVIEW=7`.

## Review-safe tracking policy checkpoint — 2026-08-28

- The real project state is intentionally left unresolved for user review: `NEED_REVIEW=7`, `problemRanges=3` (`125–128`, `136`, `139–140`); no frame in these ranges is force-accepted as `MANUAL` or silently converted to `INTERPOLATED`.
- Confirmed tracking totals remain `OCCLUDED=142`, `MANUAL=10`, `INTERPOLATED=745`, `NEED_REVIEW=7` across all 904 frames.
- Render safety is preserved: both the UI and Rust renderer refuse to render while `NEED_REVIEW`/`AUTO_WEAK` frames remain, while good/locked/interpolated frames retain their existing bbox data.
- The user will review the three unresolved ranges manually before any new restoration render is accepted as a quality result.

## Safety note — current local sample state (2026-08-28, supersedes earlier note)

- Project `7d4d4da4-8b57-42d1-a786-62b80fcaa758` was backed up before tracking interpolation at `project.json.pre-interpolation-20260828.bak`.
- The latest real project state is intentionally review-gated: `OCCLUDED=142`, `MANUAL=10`, `INTERPOLATED=745`, `NEED_REVIEW=7`, with `problemRanges=3`.
- No unresolved frame is force-accepted as `MANUAL`; existing manual anchors remain locked and existing good/interpolated frame data is preserved.
- Older renders below were produced under an earlier clean checkpoint and must not be treated as validation for the current review-gated state.

## Goal Set 1 — Tracking and manual enrollment

- Tracking: `PARTIAL / REVIEW-GATED` — project case covers 904 frames; three unresolved ranges remain intentionally available for user review.
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

## AutoBest and explicit Inpaint re-render checkpoint — 2026-08-28

- Computer Use render of `RemovalMode=AutoBest` completed successfully at 100% and wrote `output-auto-best.mp4`.
- Output validation passed: H.264 video, `1080×1920`, `904` frames, `7232/241` FPS, duration `30.125s`, AAC audio preserved; full audio/video decode completed without FFmpeg errors.
- Visual QA reviewed fresh output frames `170, 292, 350, 530, 710`. Frames `170`, `292`, `350`, and `710` do not show readable `Learna AI` residue in the reviewed regions; frame `530` still shows the watermark clearly.
- The frame `530` residual is explained by the existing tracking state: its interpolated bbox is around `(x=718.99, y=620.23)`, while the visible watermark is around the upper-left area of the source frame. Tracking was not changed or force-accepted during this checkpoint.
- Computer Use render of explicit `Spatial Inpaint` also completed successfully and wrote `output-inpaint.mp4`. Its SHA-256 is byte-identical to the AutoBest output for this project/settings; it remains a functional selectable mode, but this run does not prove consistent visual superiority.
- Validated commands: `cargo fmt --all -- --check`, `cargo test` (32 passed), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build` from `src-tauri`, and `npm run build` from the workspace root.
- Acceptance remains `PARTIAL` for H, I, J, K, and M because the real sample still contains a clearly visible residual in a difficult tracking range. QA artifacts remain outside the repository under `C:\Users\quant\AppData\Local\Temp\watermark-studio-qa-20260828`.

## Full-mode audit and temporal scoring-context checkpoint — 2026-08-28

- Rebuilt the Rust binary after adding an inactive context ring to Temporal artifact scoring; the render mask remains unchanged and tracking data remains untouched.
- Reopened the rebuilt application, confirmed the saved project loads with `0 problem range(s)`, and rendered `Temporal Restore` successfully through the UI. The output again passed full audio/video decode with `1080×1920`, `904` frames, `7232/241` FPS, `30.125s`, and AAC audio.
- The context-ring scorer and its unit test pass, but fresh frame QA shows frame `170` still has a dark patch at the interpolated ROI while the source watermark is elsewhere. The current interpolated bbox is approximately `(x=436.55, y=449.12)` and does not cover the visible source watermark; this confirms the remaining failure is an upstream tracking mismatch, not a reason to expand the render mask or force-accept the frame.
- Re-rendered explicit `Blur mask` and `Replacement PNG` through the UI; both completed successfully and passed the same metadata/decode audit. Together with the current `Spatial inpaint`, `Temporal Restore`, and `AutoBest` renders, all five selectable modes are operational.
- Current code gates: `cargo fmt --all -- --check`, `cargo test` (33 passed), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build`, and `npm run build` all pass. H, I, J, K, and M remain `PARTIAL` because the real sample still contains tracking-mismatch residue and a Temporal artifact case.

## Render-mask refinement checkpoint — 2026-08-28

- Replaced the restoration renderer's full-bbox fill with the existing solidified/dilated glyph mask. Temporal Restore and Spatial Inpaint now modify only the detected glyph coverage plus its feathered boundary; persisted tracking coordinates and the stored mask are unchanged.
- Rebuilt and opened the updated executable in an isolated temporary target, confirmed the saved project loads with `0 problem range(s)`, and rendered Temporal Restore through the UI without cancellation. The UI reported `Render complete` for `output-temporal.mp4`.
- Re-rendered explicit Spatial Inpaint through the UI after the user selected that mode. The UI reported `Render complete` for `output-inpaint.mp4`.
- Both fresh outputs passed ffprobe/decode checks: H.264 `1080×1920`, `904` frames, `7232/241` FPS, `30.125s`, and AAC 48 kHz stereo audio preserved.
- Visual QA on frames `170, 292, 350, 530, 710` confirms the large dark patch caused by the prior full-bbox restoration at frame 170 is no longer present in Temporal Restore or Spatial Inpaint. Frame 292 retains faint residue on textured hair and frame 530 retains the watermark outside the interpolated bbox; these remain upstream tracking/mask limitations and were not hidden by expanding the render area.
- The redundant old application window was closed after verifying its executable path; one updated watermark-studio window remains open.
- Post-change gates pass: `cargo fmt --all --manifest-path src-tauri/Cargo.toml -- --check`, `cargo test --manifest-path src-tauri/Cargo.toml` (33 passed), `cargo clippy --all-targets --all-features --manifest-path src-tauri/Cargo.toml -- -D warnings`, `cargo build --manifest-path src-tauri/Cargo.toml`, and `npm run build`.

## Current review-gated verification checkpoint — 2026-08-28

- Rebuilt application is running with exactly one responsive `watermark-studio` window.
- The UI exposes `Review` for an existing track and `Re-analyze track` as the explicit full-analysis action; `Render video` remains disabled while unresolved ranges exist.
- Current saved project state is unchanged and review-safe across all 904 frames: `INTERPOLATED=745`, `MANUAL=10`, `OCCLUDED=142`, `NEED_REVIEW=7`.
- The seven unresolved frames remain in three user-controlled ranges: `125–128`, `136`, and `139–140`; no weak frame was auto-accepted, interpolated, or marked occluded.
- Tracker policy confirmed: only `AUTO_GOOD` advances the prediction trajectory; `AUTO_WEAK` and `NEED_REVIEW` remain visible for manual review. Rust and UI render gates reject unresolved weak/review frames.
- Repository verification: working tree clean and `HEAD` matches `origin/main` at `8ac9876` (`make tracking review action explicit`).
- Goal Set 2 remains active and not complete until the user reviews the three ranges and fresh rendered outputs pass visual quality validation.

## Quality-gate verification checkpoint — 2026-08-28

- Re-ran the required checks on the current source after the review-gated changes: `cargo fmt --all --manifest-path src-tauri/Cargo.toml -- --check`, `cargo test --manifest-path src-tauri/Cargo.toml` (37 passed), `cargo clippy --all-targets --all-features --manifest-path src-tauri/Cargo.toml -- -D warnings`, `cargo build --manifest-path src-tauri/Cargo.toml`, and `npm run build`.
- All commands passed with no warnings/errors requiring action.
- No tracking data was changed by this validation. The project remains intentionally blocked from rendering until the user reviews `125–128`, `136`, and `139–140`.

## Synthetic temporal acceptance checkpoint — 2026-08-28

- Added a deterministic synthetic restoration test with a moving semi-transparent watermark over a textured gradient. It verifies that clean neighboring candidates reconstruct the target ROI while pixels outside the ROI remain unchanged.
- The full Rust suite now passes `38/38`; formatting was normalized and `cargo fmt --check` passes.
- This strengthens the engine-level temporal acceptance evidence, but does not replace visual QA on the real 904-frame project. That QA remains pending until the user resolves the three review ranges.

## Render-gate regression checkpoint — 2026-08-28

- Extracted the unresolved-tracking render gate into a dedicated validator and added a regression test: `AUTO_WEAK` and `NEED_REVIEW` are rejected, while `OCCLUDED`, `MANUAL`, and `INTERPOLATED` remain render-eligible (`OCCLUDED` is a no-op at frame processing).
- Full Rust suite passes `39/39`; `cargo fmt --check`, Clippy, Rust build, and `npm run build` pass.
- Rebuilt and restarted the app after the change. Computer Use confirms exactly one responsive window, the saved project loads, `Review` and `Re-analyze track` are visible, and `Render video` is disabled while the three unresolved ranges remain.
- The real project state is unchanged: `INTERPOLATED=745`, `MANUAL=10`, `OCCLUDED=142`, `NEED_REVIEW=7`; no review frame was auto-resolved.

## Runtime regression verification checkpoint — 2026-08-28

- Rebuilt the debug executable after adding the render-gate regression test; the only build interruption was the expected lock from the previously running executable, which was stopped and restarted cleanly.
- Current runtime has one `watermark-studio` process/window and reloads the saved project with the review gate active.
- The added validator is now covered by the full `39/39` Rust test suite, including both unresolved-state rejection and safe `OCCLUDED` handling.

## Review queue UI checkpoint — 2026-08-28

- Added an explicit review-queue summary to the inspector so the exact unresolved ranges are visible: `125–128`, `136`, and `139–140`.
- This is display/navigation-only: it does not modify tracking data, accept frames, interpolate ranges, or bypass the render gate.
- `npm run build` passes, and the rebuilt runtime visibly loads the saved project with the queue summary and disabled `Render video` action.

## Goal Set 2 blocked checkpoint — 2026-08-28

- All implementation, unit-test, build, render-gate, UI, and runtime work that can be completed without changing unresolved tracking decisions is complete and pushed.
- The real project remains intentionally unresolved: 7 `NEED_REVIEW` frames in `125–128`, `136`, and `139–140`; rendering and final real-video visual QA remain blocked by the review gate.
- This is a user-input blocker, not an implementation failure: the user must inspect each range and save a manual correction, re-track result, or explicit occlusion decision before rendering can proceed.
