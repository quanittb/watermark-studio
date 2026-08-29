# Task status checkpoints

## Best-quality calibration/queue hardening — 2026-08-29

- Final renders no longer use `--anchor-mode`. The new `CalibrationProfileV2` stores a confirmed sample, fixed Learna AI periodic trajectory, per-frame offsets/bboxes, visibility/occlusion flags, confidence and a SHA-256 mask hash.
- Calibration now treats the confirmed visual sample as authoritative and uses the detector only for diagnostics. This prevents a rough anchor over a subtitle/UI element from moving the removal mask onto unrelated background content.
- The calibrated mask is built from trajectory-aligned temporal consensus and connected-component filtering; the resulting mask is glyph-shaped (`Learna AI`) instead of a full noisy anchor crop.
- Added `quality_qa.py`: decode/frame/audio checks plus residual correlation, outside-mask MAE and a source/output contact sheet. A failed gate returns `NEEDS_REVIEW` and blocks completion; the old `decode_passed` report is no longer accepted.
- Added persistent sequential GPU jobs (`jobs.json`) with normalized lifecycle states, cancel, rescan/regen and restart interruption recovery. The GPU mutex guarantees one ProPainter job at a time on 4 GB cards.
- Added output-root selection, History actions (Open output/Open folder/View sidecar/Regenerate), bilingual core settings, QuanPH branding/icon assets and signed Tauri updater configuration/workflow.
- Regression evidence: the legacy `_best_2.mp4` fails the new QA gate (glyph-energy ratio ≈0.94, exit 2), while the accepted reference passes (ratio ≈0.62, exit 0). A fresh calibrated ProPainter run is being used to validate the mask against the same source before promotion.
- Fresh isolated validation output: `D:\watermark-studio-ai-work\quality-profile-test-output\calibrated-best-static.mp4` (1080×1920, 904 frames, AAC retained, 30.144s). QA status is `passed`, max glyph-energy ratio ≈0.761, and the generated contact sheet is `calibrated-best-static.qa.png`. SHA-256: `E3DEAEEA760A738B697DCCE5FAB8A014836CB847B786DCC4238A24588B960F20`.

## One-anchor Best-quality dev workflow — 2026-08-29

- The workspace now opens in `Best-quality` mode. The final path exposes only sample finding/inspection/confirmation and the ProPainter render; the old tracker, AutoBest, Temporal, Blur, Inpaint, and legacy PNG replacement controls are separated behind an explicit `Legacy / Preview` switch and are labeled non-final.
- `Find 5 alternatives` now exists for human rejection of the contact-sheet. Every new pass shifts the periodic trajectory sampling phase and excludes all previously presented frames within a temporal safety gap, so repeatedly pressing it cannot simply return the same five cards. Candidate scans are cache-only and do not change the saved profile or render input until a user confirms one sample.
- The Best-quality renderer now performs a full FFmpeg decode after composite and writes a `*.qa.json` sidecar with the render method, source/output metadata, and hard-frame review locations. A decode or report failure makes the render fail instead of reporting completion.
- Best-quality replacement labels are optional and default off: text or transparent PNG can follow the validated watermark path or use fixed source coordinates; the inspector renders a placement preview and the Python composite applies it only after ProPainter removal. Best-quality GPU jobs are serialized by an application-wide mutex, preserving per-project workspaces/outputs and preventing multiple 4 GB-VRAM jobs from running together. A multi-project queue screen is still pending; profile confirmation remains per-video by design.
- Replaced the unsafe direct-anchor assumption with a required quality gate: `Find best samples` now scans periodic-path candidates using the exact persisted-mask generator, ranks only masks with at least 350 pixels at intensity 64 or above, and presents up to five time-separated frames for visual confirmation.
- The sample scan no longer requires any hand-drawn anchor: it begins from the verified default Learna AI path and deliberately ignores a previously saved weak mask/anchor, so a rough or faint first selection cannot bias the candidate boxes.
- The five returned samples are now retained as a contact sheet: each card shows the source crop beside its generated glyph mask and its mask coverage, background-complexity, and local temporal-instability scores. The scorer follows the watermark path in the neighbouring frames, so it measures the background behind the moving watermark rather than an unrelated fixed part of the image.
- Added an inspection-preview command and UI mode: candidate verification now uses an extracted source crop with a fixed, high-contrast watermark frame and 1.00x–3.00x zoom. The original full-frame view remains available, while normal navigation clears the inspection box so it cannot become stale.
- Selecting a card locks its exact frame/bbox; confirmation persists that selected sample directly instead of trusting the current video-playback position. This avoids the prior focus-jump path saving a different frame than the user inspected.
- Confirming a candidate refreshes the primary template/mask even when legacy tracking data exists; Best-quality rendering then rejects a weak saved mask before the GPU job can start. This prevents a full ProPainter job with an empty effective mask, which previously produced an output visually indistinguishable from input when the faint frame-136 anchor had only 26 pixels below intensity 64.
- Validation after this change: `cargo test` (47 passed, 2 intentional real-video ignores), `cargo clippy -D warnings`, `cargo build`, TypeScript production build, formatting, and diff checks pass.

- Added the dedicated `Best-quality AI render` action to the Tauri dev UI. It uses the original saved anchor and does not depend on, alter, or unblock the conservative manual-tracking review queue.
- The action validates the known 1080×1920 repeating Learna AI layout and tight anchor dimensions before work starts, then invokes the full-frame FP32 ProPainter chunk pipeline and composites only the feathered glyph mask back into the original frames.
- The source remains read-only. A unique MP4 is created under the source video's `output` folder, and generated AI cache frames are removed after a successful render. Missing model/runtime paths or an incompatible anchor cause an explicit error rather than a guessed render.
- Dev-machine defaults point to the verified local ProPainter environment; `WATERMARK_STUDIO_PROPAINTER_PYTHON`, `WATERMARK_STUDIO_PROPAINTER_ROOT`, and `WATERMARK_STUDIO_WORK_ROOT` can override those locations.
- Validation: TypeScript production build, Rust formatting, `cargo test` (47 passed, 2 intentional real-video ignores), `cargo clippy -D warnings`, `cargo build`, Python syntax compilation, and a two-frame real-anchor preparation smoke test all pass.

## Production-quality periodic watermark removal checkpoint — 2026-08-29

- Reverse-engineered the verified deterministic `Learna AI` trajectory across the 904-frame source, generated a clean glyph-only mask, and preserved the source video unchanged.
- The local Temporal/AutoBest renderer was improved to reject candidate glyph pixels rather than an entire bounding box, honor a larger requested candidate budget, and retain a temporal result once its coverage, consensus, alignment, and seam gates pass. This removes the previous rectangular fallback artifacts from the highest-confidence temporal frames.
- The remaining hard crossings (face, subtitles, car reflections, robot and phone UI) were rendered with FP32 ProPainter at `288x512`, split into 15 chunks with 8-frame temporal context on each side. Only the feathered glyph mask was composited back into the original `1080x1920` frames, preserving the untouched source pixels and original AAC audio outside the removal area.
- Final output: `C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best.mp4`.
- Final validation: H.264 `1080x1920`, `904` frames, `7232/241` FPS; AAC stereo `48 kHz`; duration `30.144s`; full FFmpeg decode succeeds. SHA-256: `F223F5BC8988C8C8B6B3D224AED69E20D624810E6E7970B0C27E4EB064882758`.
- Visual acceptance: source-vs-final crop review at 35 points spanning frames `48–903` confirms no readable `Learna AI` residue and no rectangular/black restoration patches in the tested face, subtitle, vehicle, robot, or phone-interface crossings.

## Anchor-local real-video tracking checkpoint — 2026-08-28

- Replaced the single frame-480 template dependency inside tracking with templates cropped from each manual anchor's actual analysis frame. Forward and backward passes for every bounded segment now start from their own endpoint appearance without changing the project schema or manual-anchor contract.
- Removed the redundant full-video baseline pass; only the leading one-way range, anchor-bounded bidirectional ranges, and an optional trailing one-way range are analyzed.
- Locally coherent weak matches may advance only the internal trajectory; persisted `AUTO_GOOD` still requires bidirectional validation. A second strong-image-consensus path handles fast motion only when gray, high-pass, and edge channels are all strong and endpoint tracks agree.
- Isolated 904-frame audit result: `AUTO_GOOD=19`, `NEED_REVIEW=799`, `MANUAL=20`, `OCCLUDED=66`, `AUTO_WEAK=0`, `INTERPOLATED=0`, with 12 review ranges.
- All 19 `AUTO_GOOD` frames are the bounded segment `531–549`. Source overlays were inspected at frames `531`, `534`, `535`, `536`, `537`, `538`, `541`, `544`, `547`, `548`, and `549`; every reviewed bbox covers the visible watermark. Reported bad frames `170`, `235`, `350`, `654`, `700`, `710`, `800`, and `902` remain `NEED_REVIEW`.
- Added an ignored environment-configured real-project audit test so the full 904-frame measurement can be repeated on an isolated project copy without mutating user data.

## Real-video conservative tracker audit and project recovery checkpoint — 2026-08-28

- Ran the updated tracker on an isolated copy of the 904-frame real project. Result: `AUTO_GOOD=0`, `AUTO_WEAK=34`, `NEED_REVIEW=784`, `MANUAL=20`, `OCCLUDED=66`, across 13 unresolved ranges.
- Every reported bad sample (`170`, `235`, `350`, `654`, `700`, `710`, `800`, `902`) remained `NEED_REVIEW`; no misplaced bbox was certified for rendering.
- The result confirms the conservative validator prevents false-positive acceptance, but the underlying detector still cannot image-validate enough frames to be considered production-quality.
- During a debug restart, the Windows remove-then-rename compatibility path exposed an interruption window that could leave `project.json` missing. The real project was reconstructed from its persisted pre-interpolation snapshot plus all 20 reviewed manual anchors and validated back to `INTERPOLATED=604`, `MANUAL=20`, `NEED_REVIEW=145`, `OCCLUDED=135` with the same 10 ranges.
- Project persistence now writes a `project.json.last-good.bak` snapshot before the Windows replacement window and automatically restores it when the primary file is missing or unreadable.

## Conservative AUTO_GOOD validation checkpoint — 2026-08-28

- A frame can now become `AUTO_GOOD` only when gray-template, high-pass, edge, match-margin, optical-flow, motion-smoothness, and forward/backward agreement all pass independent minimums in addition to the aggregate confidence threshold.
- A one-direction-only match is capped at `AUTO_WEAK`; it remains visible and review-gated because there is no independent reverse trajectory to confirm it.
- This policy deliberately favors false negatives (more user review) over false positives that would remove pixels at the wrong location.

## Provisional bbox visibility and confidence correction checkpoint — 2026-08-28

- Visual source-overlay audit confirmed the current tracking is not production-safe: frame `654` is slightly high and clips the watermark baseline, while frames `170`, `235`, `350`, `700`, `710`, and `800` are materially misplaced.
- Provisional `AUTO_WEAK`, `NEED_REVIEW`, and `INTERPOLATED` bboxes remain visible as dashed orange overlays labeled with their status, so a wrong stored focus can be diagnosed instead of silently disappearing.
- Saving a manual correction now requires a fresh user-drawn selection; the visible provisional bbox cannot be saved accidentally. The separate explicit accept action remains available.
- On project load, legacy `INTERPOLATED` confidence is normalized to `0` because linear geometry is not image evidence. Review and Rust render gates remain unchanged.

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

## Direct review navigation checkpoint — 2026-08-28

- Added accessible direct-jump buttons for each unresolved range in the inspector: `125–128`, `Frame 136`, and `139–140`. Each button only seeks to the range's worst frame and leaves tracking unchanged.
- `npm run build` passes, and runtime accessibility inspection confirms all three buttons are present while `Render video` remains disabled.
- The goal remains blocked on the same user-controlled review decision; no unresolved frame was modified.

## Interpolation safety and bbox mismatch checkpoint — 2026-08-28

- Visual QA confirmed the reported focus mismatch is upstream tracking data, not UI source-to-display scaling: frame `235` had an interpolated bbox around `(572.9, 948.6)` while the source watermark is around `(370, 850)`.
- Re-tracking the affected section with the current manual anchors correctly refused to force-accept the ambiguous result; the section `151–291` is now `NEED_REVIEW` and rendering is blocked.
- Interpolated frames are now review-gated in both the persisted review queue and Rust render validator. Their bbox is retained for inspection but confidence is set to `0` because interpolation is not image validation for a moving watermark.
- On project load, the review queue is recomputed so older projects cannot silently treat stale interpolated bboxes as render-safe.
- The UI hides provisional boxes for `AUTO_WEAK`, `NEED_REVIEW`, and `INTERPOLATED` frames, prompting the user to draw the actual source-coordinate box before saving a manual correction.
- Validation passed: Rust tests `40/40`, `cargo fmt --check`, Clippy with `-D warnings`, Rust build, and `npm run build`.
- Current real project is intentionally review-gated with 12 ranges: `19–21`, `111`, `119`, `141`, `151–291`, `293–479`, `481–529`, `531–549`, `551–599`, `601–629`, `631–652`, and `654–902`. No render was accepted from this state.
