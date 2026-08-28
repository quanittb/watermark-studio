# Task status checkpoints

## Goal Set 1 — Tracking and manual enrollment

- Tracking: `COMPLETE` — project case covers 904 frames; unresolved problem ranges remain 0.
- Manual correction: `COMPLETE` — manual anchors are preserved and locked.
- Masking: `COMPLETE` — persisted template/mask and render-time derived mask remain available.
- Existing render modes: `COMPLETE` — Blur, Replacement, and Inpaint paths remain selectable.

## Goal Set 2 — Restoration quality

- G Restoration Architecture: `COMPLETE` — restoration layer, model, alignment, artifact, temporal, and fallback modules exist.
- H Temporal Restoration Core: `PARTIAL` — neighboring-frame selection, lookbehind/lookahead buffering, local translation alignment, median reconstruction, and conservative rejection are implemented; the real sample still has difficult scenes with visible residue.
- I Artifact Reduction / Anti-Flicker: `PARTIAL` — alignment outlier filtering, post-composite artifact checks, glyph-residual heuristic, and deterministic fallback reduce bad patches; visual review still found residual blur in some ranges.
- J AutoBest / Fallback Cascade: `PARTIAL` — AutoBest is wired and deterministic, with Temporal → Inpaint → Blur fallback; it currently uses the same per-frame scoring path as Temporal Restore and is not yet segment-optimized.
- K Inpaint Quality Improvement: `PARTIAL` — inpaint remains functional and can be evaluated before Blur fallback, but the real sample does not yet prove a consistent visual improvement over the previous MVP in every range.
- L UI Integration: `COMPLETE` — Temporal Restore, Auto Best, restoration settings, ROI padding, fallback policy, and progress behavior are integrated without redesigning the UI.
- M Tests / Validation: `PARTIAL` — unit tests, Rust/TypeScript quality gates, and real 904-frame Temporal/AutoBest renders pass container/decode checks; visual quality acceptance is intentionally not marked complete.

## Latest real-case validation

- Project: `7d4d4da4-8b57-42d1-a786-62b80fcaa758`
- Video: `clip_test.mp4`, 1080×1920, 904 frames, `7232/241` FPS.
- Audio: AAC preserved in both rendered outputs.
- Outputs: `output-temporal.mp4` and `output-auto-best.mp4` render and decode successfully.
- Current quality caveat: faint watermark residue remains in difficult textured/background ranges; do not treat this checkpoint as production-quality completion.
