from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from find_learna_samples import load_canonical
from render_periodic_dewatermark import aligned_positions, periodic_position


FIRST_WATERMARK_FRAME = 48
DETECTOR_VERSION = "learna-canonical-v4.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a verified CalibrationProfileV4")
    parser.add_argument("project_json", type=Path)
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("--sample-frame", type=int, required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--candidate-json", default="{}")
    parser.add_argument("--edited-mask", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--route", choices=("AUTO_FIND", "CLEAN_MASK_TEMPLATE", "ROI_FALLBACK"), default="AUTO_FIND")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_large_holes(mask: np.ndarray, canonical: np.ndarray) -> int:
    # Letter counters (a/e/A) are intentional. A hole is a connected stroke
    # segment present in the canonical glyph but erased from the edited mask.
    missing = np.where((canonical > 0) & (mask == 0), 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(missing, 8)
    return sum(1 for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= 8)


def validate_edited_mask(mask: np.ndarray, canonical: np.ndarray) -> dict:
    if mask.shape != canonical.shape:
        mask = cv2.resize(mask, (canonical.shape[1], canonical.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask >= 64, 255, 0).astype(np.uint8)
    canonical_on = canonical > 0
    mask_on = mask > 0
    coverage = float(np.count_nonzero(mask_on & canonical_on)) / max(1, int(np.count_nonzero(canonical_on)))
    contamination = float(np.count_nonzero(mask_on & ~canonical_on)) / max(1, int(np.count_nonzero(mask_on)))
    large_holes = count_large_holes(mask, canonical)
    return {
        "mask": mask,
        "coverage": coverage,
        "contamination": contamination,
        "largeHoles": large_holes,
        "passed": coverage >= 0.95 and contamination <= 0.05 and large_holes == 0,
    }


def explicit_occlusion(row: dict[str, object]) -> bool:
    """Return only an explicitly measured occlusion signal.

    ``modelScore`` is detector evidence, not visibility.  Motion blur,
    textured backgrounds, subtitles and scene transitions can all lower the
    score while the Learna AI glyph is still present.  Treating every low
    score as occluded was the reason the final pipeline emitted the source
    frame unchanged for a number of difficult frames.
    """
    value = row.get("occlusion")
    if isinstance(value, bool):
        return value
    value = row.get("occluded")
    return bool(value) if isinstance(value, bool) else False


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8-sig"))
    source = Path(project["source"]["path"])
    if not source.is_file():
        raise RuntimeError(f"Source video not found: {source}")
    width = int(project["video"]["width"])
    height = int(project["video"]["height"])
    frame_count = int(project["video"]["frameCount"])
    if (width, height) != (1080, 1920):
        raise RuntimeError("CalibrationProfileV4 only supports the verified 1080x1920 Learna AI preset")
    if not FIRST_WATERMARK_FRAME <= args.sample_frame < frame_count:
        raise RuntimeError("Confirmed sample frame is outside the supported watermark interval")

    bbox = json.loads(args.bbox_json)
    candidate = json.loads(args.candidate_json)
    # CLEAN_MASK_TEMPLATE is an explicit fallback: the canonical glyph is
    # trusted, while the trajectory audit/QA still decides whether any frame
    # can be rendered. AUTO_FIND candidates must satisfy the hard evidence
    # gate before a profile is created.
    if args.route == "AUTO_FIND" and (float(candidate.get("glyphCorrelation", 0.0)) < 0.65 or float(candidate.get("glyphIou", 0.0)) < 0.55):
        raise RuntimeError("Confirmed sample did not pass the Learna AI glyph hard gate")
    if args.route == "AUTO_FIND" and (float(candidate.get("contamination", 1.0)) > 0.20 or int(candidate.get("temporalPassCount", 0)) < 3):
        raise RuntimeError("Confirmed sample is contaminated or temporally unstable")

    canonical = load_canonical()
    edited = canonical.copy()
    if args.edited_mask:
        loaded = cv2.imread(str(args.edited_mask), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise RuntimeError(f"Unable to read edited mask: {args.edited_mask}")
        edited = loaded
    gate = validate_edited_mask(edited, canonical)
    if not gate["passed"]:
        raise RuntimeError("Mask gate failed: coverage >=95%, contamination <=5%, no large holes required")

    normalized = cv2.morphologyEx(gate["mask"], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inference = cv2.dilate(normalized, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blend = cv2.GaussianBlur(inference, (0, 0), 1.75)
    calibration_dir = args.profile_json.parent
    calibration_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = calibration_dir / "canonical_mask.png"
    auto_path = calibration_dir / "auto_mask.png"
    inference_path = calibration_dir / "inference_mask.png"
    blend_path = calibration_dir / "blend_mask.png"
    brush_path = calibration_dir / "brush_delta.json"
    for path, image in ((canonical_path, canonical), (auto_path, normalized), (inference_path, inference), (blend_path, blend)):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Unable to write calibration artifact: {path}")
    brush_path.write_text(json.dumps({"version": 1, "edited": bool(args.edited_mask)}, indent=2), encoding="utf-8")

    phase_shift = int(candidate.get("trajectoryPhaseOffset", 0)) % 360
    expected_x, expected_y = periodic_position((args.sample_frame + phase_shift) % 360)
    offset_x = float(bbox["x"]) - expected_x
    offset_y = float(bbox["y"]) - expected_y
    offset_limit = 240 if bool(candidate.get("roiFallback", False)) else 80
    if abs(offset_x) > offset_limit or abs(offset_y) > offset_limit:
        raise RuntimeError("Confirmed sample is too far from the verified Learna AI trajectory")

    audit_rows: list[dict[str, object]] = []
    offsets_x = np.zeros(frame_count, dtype=np.float64)
    offsets_y = np.zeros(frame_count, dtype=np.float64)
    trajectory_source = "periodic-prior"
    reliable_frames: list[int] = []
    residuals: list[float] = []
    if args.audit_json:
        audit_rows = json.loads(args.audit_json.read_text(encoding="utf-8-sig"))
        if len(audit_rows) != frame_count:
            raise RuntimeError("Trajectory audit frame count does not match the source video")
        offsets_x, offsets_y = aligned_positions(audit_rows, phase_shift=phase_shift)
        trajectory_source = "full-video-audit"
        for row in audit_rows:
            score = float(row.get("modelScore", 0.0))
            frame_number = int(row.get("frame", -1))
            if score >= 0.30 and 0 <= frame_number < frame_count:
                reliable_frames.append(frame_number)
                periodic_x, periodic_y = periodic_position((frame_number + phase_shift) % 360)
                residuals.append(float(np.hypot(float(row.get("modelX", periodic_x)) - (periodic_x + offsets_x[frame_number]), float(row.get("modelY", periodic_y)) - (periodic_y + offsets_y[frame_number]))))

    frame_data = []
    difficult_frames: list[int] = []
    trajectory_inferred_frames: list[int] = []
    explicit_occluded_frames: list[int] = []
    for frame in range(frame_count):
        watermark_active = frame >= FIRST_WATERMARK_FRAME
        score = 1.0 if watermark_active else 0.0
        measured = False
        occluded = False
        if audit_rows:
            row = audit_rows[frame]
            score = float(np.clip(float(row.get("modelScore", 0.0)), 0.0, 1.0))
            measured = score >= 0.30
            occluded = explicit_occlusion(row)
            # Low detector confidence does not mean that the watermark is
            # absent.  Its position is already recovered from the periodic
            # trajectory and neighboring measured frames, so keep the mask
            # active and record the frame for targeted QA instead.
            if watermark_active and not measured:
                difficult_frames.append(frame)
                trajectory_inferred_frames.append(frame)
            if watermark_active and occluded:
                explicit_occluded_frames.append(frame)
        x, y = periodic_position((frame + phase_shift) % 360)
        x += float(offsets_x[frame] if audit_rows else offset_x)
        y += float(offsets_y[frame] if audit_rows else offset_y)
        visible = watermark_active and not occluded
        frame_data.append({
            "frame": frame,
            "bbox": {"x": float(x), "y": float(y), "width": float(bbox["width"]), "height": float(bbox["height"])},
            "visibility": visible,
            "confidence": score,
            "detectorScore": score,
            "occlusion": occluded,
            "maskRequired": watermark_active and not occluded,
            "positionSource": "detector" if measured else ("trajectory-interpolated" if watermark_active else "inactive"),
            "maskTransform": {"scaleX": 1.0, "scaleY": 1.0, "offsetX": 0.0, "offsetY": 0.0},
        })

    source_fingerprint = {
        "sha256": sha256_file(source),
        "sizeBytes": source.stat().st_size,
        "frameCount": frame_count,
        "width": width,
        "height": height,
    }
    relative = lambda path: path.relative_to(args.project_json.parent).as_posix()
    profile = {
        "version": 4,
        "status": "READY",
        "preset": "LEARNA_AI_PERIODIC",
        "detectorVersion": DETECTOR_VERSION,
        "route": args.route,
        "sourceFingerprint": source_fingerprint,
        "frameCount": frame_count,
        "firstWatermarkFrame": FIRST_WATERMARK_FRAME,
        "sampleFrame": args.sample_frame,
        "candidate": candidate,
        "canonicalMaskPath": relative(canonical_path),
        "autoMaskPath": relative(auto_path),
        "brushDeltaPath": relative(brush_path),
        "inferenceMaskPath": relative(inference_path),
        "blendMaskPath": relative(blend_path),
        "maskPath": relative(inference_path),
        "canonicalMaskSha256": sha256_file(canonical_path),
        "maskSha256": sha256_file(inference_path),
        "frameData": frame_data,
        "qualityGate": {
            "status": "PASSED",
            "glyphCoverage": gate["coverage"],
            "contamination": gate["contamination"],
            "largeHoles": gate["largeHoles"],
            "reliableFrames": len(reliable_frames) if audit_rows else frame_count - FIRST_WATERMARK_FRAME,
            "lowConfidenceFrames": len(difficult_frames) if audit_rows else FIRST_WATERMARK_FRAME,
            "trajectoryInferredFrames": len(trajectory_inferred_frames),
            "explicitOccludedFrames": len(explicit_occluded_frames),
            "maskedFrames": sum(1 for item in frame_data if item["maskRequired"]),
            "maskPixels": int(np.count_nonzero(inference)),
            "trajectoryResidualMedian": float(np.median(residuals)) if residuals else 0.0,
            "trajectoryResidualP95": float(np.percentile(residuals, 95)) if residuals else 0.0,
        },
        "difficultFrames": difficult_frames[:256],
        "trajectoryInferredFrames": trajectory_inferred_frames,
        "explicitOccludedFrames": explicit_occluded_frames,
        "trajectoryModel": {
            "type": "periodic-4-segment",
            "source": trajectory_source,
            "positionPolicy": "measured-plus-neighbor-interpolation",
            "phaseShift": phase_shift,
            "periodFrames": 360,
            "reliableFrames": len(reliable_frames) if audit_rows else frame_count - FIRST_WATERMARK_FRAME,
            "residualMedian": float(np.median(residuals)) if residuals else 0.0,
            "residualP95": float(np.percentile(residuals, 95)) if residuals else 0.0,
        },
    }
    profile["profileSha256"] = canonical_json_sha(profile)
    args.profile_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps({"profile": str(args.profile_json), "profileSha256": profile["profileSha256"], "maskSha256": profile["maskSha256"], "sampleFrame": args.sample_frame, "qualityGate": profile["qualityGate"]}), flush=True)


if __name__ == "__main__":
    main()
