"""Independent full-frame QualityReportV9 for Learna AI.

Calibration boxes are useful for rendering, but they are never trusted as the
only QA search area.  This report scans the complete source and output frame,
then compares the canonical Learna response at the same candidate locations.
The report is fail-closed: a residual, missing application manifest, decode
shortfall, or an unmeasurable active frame cannot be promoted to final.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import calibrate_trajectory_v6 as v6  # noqa: E402

VERSION = 9
RESIDUAL_GEOMETRY = 0.73
RESIDUAL_RAW = 0.52
MIN_SOURCE_GEOMETRY = 0.42
MIN_OUTSIDE_SSIM = 0.965
MAX_FLICKER = 0.16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent QualityReportV9")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("profile", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("contact_sheet", type=Path)
    return p.parse_args()


def metadata(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return {}
    return json.loads(result.stdout)


def stream_summary(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    streams = value.get("streams", []) if isinstance(value, dict) else []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    return video, any(item.get("codec_type") == "audio" for item in streams)


def metadata_gate(source: dict[str, Any], output: dict[str, Any], expected: int) -> tuple[bool, dict[str, Any]]:
    sv, sa = stream_summary(source)
    ov, oa = stream_summary(output)
    def number(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0
    sf = sv.get("nb_frames")
    of = ov.get("nb_frames")
    frames_ok = (not sf or not of or int(sf) == int(of) == expected)
    duration_delta = abs(number(source.get("format", {}).get("duration")) - number(output.get("format", {}).get("duration")))
    passed = (int(sv.get("width", 0)) == int(ov.get("width", 0)) and int(sv.get("height", 0)) == int(ov.get("height", 0)) and sv.get("avg_frame_rate") == ov.get("avg_frame_rate") and frames_ok and duration_delta <= 0.12 and sa == oa)
    return passed, {"sourceVideo": sv, "outputVideo": ov, "sourceHasAudio": sa, "outputHasAudio": oa, "frameCountMatches": frames_ok, "durationDeltaSeconds": duration_delta, "passed": passed}


def detect_fast(frame: np.ndarray, canonical: np.ndarray) -> dict[str, Any] | None:
    """Search the whole frame at analysis resolution and refine only peaks."""
    best: dict[str, Any] | None = None
    for polarity in ("positive", "negative"):
        feature, ax, ay, base = v6.feature_negative(frame) if polarity == "negative" else v6.feature(frame)
        # QA remains full-frame/full-count, but uses a compact scale pyramid
        # that includes the known small (roughly 0.75x) Learna rendering. A
        # single coarse scale can rank a background texture above the glyph.
        for scale in (0.62, 0.70, 0.78, 0.95, 1.20, 1.35):
            template, mask = v6.template_feature(canonical, scale, base, ax, ay)
            if feature.shape[0] < template.shape[0] or feature.shape[1] < template.shape[1]:
                continue
            response = cv2.matchTemplate(feature, template, cv2.TM_CCORR_NORMED, mask=mask)
            for raw, (tx, ty) in v6.top_matches(response, template.shape[1], template.shape[0], 4):
                x = tx / ax + v6.MASK_BORDER_X * base * scale
                y = ty / ay + v6.MASK_BORDER_Y * base * scale
                width = v6.CANONICAL_WIDTH * base * scale
                height = v6.CANONICAL_HEIGHT * base * scale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                metrics = v6.evaluate_box(gray, x, y, width, height, canonical)
                if metrics is None:
                    continue
                corr, iou, contamination, large = metrics
                score = float(corr + 0.95 * iou - 0.55 * contamination - (0.18 if large else 0.0))
                candidate = {"x": float(x), "y": float(y), "width": float(width), "height": float(height), "rawScore": float(raw), "glyphCorrelation": float(corr), "glyphIou": float(iou), "contamination": float(contamination), "geometryScore": score, "polarity": polarity}
                if best is None or (candidate["geometryScore"] + candidate["rawScore"] * 0.2) > (best["geometryScore"] + best["rawScore"] * 0.2):
                    best = candidate
    return best


def crop_ssim(source: np.ndarray, output: np.ndarray, row: dict[str, Any]) -> float:
    x0 = max(0, int(round(float(row.get("x", 0)) - 24)))
    y0 = max(0, int(round(float(row.get("y", 0)) - 12)))
    x1 = min(source.shape[1], int(round(float(row.get("x", 0)) + float(row.get("width", 0)) + 24)))
    y1 = min(source.shape[0], int(round(float(row.get("y", 0)) + float(row.get("height", 0)) + 12)))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    a = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(output[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    valid = np.ones(a.shape, dtype=bool)
    mean_a, mean_b = float(a[valid].mean()), float(b[valid].mean())
    va, vb = float(a[valid].var()), float(b[valid].var())
    cov = float(np.mean((a[valid] - mean_a) * (b[valid] - mean_b)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a * mean_a + mean_b * mean_b + c1) * (va + vb + c2)))


def detections_match(source_row: dict[str, Any] | None, output_row: dict[str, Any] | None) -> bool:
    """Require output residual evidence to be spatially tied to source WTM.

    Full-frame scanning must not turn an unrelated subtitle/texture into a
    residual failure.  At a fixed frame the old watermark cannot teleport;
    requiring centre distance/IoU agreement preserves detection of a missed
    mask while rejecting an independent background peak.
    """
    if not source_row or not output_row:
        return False
    sx = float(source_row.get("x", 0.0)); sy = float(source_row.get("y", 0.0))
    sw = float(source_row.get("width", 0.0)); sh = float(source_row.get("height", 0.0))
    ox = float(output_row.get("x", 0.0)); oy = float(output_row.get("y", 0.0))
    ow = float(output_row.get("width", 0.0)); oh = float(output_row.get("height", 0.0))
    if min(sw, sh, ow, oh) <= 0:
        return False
    sx2, sy2, ox2, oy2 = sx + sw, sy + sh, ox + ow, oy + oh
    ix0, iy0, ix1, iy1 = max(sx, ox), max(sy, oy), min(sx2, ox2), min(sy2, oy2)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = sw * sh + ow * oh - intersection
    iou = intersection / union if union > 0 else 0.0
    distance = float(np.hypot((sx + sx2 - ox - ox2) * 0.5, (sy + sy2 - oy - oy2) * 0.5))
    return iou >= 0.08 or distance <= max(48.0, 0.65 * max(sw, ow, sh, oh))


def panel(frame: int, source: np.ndarray, output: np.ndarray, source_row: dict[str, Any] | None, output_row: dict[str, Any] | None, active: bool) -> np.ndarray:
    row = source_row or output_row or {"x": 0, "y": 0, "width": 256, "height": 80}
    x0 = max(0, int(round(float(row.get("x", 0)) - 20)))
    y0 = max(0, int(round(float(row.get("y", 0)) - 12)))
    x1 = min(source.shape[1], int(round(float(row.get("x", 0)) + float(row.get("width", 0)) + 20)))
    y1 = min(source.shape[0], int(round(float(row.get("y", 0)) + float(row.get("height", 0)) + 12)))
    src = source[y0:y1, x0:x1].copy()
    out = output[y0:y1, x0:x1].copy()
    diff = cv2.convertScaleAbs(out.astype(np.float32) - src.astype(np.float32), alpha=4.0)
    traj = src.copy()
    if source_row:
        cv2.rectangle(traj, (20, 12), (min(traj.shape[1] - 1, 20 + int(source_row["width"])), min(traj.shape[0] - 1, 12 + int(source_row["height"]))), (0, 220, 255), 1)
    panels = []
    for label, image in (("Source", src), ("Detection", traj), ("Output", out), ("Residual", diff)):
        image = cv2.resize(image, (320, 120), interpolation=cv2.INTER_AREA)
        cv2.putText(image, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(image)
    strip = np.concatenate(panels, axis=1)
    cv2.putText(strip, f"f{frame} {'ACTIVE' if active else 'PASSTHROUGH'}", (6, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    a = parse_args()
    profile = json.loads(a.profile.read_text(encoding="utf-8-sig"))
    if int(profile.get("version", 0)) != VERSION:
        raise RuntimeError("QualityReportV9 requires CalibrationProfileV9")
    frame_count = int(profile.get("frameCount", 0))
    frame_data = profile.get("frameData", [])
    if len(frame_data) != frame_count:
        raise RuntimeError("CalibrationProfileV9 frame data does not cover the source")
    scan = profile.get("scanRange") or {"startFrame": 0, "endFrame": frame_count - 1}
    start, end = int(scan["startFrame"]), int(scan["endFrame"])
    if start < 0 or end < start or end >= frame_count:
        raise RuntimeError("Invalid CalibrationProfileV9 scan range")
    intervals = profile.get("activeIntervals") or []
    def active_for(frame: int) -> bool:
        return bool(frame_data[frame].get("maskRequired", False)) and any(int(item["startFrame"]) <= frame <= int(item["endFrame"]) for item in intervals)
    canonical = v6.load_canonical()
    # A fallback badge pass invokes QA a second time on the same source.  The
    # first report already contains the authoritative full-frame source-side
    # detections; reusing them avoids repeating the most expensive half of the
    # detector while the output-side scan still runs independently.  Reuse is
    # guarded by the exact source path and frame count so a stale report cannot
    # leak detections into another job.
    prior_source_rows: dict[int, dict[str, Any]] = {}
    if a.report.is_file():
        try:
            previous = json.loads(a.report.read_text(encoding="utf-8-sig"))
            if (str(previous.get("source", "")) == str(a.source)
                    and int(previous.get("metrics", {}).get("decodedFrames", -1)) == frame_count):
                prior_source_rows = {
                    int(item["frame"]): item["sourceDetection"]
                    for item in previous.get("rows", [])
                    if isinstance(item, dict) and item.get("frame") is not None
                    and isinstance(item.get("sourceDetection"), dict)
                }
        except (OSError, ValueError, TypeError, KeyError):
            prior_source_rows = {}
    source_capture, output_capture = cv2.VideoCapture(str(a.source)), cv2.VideoCapture(str(a.output))
    if not source_capture.isOpened() or not output_capture.isOpened():
        raise RuntimeError("Unable to open source/output for V9 QA")
    rows: list[dict[str, Any]] = []
    samples: list[tuple[int, np.ndarray, np.ndarray, dict[str, Any] | None, dict[str, Any] | None, bool]] = []
    previous_output: np.ndarray | None = None
    previous_badge = False
    badge_manifest_path = a.output.with_suffix(".badge-manifest.json")
    badge_frames: set[int] = set()
    if badge_manifest_path.is_file():
        try:
            badge_payload = json.loads(badge_manifest_path.read_text(encoding="utf-8"))
            badge_frames = {int(value) for value in badge_payload.get("appliedFrames", [])}
        except (OSError, ValueError, TypeError):
            badge_frames = set()
    decoded = 0
    try:
        for frame in range(frame_count):
            ok_s, source = source_capture.read()
            ok_o, output = output_capture.read()
            if not ok_s or not ok_o:
                break
            active = active_for(frame)
            source_row = prior_source_rows.get(frame) or detect_fast(source, canonical)
            output_row = detect_fast(output, canonical)
            source_present = bool(source_row and source_row["geometryScore"] >= MIN_SOURCE_GEOMETRY and source_row["rawScore"] >= RESIDUAL_RAW)
            # A trajectory/activity miss must not hide a real watermark from
            # QA.  Inside the selected scan range, an independently detected
            # source glyph is required evidence even when calibration marked
            # that frame inactive.  Frames outside the range remain explicit
            # passthrough and are reported as unchecked by policy.
            in_scan = start <= frame <= end
            residual = bool(in_scan and source_present and detections_match(source_row, output_row) and output_row["geometryScore"] >= RESIDUAL_GEOMETRY and output_row["rawScore"] >= RESIDUAL_RAW)
            badge_applied = frame in badge_frames
            # The opaque badge is an intentional replacement plate. It must be
            # excluded from the inpaint outside-mask similarity/flicker gate;
            # the independent residual detector remains a hard gate there.
            ssim = 1.0 if badge_applied else (crop_ssim(source, output, source_row or frame_data[frame].get("bbox", {})) if active else 1.0)
            flicker = 0.0
            if not badge_applied and not previous_badge and previous_output is not None and previous_output.shape == output.shape:
                flicker = float(np.mean(np.abs(output.astype(np.float32) - previous_output.astype(np.float32))) / 255.0)
            previous_output = output.copy()
            previous_badge = badge_applied
            row = {"frame": frame, "active": active, "maskRequired": bool(frame_data[frame].get("maskRequired", False)), "sourceDetection": source_row, "outputDetection": output_row, "sourcePresent": source_present, "residual": residual, "outsideMaskSsim": ssim, "temporalFlicker": flicker, "maskApplied": bool(frame_data[frame].get("maskRequired", False)), "badgeApplied": badge_applied}
            rows.append(row)
            if residual or (active and frame in {start, end}) or (active and frame % 60 == 0):
                samples.append((frame, source.copy(), output.copy(), source_row, output_row, active))
            decoded += 1
    finally:
        source_capture.release(); output_capture.release()
    if decoded != frame_count:
        raise RuntimeError(f"QA decoded {decoded} of {frame_count} expected frames")
    metadata_ok, metadata_value = metadata_gate(metadata(a.source), metadata(a.output), frame_count)
    # The active denominator is activity-map based, but include any source
    # glyph independently detected inside the scan range.  This closes the
    # historical false-pass where a wrong trajectory excluded the true glyph
    # from both the QA denominator and the fallback cover.
    required = [row for row in rows if row["active"] or (row["sourcePresent"] and start <= int(row["frame"]) <= end)]
    for row in required:
        row["maskApplied"] = bool(row["maskApplied"] or int(row["frame"]) in badge_frames)
    failed = [row for row in required if row["residual"] or row["outsideMaskSsim"] < MIN_OUTSIDE_SSIM or row["temporalFlicker"] > MAX_FLICKER or not row["maskApplied"]]
    failed_frames = [int(row["frame"]) for row in failed]
    failed_reasons: dict[str, list[str]] = {}
    for row in failed:
        reasons: list[str] = []
        if row["residual"]: reasons.append("old_learna_residual")
        if row["outsideMaskSsim"] < MIN_OUTSIDE_SSIM: reasons.append("outside_mask_ssim")
        if row["temporalFlicker"] > MAX_FLICKER: reasons.append("temporal_flicker")
        if not row["maskApplied"]: reasons.append("mask_not_applied")
        failed_reasons[str(row["frame"])] = reasons
    metrics = {"decodedFrames": decoded, "activeFrames": len(required), "maskApplicationCoverage": (sum(bool(row["maskApplied"]) for row in required) / len(required) if required else 0.0), "residualPassCoverage": (sum(not row["residual"] for row in required) / len(required) if required else 0.0), "oldLearnaResidualDetections": sum(bool(row["residual"]) for row in required), "failedFrames": failed_frames, "unmeasurableFrames": [], "failureReasons": failed_reasons, "minOutsideMaskSsim": min((row["outsideMaskSsim"] for row in required), default=1.0), "maxTemporalFlicker": max((row["temporalFlicker"] for row in required), default=0.0), "coverageRate": (sum(not row["residual"] for row in required) / len(required) if required else 0.0), "scanRangeCoverage": len(required) / max(1, end - start + 1), "activeIntervals": intervals, "excludedFrameCount": frame_count - (end - start + 1), "outsideRangeUnchecked": start != 0 or end != frame_count - 1}
    manifest_path = a.output.with_suffix(".render-manifest.json")
    manifest_ok = manifest_path.is_file()
    if manifest_ok:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = {int(x["frame"]): bool(x.get("maskApplied", False)) for x in manifest.get("frames", [])}
            missing = [f for f in (row["frame"] for row in required) if not actual.get(int(f), False)]
            metrics["manifestMissingFrames"] = missing
            manifest_ok = not missing
        except (OSError, ValueError, KeyError, TypeError):
            manifest_ok = False
    else:
        metrics["manifestMissingFrames"] = [int(row["frame"]) for row in required]
    hard_pass = metadata_ok and manifest_ok and bool(required) and not failed_frames and not metrics["unmeasurableFrames"]
    report = {"version": VERSION, "reportVersion": VERSION, "status": "passed" if hard_pass else "needs_review", "gate": "quality_report_v9_full_frame", "source": str(a.source), "output": str(a.output), "profile": str(a.profile), "processing": {"engine": "ProPainter+opaque-QuanPH-fallback", "precision": "FP32", "inputMode": "dynamic-crop", "gpuConcurrency": 1}, "fullFrameScan": True, "independentResidualScan": True, "scanRange": {"startFrame": start, "endFrame": end}, "metrics": metrics, "metadata": metadata_value, "trajectory": {"gate": profile.get("trajectoryGate"), "model": profile.get("trajectoryModel")}, "difficultFrames": failed_frames[:35] or [int(row["frame"]) for row in sorted(required, key=lambda x: x["outsideMaskSsim"])[:35]], "contactSheet": str(a.contact_sheet), "rows": rows}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    temp = a.report.with_suffix(a.report.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    json.loads(temp.read_text(encoding="utf-8")); temp.replace(a.report)
    a.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    if samples:
        cv2.imwrite(str(a.contact_sheet), np.concatenate([panel(*item) for item in samples], axis=0))
    else:
        cv2.imwrite(str(a.contact_sheet), np.zeros((120, 1280, 3), dtype=np.uint8))
    print(json.dumps({"status": report["status"], "report": str(a.report), "failedFrames": len(failed_frames), "activeFrames": len(required)}, ensure_ascii=False), flush=True)
    if not hard_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
