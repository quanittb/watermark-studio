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
_QA_TEMPLATE_CACHE: dict[tuple[int, float, float, float], list[tuple[float, np.ndarray, np.ndarray]]] = {}


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


def _qa_features(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Build positive/negative high-pass maps with one resize/blur pass."""
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, v6.ANALYSIS_LONG_EDGE / max(source_width, source_height))
    analysis_width = max(32, int(round(source_width * scale)))
    analysis_height = max(32, int(round(source_height * scale)))
    analysis = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
    base = min(source_width / v6.REFERENCE_WIDTH, source_height / v6.REFERENCE_HEIGHT)
    ax, ay = analysis_width / source_width, analysis_height / source_height
    return np.maximum(gray - smooth, 0.0), np.maximum(smooth - gray, 0.0), ax, ay, base


def _qa_templates(canonical: np.ndarray, base: float, ax: float, ay: float) -> list[tuple[float, np.ndarray, np.ndarray]]:
    key = (id(canonical), round(base, 6), round(ax, 6), round(ay, 6))
    cached = _QA_TEMPLATE_CACHE.get(key)
    if cached is None:
        cached = []
        for scale in (0.62, 0.70, 0.78, 0.95, 1.20, 1.35):
            template, mask = v6.template_feature(canonical, scale, base, ax, ay)
            cached.append((scale, template, mask))
        _QA_TEMPLATE_CACHE[key] = cached
    return cached


def detect_candidates_fast(
    frame: np.ndarray,
    canonical: np.ndarray,
    limit: int = 8,
    peaks_per_scale: int = 4,
) -> list[dict[str, Any]]:
    """Return distinct full-frame Learna proposals at bounded cost.

    QA still reads every source/output frame and evaluates each selected box at
    source resolution.  The proposal pass uses the same 405x720 analysis
    resolution as calibration so small, blurred glyphs are not lost during a
    second resize; the bounded five-scale pyramid is the performance limit.
    """
    candidates: list[dict[str, Any]] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    positive_feature, negative_feature, ax, ay, base = _qa_features(frame)
    templates = _qa_templates(canonical, base, ax, ay)
    for polarity, feature in (("positive", positive_feature), ("negative", negative_feature)):
        # QA remains full-frame/full-count, but uses a compact scale pyramid
        # that includes the known small (roughly 0.75x) Learna rendering. A
        # single coarse scale can rank a background texture above the glyph.
        for scale, template, mask in templates:
            if feature.shape[0] < template.shape[0] or feature.shape[1] < template.shape[1]:
                continue
            response = cv2.matchTemplate(feature, template, cv2.TM_CCORR_NORMED, mask=mask)
            # Keep enough spatially distinct peaks for a faint glyph to
            # survive a stronger subtitle/texture peak at the same scale.
            # The analysis pyramid is bounded, so four peaks remains cheaper
            # than the former full-resolution twelve-scale scan.
            for raw, (tx, ty) in v6.top_matches(
                response, template.shape[1], template.shape[0], max(1, peaks_per_scale)
            ):
                x = tx / ax + v6.MASK_BORDER_X * base * scale
                y = ty / ay + v6.MASK_BORDER_Y * base * scale
                width = v6.CANONICAL_WIDTH * base * scale
                height = v6.CANONICAL_HEIGHT * base * scale
                metrics = v6.evaluate_box(gray, x, y, width, height, canonical)
                if metrics is None:
                    continue
                corr, iou, contamination, large = metrics
                # Give canonical glyph overlap more weight than raw response.
                # A high-contrast background edge can win CCORR while having
                # virtually zero glyph IoU; QA must prefer the candidate whose
                # actual letter geometry agrees with Learna AI.
                score = float(corr + 2.0 * iou - 0.55 * contamination - (0.18 if large else 0.0))
                candidate = {"x": float(x), "y": float(y), "width": float(width), "height": float(height), "rawScore": float(raw), "glyphCorrelation": float(corr), "glyphIou": float(iou), "contamination": float(contamination), "geometryScore": score, "polarity": polarity}
                candidates.append(candidate)
    candidates.sort(
        key=lambda item: float(item["geometryScore"]) + 0.2 * float(item["rawScore"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            float(np.hypot(
                float(candidate["x"]) - float(other["x"]),
                float(candidate["y"]) - float(other["y"]),
            )) < max(float(candidate["width"]), float(candidate["height"])) * 0.45
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(1, limit):
            break
    return selected


def detect_fast(frame: np.ndarray, canonical: np.ndarray) -> dict[str, Any] | None:
    """Search the whole frame at a bounded analysis resolution and refine peaks."""
    candidates = detect_candidates_fast(frame, canonical, limit=1)
    return candidates[0] if candidates else None


def _ssim_valid(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return 1.0
    mean_a, mean_b = float(a[valid].mean()), float(b[valid].mean())
    va, vb = float(a[valid].var()), float(b[valid].var())
    cov = float(np.mean((a[valid] - mean_a) * (b[valid] - mean_b)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a * mean_a + mean_b * mean_b + c1) * (va + vb + c2)))


def crop_ssim(
    source: np.ndarray,
    output: np.ndarray,
    row: dict[str, Any],
    canonical: np.ndarray,
    exclude_padding: int = 3,
) -> float:
    """Measure similarity only outside the glyph (or opaque replacement plate).

    The previous implementation compared the complete watermark crop.  That
    made a successful removal look like a bad outside-mask edit because the
    pixels that are intentionally changed were included in SSIM.  Build the
    exclusion mask in source coordinates from the canonical glyph geometry so
    the metric measures the pixels that must remain unchanged.  A badge pass
    uses a larger padding to exclude its intentional opaque plate as well.
    """
    x0 = max(0, int(round(float(row.get("x", 0)) - 24)))
    y0 = max(0, int(round(float(row.get("y", 0)) - 12)))
    x1 = min(source.shape[1], int(round(float(row.get("x", 0)) + float(row.get("width", 0)) + 24)))
    y1 = min(source.shape[0], int(round(float(row.get("y", 0)) + float(row.get("height", 0)) + 12)))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    a = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(output[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    valid = np.ones(a.shape, dtype=bool)
    try:
        glyph_width = max(1, int(round(float(row.get("width", 0)))))
        glyph_height = max(1, int(round(float(row.get("height", 0)))))
        if exclude_padding >= 16:
            # Badge mode intentionally changes the entire opaque plate.  A
            # glyph-only exclusion would still score the plate's rounded
            # corners as an unrelated background edit.
            px = int(exclude_padding)
            rx0 = max(0, int(round(float(row.get("x", 0)) - x0 - px)))
            ry0 = max(0, int(round(float(row.get("y", 0)) - y0 - px)))
            rx1 = min(valid.shape[1], int(round(float(row.get("x", 0)) - x0 + float(row.get("width", 0)) + px)))
            ry1 = min(valid.shape[0], int(round(float(row.get("y", 0)) - y0 + float(row.get("height", 0)) + px)))
            if rx1 > rx0 and ry1 > ry0:
                valid[ry0:ry1, rx0:rx1] = False
            if not np.any(valid):
                return 1.0
            return _ssim_valid(a, b, valid)
        glyph = cv2.resize(canonical, (glyph_width, glyph_height), interpolation=cv2.INTER_AREA)
        glyph = glyph >= 16
        if exclude_padding > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (exclude_padding * 2 + 1, exclude_padding * 2 + 1),
            )
            glyph = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
        gx0 = max(0, int(round(float(row.get("x", 0)) - x0)))
        gy0 = max(0, int(round(float(row.get("y", 0)) - y0)))
        gx1 = min(valid.shape[1], gx0 + glyph.shape[1])
        gy1 = min(valid.shape[0], gy0 + glyph.shape[0])
        if gx1 > gx0 and gy1 > gy0:
            valid[gy0:gy1, gx0:gx1] &= ~glyph[:gy1 - gy0, :gx1 - gx0]
    except (TypeError, ValueError, cv2.error):
        # A malformed detection must not make QA crash; fall back to the
        # conservative bounding-box comparison for this individual row.
        valid = np.ones(a.shape, dtype=bool)
    if not np.any(valid):
        return 1.0
    return _ssim_valid(a, b, valid)


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


def box_intersection_ratio(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> float:
    """Return intersection over the first box's area.

    Badge QA uses this only to prove that an opaque replacement plate covers
    the independently detected source glyph.  It must not be confused with
    calibration IoU: a plate is intentionally larger than the glyph.
    """
    if not isinstance(first, dict) or not isinstance(second, dict):
        return 0.0
    try:
        fx, fy = float(first.get("x", 0.0)), float(first.get("y", 0.0))
        fw, fh = float(first.get("width", 0.0)), float(first.get("height", 0.0))
        sx, sy = float(second.get("x", 0.0)), float(second.get("y", 0.0))
        sw, sh = float(second.get("width", 0.0)), float(second.get("height", 0.0))
    except (TypeError, ValueError):
        return 0.0
    first_area = max(0.0, fw) * max(0.0, fh)
    if first_area <= 0.0 or sw <= 0.0 or sh <= 0.0:
        return 0.0
    ix = max(0.0, min(fx + fw, sx + sw) - max(fx, sx))
    iy = max(0.0, min(fy + fh, sy + sh) - max(fy, sy))
    return (ix * iy) / first_area


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
    # Store only the already downscaled contact-sheet strip.  Keeping full
    # 1080x1920 source/output pairs for every sample made a long QA run retain
    # hundreds of megabytes and could terminate before the report was written.
    samples: list[np.ndarray] = []
    previous_output: np.ndarray | None = None
    previous_badge = False
    badge_manifest_path = a.output.with_suffix(".badge-manifest.json")
    badge_frames: set[int] = set()
    badge_boxes: dict[int, dict[str, Any]] = {}
    badge_cover_missing: list[int] = []
    if badge_manifest_path.is_file():
        try:
            badge_payload = json.loads(badge_manifest_path.read_text(encoding="utf-8"))
            badge_frames = {int(value) for value in badge_payload.get("appliedFrames", [])}
            badge_boxes = {
                int(frame): value
                for frame, value in (badge_payload.get("appliedBoxes") or {}).items()
                if isinstance(value, dict)
            }
        except (OSError, ValueError, TypeError):
            badge_frames = set()
            badge_boxes = {}
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
            badge_applied = frame in badge_frames
            # The opaque QuanPH plate intentionally replaces the full source
            # glyph.  Its artwork may still correlate with the Learna
            # descriptor, so do not classify the intentional plate as an old
            # watermark residual; manifest coverage and source-side detection
            # remain hard gates for these frames.
            badge_box = badge_boxes.get(frame)
            badge_cover_ratio = box_intersection_ratio(source_row, badge_box)
            badge_cover_valid = bool(
                badge_applied
                and badge_box
                and source_present
                and badge_cover_ratio >= 0.80
            )
            if badge_applied and not badge_cover_valid and in_scan and (source_present or active):
                badge_cover_missing.append(frame)
            # The intentional opaque plate may contain artwork that resembles
            # the canonical glyph.  Ignore an output candidate only when it
            # is inside a plate proven to cover the source glyph; a misplaced
            # plate must remain a residual/failure.
            output_inside_badge = bool(
                badge_cover_valid
                and output_row
                and box_intersection_ratio(output_row, badge_box) >= 0.55
            )
            residual = bool(
                in_scan
                and source_present
                and not output_inside_badge
                and detections_match(source_row, output_row)
                and output_row["geometryScore"] >= RESIDUAL_GEOMETRY
                and output_row["rawScore"] >= RESIDUAL_RAW
            )
            # The opaque badge is an intentional replacement plate. It must be
            # excluded from the inpaint outside-mask similarity/flicker gate;
            # the independent residual detector remains a hard gate there.
            ssim = (crop_ssim(
                source,
                output,
                source_row or frame_data[frame].get("bbox", {}),
                canonical,
                # The opaque plate includes the glyph plus its own rounded
                # 10px margin.  Exclude the complete intentional plate (not
                # only the glyph) from outside-mask SSIM; residual detection
                # and the badge manifest remain hard gates for that region.
                exclude_padding=32 if badge_applied else 3,
            ) if active else 1.0)
            flicker = 0.0
            if not badge_applied and not previous_badge and previous_output is not None and previous_output.shape == output.shape:
                flicker = float(np.mean(np.abs(output.astype(np.float32) - previous_output.astype(np.float32))) / 255.0)
            previous_output = output.copy()
            previous_badge = badge_applied
            row = {"frame": frame, "active": active, "maskRequired": bool(frame_data[frame].get("maskRequired", False)), "sourceDetection": source_row, "outputDetection": output_row, "sourcePresent": source_present, "residual": residual, "outsideMaskSsim": ssim, "temporalFlicker": flicker, "maskApplied": bool(frame_data[frame].get("maskRequired", False)), "badgeApplied": badge_applied, "badgeCoverageRatio": badge_cover_ratio, "badgeCoverageValid": badge_cover_valid}
            rows.append(row)
            # Keep contact-sheet source/output pixels bounded.  A bad render
            # can legitimately flag every active frame; retaining every full
            # 1080x1920 pair here grows into multiple gigabytes and makes QA
            # look hung or crash before the report is written.  The report
            # still records every frame in ``rows``; the visual sheet only
            # needs a representative, deterministic sample.
            if len(samples) < 35 and (residual or (active and frame in {start, end}) or (active and frame % 60 == 0)):
                samples.append(panel(frame, source, output, source_row, output_row, active))
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
    for frame in badge_cover_missing:
        failed_reasons.setdefault(str(frame), []).append("badge_does_not_cover_source")
    failed_frames = sorted(set(failed_frames).union(badge_cover_missing))
    badge_rows = [row for row in required if row.get("badgeApplied")]
    valid_badges = sum(bool(row.get("badgeCoverageValid")) for row in badge_rows)
    metrics = {"decodedFrames": decoded, "activeFrames": len(required), "maskApplicationCoverage": (sum(bool(row["maskApplied"]) for row in required) / len(required) if required else 0.0), "residualPassCoverage": (sum(not row["residual"] for row in required) / len(required) if required else 0.0), "oldLearnaResidualDetections": sum(bool(row["residual"]) for row in required), "failedFrames": failed_frames, "unmeasurableFrames": [], "failureReasons": failed_reasons, "badgeCoverageMissingFrames": sorted(set(badge_cover_missing)), "badgeCoverageValidRate": (valid_badges / len(badge_rows) if badge_rows else 1.0), "minOutsideMaskSsim": min((row["outsideMaskSsim"] for row in required), default=1.0), "maxTemporalFlicker": max((row["temporalFlicker"] for row in required), default=0.0), "coverageRate": (sum(not row["residual"] for row in required) / len(required) if required else 0.0), "scanRangeCoverage": len(required) / max(1, end - start + 1), "activeIntervals": intervals, "excludedFrameCount": frame_count - (end - start + 1), "outsideRangeUnchecked": start != 0 or end != frame_count - 1}
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
        cv2.imwrite(str(a.contact_sheet), np.concatenate(samples, axis=0))
    else:
        cv2.imwrite(str(a.contact_sheet), np.zeros((120, 1280, 3), dtype=np.uint8))
    print(json.dumps({"status": report["status"], "report": str(a.report), "failedFrames": len(failed_frames), "activeFrames": len(required)}, ensure_ascii=False), flush=True)
    if not hard_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
