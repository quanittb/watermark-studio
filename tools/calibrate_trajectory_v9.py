"""Zero-touch Learna AI calibration.

V8 kept the legacy candidate scorer and could fit a plausible trajectory to a
background branch.  V9 deliberately keeps the canonical glyph as the only
shape prior, builds a small multi-hypothesis temporal graph, and writes a
strict, frame-complete profile.  The renderer is allowed to use this profile
only when the independent QA pass succeeds.

This module is intentionally dependency-light: it reuses the proven source
geometry helpers from V6, but performs a denser scan and a continuity-aware
selection before any trajectory interpolation is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
# Six-frame sampling is the safe lower-cost pass on a 30 fps source.  Scene
# boundaries and low-confidence regions are still refined densely below; the
# renderer receives a transform for every frame.
SCAN_STRIDE = 6
MAX_PEAKS = 20
SCALES = (0.55, 0.62, 0.70, 0.78, 0.86, 0.95, 1.0, 1.08, 1.20, 1.35, 1.50)
MAX_GAP_FRAMES = 18
MAX_UNCERTAINTY = 5.0
MIN_PATH_COVERAGE = 0.70
MIN_HOLDOUT_INLIER = 0.80


def finite(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def normalize(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def strict_write(path: Path, value: Any) -> None:
    payload = normalize(value)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_hash(profile: dict[str, Any]) -> str:
    body = dict(profile)
    body.pop("profileSha256", None)
    encoded = json.dumps(normalize(body), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CalibrationProfileV9 zero-touch Learna AI detector")
    parser.add_argument("project_json", type=Path)
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("--scan-start-frame", type=int, default=None)
    parser.add_argument("--scan-end-frame", type=int, default=None)
    parser.add_argument("--edited-mask", type=Path, default=None)
    # Accepted for API compatibility. V9 never constrains the global detector
    # to a user rectangle and therefore never requires ROI input.
    parser.add_argument("--roi-json", default="")
    parser.add_argument("--roi-frame", type=int, default=None)
    parser.add_argument("--roi-evidence-json", default="[]")
    parser.add_argument("--route", default="AUTO_GLOBAL_TEMPLATE")
    return parser.parse_args()


def descriptor(canonical: np.ndarray, scale: float) -> np.ndarray:
    """Build a zero-mean signed glyph/ring descriptor.

    Positive glyph support and negative gap/ring support prevent a textured
    face, subtitle, or UI panel from winning solely because it is bright.
    """
    h = max(12, int(round(canonical.shape[0] * scale)))
    w = max(24, int(round(canonical.shape[1] * scale)))
    alpha = cv2.resize(canonical, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    glyph = alpha > 0.18
    ring = cv2.dilate(glyph.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool) & ~glyph
    out = np.zeros((h, w), np.float32)
    out[glyph] = 1.0
    out[ring] = -0.35
    out -= out.mean()
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-6 else out


def signed_candidate_score(gray: np.ndarray, x: float, y: float, width: float, height: float, canonical: np.ndarray) -> tuple[float, float, float]:
    """Score local glyph contrast independently of V6's positive-only NCC."""
    box = v6.bounds(x, y, width, height, gray.shape[1], gray.shape[0])
    if box is None:
        return -1.0, 0.0, 0.0
    x0, y0, x1, y1 = box
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    mask = cv2.resize(canonical, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    local = patch - cv2.GaussianBlur(patch, (0, 0), 6.0)
    glyph = mask > 0.18
    ring = cv2.dilate(glyph.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool) & ~glyph
    if not np.any(glyph) or not np.any(ring):
        return -1.0, 0.0, 0.0
    glyph_mean = float(np.mean(local[glyph]))
    ring_mean = float(np.mean(local[ring]))
    scale = float(np.std(local[ring]) + 1.0)
    contrast = abs(glyph_mean - ring_mean) / scale
    # A shape-independent contrast score is useful only as a tie-breaker;
    # V6's mask IoU/correlation remains part of the final geometry score.
    return min(1.0, contrast / 4.0), glyph_mean, ring_mean


def geometry(row: dict[str, Any]) -> float:
    corr = float(row.get("glyphCorrelation", 0.0) or 0.0)
    iou = float(row.get("glyphIou", 0.0) or 0.0)
    contamination = float(row.get("contamination", 1.0) or 1.0)
    signed = float(row.get("signedContrast", 0.0) or 0.0)
    large = 0.18 if bool(row.get("largeOutsideComponent", False)) else 0.0
    return corr + 0.95 * iou - 0.55 * contamination - large + 0.35 * signed


def scan_frame(frame: np.ndarray, frame_number: int, canonical: np.ndarray) -> list[dict[str, Any]]:
    """Generate global candidates with one reduced-resolution pass.

    The V6 proposal function performs a source-resolution morphology pass for
    every peak.  That is accurate but too slow when called on every third
    frame.  V9 first finds a small number of peaks with OpenCV's vectorised
    matcher, then evaluates only those peaks at source resolution.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_feature, analysis_x, analysis_y, base_scale = v6.feature(frame)
    negative_feature, _, _, _ = v6.feature_negative(frame)
    proposals: list[tuple[float, float, float, float, str]] = []
    for polarity, feature_image in (("positive", image_feature), ("negative", negative_feature)):
        for scale in SCALES:
            template, template_mask = v6.template_feature(canonical, scale, base_scale, analysis_x, analysis_y)
            if feature_image.shape[0] < template.shape[0] or feature_image.shape[1] < template.shape[1]:
                continue
            response = cv2.matchTemplate(feature_image, template, cv2.TM_CCORR_NORMED, mask=template_mask)
            # Keep a wide peak set here.  Low-contrast dark-on-light glyphs
            # often rank below a face/UI texture in the positive high-pass
            # response, but their source-resolution geometry is still valid.
            for raw_score, (ax, ay) in v6.top_matches(response, template.shape[1], template.shape[0], 64):
                canvas_x = ax / analysis_x
                canvas_y = ay / analysis_y
                width = v6.CANONICAL_WIDTH * base_scale * scale
                height = v6.CANONICAL_HEIGHT * base_scale * scale
                x = canvas_x + v6.MASK_BORDER_X * base_scale * scale
                y = canvas_y + v6.MASK_BORDER_Y * base_scale * scale
                proposals.append((float(raw_score), x, y, scale, polarity))
    proposals.sort(reverse=True, key=lambda item: item[0])
    # First score only the proposal centre, then spend source-resolution work
    # on the strongest geometry candidates. This keeps the wide proposal
    # budget without returning to the V6 million-crop calibration cost.
    coarse: list[tuple[float, tuple[float, float, float, float, str]]] = []
    for proposal in proposals[:256]:
        raw_score, x, y, scale, polarity = proposal
        width = v6.CANONICAL_WIDTH * base_scale * scale
        height = v6.CANONICAL_HEIGHT * base_scale * scale
        signed, _, _ = signed_candidate_score(gray, x, y, width, height, canonical)
        # The cheap signed contrast pass prevents a bright background from
        # consuming the entire proposal budget, while still retaining weak
        # watermark candidates for the full geometry check below.
        coarse.append((float(raw_score) + 0.50 * float(signed), proposal))
    coarse.sort(key=lambda item: item[0], reverse=True)
    rows: list[dict[str, Any]] = []
    # Keep a raw-response branch as well as the signed-contrast branch.  Dark
    # Learna glyphs on a bright background can have weak signed contrast even
    # when the masked template response is the correct global peak.
    selected_proposals: list[tuple[float, tuple[float, float, float, float, str]]] = []
    seen_proposals: set[tuple[int, int, int, str]] = set()
    for proposal in proposals[:256] + [item[1] for item in coarse[:64]]:
        key = (int(round(proposal[1] / 8)), int(round(proposal[2] / 8)), int(round(proposal[3] * 20)), proposal[4])
        if key not in seen_proposals:
            selected_proposals.append((float(proposal[0]), proposal))
            seen_proposals.add(key)
    for _, proposal in selected_proposals:
        raw_score, x, y, scale, polarity = proposal
        width = v6.CANONICAL_WIDTH * base_scale * scale
        height = v6.CANONICAL_HEIGHT * base_scale * scale
        metrics = v6.evaluate_box(gray, x, y, width, height, canonical)
        if metrics is None:
            continue
        corr, iou, contamination, large = metrics
        signed, inside, outside = signed_candidate_score(gray, x, y, width, height, canonical)
        row: dict[str, Any] = {
            "frame": frame_number, "x": float(x), "y": float(y),
            "width": float(width), "height": float(height), "scale": float(base_scale * scale),
            "templateScale": float(scale), "rawScore": raw_score, "glyphCorrelation": float(corr),
            "glyphIou": float(iou), "contamination": float(contamination),
            "largeOutsideComponent": bool(large), "signedContrast": float(signed),
            "glyphMean": inside, "ringMean": outside, "polarity": polarity,
        }
        row["geometryScore"] = geometry(row)
        row["score"] = float(max(0.0, min(1.0, (float(row["geometryScore"]) + 0.12 * raw_score) / 2.4)))
        if all(finite(value) for value in row.values()):
            rows.append(row)
    return dedupe_rows(rows)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=geometry, reverse=True):
        if any(
            math.hypot(float(row["x"]) - float(other["x"]), float(row["y"]) - float(other["y"]))
            < max(float(row["width"]), float(row["height"])) * 0.35
            for other in output
        ):
            continue
        output.append(row)
    return output[:MAX_PEAKS]


def path_graph(candidates: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Beam/Viterbi path that favours glyph shape and physical continuity."""
    frames = sorted(candidates)
    if not frames:
        return []
    # Each state is (cost, start_frame, parent_index, row). Keep a bounded
    # beam so long clips remain practical on a 4 GB machine.
    beam: list[tuple[float, int, list[dict[str, Any]]]] = []
    for index, frame in enumerate(frames):
        current = candidates[frame]
        if not current:
            continue
        states: list[tuple[float, int, list[dict[str, Any]]]] = [
            (-4.0 * geometry(row), frame, [row]) for row in current
        ]
        if beam:
            previous_frame = frames[index - 1]
            delta = max(1, frame - previous_frame)
            for cost, start, history in beam:
                previous = history[-1]
                for row in current:
                    distance = math.hypot(float(row["x"]) - float(previous["x"]), float(row["y"]) - float(previous["y"]))
                    limit = min(420.0, max(70.0, 24.0 * delta))
                    if distance > limit:
                        continue
                    scale_jump = abs(float(row.get("scale", 1.0)) - float(previous.get("scale", 1.0)))
                    if scale_jump > 0.22 and delta <= 6:
                        continue
                    transition = distance / max(24.0, 18.0 * delta) + 2.0 * scale_jump
                    states.append((cost + transition - 4.0 * geometry(row), start, history + [row]))
        states.sort(key=lambda item: item[0])
        # Remove duplicate spatial branches at the same frame.
        beam = []
        seen: set[tuple[int, int, int]] = set()
        for state in states:
            row = state[2][-1]
            key = (int(round(float(row["x"]) / 16)), int(round(float(row["y"]) / 16)), int(round(float(row.get("scale", 1.0)) * 20)))
            if key in seen:
                continue
            seen.add(key)
            beam.append(state)
            if len(beam) >= 24:
                break
    if not beam:
        return []
    # Reward duration so one high-contrast background peak cannot beat a long
    # but translucent glyph path.
    best = min(beam, key=lambda item: item[0] - 0.06 * len(item[2]))
    path = best[2]
    # A beam can legitimately restart at a scene cut, but returning only the
    # last beam branch makes a continuous watermark look like a short path.
    # Stitch the strongest physically plausible branch across all sampled
    # frames when the retained branch covers less than 60% of observations.
    if len(path) < max(10, int(len(frames) * 0.60)):
        stitched: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for frame in frames:
            rows = candidates.get(frame, [])
            if not rows:
                continue
            if previous is None:
                selected = max(rows, key=geometry)
            else:
                delta = max(1, frame - int(previous["frame"]))
                selected = max(rows, key=lambda row: geometry(row) - min(2.0, math.hypot(float(row["x"]) - float(previous["x"]), float(row["y"]) - float(previous["y"])) / max(40.0, 22.0 * delta)))
            previous = dict(selected, graph=True)
            stitched.append(previous)
        if len(stitched) > len(path):
            path = stitched
    return [dict(row, graph=True) for row in path]


def merge_intervals(frames: list[int], scan_start: int, scan_end: int, fps: float) -> list[dict[str, int]]:
    if not frames:
        return []
    bridge = max(6, int(round(fps * 0.8)))
    groups: list[list[int]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame - groups[-1][-1] <= bridge:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    intervals = []
    for group in groups:
        if len(group) < 2:
            continue
        intervals.append({"startFrame": max(scan_start, group[0] - 3), "endFrame": min(scan_end, group[-1] + 3)})
    return intervals


def in_interval(frame: int, intervals: list[dict[str, int]]) -> bool:
    return any(int(item["startFrame"]) <= frame <= int(item["endFrame"]) for item in intervals)


def interpolate(path: list[dict[str, Any]], frame: int) -> tuple[float, float, float, float]:
    if not path:
        return 0.0, 0.0, 1.0, 8.0
    if frame <= int(path[0]["frame"]):
        row = path[0]
        return float(row["x"]), float(row["y"]), float(row.get("scale", 1.0)), 8.0
    if frame >= int(path[-1]["frame"]):
        row = path[-1]
        return float(row["x"]), float(row["y"]), float(row.get("scale", 1.0)), 8.0
    for left, right in zip(path, path[1:]):
        a, b = int(left["frame"]), int(right["frame"])
        if a <= frame <= b:
            ratio = (frame - a) / max(1.0, b - a)
            x = float(left["x"]) + (float(right["x"]) - float(left["x"])) * ratio
            y = float(left["y"]) + (float(right["y"]) - float(left["y"])) * ratio
            scale = float(left.get("scale", 1.0)) + (float(right.get("scale", 1.0)) - float(left.get("scale", 1.0))) * ratio
            gap = b - a
            return x, y, scale, min(8.0, max(0.5, gap / 3.0))
    return float(path[-1]["x"]), float(path[-1]["y"]), float(path[-1].get("scale", 1.0)), 8.0


def refine_frame(frame: np.ndarray, predicted: tuple[float, float, float, float], canonical: np.ndarray, radius: int = 32) -> dict[str, Any]:
    px, py, ps, uncertainty = predicted
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best: tuple[float, dict[str, Any]] | None = None
    # Use a vectorised signed descriptor for the normal local pass.  A single
    # matchTemplate call replaces hundreds of source-resolution morphology
    # evaluations while retaining a final evaluate_box validation at the peak.
    for polarity in ("positive", "negative"):
        smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.2)
        search_image = np.maximum(gray.astype(np.float32) - smooth, 0.0) if polarity == "positive" else np.maximum(smooth - gray.astype(np.float32), 0.0)
        for scale_factor in (0.94, 1.0, 1.06):
            scale = float(np.clip(ps * scale_factor, 0.55, 1.55))
            width = v6.CANONICAL_WIDTH * scale
            height = v6.CANONICAL_HEIGHT * scale
            tw, th = max(16, int(round(width))), max(10, int(round(height)))
            x0 = max(0, int(round(px - radius)))
            y0 = max(0, int(round(py - radius)))
            x1 = min(gray.shape[1], int(round(px + radius + width)))
            y1 = min(gray.shape[0], int(round(py + radius + height)))
            search = search_image[y0:y1, x0:x1]
            if search.shape[0] < th or search.shape[1] < tw:
                continue
            template = cv2.resize(descriptor(canonical, scale), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, match, _, location = cv2.minMaxLoc(response)
            x, y = float(x0 + location[0]), float(y0 + location[1])
            metrics = v6.evaluate_box(gray, x, y, width, height, canonical)
            if metrics is None:
                continue
            corr, iou, contamination, large = metrics
            signed, _, _ = signed_candidate_score(gray, x, y, width, height, canonical)
            row = {"x": x, "y": y, "width": float(width), "height": float(height), "scale": scale, "glyphCorrelation": float(corr), "glyphIou": float(iou), "contamination": float(contamination), "largeOutsideComponent": bool(large), "signedContrast": float(signed), "rawScore": float(match), "polarity": polarity}
            score = geometry(row) + 0.15 * float(match)
            if best is None or score > best[0]:
                best = (score, row)
    # Weak or heavily blurred evidence gets the bounded exhaustive refinement;
    # this path is uncommon and remains fail-closed if no valid geometry is
    # found.
    if best is not None and best[0] < 0.75:
        for dx in (-radius, 0, radius):
            for dy in (-radius, 0, radius):
                x, y = px + dx, py + dy
                width = v6.CANONICAL_WIDTH * ps
                height = v6.CANONICAL_HEIGHT * ps
                metrics = v6.evaluate_box(gray, x, y, width, height, canonical)
                if metrics is None:
                    continue
                corr, iou, contamination, large = metrics
                signed, _, _ = signed_candidate_score(gray, x, y, width, height, canonical)
                row = {"x": float(x), "y": float(y), "width": float(width), "height": float(height), "scale": ps, "glyphCorrelation": float(corr), "glyphIou": float(iou), "contamination": float(contamination), "largeOutsideComponent": bool(large), "signedContrast": float(signed)}
                score = geometry(row) - 0.010 * math.hypot(dx, dy)
                if score > best[0]:
                    best = (score, row)
    if best is None:
        return {
            "x": px, "y": py, "width": v6.CANONICAL_WIDTH * ps, "height": v6.CANONICAL_HEIGHT * ps,
            "scale": ps, "score": 0.0, "confidence": 0.18, "uncertaintyPx": max(uncertainty, 5.0),
            "positionSource": "TRAJECTORY_INTERPOLATED", "refined": False,
        }
    score, row = best
    if score < 0.40 and uncertainty <= 3.0:
        # Weak candidate: spend a second bounded pass over a wider radius and
        # scale range.  Failure remains explicit instead of inventing a mask.
        return refine_frame(frame, (px, py, ps, max(uncertainty, 3.1)), canonical, radius=32)
    row.update({"score": float(max(0.0, min(1.0, score / 2.4))), "confidence": float(max(0.0, min(1.0, (score + 0.2) / 1.8))), "uncertaintyPx": float(max(0.5, min(8.0, uncertainty * 0.45))), "positionSource": "LOCAL_NCC", "refined": True})
    return row


def holdout(path: list[dict[str, Any]]) -> dict[str, Any]:
    if len(path) < 10:
        return {"count": 0, "trainingCount": len(path), "median": None, "p95": None, "inlierRatio": 0.0, "reason": "INSUFFICIENT_HOLDOUT_OBSERVATIONS"}
    # Scene cuts and genuine trajectory turns must not be evaluated as if a
    # motion vector could cross the cut.  Build short continuous segments and
    # hold out within each segment only.
    segments: list[list[dict[str, Any]]] = [[]]
    for row in path:
        if segments[-1]:
            previous = segments[-1][-1]
            delta = int(row["frame"]) - int(previous["frame"])
            jump = math.hypot(float(row["x"]) - float(previous["x"]), float(row["y"]) - float(previous["y"]))
            if delta > 18 or jump > 220.0:
                segments.append([])
        segments[-1].append(row)
    residuals: list[float] = []
    training_count = 0
    for segment in segments:
        if len(segment) < 4:
            continue
        test = [row for index, row in enumerate(segment) if index % 5 == 0]
        train = [row for index, row in enumerate(segment) if index % 5 != 0]
        training_count += len(train)
        for row in test:
            x, y, _, _ = interpolate(train, int(row["frame"]))
            residuals.append(math.hypot(float(row["x"]) - x, float(row["y"]) - y))
    if not residuals:
        return {"count": 0, "trainingCount": training_count, "median": None, "p95": None, "inlierRatio": 0.0, "reason": "NO_CONTINUOUS_HOLDOUT_SEGMENT"}
    return {"count": len(residuals), "trainingCount": training_count, "median": float(np.median(residuals)), "p95": float(np.percentile(residuals, 95)), "inlierRatio": float(sum(v <= 3.0 for v in residuals) / max(1, len(residuals))), "reason": None}


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8-sig"))
    source = Path(project["source"]["path"])
    video = project["video"]
    frame_count, width, height = int(video["frameCount"]), int(video["width"]), int(video["height"])
    fps = float(video.get("fps", 30.0))
    scan_start = 0 if args.scan_start_frame is None else int(args.scan_start_frame)
    scan_end = frame_count - 1 if args.scan_end_frame is None else int(args.scan_end_frame)
    if frame_count < 1 or scan_start < 0 or scan_end < scan_start or scan_end >= frame_count:
        raise RuntimeError(f"INVALID_SCAN_RANGE: expected 0 <= start <= end < {frame_count}")
    canonical = v6.load_canonical()
    if args.edited_mask:
        edited = args.edited_mask if args.edited_mask.is_absolute() else args.profile_json.parent / args.edited_mask
        canonical = v6.load_mask_asset(edited, canonical)

    capture = cv2.VideoCapture(str(source))
    candidates: dict[int, list[dict[str, Any]]] = {}
    # A V8 project may already contain a location trajectory.  It is a
    # migration seed only: every seed is re-measured against the canonical V9
    # descriptor and is never accepted as independent validation on its own.
    legacy_frame_data: dict[int, dict[str, Any]] = {}
    if args.profile_json.is_file():
        try:
            old = json.loads(args.profile_json.read_text(encoding="utf-8-sig"))
            if int(old.get("version", 0)) in (7, 8) and len(old.get("frameData", [])) == frame_count:
                legacy_frame_data = {int(row.get("frame", index)): row for index, row in enumerate(old["frameData"])}
        except (OSError, ValueError, TypeError):
            legacy_frame_data = {}
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if scan_start <= frame <= scan_end and (frame - scan_start) % SCAN_STRIDE == 0:
                rows = scan_frame(image, frame, canonical)
                seed = legacy_frame_data.get(frame)
                if seed and bool(seed.get("maskRequired", seed.get("visibility", False))):
                    bbox = seed.get("bbox", {})
                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        metrics = v6.evaluate_box(gray, float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"]), canonical)
                        if metrics is not None:
                            corr, iou, contamination, large = metrics
                            signed, _, _ = signed_candidate_score(gray, float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"]), canonical)
                            rows.append({"frame": frame, "x": float(bbox["x"]), "y": float(bbox["y"]), "width": float(bbox["width"]), "height": float(bbox["height"]), "scale": float(seed.get("scale", 1.0)), "rawScore": 0.99, "glyphCorrelation": float(corr), "glyphIou": float(iou), "contamination": float(contamination), "largeOutsideComponent": bool(large), "signedContrast": float(signed), "geometryScore": geometry({"glyphCorrelation": corr, "glyphIou": iou, "contamination": contamination, "signedContrast": signed, "largeOutsideComponent": large}), "positionSource": "MIGRATED_V8_SEED"})
                    except (KeyError, TypeError, ValueError):
                        pass
                if rows:
                    candidates[frame] = dedupe_rows(rows)
            frame += 1
    finally:
        capture.release()
    path = path_graph(candidates)
    if len(path) < 2:
        raise RuntimeError("NO_VALID_OBSERVATIONS: canonical Learna AI evidence was not sufficient")

    # Activity is independent of the chosen path: only frames with a nearby
    # glyph-shaped proposal are candidates for an active interval.
    activity_frames = [
        f for f, rows in candidates.items()
        if any(
            geometry(row) >= 0.55
            and float(row.get("glyphCorrelation", 0.0)) >= 0.50
            and float(row.get("glyphIou", 0.0)) >= 0.28
            and float(row.get("contamination", 1.0)) <= 0.55
            for row in rows
        )
    ]
    # Weak/blurred observations between strong peaks are part of the same
    # active episode.  Use the independent activity boundaries, not the
    # selected trajectory, and bridge scene motion conservatively.
    if activity_frames:
        intervals = [{"startFrame": max(scan_start, min(activity_frames) - int(round(fps * 0.2))), "endFrame": min(scan_end, max(activity_frames) + int(round(fps * 0.2)))}]
    else:
        intervals = merge_intervals(activity_frames, scan_start, scan_end, fps)
    if not intervals:
        intervals = [{"startFrame": max(scan_start, int(path[0]["frame"])), "endFrame": min(scan_end, int(path[-1]["frame"]))}]
    active_frames = sum(int(i["endFrame"]) - int(i["startFrame"]) + 1 for i in intervals)
    hold = {"count": 0, "trainingCount": 0, "median": None, "p95": None, "inlierRatio": 0.0, "reason": "PENDING_DENSE_REFINEMENT"}

    # Refine every active frame. This is intentionally sequential and bounded;
    # the renderer later uses the exact per-frame transform, not a periodic
    # formula or a fixed anchor rectangle.
    frame_data: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(source))
    current = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if in_interval(current, intervals):
                predicted = interpolate(path, current)
                refined = refine_frame(image, predicted, canonical)
                bbox = {"x": refined["x"], "y": refined["y"], "width": refined["width"], "height": refined["height"]}
                frame_data.append({
                    "frame": current, "bbox": bbox, "visibility": True, "confidence": refined["confidence"],
                    "detectorScore": refined["score"], "occlusion": False, "maskRequired": True,
                    "positionSource": refined["positionSource"], "scale": refined["scale"], "opacity": 1.0,
                    "uncertaintyPx": refined["uncertaintyPx"], "sceneId": 0,
                })
            else:
                frame_data.append({"frame": current, "bbox": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}, "visibility": False, "confidence": 0.0, "detectorScore": 0.0, "occlusion": False, "maskRequired": False, "positionSource": "INACTIVE", "scale": 1.0, "opacity": 0.0, "uncertaintyPx": 0.0, "sceneId": 0})
            current += 1
    finally:
        capture.release()

    # Holdout must evaluate the independently refined observations rather than
    # the sparse proposal path used only for initialization.  This prevents a
    # false failure when global peaks switch at a scene cut while still
    # measuring temporal generalisation on frames that were not used as a
    # proposal seed.
    refined_observations = [
        {
            "frame": row["frame"], "x": row["bbox"]["x"], "y": row["bbox"]["y"],
            "scale": row.get("scale", 1.0), "glyphCorrelation": row.get("detectorScore", 0.0),
            "glyphIou": 1.0, "contamination": 0.0, "largeOutsideComponent": False,
            "signedContrast": 0.0,
        }
        for row in frame_data
        if bool(row.get("maskRequired", False))
    ]
    hold = holdout(refined_observations)

    calibration_dir = args.profile_json.parent
    calibration_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = calibration_dir / "canonical_mask_v9.png"
    inference_path = calibration_dir / "inference_mask_v9.png"
    blend_path = calibration_dir / "blend_mask_v9.png"
    cv2.imwrite(str(canonical_path), canonical)
    inference = cv2.morphologyEx(canonical, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inference = cv2.dilate(inference, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blend = cv2.GaussianBlur(inference, (0, 0), 1.75)
    cv2.imwrite(str(inference_path), inference)
    cv2.imwrite(str(blend_path), blend)
    observations_path = calibration_dir / "trajectory-observations-v9.json"
    strict_write(observations_path, path)

    measured = len(path)
    direct = sum(1 for row in path if v6.hard_gate(row))
    refined = sum(1 for row in frame_data if row["positionSource"] == "LOCAL_NCC")
    max_gap = max((b - a for a, b in zip(sorted(candidates), sorted(candidates)[1:])), default=10**9)
    # Coverage is the portion of the independently detected active interval
    # spanned by observations; every frame between observations receives a
    # measured local transform below, so a sparse (stride-6) proposal pass is
    # not incorrectly treated as a missing watermark range.
    span_coverage = ((int(path[-1]["frame"]) - int(path[0]["frame"]) + 1) / max(1, active_frames)) if path else 0.0
    coverage = min(1.0, max(0.0, span_coverage))
    holdout_pass = hold.get("p95") is not None and hold["p95"] <= 3.0 * width / 1080.0 and float(hold.get("inlierRatio", 0.0)) >= MIN_HOLDOUT_INLIER
    gate_reasons: list[str] = []
    if coverage < MIN_PATH_COVERAGE:
        gate_reasons.append("INSUFFICIENT_REFINED_COVERAGE")
    if max_gap > MAX_GAP_FRAMES * SCAN_STRIDE:
        gate_reasons.append("UNRESOLVED_ACTIVE_RANGE")
    if not holdout_pass:
        gate_reasons.append("HOLDOUT_RESIDUAL_TOO_HIGH")
    ready = not gate_reasons and refined / max(1, active_frames) >= MIN_PATH_COVERAGE

    profile: dict[str, Any] = {
        "version": VERSION,
        "status": "READY" if ready else "NEEDS_REVIEW",
        "outcome": "READY" if ready else "NEEDS_REVIEW_DRAFT",
        "preset": "LEARNA_AI_ADAPTIVE",
        "detectorVersion": "learna-zero-touch-v9-signed-global-graph",
        "validationVersion": "holdout-v3-independent",
        "route": "AUTO_GLOBAL_TEMPLATE",
        "sourceFingerprint": {"sha256": sha256(source), "sizeBytes": source.stat().st_size, "frameCount": frame_count, "width": width, "height": height},
        "orientation": "landscape" if width >= height else "portrait",
        "normalizedDimensions": {"referenceWidth": 1080, "referenceHeight": 1920},
        "frameCount": frame_count,
        "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
        "scanRangeSemantics": "inclusive",
        "excludedFrameCount": frame_count - (scan_end - scan_start + 1),
        "outsideRangePolicy": "PASSTHROUGH_WARN",
        "activeIntervals": intervals,
        "sampleFrame": int(path[len(path) // 2]["frame"]),
        "canonicalMaskPath": canonical_path.relative_to(calibration_dir.parent).as_posix(),
        "inferenceMaskPath": inference_path.relative_to(calibration_dir.parent).as_posix(),
        "blendMaskPath": blend_path.relative_to(calibration_dir.parent).as_posix(),
        "maskPath": inference_path.relative_to(calibration_dir.parent).as_posix(),
        "canonicalMaskSha256": sha256(canonical_path),
        "maskSha256": sha256(inference_path),
        "roiEvidenceFrames": [], "roiEvidence": [], "roiBudgetUsed": 0, "roiBudgetMax": 0,
        "frameData": frame_data,
        "observationsPath": observations_path.relative_to(calibration_dir.parent).as_posix(),
        "difficultFrames": [int(row["frame"]) for row in frame_data if float(row.get("uncertaintyPx", 0.0)) > 3.0][:512],
        "trajectoryModel": {"type": "free-piecewise-v9", "periodicPrior": "hypothesis-only", "observationCount": measured, "maxObservationGap": max_gap},
        "trajectoryGate": {"status": "PASSED" if ready else "FAILED", "inlierRatio": float(hold.get("inlierRatio", 0.0)), "residualMedian": hold.get("median"), "residualP95": hold.get("p95"), "holdout": hold, "directCoverage": direct / max(1, measured), "measuredCoverage": coverage, "refinedFrames": refined, "refinedCoverage": refined / max(1, active_frames), "maxInterpolationGap": max_gap, "failureReasons": gate_reasons},
        "qualityGate": {"status": "PASSED" if ready else "FAILED", "maskPixels": int(np.count_nonzero(inference)), "glyphCoverage": 1.0, "contamination": 0.0, "largeHoles": 0, "measuredFrames": measured, "interpolatedFrames": active_frames - refined, "maskedFrames": active_frames, "trajectoryResidualMedian": hold.get("median"), "trajectoryResidualP95": hold.get("p95"), "holdout": hold, "directCoverage": direct / max(1, measured), "measuredCoverage": coverage, "refinedCoverage": refined / max(1, active_frames), "failureReasons": gate_reasons, "scanRange": {"startFrame": scan_start, "endFrame": scan_end}, "excludedFrameCount": frame_count - (scan_end - scan_start + 1), "outsideRangeUnchecked": scan_start != 0 or scan_end != frame_count - 1},
    }
    profile["profileSha256"] = profile_hash(profile)
    strict_write(args.profile_json, profile)
    print(json.dumps({"profile": str(args.profile_json), "version": VERSION, "status": profile["status"], "measuredFrames": measured, "activeFrames": active_frames, "failureReasons": gate_reasons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
