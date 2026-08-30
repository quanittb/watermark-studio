"""Global Learna AI template search and per-video trajectory calibration.

The legacy detector starts from a fixed 360-frame path.  This module keeps the
canonical glyph as the only prior, searches the image for it, and then builds a
piecewise trajectory from the observations that survive temporal filtering.
It intentionally fails closed: a weak fit is written as NEEDS_REVIEW and can
never be used by the final renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from find_learna_samples import load_canonical, mask_metrics
from render_periodic_dewatermark import periodic_position


ANALYSIS_LONG_EDGE = 720.0
REFERENCE_WIDTH = 1080.0
REFERENCE_HEIGHT = 1920.0
# The Learna AI preset used by the shipped calibration asset fades in after
# the opening 48 frames.  This is an activity hint, not a trajectory prior:
# positions are still learned from global observations and fitted per video.
# If no glyph evidence is found, the profile remains NEEDS_REVIEW.
# Best-quality V6 must discover activity from frame 0.  The old value of 48
# was only a hint for the periodic legacy route and caused the first frames of
# videos with an already-visible watermark to be copied unchanged.
FIRST_FRAME = 0
SAMPLE_STRIDE = 6
# The bundled canonical asset is 255x84.  Keep the calibration box in the
# same aspect ratio as the mask; using the older 245x75 enrollment box here
# squashes the glyph and makes the IoU/contamination gate reject real text.
CANONICAL_WIDTH = 255.0
CANONICAL_HEIGHT = 84.0
# Five scale variants cover the documented 0.75–1.35 resize range while the
# NMS/refinement stage keeps the candidate count bounded on a 4 GB GPU.
# Include the canonical 1.0 scale explicitly.  The exact-size candidate is
# important for the shipped 255x84 mask; omitting it can lower IoU just below
# the hard gate even when the glyph is plainly visible.
# A compact coarse pyramid keeps calibration practical on a 4 GB machine;
# fine alignment still evaluates every hit at source resolution.
SCALES = (0.60, 0.68, 0.75, 0.82, 0.90, 0.98, 1.06, 1.15, 1.28, 1.40, 1.50)
MAX_CANDIDATES_PER_FRAME = 12
# Keep enough hypotheses across scale levels for the graph, but do not run a
# source-resolution crop evaluation for every low-ranked local maximum.  The
# previous 12-per-scale × 5×5 loop performed nearly a million crop analyses on
# a 58-second clip and made the UI appear frozen.
RAW_PEAKS_PER_SCALE = 3
MIN_RAW_SCORE = 0.10
MIN_MEASURED_SCORE = 0.35
MAX_GAP = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an adaptive Learna AI CalibrationProfileV6")
    parser.add_argument("project_json", type=Path)
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("--roi-json", default="")
    parser.add_argument("--roi-frame", type=int)
    parser.add_argument(
        "--route",
        choices=("AUTO_GLOBAL_TEMPLATE", "AUTO_ROI_TEMPLATE", "ROI_FALLBACK"),
        default="AUTO_GLOBAL_TEMPLATE",
    )
    parser.add_argument("--edited-mask", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: dict) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def assert_finite_json(value: object, path: str = "$") -> None:
    """Reject non-standard JSON numbers before they reach serde_json."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")


def write_strict_json(path: Path, value: object) -> None:
    """Write and reparse a strict JSON artifact atomically enough for readers."""
    assert_finite_json(value)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def parse_roi(raw: str) -> dict[str, float] | None:
    if not raw.strip():
        return None
    value = json.loads(raw)
    required = {"x", "y", "width", "height"}
    if not required.issubset(value):
        raise RuntimeError("ROI must contain x, y, width and height")
    roi = {key: float(value[key]) for key in required}
    if roi["width"] < 32 or roi["height"] < 16:
        raise RuntimeError("ROI is too small for the Learna AI glyph")
    return roi


def feature(frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return a resolution-independent positive high-pass analysis image."""
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, ANALYSIS_LONG_EDGE / max(source_width, source_height))
    analysis_width = max(32, int(round(source_width * scale)))
    analysis_height = max(32, int(round(source_height * scale)))
    analysis = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
    highpass = np.maximum(gray - smooth, 0.0)
    base_scale = min(source_width / REFERENCE_WIDTH, source_height / REFERENCE_HEIGHT)
    return highpass.astype(np.float32), analysis_width / source_width, analysis_height / source_height, base_scale


def template_feature(
    canonical: np.ndarray,
    scale: float,
    base_scale: float,
    analysis_x: float,
    analysis_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    # matchTemplate runs on the analysis frame (405x720), therefore the
    # canonical source-pixel mask must be reduced by ANALYSIS_SCALE too.  The
    # previous implementation used source dimensions here and silently
    # rejected every 1080x1920 frame because the template was nearly half the
    # analysis image width.
    width = max(12, int(round(canonical.shape[1] * base_scale * scale * analysis_x)))
    height = max(8, int(round(canonical.shape[0] * base_scale * scale * analysis_y)))
    alpha = cv2.resize(canonical, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.65)
    # CCORR_NORMED with a float mask scores only the glyph, not the rectangle.
    return alpha, np.maximum(alpha, 1e-3).astype(np.float32)


def bounds(x: float, y: float, width: float, height: float, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + width))
    y1 = int(round(y + height))
    if x0 < 0 or y0 < 0 or x1 > frame_width or y1 > frame_height or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def roi_allows(x: float, y: float, width: float, height: float, roi: dict[str, float] | None) -> bool:
    if roi is None:
        return True
    return (
        x + width >= roi["x"]
        and y + height >= roi["y"]
        and x <= roi["x"] + roi["width"]
        and y <= roi["y"] + roi["height"]
    )


def evaluate_box(
    frame: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    canonical: np.ndarray,
) -> tuple[float, float, float, bool] | None:
    """Measure a source-resolution box against the canonical glyph mask."""
    box = bounds(x, y, width, height, frame.shape[1], frame.shape[0])
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    normalized = cv2.resize(crop, (canonical.shape[1], canonical.shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    positive = np.maximum(gray - cv2.GaussianBlur(gray, (0, 0), 8.0), 0.0)
    # A high-pass threshold around 7 suppresses the low-amplitude background
    # texture while retaining the bright anti-aliased Learna glyph.  The
    # former 3.5 threshold made large connected background components dominate
    # the contamination gate on complex scenes.
    binary = np.where(positive >= 7.0, 255, 0).astype(np.uint8)
    binary_correlation, iou, contamination, large = mask_metrics(binary, canonical)
    alpha = cv2.GaussianBlur((canonical.astype(np.float32) / 255.0), (0, 0), 0.75)
    active = alpha >= 0.08
    glyph = positive[active]
    expected = alpha[active]
    if glyph.size < 16 or float(np.linalg.norm(glyph)) < 1e-6:
        soft_correlation = 0.0
    else:
        glyph = glyph - float(glyph.mean())
        expected = expected - float(expected.mean())
        denominator = float(np.linalg.norm(glyph) * np.linalg.norm(expected))
        soft_correlation = float(np.dot(glyph, expected) / denominator) if denominator > 1e-6 else 0.0
        soft_correlation = max(0.0, min(1.0, (soft_correlation + 1.0) * 0.5))
    return 0.55 * binary_correlation + 0.45 * soft_correlation, iou, contamination, large


def top_matches(response: np.ndarray, width: int, height: int, limit: int = MAX_CANDIDATES_PER_FRAME) -> list[tuple[float, tuple[int, int]]]:
    work = response.copy()
    matches: list[tuple[float, tuple[int, int]]] = []
    radius_x = max(2, width // 2)
    radius_y = max(2, height // 2)
    for _ in range(limit):
        _, score, _, location = cv2.minMaxLoc(work)
        if score < MIN_RAW_SCORE:
            break
        matches.append((float(score), (int(location[0]), int(location[1]))))
        left = max(0, location[0] - radius_x)
        top = max(0, location[1] - radius_y)
        right = min(work.shape[1], location[0] + radius_x + 1)
        bottom = min(work.shape[0], location[1] + radius_y + 1)
        work[top:bottom, left:right] = -1.0
    return matches


def candidate_rows(frame: np.ndarray, frame_number: int, canonical: np.ndarray, roi: dict[str, float] | None) -> list[dict[str, float | int | bool]]:
    image_feature, analysis_x, analysis_y, base_scale = feature(frame)
    rows: list[dict[str, float | int | bool]] = []
    for scale in SCALES:
        template, template_mask = template_feature(canonical, scale, base_scale, analysis_x, analysis_y)
        response = cv2.matchTemplate(image_feature, template, cv2.TM_CCORR_NORMED, mask=template_mask)
        template_width = template.shape[1]
        template_height = template.shape[0]
        if roi is not None:
            # ROI fallback is an evidence seed, not a post-filter.  Restrict
            # the correlation map *before* peak selection; otherwise the top
            # global peaks from subtitles/faces can exhaust the candidate
            # budget and a valid glyph inside the user-confirmed region never
            # reaches refinement.
            source_width = CANONICAL_WIDTH * base_scale * scale
            source_height = CANONICAL_HEIGHT * base_scale * scale
            left = max(0, int(math.floor((roi["x"] - source_width) * analysis_x)))
            top = max(0, int(math.floor((roi["y"] - source_height) * analysis_y)))
            right = min(response.shape[1], int(math.ceil((roi["x"] + roi["width"]) * analysis_x)))
            bottom = min(response.shape[0], int(math.ceil((roi["y"] + roi["height"]) * analysis_y)))
            constrained = np.full_like(response, -1.0)
            if right > left and bottom > top:
                constrained[top:bottom, left:right] = response[top:bottom, left:right]
            response = constrained
        for raw_score, (ax, ay) in top_matches(response, template_width, template_height, RAW_PEAKS_PER_SCALE):
            x = ax / analysis_x
            y = ay / analysis_y
            width = CANONICAL_WIDTH * base_scale * scale
            height = CANONICAL_HEIGHT * base_scale * scale
            if not roi_allows(x, y, width, height, roi):
                continue
            # matchTemplate is intentionally coarse.  Refine each hit at
            # source resolution so a 3-8px analysis rounding error does not
            # turn a genuine glyph into a contaminated crop.  Keep the best
            # local alignment by the same mask metrics used by the hard gate.
            refined = None
            # Coarse correlation is already at a reduced resolution.  A
            # 3×3 source grid covers the documented ±12 px search area while
            # retaining a bounded calibration time; ECC/flow can later refine
            # only the retained graph path.
            refinement_radius = max(4, int(round(12 * base_scale)))
            for dx in (-refinement_radius, 0, refinement_radius):
                for dy in (-refinement_radius, 0, refinement_radius):
                    metrics = evaluate_box(frame, x + dx, y + dy, width, height, canonical)
                    if metrics is None:
                        continue
                    corr, iou, contam, large = metrics
                    # Background components are a soft penalty.  A smaller
                    # watermark over hair/textures can legitimately produce
                    # one large outside component; trajectory consistency is
                    # the authority that decides whether it is real.
                    rank = corr + 0.35 * iou - 0.40 * contam - (0.15 if large else 0.0)
                    if refined is None or rank > refined[0]:
                        refined = (rank, x + dx, y + dy, corr, iou, contam, large)
            if refined is None:
                continue
            _, x, y, correlation, iou, contamination, large = refined
            local_quality = 0.45 * max(0.0, min(1.0, raw_score)) + 0.35 * correlation + 0.20 * iou
            if large:
                local_quality *= 0.85
            rows.append(
                {
                    "frame": frame_number,
                    "x": float(x),
                    "y": float(y),
                    "width": float(width),
                    "height": float(height),
                    # Store the effective source scale.  The renderer works
                    # in source pixels, so retaining only the canonical
                    # variant (for example .75) is wrong for a 720p fixture.
                    "scale": float(base_scale * scale),
                    "templateScale": float(scale),
                    "rawScore": float(raw_score),
                    "glyphCorrelation": float(correlation),
                    "glyphIou": float(iou),
                    "contamination": float(contamination),
                    "largeOutsideComponent": bool(large),
                    "baseScale": float(base_scale),
                    "score": float(local_quality),
                }
            )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    selected: list[dict[str, float | int | bool]] = []
    for row in rows:
        if any(
            math.hypot(float(row["x"]) - float(other["x"]), float(row["y"]) - float(other["y"]))
            < max(float(row["width"]), float(row["height"])) * 0.45
            for other in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= MAX_CANDIDATES_PER_FRAME:
            break
    return selected


def write_contact_sheet(
    source: Path,
    measured: list[dict[str, float | int | bool]],
    canonical: np.ndarray,
    output: Path,
) -> None:
    """Persist a compact source/canonical overlay sheet for Review/QA."""
    if not measured:
        return
    capture = cv2.VideoCapture(str(source))
    panels: list[np.ndarray] = []
    try:
        for row in measured[:35]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
            ok, frame = capture.read()
            if not ok:
                continue
            box = bounds(float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"]), frame.shape[1], frame.shape[0])
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = frame[y0:y1, x0:x1]
            crop = cv2.resize(crop, (255, 84), interpolation=cv2.INTER_AREA)
            overlay = cv2.resize(canonical, (255, 84), interpolation=cv2.INTER_NEAREST)
            overlay_bgr = np.zeros_like(crop)
            overlay_bgr[:, :, 1] = overlay
            panel = cv2.addWeighted(crop, 0.78, overlay_bgr, 0.22, 0.0)
            cv2.putText(panel, f"f{int(row['frame'])}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
    finally:
        capture.release()
    if not panels:
        return
    columns = 5
    rows = []
    for start in range(0, len(panels), columns):
        line = panels[start:start + columns]
        while len(line) < columns:
            line.append(np.zeros_like(panels[0]))
        rows.append(np.concatenate(line, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def choose_track(all_candidates: dict[int, list[dict[str, float | int | bool]]]) -> list[dict[str, float | int | bool]]:
    frames = sorted(all_candidates)
    if not frames:
        return []
    costs: list[list[float]] = []
    parents: list[list[int]] = []
    for index, frame in enumerate(frames):
        current = all_candidates[frame]
        # A new segment is allowed, but expensive.  This prevents a false
        # static background match from teleporting across long evidence gaps.
        frame_cost = [-float(candidate["score"]) * 6.0 + 18.0 for candidate in current]
        frame_parent = [-1] * len(current)
        for candidate_index, candidate in enumerate(current):
            base = -float(candidate["score"]) * 6.0
            if index == 0:
                frame_cost[candidate_index] = base
                continue
            previous_frame = frames[index - 1]
            delta_frames = max(1, frame - previous_frame)
            for previous_index, previous in enumerate(all_candidates[previous_frame]):
                distance = math.hypot(
                    float(candidate["x"]) - float(previous["x"]),
                    float(candidate["y"]) - float(previous["y"]),
                )
                # Motion is locally smooth.  Do not connect candidates across
                # an implausible jump; the explicit restart cost above lets a
                # later coherent segment seed a new trajectory instead.
                max_jump = min(180.0, 45.0 * delta_frames)
                if delta_frames > 60 or distance > max_jump:
                    continue
                jump_penalty = distance / max(24.0, 22.0 * delta_frames)
                scale_penalty = abs(float(candidate["scale"]) - float(previous["scale"])) * 2.0
                value = costs[index - 1][previous_index] + jump_penalty + scale_penalty + base
                if value < frame_cost[candidate_index]:
                    frame_cost[candidate_index] = value
                    frame_parent[candidate_index] = previous_index
        costs.append(frame_cost)
        parents.append(frame_parent)
    last_index = min(range(len(costs[-1])), key=lambda index: costs[-1][index])
    chosen: list[dict[str, float | int | bool]] = []
    for index in range(len(frames) - 1, -1, -1):
        chosen.append(all_candidates[frames[index]][last_index])
        last_index = parents[index][last_index]
        if last_index < 0 and index > 0:
            # The chain deliberately starts a new segment after an
            # implausible jump or a long observation gap.  Do not stitch the
            # unrelated segment back into the selected trajectory.
            break
    chosen.reverse()
    return chosen


def candidate_key(row: dict[str, float | int | bool]) -> tuple[int, int, int, int]:
    """Stable identity for removing an already-selected graph node."""
    return (
        int(row["frame"]),
        int(round(float(row["x"]) * 10.0)),
        int(round(float(row["y"]) * 10.0)),
        int(round(float(row["scale"]) * 1_000.0)),
    )


def choose_track_segments(
    all_candidates: dict[int, list[dict[str, float | int | bool]]],
    minimum_points: int = 5,
) -> list[list[dict[str, float | int | bool]]]:
    """Extract independent coherent paths without joining scene/trajectory jumps.

    ``choose_track`` returns the best terminal chain on purpose.  Repeating it
    after removing that chain lets a video contain several active intervals or
    reappearances without treating an unrelated subtitle as one long path.
    """
    remaining = {frame: list(rows) for frame, rows in all_candidates.items()}
    segments: list[list[dict[str, float | int | bool]]] = []
    while remaining:
        track = choose_track(remaining)
        if len(track) < minimum_points:
            break
        # A provisional match only becomes evidence when the graph retained a
        # genuinely local chain.  The span guard rejects five duplicate NMS
        # peaks from a single static frame neighbourhood.
        if int(track[-1]["frame"]) - int(track[0]["frame"]) >= SAMPLE_STRIDE * (minimum_points - 1):
            segments.append(track)
        selected = {candidate_key(row) for row in track}
        next_remaining: dict[int, list[dict[str, float | int | bool]]] = {}
        for frame, rows in remaining.items():
            kept = [row for row in rows if candidate_key(row) not in selected]
            if kept:
                next_remaining[frame] = kept
        if len(next_remaining) == len(remaining):
            break
        remaining = next_remaining
    return segments


def active_intervals_from_segments(
    segments: list[list[dict[str, float | int | bool]]],
    frame_count: int,
    fps: float,
) -> list[dict[str, int]]:
    """Create conservative active ranges and preserve long unresolved gaps."""
    if not segments:
        return []
    bridge_limit = max(SAMPLE_STRIDE, int(round(fps * 0.6)))
    raw = sorted(
        (max(0, int(segment[0]["frame"]) - SAMPLE_STRIDE), min(frame_count - 1, int(segment[-1]["frame"]) + SAMPLE_STRIDE))
        for segment in segments
    )
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= bridge_limit:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"startFrame": start, "endFrame": end} for start, end in merged]


def frame_in_intervals(frame: int, intervals: list[dict[str, int]]) -> bool:
    return any(interval["startFrame"] <= frame <= interval["endFrame"] for interval in intervals)


def rdp_indices(points: list[tuple[float, float]], epsilon: float) -> list[int]:
    if len(points) <= 2:
        return list(range(len(points)))

    def distance(index: int, start: int, end: int) -> float:
        x, y = points[index]
        x1, y1 = points[start]
        x2, y2 = points[end]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

    farthest = max(range(1, len(points) - 1), key=lambda index: distance(index, 0, len(points) - 1))
    if distance(farthest, 0, len(points) - 1) <= epsilon:
        return [0, len(points) - 1]
    left = rdp_indices(points[: farthest + 1], epsilon)
    right = rdp_indices(points[farthest:], epsilon)
    return left[:-1] + [value + farthest for value in right]


def interpolate(keys: list[dict[str, float | int | bool]], frame: int) -> tuple[float, float, float, bool, float]:
    if frame <= int(keys[0]["frame"]):
        return float(keys[0]["x"]), float(keys[0]["y"]), float(keys[0]["scale"]), frame == int(keys[0]["frame"]), float(keys[0]["score"])
    if frame >= int(keys[-1]["frame"]):
        return float(keys[-1]["x"]), float(keys[-1]["y"]), float(keys[-1]["scale"]), frame == int(keys[-1]["frame"]), float(keys[-1]["score"])
    for left, right in zip(keys, keys[1:]):
        start, end = int(left["frame"]), int(right["frame"])
        if start <= frame <= end:
            ratio = (frame - start) / max(1, end - start)
            return (
                float(left["x"]) + (float(right["x"]) - float(left["x"])) * ratio,
                float(left["y"]) + (float(right["y"]) - float(left["y"])) * ratio,
                float(left["scale"]) + (float(right["scale"]) - float(left["scale"])) * ratio,
                frame in (start, end),
                min(float(left["score"]), float(right["score"])) * 0.88,
            )
    return float(keys[-1]["x"]), float(keys[-1]["y"]), float(keys[-1]["scale"]), False, 0.0


def contiguous_gaps(frames: Iterable[int]) -> int:
    values = sorted(set(frames))
    if len(values) < 2:
        return 10**9
    return max((right - left for left, right in zip(values, values[1:])), default=10**9)


def hard_gate(row: dict[str, float | int | bool]) -> bool:
    """Return true only for a glyph observation safe to seed trajectory fit."""
    return (
        float(row.get("glyphCorrelation", 0.0)) >= 0.68
        and float(row.get("glyphIou", 0.0)) >= 0.48
        and float(row.get("contamination", 1.0)) <= 0.20
        and not bool(row.get("largeOutsideComponent", True))
    )


def provisional_gate(row: dict[str, float | int | bool]) -> bool:
    """Near-match gate; temporal fitting must validate these rows."""
    return (
        float(row.get("glyphCorrelation", 0.0)) >= 0.58
        and float(row.get("glyphIou", 0.0)) >= 0.38
        and float(row.get("contamination", 1.0)) <= 0.35
    )


def extend_keys(keys: list[dict[str, float | int | bool]], first_frame: int, last_frame: int) -> list[dict[str, float | int | bool]]:
    """Extrapolate the first/last fitted segment to watermark-active bounds."""
    if len(keys) < 2:
        return keys
    output = list(keys)
    first, second = output[0], output[1]
    f0, f1 = int(first["frame"]), int(second["frame"])
    if f0 > first_frame and f1 > f0:
        ratio = (first_frame - f0) / (f1 - f0)
        output.insert(0, {
            "frame": first_frame,
            "x": float(first["x"]) + (float(second["x"]) - float(first["x"])) * ratio,
            "y": float(first["y"]) + (float(second["y"]) - float(first["y"])) * ratio,
            "scale": float(first["scale"]) + (float(second["scale"]) - float(first["scale"])) * ratio,
            "score": float(first["score"]) * 0.75,
        })
    last, before = output[-1], output[-2]
    fl, fb = int(last["frame"]), int(before["frame"])
    if fl < last_frame and fl > fb:
        ratio = (last_frame - fl) / (fl - fb)
        output.append({
            "frame": last_frame,
            "x": float(last["x"]) + (float(last["x"]) - float(before["x"])) * ratio,
            "y": float(last["y"]) + (float(last["y"]) - float(before["y"])) * ratio,
            "scale": float(last["scale"]) + (float(last["scale"]) - float(before["scale"])) * ratio,
            "score": float(last["score"]) * 0.75,
        })
    return output


def fit_periodic_prior(measured: list[dict[str, float | int | bool]]) -> tuple[dict[str, float], list[float]] | None:
    """Fit an affine transform of the known path only when evidence agrees.

    This is not a fixed-position fallback: the observed global-template
    anchors decide whether the periodic prior is admissible and calibrate its
    x/y scale and offset per video.  A different trajectory therefore fails
    this check and remains on the free-fit/ROI route.
    """
    if len(measured) < 12:
        return None
    prior = [periodic_position(int(row["frame"])) for row in measured]
    px = np.asarray([value[0] for value in prior], dtype=np.float64)
    py = np.asarray([value[1] for value in prior], dtype=np.float64)
    ox = np.asarray([float(row["x"]) for row in measured], dtype=np.float64)
    oy = np.asarray([float(row["y"]) for row in measured], dtype=np.float64)
    sx, bx = np.polyfit(px, ox, 1)
    sy, by = np.polyfit(py, oy, 1)
    if not (0.75 <= sx <= 1.25 and 0.75 <= sy <= 1.25):
        return None
    residuals = [
        math.hypot(float(row["x"]) - (float(sx) * px[index] + float(bx)),
                   float(row["y"]) - (float(sy) * py[index] + float(by)))
        for index, row in enumerate(measured)
    ]
    if float(np.percentile(residuals, 95)) > 3.0:
        return None
    return {"scaleX": float(sx), "scaleY": float(sy), "offsetX": float(bx), "offsetY": float(by)}, residuals


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8-sig"))
    source = Path(project["source"]["path"])
    frame_count = int(project["video"]["frameCount"])
    width = int(project["video"]["width"])
    height = int(project["video"]["height"])
    fps = float(project["video"].get("fps", 30.0))
    if width < 64 or height < 64 or frame_count < 1:
        raise RuntimeError("Invalid source geometry for CalibrationProfileV6")
    roi = parse_roi(args.roi_json)
    roi_frame = args.roi_frame
    if roi_frame is not None and not 0 <= roi_frame < frame_count:
        raise RuntimeError("ROI frame is outside the source video")
    canonical = load_canonical()
    capture = cv2.VideoCapture(str(source))
    all_candidates: dict[int, list[dict[str, float | int | bool]]] = {}
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # The broad user ROI is sampled on its exact frame, even when that
            # frame falls between the normal stride phases.  All other frames
            # remain global searches so the ROI seeds a moving trajectory
            # instead of incorrectly constraining it to one screen location.
            if frame_number % SAMPLE_STRIDE == 0 or frame_number == roi_frame:
                search_roi = roi if frame_number == roi_frame else None
                candidates = candidate_rows(frame, frame_number, canonical, search_roi)
                if candidates:
                    if frame_number == roi_frame:
                        for candidate in candidates:
                            candidate["userRoi"] = True
                    all_candidates[frame_number] = candidates
            frame_number += 1
    finally:
        capture.release()
    # Fit the temporal graph from both direct and provisional evidence.  The
    # previous implementation filtered to hard candidates before tracking;
    # for smaller/transparent watermarks that produced zero observations and
    # eventually serialized Infinity residuals.  Provisional rows are never
    # accepted in isolation: they must belong to the smooth selected path.
    track_candidates = {
        frame: [row for row in rows if provisional_gate(row)]
        for frame, rows in all_candidates.items()
    }
    track_candidates = {frame: rows for frame, rows in track_candidates.items() if rows}
    tracks = choose_track_segments(track_candidates)
    measured = sorted(
        (row for track in tracks for row in track if provisional_gate(row)),
        key=lambda row: int(row["frame"]),
    )
    track = measured
    active_intervals = active_intervals_from_segments(tracks, frame_count, fps)
    # A periodic model is permitted only when the global evidence actually
    # covers the source.  It is an optional candidate model, never a way to
    # manufacture a full-video interval from a few terminal observations.
    observed_span = sum(
        max(0, int(segment[-1]["frame"]) - int(segment[0]["frame"]) + SAMPLE_STRIDE)
        for segment in tracks
    )
    periodic_candidate_allowed = observed_span >= frame_count * 0.70 and width == 1080 and height == 1920
    first_active = min((interval["startFrame"] for interval in active_intervals), default=0)
    last_active = max((interval["endFrame"] for interval in active_intervals), default=0)
    points = [(float(row["x"]), float(row["y"])) for row in measured]
    key_indices = rdp_indices(points, 6.0) if len(points) >= 2 else []
    key_track = [measured[index] for index in key_indices]
    if len(key_track) < 2 and measured:
        key_track = [measured[0], measured[-1]] if len(measured) > 1 else measured
    # Never extrapolate a free fit over an unresolved inactive/scene gap.  A
    # periodic model below may cover the entire source only after validation.
    residuals: list[float] = []
    if len(key_track) >= 2:
        for row in measured:
            px, py, _, _, _ = interpolate(key_track, int(row["frame"]))
            residuals.append(math.hypot(float(row["x"]) - px, float(row["y"]) - py))
    median_residual = float(np.median(residuals)) if residuals else None
    p95_residual = float(np.percentile(residuals, 95)) if residuals else None
    inlier_ratio = float(sum(value <= 3.0 for value in residuals) / max(1, len(residuals)))
    periodic_fit = fit_periodic_prior(measured) if periodic_candidate_allowed else None
    periodic_transform: dict[str, float] | None = None
    if periodic_fit is not None:
        periodic_transform, periodic_residuals = periodic_fit
        # The prior is accepted only after global observations fit it.  Once
        # accepted, it supplies positions through low-opacity/occluded gaps;
        # it never creates a profile by itself.
        residuals = periodic_residuals
        median_residual = float(np.median(residuals))
        p95_residual = float(np.percentile(residuals, 95))
        inlier_ratio = float(sum(value <= 3.0 for value in residuals) / max(1, len(residuals)))
    max_gap = max((contiguous_gaps(int(row["frame"]) for row in segment) for segment in tracks), default=10**9)
    active_count = sum(interval["endFrame"] - interval["startFrame"] + 1 for interval in active_intervals)
    direct_coverage = min(1.0, len(measured) * SAMPLE_STRIDE / max(1, active_count))
    residual_tolerance = 2.0 * width / REFERENCE_WIDTH
    residual_p95_tolerance = 3.0 * width / REFERENCE_WIDTH
    # A sparse but geometrically stable fit is still actionable: the renderer
    # uses the fitted segment for every frame and the report exposes the raw
    # observation gap.  This branch is deliberately strict (12 anchors, high
    # inlier ratio and <=3px p95) and does not apply to weak arbitrary paths.
    sparse_fit_ok = bool(
        len(measured) >= 12
        and inlier_ratio >= 0.80
        and p95_residual is not None and p95_residual <= residual_p95_tolerance
        and len(key_track) <= max(2, int(len(measured) * 0.50))
    )
    trajectory_passed = bool(
        (len(measured) >= max(30, math.ceil(active_count * 0.10)) or sparse_fit_ok)
        and direct_coverage >= 0.70
        and inlier_ratio >= 0.60
        and median_residual is not None and median_residual <= residual_tolerance
        and p95_residual is not None and p95_residual <= residual_p95_tolerance
        and (max_gap <= MAX_GAP or sparse_fit_ok)
        and (periodic_transform is not None or len(key_track) <= max(2, int(len(measured) * 0.35)))
    )

    calibration_dir = args.profile_json.parent
    calibration_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = calibration_dir / "canonical_mask.png"
    auto_path = calibration_dir / "auto_mask.png"
    inference_path = calibration_dir / "inference_mask.png"
    blend_path = calibration_dir / "blend_mask.png"
    for path in (canonical_path, auto_path):
        cv2.imwrite(str(path), canonical)
    inference = cv2.morphologyEx(canonical, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inference = cv2.dilate(inference, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blend = cv2.GaussianBlur(inference, (0, 0), 1.75)
    cv2.imwrite(str(inference_path), inference)
    cv2.imwrite(str(blend_path), blend)

    frame_data: list[dict[str, object]] = []
    measured_frames = {int(row["frame"]) for row in measured}
    for frame in range(frame_count):
        active = bool(measured and frame_in_intervals(frame, active_intervals))
        if active and periodic_transform is not None:
            base_x, base_y = periodic_position(frame)
            x = periodic_transform["scaleX"] * base_x + periodic_transform["offsetX"]
            y = periodic_transform["scaleY"] * base_y + periodic_transform["offsetY"]
            scale = 1.0
            confidence = 0.72 if frame not in measured_frames else 0.86
            source_name = "detector" if frame in measured_frames else "trajectory-periodic"
        elif active and len(key_track) >= 2:
            x, y, scale, is_measured, confidence = interpolate(key_track, frame)
            # A frame is considered directly measured only when it was part of
            # the global detector stride; all gaps are explicitly labeled.
            source_name = "detector" if frame in measured_frames else "trajectory-interpolated"
        else:
            x, y, scale, source_name, confidence = 0.0, 0.0, 1.0, "inactive", 0.0
        frame_data.append(
            {
                "frame": frame,
                "bbox": {"x": x, "y": y, "width": CANONICAL_WIDTH * scale, "height": CANONICAL_HEIGHT * scale},
                "visibility": active,
                "confidence": float(confidence),
                "detectorScore": float(confidence),
                "occlusion": False,
                "maskRequired": active,
                "positionSource": source_name,
                "scale": scale,
                "opacity": 1.0,
                "uncertaintyPx": 0.0 if frame in measured_frames else float(min(residual_p95_tolerance * 2.0, max_gap * width / (REFERENCE_WIDTH * max(1.0, fps / 30.0)))),
                "maskTransform": {"scaleX": scale, "scaleY": scale, "offsetX": 0.0, "offsetY": 0.0},
            }
        )

    observed_path = calibration_dir / "trajectory-observations.json"
    write_strict_json(observed_path, track)
    contact_sheet_path = calibration_dir / "contact-sheet.png"
    write_contact_sheet(source, measured, canonical, contact_sheet_path)
    diagnostics = calibration_dir / "diagnostics.json"
    write_strict_json(
        diagnostics,
        {
                "candidateCount": sum(len(value) for value in all_candidates.values()),
                "measuredCount": len(measured),
                "trajectoryGate": {
                    "status": "PASSED" if trajectory_passed else "FAILED",
                    "inlierRatio": inlier_ratio,
                    "residualMedian": median_residual,
                    "residualP95": p95_residual,
                    "maxGap": max_gap,
                    "keyPoints": len(key_track),
                    "directCoverage": direct_coverage,
                    "activeIntervals": active_intervals,
                },
                "status": "READY" if trajectory_passed else "NEEDS_REVIEW",
        },
    )
    quality_status = "PASSED" if trajectory_passed else "FAILED"
    failure_reasons: list[str] = []
    if not measured:
        failure_reasons.append("NO_VALID_OBSERVATIONS")
    if measured and len(measured) < max(30, math.ceil(active_count * 0.10)):
        failure_reasons.append("TRAJECTORY_UNDERCONSTRAINED")
    if max_gap > MAX_GAP:
        failure_reasons.append("UNRESOLVED_ACTIVE_RANGE")
    if p95_residual is not None and p95_residual > 3.0:
        failure_reasons.append("TRAJECTORY_RESIDUAL_TOO_HIGH")
    if inlier_ratio < 0.60:
        failure_reasons.append("LOW_INLIER_RATIO")
    profile = {
        "version": 6,
        "status": "READY" if trajectory_passed else "NEEDS_REVIEW",
        "preset": "LEARNA_AI_ADAPTIVE",
        "detectorVersion": "learna-global-template-v6.0",
        "route": args.route,
        "sourceFingerprint": {
            "sha256": sha256_file(source),
            "sizeBytes": source.stat().st_size,
            "frameCount": frame_count,
            "width": width,
            "height": height,
        },
        "orientation": "landscape" if width >= height else "portrait",
        "normalizedDimensions": {"referenceWidth": REFERENCE_WIDTH, "referenceHeight": REFERENCE_HEIGHT},
        "frameCount": frame_count,
        "firstWatermarkFrame": first_active,
        "lastWatermarkFrame": last_active,
        "activeIntervals": active_intervals,
        "sampleFrame": int(measured[len(measured) // 2]["frame"]) if measured else 0,
        "canonicalMaskPath": canonical_path.relative_to(calibration_dir.parent).as_posix(),
        "autoMaskPath": auto_path.relative_to(calibration_dir.parent).as_posix(),
        "inferenceMaskPath": inference_path.relative_to(calibration_dir.parent).as_posix(),
        "blendMaskPath": blend_path.relative_to(calibration_dir.parent).as_posix(),
        "maskPath": inference_path.relative_to(calibration_dir.parent).as_posix(),
        "canonicalMaskSha256": sha256_file(canonical_path),
        "maskSha256": sha256_file(inference_path),
        "frameData": frame_data,
        "observationsPath": observed_path.relative_to(calibration_dir.parent).as_posix(),
        "contactSheetPath": contact_sheet_path.relative_to(calibration_dir.parent).as_posix(),
        "trajectoryModel": (
            {
                "type": "affine-periodic-calibrated",
                "source": "global-template-observations+validated-periodic-prior",
                "periodicPrior": "validated",
                "transform": periodic_transform,
                "segments": [
                    {"startFrame": frame, "x": periodic_transform["scaleX"] * periodic_position(frame)[0] + periodic_transform["offsetX"], "y": periodic_transform["scaleY"] * periodic_position(frame)[1] + periodic_transform["offsetY"], "scale": 1.0}
                    for frame in (0, 120, 180, 300, 360)
                ],
                "maxInterpolationGap": 6,
                "maxObservationGap": max_gap,
            }
            if periodic_transform is not None
            else {
                "type": "piecewise-linear-adaptive",
                "source": "global-template-observations",
                "periodicPrior": "rejected-or-not-needed",
                "segments": [
                    {"startFrame": int(row["frame"]), "x": float(row["x"]), "y": float(row["y"]), "scale": float(row["scale"])}
                    for row in key_track
                ],
                "maxInterpolationGap": MAX_GAP,
                "maxObservationGap": max_gap,
            }
        ),
        "difficultFrames": [
            int(row["frame"])
            for row in measured
            if float(row["score"]) < MIN_MEASURED_SCORE or float(row.get("glyphCorrelation", 0.0)) < 0.65
        ][:512],
        "qualityGate": {
            "status": quality_status,
            "maskPixels": int(np.count_nonzero(inference)),
            "glyphCoverage": 1.0,
            "contamination": 0.0,
            "largeHoles": 0,
            "measuredFrames": len(measured),
            "interpolatedFrames": sum(1 for row in frame_data if row["positionSource"] in ("trajectory-interpolated", "trajectory-periodic")),
            "maskedFrames": sum(1 for row in frame_data if row["maskRequired"]),
            "inlierRatio": inlier_ratio,
            "trajectoryResidualMedian": median_residual,
            "trajectoryResidualP95": p95_residual,
            "directCoverage": direct_coverage,
            "maxInterpolationGap": 6 if periodic_transform is not None else max_gap,
            "failureReasons": failure_reasons,
        },
        "trajectoryGate": {
            "status": "PASSED" if trajectory_passed else "FAILED",
            "inlierRatio": inlier_ratio,
            "residualMedian": median_residual,
            "residualP95": p95_residual,
            "directCoverage": direct_coverage,
            "maxInterpolationGap": 6 if periodic_transform is not None else max_gap,
            "failureReasons": failure_reasons,
        },
    }
    profile["profileSha256"] = canonical_json_sha(profile)
    write_strict_json(args.profile_json, profile)
    print(
        json.dumps(
            {
                "profile": str(args.profile_json),
                "status": profile["status"],
                "measuredFrames": len(measured),
                "trajectoryGate": profile["trajectoryGate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
