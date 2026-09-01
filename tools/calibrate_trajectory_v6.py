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
# The canonical PNG is a 255x84 mask canvas that contains a 4 px context
# border around the actual 245.33x74.67 source glyph box. Keep those two
# coordinate spaces separate: template matching searches the padded canvas,
# while profile bboxes describe the source glyph box consumed by the renderer.
CANONICAL_WIDTH = 245.33333333333334
CANONICAL_HEIGHT = 74.66666666666667
MASK_CANVAS_WIDTH = 255.0
MASK_CANVAS_HEIGHT = 84.0
MASK_BORDER_X = (MASK_CANVAS_WIDTH - CANONICAL_WIDTH) / 2.0
MASK_BORDER_Y = (MASK_CANVAS_HEIGHT - CANONICAL_HEIGHT) / 2.0
# The scale pyramid covers the documented 0.75–1.35 resize range plus a small
# margin for encoded/rescaled clips.  The NMS/refinement stage keeps the
# candidate count bounded on a 4 GB GPU.
# Include the canonical 1.0 scale explicitly.  The exact-size candidate is
# important for the shipped 255x84 mask; omitting it can lower IoU just below
# the hard gate even when the glyph is plainly visible.
# A compact coarse pyramid keeps calibration practical on a 4 GB machine;
# fine alignment still evaluates every hit at source resolution.
SCALES = (0.60, 0.68, 0.75, 0.82, 0.90, 0.98, 1.06, 1.15, 1.28, 1.40, 1.50)
MAX_CANDIDATES_PER_FRAME = 20
# Keep enough hypotheses across scale levels for the graph, but do not run a
# source-resolution crop evaluation for every low-ranked local maximum.  The
# previous 12-per-scale × 5×5 loop performed nearly a million crop analyses on
# a 58-second clip and made the UI appear frozen.
# Busy shots can contain a face, subtitle or UI edge that scores above the
# faint translucent watermark. Retain more raw peaks so the real glyph is not
# discarded before source-resolution refinement; final NMS still caps rows.
RAW_PEAKS_PER_SCALE = 8
MIN_RAW_SCORE = 0.10
MIN_MEASURED_SCORE = 0.35
MAX_GAP = 18
# ROI hints are guidance for a human, not a frame-by-frame to-do list.  A
# detector failure can otherwise emit one range for every stride-sized hole,
# even when all holes belong to the same motion segment.  Keep the individual
# ranges available in diagnostics, but present only representative clusters
# in the quality-gate result.
REVIEW_MERGE_GAP = 72
REVIEW_MAX_CLUSTER_SPAN = 360
MAX_REVIEW_RANGES = 8
# Once enough independent ROI anchors cover the motion, repeatedly asking the
# user for more anchors is not useful: a high residual then indicates that the
# fitted path itself needs refinement.  Keep these thresholds deliberately
# conservative so this is only a UX/convergence guard; it never relaxes the
# trajectory or render quality gates.
ROI_SATURATION_MIN_EVIDENCE = 24
ROI_SATURATION_MIN_CONFIRMED_COVERAGE = 0.15
ROI_SATURATION_MIN_PATH_COVERAGE = 0.70
# V7 changes the acceptance contract: the selected path must be locally
# refined and validated on held-out observations.  Never reuse a V6 cache for
# final calibration, even when its source and mask hashes match.
CANDIDATE_CACHE_VERSION = 3
CALIBRATION_VERSION = 7
MIN_INLIER_RATIO = 0.80
LOCAL_REFINE_STRIDE = 2
LOCAL_REFINE_RADIUS = 12
LOCAL_REFINE_STEP = 3


def should_suppress_roi_review(
    roi_evidence_count: int,
    confirmed_coverage: float,
    measured_coverage: float,
    residual_p95: float | None,
    residual_p95_tolerance: float,
    max_gap: int,
) -> bool:
    """Stop an endless ROI loop after evidence saturation.

    This helper only controls actionable UI hints.  A saturated profile still
    fails closed when residual/gap quality is not within the normal gate and
    must be refined automatically or reviewed by the user.
    """
    enough_evidence = (
        roi_evidence_count >= ROI_SATURATION_MIN_EVIDENCE
        and confirmed_coverage >= ROI_SATURATION_MIN_CONFIRMED_COVERAGE
        and measured_coverage >= ROI_SATURATION_MIN_PATH_COVERAGE
    )
    needs_path_refinement = (
        max_gap > MAX_GAP
        or (residual_p95 is not None and residual_p95 > residual_p95_tolerance)
    )
    return enough_evidence and needs_path_refinement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an independently validated Learna AI CalibrationProfileV7")
    parser.add_argument("project_json", type=Path)
    parser.add_argument("profile_json", type=Path)
    parser.add_argument("--roi-json", default="")
    parser.add_argument("--roi-frame", type=int)
    parser.add_argument("--roi-evidence-json", default="[]")
    parser.add_argument("--scan-start-frame", type=int)
    parser.add_argument("--scan-end-frame", type=int)
    parser.add_argument(
        "--route",
        choices=("AUTO_GLOBAL_TEMPLATE", "AUTO_ROI_TEMPLATE", "ROI_FALLBACK"),
        default="AUTO_GLOBAL_TEMPLATE",
    )
    parser.add_argument("--edited-mask", type=Path)
    return parser.parse_args()


def load_mask_asset(path: Path, canonical: np.ndarray) -> np.ndarray:
    """Load a user-edited canonical mask without changing its coordinate space."""
    if not path.is_file():
        raise RuntimeError(f"Edited canonical mask was not found: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError(f"Edited canonical mask is not a readable image: {path}")
    if decoded.shape != canonical.shape:
        decoded = cv2.resize(decoded, (canonical.shape[1], canonical.shape[0]), interpolation=cv2.INTER_AREA)
    # Keep anti-aliased glyph coverage while removing transparent/background
    # noise introduced by an editor export.
    return np.where(decoded >= 32, 255, 0).astype(np.uint8)


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
    # ``numpy`` scalars can be produced by OpenCV/NumPy even when callers
    # explicitly cast most values to ``float``.  Treat them exactly like
    # native JSON numbers so a bad match is reported with its real path.
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")


def normalize_json_value(value: object) -> object:
    """Convert NumPy scalars and unavailable metrics to strict JSON values.

    A failed fit is a calibration result, not a process crash.  OpenCV can
    legitimately produce NaN/Inf for a zero-energy crop; preserve that fact as
    JSON ``null`` so Rust can open the diagnostics and keep the profile
    fail-closed in NEEDS_REVIEW.  NumPy integer/bool values are also converted
    because ``json.dumps`` does not serialize every scalar implementation.
    """
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): normalize_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(child) for child in value]
    if isinstance(value, np.ndarray):
        return normalize_json_value(value.tolist())
    return value


def write_strict_json(path: Path, value: object) -> None:
    """Write and reparse a strict JSON artifact atomically enough for readers."""
    normalized = normalize_json_value(value)
    assert_finite_json(normalized)
    encoded = json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def finite_candidate_row(row: dict[str, object]) -> bool:
    """Return whether a detector row is safe to persist and fit.

    OpenCV's masked ``TM_CCORR_NORMED`` is allowed to return NaN when the
    sampled patch/template has zero energy (common on a fully blurred frame).
    Such a row must be discarded before it reaches the cache or trajectory
    fitter; serializing it as ``Infinity`` would make Rust reject the whole
    calibration profile.
    """
    for value in row.values():
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                return False
    return True


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


def parse_roi_evidence(
    raw: str,
    frame_count: int,
    scan_start: int = 0,
    scan_end: int | None = None,
) -> dict[int, dict[str, float]]:
    """Parse broad ROI seeds keyed by their exact source frame."""
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, list):
        raise RuntimeError("ROI evidence must be a JSON array")
    evidence: dict[int, dict[str, float]] = {}
    for item in value:
        if not isinstance(item, dict) or "frame" not in item or "bbox" not in item:
            raise RuntimeError("Each ROI evidence item must contain frame and bbox")
        frame = int(item["frame"])
        if frame < 0 or frame >= frame_count:
            raise RuntimeError("ROI evidence frame is outside the source video")
        if frame < scan_start or (scan_end is not None and frame > scan_end):
            raise RuntimeError("ROI evidence is outside the selected scan range")
        roi = item["bbox"]
        if not isinstance(roi, dict):
            raise RuntimeError("ROI evidence bbox must be an object")
        parsed = {key: float(roi[key]) for key in ("x", "y", "width", "height") if key in roi}
        if len(parsed) != 4 or parsed["width"] < 32 or parsed["height"] < 16:
            raise RuntimeError("ROI evidence must contain a broad region of at least 32 x 16 source pixels")
        evidence[frame] = parsed
    return evidence


def normalize_scan_range(frame_count: int, start: int | None, end: int | None) -> tuple[int, int]:
    """Resolve and validate the inclusive scan range used by every V6 stage."""
    if frame_count < 1:
        raise RuntimeError("Cannot scan an empty source video")
    scan_start = 0 if start is None else int(start)
    scan_end = frame_count - 1 if end is None else int(end)
    if scan_start < 0 or scan_end >= frame_count or scan_start > scan_end:
        raise RuntimeError(
            f"Invalid scan range {scan_start}–{scan_end}; expected 0–{frame_count - 1} with start <= end"
        )
    return scan_start, scan_end


def feature(frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return a resolution-independent positive high-pass analysis image."""
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, ANALYSIS_LONG_EDGE / max(source_width, source_height))
    analysis_width = max(32, int(round(source_width * scale)))
    analysis_height = max(32, int(round(source_height * scale)))
    analysis = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
    # The positive polarity remains the primary scan because the shipped
    # canonical asset is a bright Learna glyph.  ``candidate_rows`` also runs
    # a negative-polarity pass for shots where the same gray watermark sits on
    # a bright robot/skin background.
    highpass = np.maximum(gray - smooth, 0.0)
    base_scale = min(source_width / REFERENCE_WIDTH, source_height / REFERENCE_HEIGHT)
    return highpass.astype(np.float32), analysis_width / source_width, analysis_height / source_height, base_scale


def feature_negative(frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return the dark-on-light counterpart of :func:`feature`."""
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, ANALYSIS_LONG_EDGE / max(source_width, source_height))
    analysis_width = max(32, int(round(source_width * scale)))
    analysis_height = max(32, int(round(source_height * scale)))
    analysis = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.2)
    highpass = np.maximum(smooth - gray, 0.0)
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
    # A broad ROI is a location hint, but mere one-pixel intersection lets a
    # background peak that is mostly outside the user box win the match.  Such
    # peaks were responsible for large trajectory jumps on the 14_7 clip.
    # Accept a candidate when its center is inside the hint or when a
    # meaningful fraction of the candidate area overlaps it.  This keeps the
    # ROI tolerant of a little padding while rejecting near-by unrelated
    # text/texture.
    roi_x1 = float(roi["x"])
    roi_y1 = float(roi["y"])
    roi_x2 = roi_x1 + float(roi["width"])
    roi_y2 = roi_y1 + float(roi["height"])
    candidate_x2 = x + width
    candidate_y2 = y + height
    overlap_width = max(0.0, min(candidate_x2, roi_x2) - max(x, roi_x1))
    overlap_height = max(0.0, min(candidate_y2, roi_y2) - max(y, roi_y1))
    candidate_area = max(1.0, width * height)
    overlap_ratio = overlap_width * overlap_height / candidate_area
    center_x = x + width * 0.5
    center_y = y + height * 0.5
    center_inside = roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2
    return center_inside or overlap_ratio >= 0.35


def evaluate_box(
    frame: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    canonical: np.ndarray,
) -> tuple[float, float, float, bool] | None:
    """Measure a source-resolution box against the canonical glyph mask."""
    # Match the same padded mask canvas used by the canonical descriptor.
    # Evaluating only the tight glyph box shifts the letters against the
    # descriptor and makes clear evidence score as a weak match.
    scale_x = width / max(1.0, CANONICAL_WIDTH)
    scale_y = height / max(1.0, CANONICAL_HEIGHT)
    padding_x = max(1, int(round(MASK_BORDER_X * scale_x)))
    padding_y = max(1, int(round(MASK_BORDER_Y * scale_y)))
    box = bounds(
        x - padding_x,
        y - padding_y,
        width + 2 * padding_x,
        height + 2 * padding_y,
        frame.shape[1],
        frame.shape[0],
    )
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    normalized = cv2.resize(crop, (canonical.shape[1], canonical.shape[0]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Evaluate both bright-on-dark and dark-on-light polarity.  A gray
    # Learna glyph is sometimes darker than a white robot face; considering
    # only positive contrast made those real observations disappear.  Each
    # polarity is scored independently and the stronger structural evidence
    # wins, while contamination/connected-component penalties remain active.
    smooth = cv2.GaussianBlur(gray, (0, 0), 8.0)
    alpha = cv2.GaussianBlur((canonical.astype(np.float32) / 255.0), (0, 0), 0.75)
    active = alpha >= 0.08

    def score_contrast(contrast: np.ndarray) -> tuple[float, float, float, bool]:
        # A threshold around 7 suppresses low-amplitude texture while keeping
        # anti-aliased glyph strokes.  The signed passes make the threshold
        # independent of whether the local background is lighter or darker.
        binary = np.where(contrast >= 7.0, 255, 0).astype(np.uint8)
        binary_correlation, iou, contamination, large = mask_metrics(binary, canonical)
        glyph = contrast[active]
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

    metrics = [score_contrast(np.maximum(gray - smooth, 0.0)), score_contrast(np.maximum(smooth - gray, 0.0))]
    finite_metrics = [
        item
        for item in metrics
        if all(math.isfinite(float(value)) for value in item[:3])
    ]
    if not finite_metrics:
        return 0.0, 0.0, 1.0, False
    return max(
        finite_metrics,
        key=lambda item: item[0] + 0.35 * item[1] - 0.40 * item[2] - (0.15 if item[3] else 0.0),
    )


def top_matches(response: np.ndarray, width: int, height: int, limit: int = MAX_CANDIDATES_PER_FRAME) -> list[tuple[float, tuple[int, int]]]:
    work = response.copy()
    matches: list[tuple[float, tuple[int, int]]] = []
    # Suppress only near-identical peaks.  A half-template radius discarded
    # two legitimate hypotheses on busy frames (the real translucent glyph
    # was often 70–90 source pixels from a stronger face/texture peak).  The
    # final source-space NMS still removes duplicates after refinement.
    radius_x = max(2, width // 3)
    radius_y = max(2, height // 3)
    for _ in range(limit):
        _, score, _, location = cv2.minMaxLoc(work)
        # A zero-energy masked match can yield NaN on some OpenCV builds.
        # Never admit that value into a candidate row or the persistent cache.
        if not math.isfinite(float(score)) or score < MIN_RAW_SCORE:
            break
        matches.append((float(score), (int(location[0]), int(location[1]))))
        left = max(0, location[0] - radius_x)
        top = max(0, location[1] - radius_y)
        right = min(work.shape[1], location[0] + radius_x + 1)
        bottom = min(work.shape[0], location[1] + radius_y + 1)
        work[top:bottom, left:right] = -1.0
    return matches


def candidate_rows(
    frame: np.ndarray,
    frame_number: int,
    canonical: np.ndarray,
    roi: dict[str, float] | None,
    polarity: str = "positive",
) -> list[dict[str, float | int | bool]]:
    if polarity == "negative":
        image_feature, analysis_x, analysis_y, base_scale = feature_negative(frame)
    else:
        image_feature, analysis_x, analysis_y, base_scale = feature(frame)
    rows: list[dict[str, float | int | bool]] = []
    # ROI evidence is only a seed for the frame where it was drawn, but it is
    # also an important performance hint.  The previous implementation ran
    # masked template matching over the complete 405x720 analysis frame and
    # only then blanked the response outside the ROI.  On a 58s H.264 clip
    # this made an ROI retry take many minutes and made the Review dialog look
    # frozen.  Match a padded crop instead and translate peaks back to the
    # full-frame analysis coordinates; the final roi_allows check still keeps
    # the broad evidence semantics intact.
    feature_origin_x = 0
    feature_origin_y = 0
    search_feature = image_feature
    if roi is not None:
        max_scale = max(SCALES)
        margin_x = MASK_CANVAS_WIDTH * base_scale * max_scale + 24.0
        margin_y = MASK_CANVAS_HEIGHT * base_scale * max_scale + 24.0
        left = max(0, int(math.floor((roi["x"] - margin_x) * analysis_x)))
        top = max(0, int(math.floor((roi["y"] - margin_y) * analysis_y)))
        right = min(image_feature.shape[1], int(math.ceil((roi["x"] + roi["width"] + margin_x) * analysis_x)))
        bottom = min(image_feature.shape[0], int(math.ceil((roi["y"] + roi["height"] + margin_y) * analysis_y)))
        if right > left and bottom > top:
            feature_origin_x = left
            feature_origin_y = top
            search_feature = image_feature[top:bottom, left:right]
    for scale in SCALES:
        template, template_mask = template_feature(canonical, scale, base_scale, analysis_x, analysis_y)
        if search_feature.shape[1] < template.shape[1] or search_feature.shape[0] < template.shape[0]:
            continue
        response = cv2.matchTemplate(search_feature, template, cv2.TM_CCORR_NORMED, mask=template_mask)
        template_width = template.shape[1]
        template_height = template.shape[0]
        for raw_score, (ax, ay) in top_matches(response, template_width, template_height, RAW_PEAKS_PER_SCALE):
            # matchTemplate coordinates refer to the padded descriptor canvas;
            # convert back to the tight source glyph bbox for the profile.
            canvas_x = (ax + feature_origin_x) / analysis_x
            canvas_y = (ay + feature_origin_y) / analysis_y
            width = CANONICAL_WIDTH * base_scale * scale
            height = CANONICAL_HEIGHT * base_scale * scale
            padding_x = MASK_BORDER_X * base_scale * scale
            padding_y = MASK_BORDER_Y * base_scale * scale
            x = canvas_x + padding_x
            y = canvas_y + padding_y
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
            row = {
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
            if finite_candidate_row(row):
                rows.append(row)
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
    # Run a separate negative-polarity pass and merge it with the primary
    # results.  Keeping the passes separate preserves their top peaks; taking
    # a pixel-wise maximum would let unrelated high-contrast texture consume
    # the limited NMS budget before the real gray watermark is considered.
    if polarity == "positive":
        selected.extend(candidate_rows(frame, frame_number, canonical, roi, "negative"))
        selected.sort(key=lambda row: float(row["score"]), reverse=True)
        merged: list[dict[str, float | int | bool]] = []
        for row in selected:
            if any(
                math.hypot(float(row["x"]) - float(other["x"]), float(row["y"]) - float(other["y"]))
                < max(float(row["width"]), float(row["height"])) * 0.45
                for other in merged
            ):
                continue
            merged.append(row)
            if len(merged) >= MAX_CANDIDATES_PER_FRAME:
                break
        return merged
    return selected


def periodic_prior_candidate(
    frame: np.ndarray,
    frame_number: int,
    canonical: np.ndarray,
) -> dict[str, float | int | bool] | None:
    """Return weak, image-validated evidence near the shipped Learna path.

    This is deliberately only a *candidate* model.  It is retained for the
    temporal graph so a known Learna path can bridge motion-blurred frames,
    but it can be promoted only after independent global observations confirm
    the path and the trajectory gate passes.  A different video trajectory
    therefore cannot be rendered from this prior alone.
    """
    _, _, _, base_scale = feature(frame)
    prior_x, prior_y = periodic_position(frame_number)
    best: tuple[float, float, float, float, float, float, bool, float] | None = None
    for variant in (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25):
        width = CANONICAL_WIDTH * base_scale * variant
        height = CANONICAL_HEIGHT * base_scale * variant
        center_x = prior_x + CANONICAL_WIDTH * 0.5
        center_y = prior_y + CANONICAL_HEIGHT * 0.5
        base_x = center_x - width * 0.5
        base_y = center_y - height * 0.5
        radius = max(6, int(round(24 * base_scale)))
        for dx in range(-radius, radius + 1, max(6, int(round(6 * base_scale)))):
            for dy in range(-radius, radius + 1, max(6, int(round(6 * base_scale)))):
                metrics = evaluate_box(frame, base_x + dx, base_y + dy, width, height, canonical)
                if metrics is None:
                    continue
                corr, iou, contamination, large = metrics
                rank = corr + 0.35 * iou - 0.40 * contamination - (0.15 if large else 0.0)
                if not all(math.isfinite(float(value)) for value in (corr, iou, contamination, rank)):
                    continue
                item = (rank, base_x + dx, base_y + dy, corr, iou, contamination, large, variant)
                if best is None or rank > best[0]:
                    best = item
    if best is None:
        return None
    rank, x, y, corr, iou, contamination, large, variant = best
    # These are intentionally weaker than DIRECT/PROVISIONAL.  The graph and
    # independent global seeds remain mandatory before this route is usable.
    if corr < 0.35 or iou < 0.15 or contamination > 0.85:
        return None
    row = {
        "frame": frame_number,
        "x": float(x),
        "y": float(y),
        "width": float(CANONICAL_WIDTH * base_scale * variant),
        "height": float(CANONICAL_HEIGHT * base_scale * variant),
        "scale": float(base_scale * variant),
        "templateScale": float(variant),
        "rawScore": float(corr),
        "glyphCorrelation": float(corr),
        "glyphIou": float(iou),
        "contamination": float(contamination),
        "largeOutsideComponent": bool(large),
        "baseScale": float(base_scale),
        "score": float(max(0.0, min(1.0, 0.50 * corr + 0.25 * iou + 0.25 * (1.0 - contamination)))),
        "periodicPrior": True,
    }
    return row if finite_candidate_row(row) else None


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
        def support_bonus(candidate: dict[str, float | int | bool]) -> float:
            # Image evidence remains the score authority.  These small bonuses
            # merely keep a validated prior/ROI seed connected when a blurred
            # frame has several similarly-scored background peaks.
            if bool(candidate.get("periodicPrior", False)):
                return 0.42
            if bool(candidate.get("userRoi", False)):
                return 0.24
            return 0.0

        frame_cost = [-float(candidate["score"]) * 6.0 - support_bonus(candidate) + 18.0 for candidate in current]
        frame_parent = [-1] * len(current)
        for candidate_index, candidate in enumerate(current):
            base = -float(candidate["score"]) * 6.0 - support_bonus(candidate)
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
    # Do not force the path to terminate at the last sampled frame.  The old
    # implementation only considered ``costs[-1]`` and therefore selected a
    # short, high-scoring terminal false-positive (for example a blue
    # transition near the end) while discarding a much longer coherent path
    # that started at frame 0.  Choose the best terminal state across all
    # sampled frames; the backtracking loop below preserves its true start
    # boundary and lets ``choose_track_segments`` discover reappearances as
    # separate intervals.
    end_frame_index, last_index = min(
        (
            (frame_index, candidate_index)
            for frame_index, frame_cost in enumerate(costs)
            for candidate_index in range(len(frame_cost))
        ),
        key=lambda item: costs[item[0]][item[1]],
    )
    chosen: list[dict[str, float | int | bool]] = []
    for index in range(end_frame_index, -1, -1):
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


def choose_user_seeded_track(
    all_candidates: dict[int, list[dict[str, float | int | bool]]],
) -> list[dict[str, float | int | bool]]:
    """Build a temporally coherent path from broad ROI evidence.

    ROI evidence is a user-confirmed *location hint*, not a fixed rectangle.
    The generic Viterbi pass can nevertheless discard an exact evidence node
    when a stronger background chain wins on raw score.  This seeded pass
    keeps the highest-quality candidate at every evidence frame and selects
    nearby candidates by motion continuity between those anchors.  It is
    deliberately conservative: candidates still have to satisfy the
    provisional gate, and unresolved jumps simply terminate the seed instead
    of inventing a mask.
    """
    frames = sorted(all_candidates)
    anchors: list[dict[str, float | int | bool]] = []
    for frame in frames:
        user = [row for row in all_candidates[frame] if bool(row.get("userRoi", False)) and provisional_gate(row)]
        if user:
            anchors.append(max(user, key=lambda row: float(row.get("score", 0.0))))
    if not anchors:
        return []

    by_frame = {int(row["frame"]): row for row in anchors}
    selected: dict[int, dict[str, float | int | bool]] = dict(by_frame)
    # Interpolate only between independently confirmed anchors.  The expected
    # point is used as a tie-breaker, while glyph score remains a small quality
    # term so a nearby background peak cannot win by distance alone.
    for left, right in zip(anchors, anchors[1:]):
        left_frame, right_frame = int(left["frame"]), int(right["frame"])
        if right_frame <= left_frame:
            continue
        previous_selected = left
        for frame in frames:
            if frame <= left_frame or frame >= right_frame or frame in selected:
                continue
            ratio = (frame - left_frame) / float(right_frame - left_frame)
            expected_x = float(left["x"]) + (float(right["x"]) - float(left["x"])) * ratio
            expected_y = float(left["y"]) + (float(right["y"]) - float(left["y"])) * ratio
            candidates = [row for row in all_candidates[frame] if provisional_gate(row)]
            if not candidates:
                continue
            # A broad ROI is a location hint, but a weak background peak can
            # still be hundreds of pixels away.  The previous implementation
            # always selected the nearest candidate, which silently injected
            # jumps (e.g. a 1,000 px branch switch) into the trajectory and
            # then asked the user to repair every downstream gap.  Reject an
            # implausible candidate instead; the gap is left unresolved and
            # will be represented by one clustered ROI hint later.
            anchor_displacement = math.hypot(
                float(right["x"]) - float(left["x"]),
                float(right["y"]) - float(left["y"]),
            )
            # Scale the tolerance with the motion actually implied by this
            # anchor interval.  A fixed 72 px floor let an unrelated branch
            # jump in immediately after an anchor (the 951→954→960 pattern
            # seen in 14_7), which then inflated the fitted residual and
            # produced another ROI request.  Tight tolerances on slow motion
            # and a capped allowance on fast motion preserve real corners
            # while rejecting those branch switches.
            interval_frames = max(1, right_frame - left_frame)
            expected_step = anchor_displacement / interval_frames * max(1, frame - int(previous_selected["frame"]))
            distance_limit = min(112.0, max(28.0, expected_step * 2.5 + 18.0))
            previous_delta = max(1, frame - int(previous_selected["frame"]))
            step_limit = min(112.0, max(42.0, 14.0 * previous_delta))
            candidates = [
                row
                for row in candidates
                if math.hypot(float(row["x"]) - expected_x, float(row["y"]) - expected_y)
                <= distance_limit
                and math.hypot(
                    float(row["x"]) - float(previous_selected["x"]),
                    float(row["y"]) - float(previous_selected["y"]),
                ) <= step_limit
            ]
            if not candidates:
                continue
            selected[frame] = min(
                candidates,
                key=lambda row: (
                    math.hypot(float(row["x"]) - expected_x, float(row["y"]) - expected_y)
                    / max(24.0, 20.0 * max(1.0, (float(right_frame - left_frame) / SAMPLE_STRIDE))),
                    -float(row.get("score", 0.0)),
                ),
            )
            previous_selected = selected[frame]

    # Propagate forward/backward from the outermost anchors using the last
    # measured velocity.  This covers a moving watermark beyond the first or
    # last ROI hint without turning a long gap into an unconditional mask.
    def propagate(indices: Iterable[int], anchor: dict[str, float | int | bool], velocity: tuple[float, float]) -> None:
        previous = anchor
        # A watermark may fade out at the end of a video.  Do not let a long
        # chain of weak background peaks keep the active interval alive after
        # the last confirmed ROI.  One short weak tail is useful for motion
        # blur, but a second consecutive weak-only step terminates propagation
        # and leaves the inactive tail untouched.
        weak_steps = 0
        for frame in indices:
            candidates = [row for row in all_candidates[frame] if provisional_gate(row)]
            if not candidates:
                break
            delta = frame - int(previous["frame"])
            expected_x = float(previous["x"]) + velocity[0] * delta
            expected_y = float(previous["y"]) + velocity[1] * delta
            candidate = min(
                candidates,
                key=lambda row: (
                    math.hypot(float(row["x"]) - expected_x, float(row["y"]) - expected_y)
                    / max(24.0, 22.0 * max(1.0, delta / SAMPLE_STRIDE)),
                    -float(row.get("score", 0.0)),
                ),
            )
            distance = math.hypot(float(candidate["x"]) - expected_x, float(candidate["y"]) - expected_y)
            if distance > max(180.0, 45.0 * max(1.0, delta)):
                break
            step_distance = math.hypot(
                float(candidate["x"]) - float(previous["x"]),
                float(candidate["y"]) - float(previous["y"]),
            )
            if step_distance > min(112.0, max(42.0, 14.0 * max(1.0, delta))):
                break
            if hard_gate(candidate) or bool(candidate.get("userRoi", False)):
                weak_steps = 0
            else:
                weak_steps += 1
                if weak_steps > 1:
                    break
            selected[frame] = candidate
            previous = candidate

    if len(anchors) >= 2:
        first, second = anchors[0], anchors[1]
        dt = max(1, int(second["frame"]) - int(first["frame"]))
        velocity = ((float(second["x"]) - float(first["x"])) / dt, (float(second["y"]) - float(first["y"])) / dt)
        propagate((frame for frame in frames if frame < int(first["frame"])), first, velocity)
        last, previous = anchors[-1], anchors[-2]
        dt = max(1, int(last["frame"]) - int(previous["frame"]))
        velocity = ((float(last["x"]) - float(previous["x"])) / dt, (float(last["y"]) - float(previous["y"])) / dt)
        propagate((frame for frame in frames if frame > int(last["frame"])), last, velocity)
    return [selected[frame] for frame in sorted(selected)]


def filter_static_background_segments(
    segments: list[list[dict[str, float | int | bool]]],
    minimum_duration_frames: int = 30,
    minimum_spread_px: float = 48.0,
) -> list[list[dict[str, float | int | bool]]]:
    """Drop long, unconfirmed static chains from active watermark intervals.

    A repeated background title/UI element can match the glyph descriptor for
    an entire tail of a clip.  It is not safe to ask the user for ROI there or
    render it as a watermark when the chain has neither a hard-gated match nor
    explicit ROI evidence.  Moving segments, and any segment with direct/user
    evidence, remain untouched.  Static watermark videos fail closed and can
    be recovered through the ROI route.
    """
    filtered: list[list[dict[str, float | int | bool]]] = []
    for segment in segments:
        if not segment:
            continue
        has_confirmed_evidence = any(
            hard_gate(row) or bool(row.get("userRoi", False)) for row in segment
        )
        span = int(segment[-1]["frame"]) - int(segment[0]["frame"])
        x_values = [float(row["x"]) for row in segment]
        y_values = [float(row["y"]) for row in segment]
        spread = math.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values))
        if not has_confirmed_evidence and span >= minimum_duration_frames and spread < minimum_spread_px:
            continue
        filtered.append(segment)
    return filtered


def active_intervals_from_segments(
    segments: list[list[dict[str, float | int | bool]]],
    frame_count: int,
    fps: float,
    scan_start: int = 0,
    scan_end: int | None = None,
) -> list[dict[str, int]]:
    """Create conservative active ranges and preserve long unresolved gaps."""
    if not segments:
        return []
    bounded_end = frame_count - 1 if scan_end is None else min(frame_count - 1, scan_end)
    bridge_limit = max(SAMPLE_STRIDE, int(round(fps * 0.6)))
    raw = sorted(
        (
            max(scan_start, int(segment[0]["frame"]) - SAMPLE_STRIDE),
            min(bounded_end, int(segment[-1]["frame"]) + SAMPLE_STRIDE),
        )
        for segment in segments
        if segment
        and int(segment[-1]["frame"]) >= scan_start
        and int(segment[0]["frame"]) <= bounded_end
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
        return float(keys[0]["x"]), float(keys[0]["y"]), float(keys[0].get("scale", 1.0)), frame == int(keys[0]["frame"]), float(keys[0].get("score", 0.0))
    if frame >= int(keys[-1]["frame"]):
        return float(keys[-1]["x"]), float(keys[-1]["y"]), float(keys[-1].get("scale", 1.0)), frame == int(keys[-1]["frame"]), float(keys[-1].get("score", 0.0))
    for left, right in zip(keys, keys[1:]):
        start, end = int(left["frame"]), int(right["frame"])
        if start <= frame <= end:
            ratio = (frame - start) / max(1, end - start)
            return (
                float(left["x"]) + (float(right["x"]) - float(left["x"])) * ratio,
                float(left["y"]) + (float(right["y"]) - float(left["y"])) * ratio,
                float(left.get("scale", 1.0)) + (float(right.get("scale", 1.0)) - float(left.get("scale", 1.0))) * ratio,
                frame in (start, end),
                min(float(left["score"]), float(right["score"])) * 0.88,
            )
    return float(keys[-1]["x"]), float(keys[-1]["y"]), float(keys[-1].get("scale", 1.0)), False, float(keys[-1].get("score", 0.0))


def _path_key_rows(rows: list[dict[str, float | int | bool]], epsilon: float = 2.0) -> list[dict[str, float | int | bool]]:
    """Build a compact path without allowing an observation to validate itself."""
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    if len(ordered) < 2:
        return ordered
    points = [(float(row["x"]), float(row["y"])) for row in ordered]
    indices = rdp_indices(points, epsilon)
    compact = [ordered[index] for index in indices]
    return compact if len(compact) >= 2 else [ordered[0], ordered[-1]]


def holdout_metrics(rows: list[dict[str, float | int | bool]], epsilon: float = 2.0) -> dict[str, object]:
    """Validate a trajectory on observations excluded from fitting.

    A low residual on the same ROI anchors used to build a path is not an
    independent quality signal.  V7 therefore reserves a deterministic,
    time-stratified fifth of the accepted rows for this check.
    """
    ordered = sorted(rows, key=lambda row: int(row["frame"]))
    if len(ordered) < 10:
        return {
            "count": 0,
            "trainingCount": len(ordered),
            "median": None,
            "p95": None,
            "inlierRatio": 0.0,
            "reason": "INSUFFICIENT_HOLDOUT_OBSERVATIONS",
        }
    holdout = [row for index, row in enumerate(ordered) if index % 5 == 0]
    training = [row for index, row in enumerate(ordered) if index % 5 != 0]
    keys = _path_key_rows(training, epsilon)
    if len(keys) < 2:
        return {
            "count": len(holdout),
            "trainingCount": len(training),
            "median": None,
            "p95": None,
            "inlierRatio": 0.0,
            "reason": "HOLDOUT_PATH_UNDERCONSTRAINED",
        }
    residuals: list[float] = []
    for row in holdout:
        px, py, _, _, _ = interpolate(keys, int(row["frame"]))
        residuals.append(math.hypot(float(row["x"]) - px, float(row["y"]) - py))
    return {
        "count": len(holdout),
        "trainingCount": len(training),
        "median": float(np.median(residuals)) if residuals else None,
        "p95": float(np.percentile(residuals, 95)) if residuals else None,
        "max": max(residuals, default=None),
        "inlierRatio": float(sum(value <= 3.0 for value in residuals) / max(1, len(residuals))),
        "reason": None,
    }


def refine_local_path(
    source: Path,
    canonical: np.ndarray,
    seed_rows: list[dict[str, float | int | bool]],
    active_intervals: list[dict[str, int]],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, float | int | bool]]:
    """Refine a seeded trajectory at source resolution with local masked NCC.

    The global scan remains sparse.  This pass only evaluates a small window
    around the predicted path, so adding evidence does not trigger another
    full-frame multi-scale scan.  It is deliberately conservative: a weak
    local match is left to trajectory interpolation instead of becoming a
    false observation.
    """
    if len(seed_rows) < 2 or not active_intervals:
        return []
    ordered = sorted(seed_rows, key=lambda row: int(row["frame"]))
    targets: set[int] = set()
    for interval in active_intervals:
        start = int(interval["startFrame"])
        end = int(interval["endFrame"])
        targets.update(range(start, end + 1, LOCAL_REFINE_STRIDE))
        targets.add(start)
        targets.add(end)
    refined: list[dict[str, float | int | bool]] = []
    capture = cv2.VideoCapture(str(source))
    frame_number = 0
    try:
        while targets:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number not in targets:
                frame_number += 1
                continue
            px, py, predicted_scale, _, predicted_score = interpolate(ordered, frame_number)
            best: tuple[float, float, float, float, float, float, bool] | None = None
            for scale_factor in (0.94, 1.0, 1.06):
                scale = max(0.55, min(1.55, predicted_scale * scale_factor))
                box_width = CANONICAL_WIDTH * scale
                box_height = CANONICAL_HEIGHT * scale
                for dx in range(-LOCAL_REFINE_RADIUS, LOCAL_REFINE_RADIUS + 1, LOCAL_REFINE_STEP):
                    for dy in range(-LOCAL_REFINE_RADIUS, LOCAL_REFINE_RADIUS + 1, LOCAL_REFINE_STEP):
                        x = px + dx
                        y = py + dy
                        metrics = evaluate_box(frame, x, y, box_width, box_height, canonical)
                        if metrics is None:
                            continue
                        corr, iou, contamination, large = metrics
                        distance = math.hypot(dx, dy)
                        rank = corr + 0.35 * iou - 0.40 * contamination - 0.015 * distance - (0.15 if large else 0.0)
                        if not all(math.isfinite(float(value)) for value in (corr, iou, contamination, rank)):
                            continue
                        if best is None or rank > best[0]:
                            best = (rank, x, y, scale, corr, iou, large)
                            best_contamination = contamination
            if best is not None:
                rank, x, y, scale, corr, iou, large = best
                contamination = best_contamination
                refined_width = CANONICAL_WIDTH * scale
                refined_height = CANONICAL_HEIGHT * scale
                # Local refinement is accepted only with structural evidence;
                # path continuity alone cannot manufacture a glyph.
                if corr >= 0.42 and iou >= 0.22 and contamination <= 0.55 and rank >= 0.20:
                    row = {
                        "frame": frame_number,
                        "x": float(x),
                        "y": float(y),
                        "width": float(refined_width),
                        "height": float(refined_height),
                        "scale": float(scale),
                        "score": float(max(0.0, min(1.0, rank))),
                        "glyphCorrelation": float(corr),
                        "glyphIou": float(iou),
                        "contamination": float(contamination),
                        "largeOutsideComponent": bool(large),
                        "refined": True,
                        "positionSource": "LOCAL_NCC",
                        "predictedScore": float(predicted_score),
                    }
                    if finite_candidate_row(row):
                        refined.append(row)
            targets.remove(frame_number)
            frame_number += 1
    finally:
        capture.release()
    # Keep one refined row per physical frame and discard accidental duplicates.
    return sorted({int(row["frame"]): row for row in refined}.values(), key=lambda row: int(row["frame"]))


def contiguous_gaps(frames: Iterable[int]) -> int:
    values = sorted(set(frames))
    if len(values) < 2:
        return 10**9
    return max((right - left for left, right in zip(values, values[1:])), default=10**9)


def review_ranges_from_gaps(
    measured: list[dict[str, float | int | bool]],
    active_intervals: list[dict[str, int]],
    roi_evidence: dict[int, dict[str, float]],
    all_candidates: dict[int, list[dict[str, float | int | bool]]] | None = None,
    scan_start: int = 0,
    scan_end: int | None = None,
) -> list[dict[str, object]]:
    """Return actionable ROI hints only for unresolved active gaps.

    The UI uses these ranges to guide a user toward a few representative
    frames.  Inactive tails (for example after a watermark disappears) are
    deliberately excluded, so the dialog never asks for ROI evidence where
    no watermark is present.
    """
    inferred_end = max(
        [int(row["frame"]) for row in measured]
        + [int(frame) for frame in (all_candidates or {})]
        + [int(interval["endFrame"]) for interval in active_intervals]
        + [scan_start]
    )
    bounded_end = max(scan_start, scan_end if scan_end is not None else inferred_end)
    # ``measured`` also contains provisional graph rows.  They are useful for
    # fitting a candidate path, but they are not independently image-validated
    # evidence.  Treating them as direct evidence made the UI report 100%
    # coverage while the profile actually had zero hard-gated observations.
    # Only hard-gated matches and explicit user ROI anchors can close a review
    # gap.  The provisional pool is still inspected below for diagnostics.
    by_frame = {
        int(row["frame"]): row
        for row in measured
        if scan_start <= int(row["frame"]) <= bounded_end
        and (hard_gate(row) or bool(row.get("userRoi", False)))
    }
    ranges: list[dict[str, object]] = []
    for interval in active_intervals:
        frames = sorted(
            frame
            for frame in by_frame
            if int(interval["startFrame"]) <= frame <= int(interval["endFrame"])
        )
        for left, right in zip(frames, frames[1:]):
            if right - left <= MAX_GAP:
                continue
            # Keep the displayed range within the active interval and offer
            # the midpoint plus nearby stride-aligned alternatives.  Existing
            # ROI evidence is not suggested again.
            start = max(scan_start, int(interval["startFrame"]), left)
            end = min(bounded_end, int(interval["endFrame"]), right)
            if end <= start:
                continue
            # The endpoints (and sometimes interior frames) can already be
            # confirmed ROI evidence.  Do not show a range that the user has
            # already covered; the trajectory stage must either interpolate
            # it or report a refinement failure, rather than asking for the
            # same evidence again on every calibration pass.
            if any(start <= int(frame) <= end for frame in roi_evidence):
                continue
            midpoint = int(round((start + end) / 2.0))
            candidates = [
                midpoint,
                start + max(SAMPLE_STRIDE, (end - start) // 3),
                end - max(SAMPLE_STRIDE, (end - start) // 3),
            ]
            suggested: list[int] = []
            for candidate in candidates:
                candidate = max(start, min(end, int(candidate)))
                if candidate in roi_evidence or candidate in suggested:
                    continue
                suggested.append(candidate)
            ranges.append(
                {
                    "startFrame": start,
                    "endFrame": end,
                    "suggestedFrames": suggested[:3],
                    "reason": "UNRESOLVED_ACTIVE_RANGE",
                }
            )
        # When the selected path has a dense interpolated row at every stride,
        # looking only at measured-frame gaps hides the very ranges that need
        # another ROI.  Inspect the original candidate pool and mark a frame
        # as evidence only when it has an image-validated direct match or an
        # explicit user ROI.  Consecutive weak runs become actionable ranges.
        if all_candidates:
            start_bound = max(scan_start, int(interval["startFrame"]))
            end_bound = min(bounded_end, int(interval["endFrame"]))
            if end_bound <= start_bound:
                continue
            sampled = [
                frame for frame in sorted(all_candidates)
                if start_bound <= frame <= end_bound
            ]
            reliable_frames = {
                frame
                for frame in sampled
                if any(
                    hard_gate(row) or bool(row.get("userRoi", False))
                    for row in all_candidates.get(frame, [])
                )
            }
            reliable_frames.update(int(frame) for frame in roi_evidence if start_bound <= int(frame) <= end_bound)
            run: list[int] = []
            for frame in sampled + [None]:
                if frame is not None and frame not in reliable_frames and (not run or frame - run[-1] <= SAMPLE_STRIDE * 2):
                    run.append(frame)
                    continue
                if run and run[-1] - run[0] >= MAX_GAP:
                    start = max(scan_start, start_bound, run[0])
                    end = min(bounded_end, end_bound, run[-1])
                    if end <= start:
                        run = []
                        continue
                    if any(start <= int(frame) <= end for frame in roi_evidence):
                        run = []
                        continue
                    midpoint = int(round((start + end) / 2.0))
                    candidates = [
                        midpoint,
                        start + max(SAMPLE_STRIDE, (end - start) // 3),
                        end - max(SAMPLE_STRIDE, (end - start) // 3),
                    ]
                    suggested: list[int] = []
                    for candidate in candidates:
                        candidate = max(start, min(end, int(candidate)))
                        if candidate in roi_evidence or candidate in suggested:
                            continue
                        suggested.append(candidate)
                    ranges.append(
                        {
                            "startFrame": start,
                            "endFrame": end,
                            "suggestedFrames": suggested[:3],
                            "reason": "WEAK_DIRECT_EVIDENCE",
                        }
                    )
                run = []
    unique: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for item in ranges:
        key = (int(item["startFrame"]), int(item["endFrame"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    # Collapse neighbouring weak runs into a single actionable cluster.  The
    # full unclustered candidate information remains in diagnostics; the
    # quality dialog should ask for one or a few ROI anchors per motion phase,
    # not dozens of nearly identical frame selections.  Never merge different
    # failure classes: an unresolved active gap remains visible as such.
    clustered: list[dict[str, object]] = []
    # Group by reason before clustering so an interleaved diagnostic range of
    # another class cannot prevent two adjacent unresolved ranges from being
    # merged (the previous ordering made this happen in long clips).
    ordered_unique = sorted(
        unique,
        key=lambda value: (
            str(value.get("reason", "")),
            int(value["startFrame"]),
            int(value["endFrame"]),
        ),
    )
    for item in ordered_unique:
        if not clustered:
            clustered.append(dict(item))
            continue
        previous = clustered[-1]
        same_reason = item.get("reason") == previous.get("reason")
        merged_start = min(int(previous["startFrame"]), int(item["startFrame"]))
        merged_end = max(int(previous["endFrame"]), int(item["endFrame"]))
        close_enough = int(item["startFrame"]) - int(previous["endFrame"]) <= REVIEW_MERGE_GAP
        within_cluster = merged_end - merged_start <= REVIEW_MAX_CLUSTER_SPAN
        evidence_between = any(
            int(previous["endFrame"]) < int(frame) < int(item["startFrame"])
            for frame in roi_evidence
        )
        if same_reason and close_enough and within_cluster and not evidence_between:
            previous["startFrame"] = merged_start
            previous["endFrame"] = merged_end
            previous_suggestions = [int(frame) for frame in previous.get("suggestedFrames", [])]
            item_suggestions = [int(frame) for frame in item.get("suggestedFrames", [])]
            merged_suggestions: list[int] = []
            for frame in previous_suggestions + item_suggestions + [
                int(round((merged_start + merged_end) / 2.0)),
                merged_start + max(SAMPLE_STRIDE, (merged_end - merged_start) // 3),
                merged_end - max(SAMPLE_STRIDE, (merged_end - merged_start) // 3),
            ]:
                frame = max(merged_start, min(merged_end, frame))
                if frame in roi_evidence or frame in merged_suggestions:
                    continue
                merged_suggestions.append(frame)
            previous["suggestedFrames"] = merged_suggestions[:3]
        else:
            clustered.append(dict(item))

    # Prioritise unresolved active gaps, then the longest weak clusters.  This
    # preserves the information most likely to reconnect the trajectory while
    # keeping the dialog compact and predictable.
    clustered.sort(
        key=lambda value: (
            0 if value.get("reason") == "UNRESOLVED_ACTIVE_RANGE" else 1,
            -(int(value["endFrame"]) - int(value["startFrame"])),
            int(value["startFrame"]),
        )
    )
    return clustered[:MAX_REVIEW_RANGES]


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
    if bool(row.get("periodicPrior", False)):
        return (
            float(row.get("glyphCorrelation", 0.0)) >= 0.35
            and float(row.get("glyphIou", 0.0)) >= 0.15
            and float(row.get("contamination", 1.0)) <= 0.85
        )
    # A user-confirmed broad ROI is allowed to seed the graph with weaker
    # local evidence.  It is still never rendered by itself: the temporal
    # graph, trajectory residual and gap gates must validate it against other
    # frames.  This is what makes ROI useful when the clean glyph is blurred
    # or composited over a busy background.
    if bool(row.get("userRoi", False)):
        return (
            # A broad ROI is explicit user evidence.  On a bright face/robot
            # background the translucent glyph can have low IoU and a large
            # outside component even when its location is correct.  Keep this
            # weak candidate for temporal fitting; trajectory residual,
            # uncertainty and final QA remain mandatory before READY.
            float(row.get("glyphCorrelation", 0.0)) >= 0.28
            and float(row.get("glyphIou", 0.0)) >= 0.12
            and float(row.get("contamination", 1.0)) <= 0.75
        )
    # Global evidence can be weaker on motion-blurred frames.  A candidate is
    # still retained for the graph at this lower tier; it cannot become a
    # profile by itself because trajectory and direct-seed gates run later.
    return (
        float(row.get("glyphCorrelation", 0.0)) >= 0.48
        and float(row.get("glyphIou", 0.0)) >= 0.20
        and float(row.get("contamination", 1.0)) <= 0.65
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
    scan_start, scan_end = normalize_scan_range(
        frame_count, args.scan_start_frame, args.scan_end_frame
    )
    scan_length = scan_end - scan_start + 1
    roi = parse_roi(args.roi_json)
    roi_frame = args.roi_frame
    if roi_frame is not None and not 0 <= roi_frame < frame_count:
        raise RuntimeError("ROI frame is outside the source video")
    if roi_frame is not None and not scan_start <= roi_frame <= scan_end:
        raise RuntimeError("ROI frame is outside the selected scan range")
    roi_evidence = parse_roi_evidence(
        args.roi_evidence_json, frame_count, scan_start, scan_end
    )
    if roi is not None and roi_frame is not None:
        roi_evidence[roi_frame] = roi
    canonical = load_canonical()
    matching_mask = canonical
    edited_mask_path: Path | None = None
    if args.edited_mask:
        edited_mask_path = args.edited_mask
        if not edited_mask_path.is_absolute():
            edited_mask_path = args.profile_json.parent / edited_mask_path
        matching_mask = load_mask_asset(edited_mask_path, canonical)
    # Candidate extraction is the expensive part of a calibration retry.  A
    # cache keyed by the immutable source and canonical descriptor lets an
    # ROI-only retry rescan just the newly supplied evidence frames instead of
    # repeating every multi-scale/dual-polarity search in the video.
    candidate_cache_path = args.profile_json.parent / "candidate-cache.json"
    descriptor_hash = hashlib.sha256(matching_mask.tobytes()).hexdigest()
    candidate_cache: dict[int, list[dict[str, float | int | bool]]] = {}
    cache_reusable = False
    try:
        cached = json.loads(candidate_cache_path.read_text(encoding="utf-8"))
        assert_finite_json(cached)
        cache_reusable = bool(
            isinstance(cached, dict)
            and cached.get("version") == CANDIDATE_CACHE_VERSION
            and cached.get("sourceSha256") == sha256_file(source)
            and cached.get("descriptorSha256") == descriptor_hash
            and cached.get("scanRange") == {"startFrame": scan_start, "endFrame": scan_end}
        )
        if cache_reusable and isinstance(cached.get("candidates"), dict):
            candidate_cache = {
                int(frame): [
                    {key: value for key, value in row.items() if key != "userRoi"}
                    for row in rows
                    if isinstance(row, dict)
                ]
                for frame, rows in cached["candidates"].items()
                if isinstance(rows, list)
            }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A partial/crashed cache is disposable; calibration simply rebuilds it.
        candidate_cache = {}
        cache_reusable = False
    # A project created before candidate-cache.json may still have a valid
    # selected observation path from its last calibration.  Treat those rows
    # as a partial cache only when the source/range match; the missing sampled
    # frames are then scanned normally and the cache is completed atomically.
    if not cache_reusable:
        try:
            previous_profile = json.loads(args.profile_json.read_text(encoding="utf-8"))
            previous_observations_path = args.profile_json.parent / "trajectory-observations.json"
            previous_observations = json.loads(previous_observations_path.read_text(encoding="utf-8"))
            previous_source = previous_profile.get("sourceFingerprint", {})
            previous_range = previous_profile.get("scanRange")
            if (
                previous_source.get("sha256") == sha256_file(source)
                and previous_range == {"startFrame": scan_start, "endFrame": scan_end}
                and isinstance(previous_observations, list)
            ):
                candidate_cache = {
                    int(row["frame"]): [{key: value for key, value in row.items() if key != "userRoi"}]
                    for row in previous_observations
                    if isinstance(row, dict) and "frame" in row
                }
                cache_reusable = bool(candidate_cache)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            candidate_cache = {}
            cache_reusable = False
    cached_candidate_frames = len(candidate_cache) if cache_reusable else 0
    capture = cv2.VideoCapture(str(source))
    all_candidates: dict[int, list[dict[str, float | int | bool]]] = dict(candidate_cache)
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Re-run exact ROI frames so a new broad hint can improve a cached
            # global candidate.  All other frames already present in a valid
            # cache are reused byte-for-byte.
            if cache_reusable and frame_number in all_candidates and frame_number not in roi_evidence:
                frame_number += 1
                continue
            if cache_reusable and frame_number in roi_evidence:
                all_candidates.pop(frame_number, None)
            # The broad user ROI is sampled on its exact frame, even when that
            # frame falls between the normal stride phases.  All other frames
            # remain global searches so the ROI seeds a moving trajectory
            # instead of incorrectly constraining it to one screen location.
            if scan_start <= frame_number <= scan_end and (
                frame_number % SAMPLE_STRIDE == 0 or frame_number in roi_evidence
            ):
                # Evidence constrains only the exact frame where it was drawn.
                # Every other sampled frame is global, so ROI never becomes a
                # fixed position or a false trajectory prior.
                search_roi = roi_evidence.get(frame_number)
                candidates = candidate_rows(frame, frame_number, matching_mask, search_roi)
                if candidates:
                    if frame_number in roi_evidence:
                        for candidate in candidates:
                            candidate["userRoi"] = True
                    all_candidates[frame_number] = candidates
            frame_number += 1
    finally:
        capture.release()
    # Drop any non-finite rows defensively before cache serialization and
    # before trajectory selection.  This is the final guard for OpenCV builds
    # that return NaN from masked template matching on zero-energy patches.
    all_candidates = {
        frame: [row for row in rows if finite_candidate_row(row)]
        for frame, rows in all_candidates.items()
    }
    all_candidates = {frame: rows for frame, rows in all_candidates.items() if rows}
    write_strict_json(
        candidate_cache_path,
        {
            "version": CANDIDATE_CACHE_VERSION,
            "sourceSha256": sha256_file(source),
            "descriptorSha256": descriptor_hash,
            "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
            "candidates": {
                str(frame): [
                    {key: value for key, value in row.items() if key != "userRoi"}
                    for row in rows
                    if finite_candidate_row(row)
                ]
                for frame, rows in sorted(all_candidates.items())
            },
        },
    )
    # Mark only this run's explicit evidence after cache reuse.  The cache is
    # deliberately kept neutral so future retries can apply a different ROI
    # set without persisting stale user flags.
    for frame in roi_evidence:
        for row in all_candidates.get(frame, []):
            row["userRoi"] = True
    # The periodic model is optional evidence, never a fixed-position mask.
    # Admit it only when independent global matches agree with the shipped
    # Learna path at several phases.  This lets the known path rescue blurred
    # frames without making a different trajectory look READY.
    periodic_seed_rows = [
        row
        for frame, rows in all_candidates.items()
        for row in rows
        if hard_gate(row)
        and math.hypot(
            float(row["x"]) - periodic_position(frame)[0],
            float(row["y"]) - periodic_position(frame)[1],
        ) <= 120.0
    ]
    periodic_phase_bins = {
        int(int(row["frame"]) % 360 // 60) for row in periodic_seed_rows
    }
    periodic_seed_span = (
        max(int(row["frame"]) for row in periodic_seed_rows)
        - min(int(row["frame"]) for row in periodic_seed_rows)
        if periodic_seed_rows
        else 0
    )
    periodic_seed_valid = bool(
        width == 1080
        and height == 1920
        and len(periodic_seed_rows) >= 3
        and len(periodic_phase_bins) >= 3
        and periodic_seed_span >= max(180, int(scan_length * 0.50))
    )
    validated_periodic_track: list[dict[str, float | int | bool]] = []
    if periodic_seed_valid:
        # Calibrate a small per-video offset from independent observations;
        # do not assume the reference path is pixel-perfect for every encode.
        offset_x = float(
            np.median(
                [float(row["x"]) - periodic_position(int(row["frame"]))[0] for row in periodic_seed_rows]
            )
        )
        offset_y = float(
            np.median(
                [float(row["y"]) - periodic_position(int(row["frame"]))[1] for row in periodic_seed_rows]
            )
        )
        capture = cv2.VideoCapture(str(source))
        target_frames = set(all_candidates)
        frame_number = 0
        try:
            # Read sequentially instead of seeking once per sampled frame;
            # repeated VideoCapture seeks are disproportionately slow for
            # H.264 and made the UI look stuck during the prior pass.
            while target_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number in target_frames:
                    prior = periodic_prior_candidate(frame, frame_number, matching_mask)
                    if prior is not None:
                        prior["x"] = float(prior["x"]) + offset_x
                        prior["y"] = float(prior["y"]) + offset_y
                        prior["periodicPriorOffsetX"] = offset_x
                        prior["periodicPriorOffsetY"] = offset_y
                        # Keep both independent and prior evidence.  The graph
                        # uses motion continuity to select one path;
                        # diagnostics retain the original global candidates.
                        all_candidates[frame_number].append(prior)
                    target_frames.remove(frame_number)
                frame_number += 1
        finally:
            capture.release()
        periodic_rows = sorted(
            (
                row
                for rows in all_candidates.values()
                for row in rows
                if bool(row.get("periodicPrior", False))
            ),
            key=lambda row: (int(row["frame"]), -float(row["score"])),
        )
        # Keep one image-validated prior observation per sampled frame.  This
        # bypasses a terminal-chain tie in the generic graph (where a strong
        # background peak could reset the chain), while the independent seed
        # gate above still prevents this route on a different trajectory.
        periodic_track: list[dict[str, float | int | bool]] = []
        seen_periodic_frames: set[int] = set()
        for row in periodic_rows:
            frame = int(row["frame"])
            if frame in seen_periodic_frames:
                continue
            seen_periodic_frames.add(frame)
            periodic_track.append(row)
        if (
            len(periodic_track) >= 12
            and int(periodic_track[-1]["frame"]) - int(periodic_track[0]["frame"])
            >= max(180, int(frame_count * 0.50))
        ):
            validated_periodic_track = periodic_track
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
    # Preserve explicit ROI evidence when a high-scoring background chain wins
    # the unconstrained graph.  The seeded path is still subject to the same
    # residual, gap and coverage gates below; it can never bypass fail-closed
    # validation, but it gives the user's confirmed glyph locations a real
    # influence on the fitted trajectory.
    user_seed = choose_user_seeded_track(track_candidates)
    if len(user_seed) >= 2:
        # In a user-assisted route the confirmed ROI anchors are the source
        # of truth for the active path.  Keeping unrelated Viterbi segments
        # as additional tracks lets a high-contrast subtitle/background chain
        # create a second "active" interval after the watermark disappears.
        # Use the seeded path exclusively; it still goes through the normal
        # residual, gap and mask gates and therefore cannot bypass review.
        tracks = [user_seed]
    # A periodic prior is retained as another graph hypothesis only.  It must
    # never replace the free trajectory selected from image evidence: doing so
    # made a known 360-frame clip appear correct while silently misplacing a
    # different Learna trajectory.
    # Do not turn a long static subtitle/UI match into an active watermark
    # interval.  A real static watermark can still be recovered through
    # explicit ROI evidence; without that evidence the safe state is review,
    # not a guessed tail mask.
    tracks = filter_static_background_segments(tracks)
    # A seeded ROI path and the unconstrained Viterbi paths can contain the
    # same frame more than once.  Feeding both rows into the residual fit
    # makes one physical frame look like two conflicting observations and can
    # inflate p95 by hundreds of pixels.  Collapse to one observation per
    # frame, preferring the user-seeded candidate (the confirmed location
    # hint) and then the strongest remaining candidate.  This keeps the
    # trajectory statistics about physical frames rather than graph branches.
    user_seed_by_frame = {
        int(row["frame"]): row
        for row in user_seed
        if provisional_gate(row)
    }
    measured_by_frame: dict[int, dict[str, float | int | bool]] = dict(user_seed_by_frame)
    for segment in tracks:
        for row in segment:
            if not provisional_gate(row):
                continue
            frame = int(row["frame"])
            if frame in measured_by_frame:
                continue
            current = measured_by_frame.get(frame)
            if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                measured_by_frame[frame] = row
    measured = [measured_by_frame[frame] for frame in sorted(measured_by_frame)]
    track = measured
    active_intervals = active_intervals_from_segments(
        tracks, frame_count, fps, scan_start, scan_end
    )
    # V7 performs a dense, source-resolution local pass around the selected
    # path.  This is the important bridge between a handful of ROI seeds and
    # every active frame: it validates the predicted location instead of
    # fitting a path only through the user's anchors.
    refined_rows = refine_local_path(
        source,
        matching_mask,
        measured,
        active_intervals,
        width,
        height,
    )
    if refined_rows:
        for row in refined_rows:
            frame = int(row["frame"])
            existing = measured_by_frame.get(frame)
            # Prefer an image-refined row unless it is materially weaker than
            # a hard/direct observation already present at the same frame.
            if existing is None or not hard_gate(existing) or float(row.get("score", 0.0)) >= float(existing.get("score", 0.0)) * 0.70:
                measured_by_frame[frame] = row
        measured = [measured_by_frame[frame] for frame in sorted(measured_by_frame)]
        track = measured
    # Trim a terminal chain of background candidates after the last validated
    # glyph evidence.  This handles videos whose watermark disappears before
    # the file ends without assuming a fixed end frame.
    validated_frames = [
        int(frame)
        for frame, rows in all_candidates.items()
        if any(hard_gate(row) or bool(row.get("userRoi", False)) for row in rows)
    ]
    if validated_frames:
        last_validated = max(validated_frames)
        active_intervals = [
            {
                "startFrame": int(interval["startFrame"]),
                "endFrame": min(int(interval["endFrame"]), last_validated + MAX_GAP),
            }
            for interval in active_intervals
            if int(interval["startFrame"]) <= last_validated + MAX_GAP
        ]
        active_intervals = [interval for interval in active_intervals if interval["endFrame"] >= interval["startFrame"]]
    # A periodic model is permitted only when the global evidence actually
    # covers the source.  It is an optional candidate model, never a way to
    # manufacture a full-video interval from a few terminal observations.
    observed_span = sum(
        max(0, int(segment[-1]["frame"]) - int(segment[0]["frame"]) + SAMPLE_STRIDE)
        for segment in tracks
    )
    periodic_candidate_allowed = (
        periodic_seed_valid
        and observed_span >= scan_length * 0.50
        and width == 1080
        and height == 1920
    )
    first_active = min((interval["startFrame"] for interval in active_intervals), default=scan_start)
    last_active = max((interval["endFrame"] for interval in active_intervals), default=scan_start)
    hard_measured = [row for row in measured if hard_gate(row)]
    roi_measured = [row for row in measured if bool(row.get("userRoi", False))]
    confirmed_measured = [
        row for row in measured
        if hard_gate(row) or bool(row.get("userRoi", False)) or bool(row.get("refined", False))
    ]
    # Provisional global matches are useful for finding active intervals but
    # can jump to a subtitle/face branch on busy shots.  Once enough explicit
    # ROI/direct evidence exists, fit through that confirmed control path and
    # keep provisional rows as diagnostics/coverage hints only.  This avoids
    # an outlier branch inflating p95 and triggering another ROI round.
    fit_rows = (
        confirmed_measured
        if len(confirmed_measured) >= ROI_SATURATION_MIN_EVIDENCE
        else measured
    )
    points = [(float(row["x"]), float(row["y"])) for row in fit_rows]
    key_indices = rdp_indices(points, 2.0 if len(fit_rows) >= 24 else 6.0) if len(fit_rows) >= 2 else []
    key_track = [fit_rows[index] for index in key_indices]
    if len(key_track) < 2 and fit_rows:
        key_track = [fit_rows[0], fit_rows[-1]] if len(fit_rows) > 1 else fit_rows

    def path_residuals(rows: list[dict[str, float | int | bool]]) -> list[float]:
        if len(key_track) < 2:
            return []
        values: list[float] = []
        for row in rows:
            px, py, _, _, _ = interpolate(key_track, int(row["frame"]))
            values.append(math.hypot(float(row["x"]) - px, float(row["y"]) - py))
        return values

    raw_residuals = path_residuals(measured)
    residuals = path_residuals(fit_rows)
    if len(confirmed_measured) < ROI_SATURATION_MIN_EVIDENCE:
        residuals = raw_residuals
    median_residual = float(np.median(residuals)) if residuals else None
    p95_residual = float(np.percentile(residuals, 95)) if residuals else None
    raw_median_residual = float(np.median(raw_residuals)) if raw_residuals else None
    raw_p95_residual = float(np.percentile(raw_residuals, 95)) if raw_residuals else None
    inlier_ratio = float(sum(value <= 3.0 for value in residuals) / max(1, len(residuals)))
    # Compute the active denominator before any coverage gate.  The previous
    # V6 path calculated refined coverage before ``active_count`` existed,
    # which made a valid calibration crash with an unbound-local error.
    active_count = sum(interval["endFrame"] - interval["startFrame"] + 1 for interval in active_intervals)
    holdout = holdout_metrics(confirmed_measured, 2.0)
    refined_frames = {int(row["frame"]) for row in measured if bool(row.get("refined", False))}
    refined_coverage = min(1.0, len(refined_frames) / max(1, active_count))
    # Fit the affine prior only from independent global hard-gate rows.  The
    # periodic rows are image-validated bridge evidence and must not be able
    # to fit the model they were generated from.
    periodic_fit = fit_periodic_prior(periodic_seed_rows) if periodic_candidate_allowed else None
    periodic_transform: dict[str, float] | None = None
    if periodic_fit is not None:
        candidate_transform, periodic_residuals = periodic_fit
        # Select the prior only when it is demonstrably better than the free
        # fit on independent global observations.  Otherwise the free model
        # remains authoritative for this video.
        candidate_median = float(np.median(periodic_residuals))
        candidate_p95 = float(np.percentile(periodic_residuals, 95))
        candidate_inlier = float(sum(value <= 3.0 for value in periodic_residuals) / max(1, len(periodic_residuals)))
        if (
            candidate_inlier >= MIN_INLIER_RATIO
            and (p95_residual is None or candidate_p95 <= p95_residual)
        ):
            periodic_transform = candidate_transform
            residuals = periodic_residuals
            median_residual = candidate_median
            p95_residual = candidate_p95
            inlier_ratio = candidate_inlier
    max_gap = contiguous_gaps(int(row["frame"]) for row in measured) if measured else 10**9
    raw_review_ranges = review_ranges_from_gaps(
        measured,
        active_intervals,
        roi_evidence,
        all_candidates,
        scan_start,
        scan_end,
    )
    confirmed_frames = {
        int(row["frame"])
        for row in measured
        if hard_gate(row) or bool(row.get("userRoi", False))
    }
    # Keep the two notions separate.  ``measured`` is the selected graph path
    # (and may contain provisional rows), while direct coverage is reserved for
    # image-validated hard-gate matches.  The old calculation used
    # ``len(measured) * stride`` and therefore displayed 100% even when
    # hardMeasuredFrames was zero.  A user ROI is explicit location evidence,
    # so it is reported independently as confirmed coverage and may participate
    # in the ROI-assisted route without being mislabeled as direct detection.
    direct_coverage = min(1.0, len(hard_measured) * SAMPLE_STRIDE / max(1, active_count))
    confirmed_coverage = min(1.0, len(confirmed_frames) * SAMPLE_STRIDE / max(1, active_count))
    measured_coverage = min(1.0, len(measured) * SAMPLE_STRIDE / max(1, active_count))
    validated_frames = set(int(row["frame"]) for row in hard_measured)
    validated_frames.update(refined_frames)
    validated_coverage = min(1.0, len(validated_frames) * LOCAL_REFINE_STRIDE / max(1, active_count))
    measured_span = (
        max(int(row["frame"]) for row in measured) - min(int(row["frame"]) for row in measured) + SAMPLE_STRIDE
        if measured
        else 0
    )
    global_span_ratio = measured_span / max(1, scan_length)
    residual_tolerance = 2.0 * width / REFERENCE_WIDTH
    residual_p95_tolerance = 3.0 * width / REFERENCE_WIDTH
    # Do not make a user who has already supplied broad temporal evidence
    # chase a new list of representative frames on every pass.  At this point
    # the remaining problem is path fitting (the current clip has p95 around
    # 15 px), so expose that diagnosis and leave the profile fail-closed until
    # the automatic refinement stage brings the residual below the gate.
    roi_review_saturated = should_suppress_roi_review(
        len(roi_measured),
        confirmed_coverage,
        measured_coverage,
        # Use the raw candidate residual to decide whether more ROI hints are
        # useful.  A confirmed control path may already fit within tolerance
        # even when noisy provisional background candidates remain in the raw
        # diagnostics; that is a successful calibration, not a new manual task.
        raw_p95_residual,
        residual_p95_tolerance,
        max_gap,
    )
    review_ranges = [] if roi_review_saturated else raw_review_ranges
    # A sparse but geometrically stable fit is only actionable after the user
    # has supplied ROI evidence.  The automatic global route must not be able
    # to turn a short false-positive segment into READY merely because its
    # self-derived active interval is small (the exact failure that caused a
    # blue transition near the end of 14_7 to mask the rest of the video).
    # Auto-global therefore needs independently hard-gated glyph anchors
    # spanning a meaningful part of the source; ROI routes may use weaker
    # provisional anchors, but still need the temporal/uncertainty gates.
    # V7 never turns a sparse ROI path into READY.  Refinement must produce
    # enough independent observations for the active interval and holdout.
    sparse_fit_ok = False
    auto_global_evidence_ok = bool(
        args.route != "AUTO_GLOBAL_TEMPLATE"
        or (
            len(hard_measured) >= max(3, math.ceil(len(measured) * 0.10))
            and len(measured) >= 30
            and global_span_ratio >= 0.25
        )
    )
    holdout_p95 = holdout.get("p95")
    holdout_median = holdout.get("median")
    holdout_inlier_ratio = float(holdout.get("inlierRatio", 0.0) or 0.0)
    trajectory_passed = bool(
        auto_global_evidence_ok
        and len(measured) >= max(30, math.ceil(active_count * 0.10))
        # ROI-assisted calibration may use the temporally validated graph path
        # as its coverage signal; AUTO_GLOBAL must still prove coverage with
        # hard image matches.  ROI anchors are sparse by design, so using only
        # their count here would make every broad-ROI route fail before the
        # graph/refinement stage has a chance to validate the intervening path.
        # Residual, gap, uncertainty and mask gates remain mandatory in both
        # routes.
        and (
            (measured_coverage if args.route != "AUTO_GLOBAL_TEMPLATE" else validated_coverage)
            >= 0.70
        )
        and inlier_ratio >= MIN_INLIER_RATIO
        and median_residual is not None and median_residual <= residual_tolerance
        and p95_residual is not None and p95_residual <= residual_p95_tolerance
        and max_gap <= MAX_GAP
        and refined_coverage >= 0.70
        and holdout_p95 is not None and float(holdout_p95) <= residual_p95_tolerance
        and holdout_median is not None and float(holdout_median) <= residual_tolerance
        and holdout_inlier_ratio >= MIN_INLIER_RATIO
        and len(key_track) >= 2
    )

    calibration_dir = args.profile_json.parent
    calibration_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = calibration_dir / "canonical_mask.png"
    auto_path = calibration_dir / "auto_mask.png"
    inference_path = calibration_dir / "inference_mask.png"
    blend_path = calibration_dir / "blend_mask.png"
    cv2.imwrite(str(canonical_path), canonical)
    cv2.imwrite(str(auto_path), matching_mask)
    inference = cv2.morphologyEx(matching_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inference = cv2.dilate(inference, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    blend = cv2.GaussianBlur(inference, (0, 0), 1.75)
    cv2.imwrite(str(inference_path), inference)
    cv2.imwrite(str(blend_path), blend)

    frame_data: list[dict[str, object]] = []
    measured_frames = {int(row["frame"]) for row in measured}
    refined_by_frame = {
        int(row["frame"]): row for row in measured if bool(row.get("refined", False))
    }
    for frame in range(frame_count):
        inside_scan_range = scan_start <= frame <= scan_end
        active = bool(
            inside_scan_range
            and measured
            and frame_in_intervals(frame, active_intervals)
        )
        if active and frame in refined_by_frame:
            refined = refined_by_frame[frame]
            x = float(refined["x"])
            y = float(refined["y"])
            scale = float(refined.get("scale", 1.0))
            confidence = float(refined.get("score", 0.0))
            source_name = "local-ncc"
        elif active and periodic_transform is not None:
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
            x, y, scale, source_name, confidence = (
                0.0,
                0.0,
                1.0,
                "inactive" if inside_scan_range else "excluded-scan-range",
                0.0,
            )
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
    write_contact_sheet(source, measured, matching_mask, contact_sheet_path)
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
                "rawResidualMedian": raw_median_residual,
                "rawResidualP95": raw_p95_residual,
                "residualFitSource": "CONFIRMED_CONTROL_PATH" if len(confirmed_measured) >= ROI_SATURATION_MIN_EVIDENCE else "SELECTED_CANDIDATE_PATH",
                "residualFitFrames": len(fit_rows),
                "maxGap": max_gap,
                "keyPoints": len(key_track),
                "directCoverage": direct_coverage,
                "validatedCoverage": validated_coverage,
                "confirmedCoverage": confirmed_coverage,
                "measuredCoverage": measured_coverage,
                "hardMeasuredCount": len(hard_measured),
                "roiEvidenceCount": len(roi_measured),
                "globalSpanRatio": global_span_ratio,
                "activeIntervals": active_intervals,
                "periodicPriorUsed": periodic_transform is not None,
                "independentSeedCount": len(periodic_seed_rows),
                "independentSeedPhaseBins": sorted(periodic_phase_bins),
                "independentSeedSpan": periodic_seed_span,
                "rawReviewRanges": raw_review_ranges,
                "reviewRanges": review_ranges,
                "reviewRangesSuppressed": roi_review_saturated,
                "reviewSuppressionReason": (
                    "ROI_EVIDENCE_SATURATED_TRAJECTORY_REFINEMENT_REQUIRED"
                    if roi_review_saturated
                    else None
                ),
                "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
                "excludedFrameCount": frame_count - scan_length,
            },
            "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
            "status": "READY" if trajectory_passed else "NEEDS_REVIEW",
            "candidateCache": {
                "reused": cache_reusable,
                "seededFrames": cached_candidate_frames,
                "finalFrames": len(all_candidates),
            },
        },
    )
    quality_status = "PASSED" if trajectory_passed else "FAILED"
    failure_reasons: list[str] = []
    if not measured:
        failure_reasons.append("NO_VALID_OBSERVATIONS")
    if measured and len(measured) < max(30, math.ceil(active_count * 0.10)):
        failure_reasons.append("TRAJECTORY_UNDERCONSTRAINED")
    if args.route == "AUTO_GLOBAL_TEMPLATE" and not auto_global_evidence_ok:
        failure_reasons.append("INSUFFICIENT_GLOBAL_EVIDENCE")
    # A periodic prior may help prediction, but it cannot hide an unresolved
    # active gap from the V7 gate or make the UI appear complete.
    if max_gap > MAX_GAP:
        failure_reasons.append("UNRESOLVED_ACTIVE_RANGE")
    if roi_review_saturated and not trajectory_passed:
        failure_reasons.append("TRAJECTORY_REFINEMENT_REQUIRED")
    if p95_residual is not None and p95_residual > residual_p95_tolerance:
        failure_reasons.append("TRAJECTORY_RESIDUAL_TOO_HIGH")
    if inlier_ratio < MIN_INLIER_RATIO:
        failure_reasons.append("LOW_INLIER_RATIO")
    if refined_coverage < 0.70:
        failure_reasons.append("INSUFFICIENT_REFINED_COVERAGE")
    if holdout_p95 is None:
        failure_reasons.append("HOLDOUT_UNAVAILABLE")
    elif float(holdout_p95) > residual_p95_tolerance:
        failure_reasons.append("HOLDOUT_RESIDUAL_TOO_HIGH")
    if holdout_inlier_ratio < MIN_INLIER_RATIO:
        failure_reasons.append("LOW_HOLDOUT_INLIER_RATIO")
    # Keep trajectory segments tied to evidence found in this video.  The
    # periodic model is optional, so never bake the old 0/120/180/300/360
    # checkpoints into a V7 profile.
    trajectory_segment_frames = {
        scan_start,
        scan_end,
        *(int(row["frame"]) for row in key_track),
        *(int(interval["startFrame"]) for interval in active_intervals),
        *(int(interval["endFrame"]) for interval in active_intervals),
    }
    trajectory_segment_frames = {
        frame for frame in trajectory_segment_frames if scan_start <= frame <= scan_end
    }
    profile = {
        "version": CALIBRATION_VERSION,
        "status": "READY" if trajectory_passed else "NEEDS_REVIEW",
        "preset": "LEARNA_AI_ADAPTIVE",
        "detectorVersion": "learna-global-template-v7.0-local-refine",
        "validationVersion": "holdout-v1",
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
        "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
        "scanRangeSemantics": "inclusive",
        "excludedFrameCount": frame_count - scan_length,
        "outsideRangePolicy": "PASSTHROUGH_WARN",
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
        "editedMaskPath": (
            edited_mask_path.relative_to(calibration_dir.parent).as_posix()
            if edited_mask_path and edited_mask_path.is_relative_to(calibration_dir.parent)
            else (str(edited_mask_path) if edited_mask_path else None)
        ),
        "editedMaskSha256": sha256_file(edited_mask_path) if edited_mask_path else None,
        "roiEvidenceFrames": sorted(roi_evidence),
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
                    for frame in sorted(trajectory_segment_frames)
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
            "rawTrajectoryResidualMedian": raw_median_residual,
            "rawTrajectoryResidualP95": raw_p95_residual,
            "trajectoryResidualFitSource": "CONFIRMED_CONTROL_PATH" if len(confirmed_measured) >= ROI_SATURATION_MIN_EVIDENCE else "SELECTED_CANDIDATE_PATH",
            "trajectoryResidualFitFrames": len(fit_rows),
            "refinedFrames": len(refined_frames),
            "refinedCoverage": refined_coverage,
            "holdout": holdout,
            "holdoutMedian": holdout_median,
            "holdoutP95": holdout_p95,
            "holdoutInlierRatio": holdout_inlier_ratio,
            "directCoverage": direct_coverage,
            "validatedCoverage": validated_coverage,
            "confirmedCoverage": confirmed_coverage,
            "measuredCoverage": measured_coverage,
            "hardMeasuredFrames": len(hard_measured),
            "roiEvidenceFrames": len(roi_measured),
            "globalSpanRatio": global_span_ratio,
            "maxInterpolationGap": 6 if periodic_transform is not None else max_gap,
            "failureReasons": failure_reasons,
            "rawReviewRanges": raw_review_ranges,
            "reviewRanges": review_ranges,
            "reviewRangesSuppressed": roi_review_saturated,
            "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
            "excludedFrameCount": frame_count - scan_length,
            "outsideRangeUnchecked": scan_length != frame_count,
        },
        "trajectoryGate": {
            "status": "PASSED" if trajectory_passed else "FAILED",
            "inlierRatio": inlier_ratio,
            "residualMedian": median_residual,
            "residualP95": p95_residual,
            "rawResidualMedian": raw_median_residual,
            "rawResidualP95": raw_p95_residual,
            "residualFitSource": "CONFIRMED_CONTROL_PATH" if len(confirmed_measured) >= ROI_SATURATION_MIN_EVIDENCE else "SELECTED_CANDIDATE_PATH",
            "residualFitFrames": len(fit_rows),
            "refinedFrames": len(refined_frames),
            "refinedCoverage": refined_coverage,
            "holdout": holdout,
            "holdoutMedian": holdout_median,
            "holdoutP95": holdout_p95,
            "holdoutInlierRatio": holdout_inlier_ratio,
            "directCoverage": direct_coverage,
            "validatedCoverage": validated_coverage,
            "confirmedCoverage": confirmed_coverage,
            "measuredCoverage": measured_coverage,
            "hardMeasuredFrames": len(hard_measured),
            "roiEvidenceFrames": len(roi_measured),
            "maxInterpolationGap": 6 if periodic_transform is not None else max_gap,
            "failureReasons": failure_reasons,
            "rawReviewRanges": raw_review_ranges,
            "reviewRanges": review_ranges,
            "reviewRangesSuppressed": roi_review_saturated,
            "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
            "excludedFrameCount": frame_count - scan_length,
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
