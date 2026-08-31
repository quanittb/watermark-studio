from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from render_periodic_dewatermark import bounds_for_position, periodic_position


MIN_CORRELATION = 0.65
MIN_IOU = 0.55
MAX_CONTAMINATION = 0.20
MIN_NEIGHBOURS = 3
# Best-quality sample discovery must not silently skip an already-visible
# watermark at the start of a clip.  Legacy/Preview may retain its own
# historical start hint, but the shared detector scans from frame zero.
FIRST_FRAME = 0
FRAME_STRIDE = 6
PADDING = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find glyph-verified Learna AI samples")
    parser.add_argument("project_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--scan-round", type=int, default=0)
    parser.add_argument("--exclude-frames", default="[]")
    parser.add_argument("--exclude-signatures", default="[]")
    parser.add_argument("--roi-json", default="")
    parser.add_argument("--anchor-frame", type=int, default=0)
    parser.add_argument("--scan-start-frame", type=int)
    parser.add_argument("--scan-end-frame", type=int)
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Scan every frame-stride phase in one pass (the Best-quality UI default).",
    )
    return parser.parse_args()


def load_canonical() -> np.ndarray:
    encoded = (Path(__file__).parent / "assets" / "learna_ai_mask.b64").read_text(encoding="ascii").strip()
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("Unable to decode canonical Learna AI mask")
    return np.where(decoded >= 64, 255, 0).astype(np.uint8)


def filter_components(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output = np.zeros_like(binary)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= 3 and width >= 2 and height >= 2:
            output[labels == label] = 255
    return cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def extract_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    positive = np.maximum(gray - cv2.GaussianBlur(gray, (0, 0), 8.0), 0.0)
    binary = np.where(positive >= 4.0, 255, 0).astype(np.uint8)
    return cv2.dilate(filter_components(binary), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def mask_metrics(candidate: np.ndarray, canonical: np.ndarray) -> tuple[float, float, float, bool]:
    if candidate.shape != canonical.shape:
        candidate = cv2.resize(candidate, (canonical.shape[1], canonical.shape[0]), interpolation=cv2.INTER_NEAREST)
    candidate_on = candidate >= 64
    canonical_on = canonical >= 64
    halo = cv2.dilate(canonical, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) >= 64
    intersection = int(np.count_nonzero(candidate_on & canonical_on))
    union = int(np.count_nonzero(candidate_on | canonical_on))
    candidate_count = max(1, int(np.count_nonzero(candidate_on)))
    canonical_count = max(1, int(np.count_nonzero(canonical_on)))
    coverage = intersection / canonical_count
    precision = intersection / candidate_count
    correlation = 0.5 * coverage + 0.5 * precision
    iou = intersection / max(1, union)
    contamination = int(np.count_nonzero(candidate_on & ~halo)) / candidate_count
    outside = np.where(candidate_on & ~halo, 255, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(outside, 8)
    large_outside_component = any(int(stats[index, cv2.CC_STAT_AREA]) >= 120 for index in range(1, count))
    return correlation, iou, contamination, large_outside_component


CANONICAL_WIDTH = 245.33333333333334
CANONICAL_HEIGHT = 74.66666666666667


def resolve_trajectory(roi: dict[str, float] | None, anchor_frame: int) -> tuple[int, float, float]:
    if not roi:
        return 0, 0.0, 0.0
    best: tuple[float, int, float, float] | None = None
    for phase_shift in range(360):
        expected_x, expected_y = periodic_position((anchor_frame + phase_shift) % 360)
        target_x = float(roi["x"]) + (float(roi["width"]) - CANONICAL_WIDTH) / 2.0
        target_y = float(roi["y"]) + (float(roi["height"]) - CANONICAL_HEIGHT) / 2.0
        offset_x = target_x - expected_x
        offset_y = target_y - expected_y
        candidate = (offset_x * offset_x + offset_y * offset_y, phase_shift, offset_x, offset_y)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def crop_at(
    frame: np.ndarray,
    frame_number: int,
    width: int,
    height: int,
    roi: dict[str, float] | None = None,
    anchor_frame: int = 0,
    phase_shift: int = 0,
    local_dx: float = 0.0,
    local_dy: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    base_x, base_y = periodic_position((frame_number + phase_shift) % 360)
    if roi:
        anchor_x, anchor_y = periodic_position((anchor_frame + phase_shift) % 360)
        target_x = float(roi["x"]) + (float(roi["width"]) - width) / 2.0
        target_y = float(roi["y"]) + (float(roi["height"]) - height) / 2.0
        x = base_x + target_x - anchor_x
        y = base_y + target_y - anchor_y
        x0, y0, x1, y1 = bounds_for_position(x, y, frame.shape[1], frame.shape[0])
    else:
        x, y = base_x, base_y
        x0, y0, x1, y1 = bounds_for_position(x, y, frame.shape[1], frame.shape[0])
    crop = frame[y0:y1, x0:x1]
    if crop.shape[1] != width or crop.shape[0] != height:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    x += local_dx
    y += local_dy
    if local_dx or local_dy:
        x0, y0, x1, y1 = bounds_for_position(x, y, frame.shape[1], frame.shape[0])
        crop = frame[y0:y1, x0:x1]
        if crop.shape[1] != width or crop.shape[0] != height:
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return crop, {
        "x": x,
        "y": y,
        "width": CANONICAL_WIDTH,
        "height": CANONICAL_HEIGHT,
    }


def best_roi_crop(
    frame: np.ndarray,
    frame_number: int,
    canonical: np.ndarray,
    roi: dict[str, float] | None,
    anchor_frame: int,
    phase_shift: int,
) -> tuple[np.ndarray, dict[str, float], tuple[float, float], tuple[float, float, float, bool]]:
    if not roi:
        crop, bbox = crop_at(frame, frame_number, canonical.shape[1], canonical.shape[0])
        return crop, bbox, (0.0, 0.0), mask_metrics(extract_mask(crop), canonical)
    best = None
    offsets = [0.0] if not roi else [float(value) for value in range(-24, 25, 8)]
    for dx in offsets:
        for dy in offsets:
            crop, bbox = crop_at(frame, frame_number, canonical.shape[1], canonical.shape[0], roi, anchor_frame, phase_shift, dx, dy)
            metrics = mask_metrics(extract_mask(crop), canonical)
            ranking = metrics[0] + 0.35 * metrics[1] - 0.4 * metrics[2] - (0.001 * (abs(dx) + abs(dy)))
            if best is None or ranking > best[0]:
                best = (ranking, crop, bbox, (dx, dy), metrics)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def read_frame(capture: cv2.VideoCapture, frame_number: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    return frame if ok else None


def background_complexity(crop: np.ndarray, canonical: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    halo = cv2.dilate(canonical, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) >= 64
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    outside = np.abs(edges)[~halo]
    return float(np.mean(outside)) if outside.size else 999.0


def temporal_gate(
    capture: cv2.VideoCapture,
    frame_number: int,
    frame_count: int,
    canonical: np.ndarray,
    roi: dict[str, float] | None = None,
    anchor_frame: int = 0,
    phase_shift: int = 0,
    local_offset: tuple[float, float] = (0.0, 0.0),
    scan_start: int = 0,
    scan_end: int | None = None,
) -> tuple[int, float]:
    passed = 0
    scores: list[float] = []
    bounded_end = frame_count - 1 if scan_end is None else min(frame_count - 1, scan_end)
    for neighbour in range(max(scan_start, frame_number - 2), min(bounded_end + 1, frame_number + 3)):
        frame = read_frame(capture, neighbour)
        if frame is None:
            continue
        crop, _ = crop_at(frame, neighbour, canonical.shape[1], canonical.shape[0], roi, anchor_frame, phase_shift, *local_offset)
        correlation, iou, contamination, large = mask_metrics(extract_mask(crop), canonical)
        if correlation >= 0.52 and iou >= 0.38 and contamination <= 0.28 and not large:
            passed += 1
        scores.append(correlation)
    return passed, float(np.mean(scores)) if scores else 0.0


def scene_signature(crop: np.ndarray, canonical: np.ndarray) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    halo = cv2.dilate(canonical, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) >= 64
    clean = gray.copy()
    clean[halo] = int(np.median(gray[~halo])) if np.any(~halo) else 0
    small = cv2.resize(clean, (8, 8), interpolation=cv2.INTER_AREA)
    return "".join("1" if value >= small.mean() else "0" for value in small.reshape(-1))


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8-sig"))
    source = Path(project["source"]["path"])
    frame_count = int(project["video"]["frameCount"])
    scan_start = 0 if args.scan_start_frame is None else int(args.scan_start_frame)
    scan_end = frame_count - 1 if args.scan_end_frame is None else int(args.scan_end_frame)
    if scan_start < 0 or scan_end >= frame_count or scan_start > scan_end:
        raise RuntimeError(
            f"Invalid scan range {scan_start}–{scan_end}; expected 0–{frame_count - 1} with start <= end"
        )
    excluded = [int(value) for value in json.loads(args.exclude_frames)]
    excluded_signatures = {str(value) for value in json.loads(args.exclude_signatures)}
    roi = json.loads(args.roi_json) if args.roi_json.strip() else None
    if roi:
        required = {"x", "y", "width", "height"}
        if not required.issubset(roi) or float(roi["width"]) < 8 or float(roi["height"]) < 8:
            raise RuntimeError("ROI hint must contain x, y, width and height of at least 8 pixels")
        roi = {key: float(roi[key]) for key in required}
        if not scan_start <= int(args.anchor_frame) <= scan_end:
            raise RuntimeError("ROI anchor frame is outside the selected scan range")
    anchor_frame = max(scan_start, min(int(args.anchor_frame), scan_end))
    phase_shift, _, _ = resolve_trajectory(roi, anchor_frame)
    canonical = load_canonical()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_directory / "canonical-mask.png"), canonical)

    candidates: list[dict[str, object]] = []
    capture = cv2.VideoCapture(str(source))
    try:
        # The watermark trajectory is periodic, while the detector samples on
        # a six-frame stride.  A single phase can miss the cleanest glyph
        # evidence entirely, so the Best-quality flow scans all six phases in
        # one invocation.  Alternatives rotate the order to avoid repeatedly
        # preferring the same scene when exclusions are applied.
        phases = (
            [(args.scan_round + index) % FRAME_STRIDE for index in range(FRAME_STRIDE)]
            if args.all_phases
            else [args.scan_round % FRAME_STRIDE]
        )
        for phase in phases:
            first = scan_start + ((phase - scan_start) % FRAME_STRIDE)
            for frame_number in range(first, scan_end + 1, FRAME_STRIDE):
                if any(abs(frame_number - rejected) < 72 for rejected in excluded):
                    continue
                frame = read_frame(capture, frame_number)
                if frame is None:
                    continue
                crop, bbox, local_offset, (correlation, iou, contamination, large) = best_roi_crop(frame, frame_number, canonical, roi, anchor_frame, phase_shift)
                candidate_mask = extract_mask(crop)
                if correlation < MIN_CORRELATION or iou < MIN_IOU or contamination > MAX_CONTAMINATION or large:
                    continue
                neighbours, temporal_score = temporal_gate(capture, frame_number, frame_count, canonical, roi, anchor_frame, phase_shift, local_offset, scan_start, scan_end)
                if neighbours < MIN_NEIGHBOURS:
                    continue
                complexity = background_complexity(crop, canonical)
                score = 0.45 * correlation + 0.25 * temporal_score + 0.20 * (1.0 - contamination) + 0.10 * (1.0 / (1.0 + complexity / 20.0))
                preview = args.output_directory / f"frame-{frame_number}.png"
                mask_path = args.output_directory / f"frame-{frame_number}-mask.png"
                editor_mask_path = args.output_directory / f"frame-{frame_number}-editor-mask.png"
                cv2.imwrite(str(preview), crop)
                cv2.imwrite(str(mask_path), candidate_mask)
                cv2.imwrite(str(editor_mask_path), canonical)
                signature = scene_signature(crop, canonical)
                if signature in excluded_signatures:
                    continue
                candidates.append({
                    "frame": frame_number,
                    "timestampSeconds": frame_number / float(project["video"]["fps"]),
                    "bbox": bbox,
                    "maskCoverage": int(np.count_nonzero(candidate_mask)),
                    "maskPeak": int(candidate_mask.max()),
                    "backgroundComplexity": complexity,
                    "temporalInstability": 1.0 - temporal_score,
                    "glyphCorrelation": correlation,
                    "glyphIou": iou,
                    "contamination": contamination,
                    "temporalPassCount": neighbours,
                    "score": score,
                    "sceneSignature": signature,
                    "previewPath": str(preview),
                    "maskPath": str(mask_path),
                    "editorMaskPath": str(editor_mask_path),
                    "roiFallback": bool(roi),
                    "trajectoryPhaseOffset": phase_shift,
                })
    finally:
        capture.release()

    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    selected: list[dict[str, object]] = []
    signatures: set[str] = set()
    for candidate in candidates:
        frame_number = int(candidate["frame"])
        signature = str(candidate["sceneSignature"])
        if signature in signatures or any(abs(frame_number - int(row["frame"])) < 72 for row in selected):
            continue
        signatures.add(signature)
        selected.append(candidate)
        if len(selected) == 5:
            break
    print(json.dumps(selected, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
