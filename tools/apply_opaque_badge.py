"""Cover residual Learna AI frames with an opaque QuanPH badge.

This is deliberately a post-QA safety pass.  It never decides where a
watermark is; it only consumes the independent QA failed-frame list and the
per-frame V9 calibration boxes.  The plate is fully opaque so glyph gaps can
not reveal the old watermark underneath.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import quality_qa_v9 as qa  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("review", type=Path)
    p.add_argument("profile", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--badge", type=Path, default=None)
    p.add_argument("--margin", type=int, default=10)
    return p.parse_args()


def metadata(path: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)["streams"][0]
    return int(value["width"]), int(value["height"]), str(value["avg_frame_rate"])


def failed_frames(report: dict) -> set[int]:
    metrics = report.get("metrics", report)
    values = metrics.get("failedFrames", [])
    output: set[int] = set()
    for value in values:
        try:
            output.add(int(value))
        except (TypeError, ValueError):
            continue
    # Also accept the V9 top-level shape for reports produced by diagnostics.
    for value in report.get("failedFrames", []):
        try:
            output.add(int(value))
        except (TypeError, ValueError):
            continue
    return output


def expanded_frames(frames: set[int], count: int, padding: int = 8) -> set[int]:
    result: set[int] = set()
    for frame in frames:
        result.update(range(max(0, frame - padding), min(count, frame + padding + 1)))
    return result


def rounded_plate(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a genuinely opaque, anti-aliased rounded rectangle in-place."""
    layer = frame.copy()
    radius = max(4, min(14, (y1 - y0) // 4))
    cv2.rectangle(layer, (x0 + radius, y0), (x1 - radius, y1), (24, 20, 34), -1)
    cv2.rectangle(layer, (x0, y0 + radius), (x1, y1 - radius), (24, 20, 34), -1)
    cv2.circle(layer, (x0 + radius, y0 + radius), radius, (24, 20, 34), -1)
    cv2.circle(layer, (x1 - radius, y0 + radius), radius, (24, 20, 34), -1)
    cv2.circle(layer, (x0 + radius, y1 - radius), radius, (24, 20, 34), -1)
    cv2.circle(layer, (x1 - radius, y1 - radius), radius, (24, 20, 34), -1)
    frame[y0:y1 + 1, x0:x1 + 1] = layer[y0:y1 + 1, x0:x1 + 1]


def badge_pixels(path: Path | None, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if path is not None and path.is_file():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        image = None
    target_w = max(24, int(width * 0.82))
    target_h = max(16, int(height * 0.66))
    if image is None:
        pixels = np.zeros((target_h, target_w, 3), np.uint8)
        cv2.putText(pixels, "QuanPH", (4, int(target_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, target_h / 55.0), (240, 240, 255), max(1, target_h // 18), cv2.LINE_AA)
        return pixels, (np.any(pixels > 0, axis=2).astype(np.float32))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return image[:, :, :3], image[:, :, 3].astype(np.float32) / 255.0


def _candidate_quality(row: dict) -> float:
    """Rank a source candidate for the safety-cover trajectory.

    A fallback cover must tolerate a partially occluded glyph, but it must
    still prefer the canonical shape over a high-contrast background.  The
    temporal graph below supplies the final discrimination; this score is
    deliberately bounded and never used to mark calibration READY.
    """
    raw = float(row.get("rawScore", 0.0) or 0.0)
    corr = float(row.get("glyphCorrelation", 0.0) or 0.0)
    iou = float(row.get("glyphIou", 0.0) or 0.0)
    contamination = float(row.get("contamination", 1.0) or 1.0)
    large = 0.20 if bool(row.get("largeOutsideComponent", False)) else 0.0
    return 1.00 * raw + 1.15 * corr + 2.00 * iou - 0.45 * contamination - large


def _sample_fallback_candidates(
    capture: cv2.VideoCapture,
    frame_count: int,
    active_frames: set[int],
    canonical: np.ndarray,
    stride: int = 12,
) -> list[tuple[int, list[dict]]]:
    """Scan sparse source frames for a bounded, full-frame cover path.

    This is only used after a failed trajectory gate.  Keeping the scan sparse
    and interpolating between observations avoids producing a second full
    ProPainter job while still allowing a wrong legacy profile to be ignored.
    """
    if not active_frames:
        return []
    active_sorted = sorted(active_frames)
    start, end = active_sorted[0], active_sorted[-1]
    sampled: list[tuple[int, list[dict]]] = []
    frame = 0
    visible_y_min = 0.0
    visible_y_max = float("inf")
    while frame <= end:
        ok, image = capture.read()
        if not ok:
            break
        if frame == start:
            # The source clips used for acceptance are portrait composites
            # with blurred top/bottom padding.  Those bands generate very
            # strong template peaks but can never contain the visible
            # watermark.  Detect that layout from edge energy rather than
            # hard-coding a coordinate range; ordinary full-frame sources
            # retain the complete search area.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            height = energy.shape[0]
            border = float(np.mean(np.concatenate((energy[: max(1, height // 10)], energy[-max(1, height // 10):]))))
            middle = float(np.mean(energy[height // 5: height * 4 // 5]))
            if middle > 1.0 and border < middle * 0.70:
                # Keep a conservative margin around the sharp central
                # portrait.  The false peaks observed in the blurred bands
                # cluster around 100–220px; the real Learna glyph remains in
                # the visible content below that boundary for this layout.
                visible_y_min = height * 0.15
                visible_y_max = height * 0.88
        if frame >= start and frame in active_frames and ((frame - start) % stride == 0 or frame == end):
            # QA's bounded full-frame proposal list is sufficient here and is
            # materially cheaper than the production calibration pyramid.
            # Keep several peaks so a blurred/occluded glyph can be recovered
            # by the temporal path instead of locking onto the first texture.
            rows = qa.detect_candidates_fast(
                image, canonical, limit=12, peaks_per_scale=12
            )
            rows = [
                row for row in rows
                if float(row.get("rawScore", 0.0) or 0.0) >= 0.62
                and float(row.get("width", 0.0) or 0.0) > 0
                and float(row.get("y", 0.0) or 0.0) >= visible_y_min
                and float(row.get("y", 0.0) or 0.0) + float(row.get("height", 0.0) or 0.0) <= visible_y_max
            ]
            rows.sort(key=_candidate_quality, reverse=True)
            sampled.append((frame, rows[:30]))
        frame += 1
    return sampled


def _fit_fallback_path(
    sampled: list[tuple[int, list[dict]]],
    frame_count: int,
    active_frames: set[int],
    seed_bbox: dict | None = None,
) -> dict[int, dict]:
    """Choose a temporally continuous candidate path and interpolate gaps.

    The previous fallback reused the failed profile box, which is precisely
    the unsafe case: it could place a perfect opaque plate hundreds of pixels
    away from the old glyph.  A small Viterbi graph keeps the actual source
    candidates available and rejects teleports to static background texture.
    """
    if not sampled:
        return {}
    costs: list[list[float]] = []
    parents: list[list[int]] = []
    for index, (frame, rows) in enumerate(sampled):
        frame_costs: list[float] = []
        frame_parents: list[int] = []
        best_previous = min(costs[index - 1], default=0.0) if index else 0.0
        previous_frame, previous_rows = sampled[index - 1] if index else (frame, [])
        for candidate_index, row in enumerate(rows):
            base = -_candidate_quality(row)
            if index == 0 and seed_bbox:
                seed_x = float(seed_bbox.get("x", 0.0) or 0.0)
                seed_y = float(seed_bbox.get("y", 0.0) or 0.0)
                seed_distance = float(np.hypot(
                    float(row.get("x", 0.0)) - seed_x,
                    float(row.get("y", 0.0)) - seed_y,
                ))
                # A failed later trajectory can still have a trustworthy
                # first/global observation.  Use that as a soft seed so a
                # static background peak cannot win the initial path.
                base += min(100.0, 0.01 * seed_distance)
            # A restart is expensive.  Without this penalty a strong static
            # background edge can start a new path at every sample and beat
            # the real glyph simply because the glyph is motion-blurred.
            value = base + (0.0 if index == 0 else best_previous + 250.0)
            parent = -1
            if index:
                delta_frames = max(1, frame - previous_frame)
                # The real glyph is smooth at this sampling interval.  A
                # restart is allowed only when no nearby candidate exists.
                # The 132 clip contains fast diagonal moves; a 12-frame
                # sample can legitimately move roughly 200 source pixels.
                # Keep the graph permissive enough for that motion while the
                # restart penalty and velocity continuity reject teleports.
                max_jump = max(280.0, 24.0 * delta_frames)
                for previous_index, previous in enumerate(previous_rows):
                    distance = float(np.hypot(
                        float(row.get("x", 0.0)) - float(previous.get("x", 0.0)),
                        float(row.get("y", 0.0)) - float(previous.get("y", 0.0)),
                    ))
                    if distance > max_jump:
                        continue
                    scale_delta = abs(float(row.get("scale", 1.0)) - float(previous.get("scale", 1.0)))
                    # Position is a continuity prior, not a veto: a moving
                    # badge can cross a large portion of a portrait frame in
                    # one sparse interval.  Let the canonical evidence win
                    # when the two hypotheses are close in quality.
                    transition = 0.001 * distance + 2.5 * scale_delta
                    candidate_value = costs[index - 1][previous_index] + base + transition
                    if candidate_value < value:
                        value = candidate_value
                        parent = previous_index
            frame_costs.append(value)
            frame_parents.append(parent)
        costs.append(frame_costs)
        parents.append(frame_parents)
    if not costs[-1]:
        return {}
    terminal_index = min(range(len(costs[-1])), key=lambda item: costs[-1][item])
    selected: list[dict | None] = []
    for index in range(len(sampled) - 1, -1, -1):
        rows = sampled[index][1]
        selected.append(rows[terminal_index] if 0 <= terminal_index < len(rows) else None)
        terminal_index = parents[index][terminal_index] if 0 <= terminal_index < len(parents[index]) else -1
    selected.reverse()
    anchors = [(sampled[index][0], row) for index, row in enumerate(selected) if row is not None]
    if not anchors:
        return {}
    # Never interpolate from a disconnected later track.  That was the
    # source of the previous catastrophic fallback: when the graph lost the
    # real glyph, a background candidate became the first anchor and its box
    # was repeated over hundreds of frames.  A badge is only safe when the
    # selected chain starts at the first active sample and has bounded gaps;
    # otherwise the caller must leave the frame untouched and keep the job in
    # NEEDS_REVIEW rather than covering arbitrary content.
    if anchors[0][0] != sampled[0][0]:
        return {}
    if any(
        right_frame - left_frame > 72
        for (left_frame, _), (right_frame, _) in zip(anchors, anchors[1:])
    ):
        return {}
    result: dict[int, dict] = {}
    for frame in sorted(active_frames):
        previous = next((item for item in reversed(anchors) if item[0] <= frame), None)
        following = next((item for item in anchors if item[0] >= frame), None)
        if previous is None:
            previous = anchors[0]
        if following is None:
            following = anchors[-1]
        left_frame, left = previous
        right_frame, right = following
        if right_frame == left_frame:
            ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, (frame - left_frame) / float(right_frame - left_frame)))
        row: dict = {}
        for key in ("x", "y", "width", "height", "scale"):
            left_value = float(left.get(key, 0.0) or 0.0)
            right_value = float(right.get(key, left_value) or left_value)
            row[key] = left_value + (right_value - left_value) * ratio
        # Long gaps are intentionally covered with a wider plate.  This is a
        # safety fallback: it trades a little more replacement area for not
        # exposing an untracked fragment of Learna AI.
        gap = max(0, right_frame - left_frame)
        quality = _candidate_quality(left)
        # A low-evidence anchor is intentionally covered with a wider plate;
        # this is safer than exposing a partial glyph between two samples.
        row["uncertaintyPx"] = min(96.0, max(10.0, 36.0 - 18.0 * quality) + 1.5 * gap)
        row["positionSource"] = "BADGE_GLOBAL_PATH"
        result[frame] = row
    return result


def main() -> None:
    a = args()
    profile = json.loads(a.profile.read_text(encoding="utf-8-sig"))
    report = json.loads(a.report.read_text(encoding="utf-8-sig"))
    failed = failed_frames(report)
    report_rows = {
        int(row.get("frame")): row
        for row in report.get("rows", [])
        if isinstance(row, dict) and row.get("frame") is not None
    }
    capture = cv2.VideoCapture(str(a.review))
    source_capture = cv2.VideoCapture(str(a.source))
    if not capture.isOpened() or not source_capture.isOpened():
        raise RuntimeError("Unable to open source/review video for opaque badge pass")
    width, height, fps = metadata(a.review)
    frame_count = int(profile.get("frameCount", 0))
    frame_data = profile.get("frameData", [])
    # A failed trajectory is not safe for selective covering: the first QA
    # pass can miss a glyph on a frame where the calibration box is wrong.
    # In that case use the independent activity map and cover the whole active
    # interval.  This is the bounded REMOVE_THEN_COVER fallback, not a new
    # trajectory guess, and guarantees no old Learna glyph leaks between the
    # sparse failed-frame detections.
    trajectory_gate = profile.get("trajectoryGate") or {}
    trajectory_status = str(trajectory_gate.get("status", "")).upper()
    cover_all_active = trajectory_status not in {"READY", "PASSED"}
    if cover_all_active:
        failed.update(
            index
            for index, row in enumerate(frame_data)
            if bool(row.get("maskRequired", False))
        )
    target_frames = expanded_frames(failed, frame_count)
    canonical = qa.v6.load_canonical()
    fallback_boxes: dict[int, dict] = {}
    if cover_all_active:
        # The profile trajectory is explicitly failed, so never use its stale
        # bbox for the safety plate.  Build a source-side global path once and
        # interpolate it over the active interval instead.
        scan_capture = cv2.VideoCapture(str(a.source))
        try:
            active_frames = {
                index for index, row in enumerate(frame_data)
                if bool(row.get("maskRequired", False))
            }
            sampled = _sample_fallback_candidates(scan_capture, frame_count, active_frames, canonical)
            seed_bbox = None
            if active_frames:
                first_active = min(active_frames)
                if first_active < len(frame_data):
                    candidate_seed = frame_data[first_active].get("bbox")
                    if isinstance(candidate_seed, dict) and float(candidate_seed.get("width", 0.0) or 0.0) > 0:
                        seed_bbox = candidate_seed
            fallback_boxes = _fit_fallback_path(sampled, frame_count, active_frames, seed_bbox)
        finally:
            scan_capture.release()
    badge_path = a.badge if a.badge is not None else Path(__file__).resolve().parents[1] / "assets" / "quanph_watermark_v1.png"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", fps, "-i", "-",
               "-i", str(a.source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
               "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy",
               "-movflags", "+faststart", str(a.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    applied: list[int] = []
    applied_boxes: dict[str, dict[str, float | int | str]] = {}
    frame = 0
    try:
        while True:
            ok, image = capture.read()
            source_ok, source = source_capture.read()
            if not ok or not source_ok:
                break
            if frame in target_frames and frame < len(frame_data):
                row = frame_data[frame]
                report_row = report_rows.get(frame, {})
                bbox = row.get("bbox", {})
                if cover_all_active:
                    # A failed trajectory may not be replaced by the stale
                    # calibration box.  Only a box from the independent
                    # fallback path is eligible for an opaque plate; missing
                    # path frames are deliberately left unchanged so QA can
                    # report them as a review draft.
                    if frame not in fallback_boxes:
                        process.stdin.write(image.tobytes())
                        frame += 1
                        continue
                    bbox = fallback_boxes[frame]
                # Prefer the independent full-frame residual detection.  A
                # stale trajectory can miss the glyph by hundreds of pixels;
                # using that stale box for the cover would reproduce the old
                # transparent-badge failure.
                # Reuse the source-side full-frame detection already produced
                # by QA.  This keeps the safety pass deterministic and avoids
                # running the expensive multi-scale detector a second time for
                # every failed frame.  A fresh scan remains the guarded
                # fallback when an older/diagnostic report has no source row.
                if not cover_all_active:
                    detected = report_row.get("sourceDetection")
                    if not (isinstance(detected, dict) and float(detected.get("geometryScore", 0) or 0) >= qa.MIN_SOURCE_GEOMETRY):
                        detected = report_row.get("outputDetection")
                    if isinstance(detected, dict) and float(detected.get("width", 0) or 0) > 0:
                        bbox = detected
                    elif not report_row:
                        source_detected = qa.detect_fast(source, canonical)
                        if isinstance(source_detected, dict) and float(source_detected.get("geometryScore", 0) or 0) >= qa.MIN_SOURCE_GEOMETRY:
                            bbox = source_detected
                # A stale/under-inclusive activity interval must not prevent
                # the safety cover.  QA has already found a source/output
                # residual in this exact frame; in that case the independent
                # detection is authoritative even when V9 marked the frame
                # inactive.  This is what prevents the old glyph from leaking
                # through a transparent replacement badge.
                source_present = bool(report_row.get("sourcePresent", False))
                if (bool(row.get("maskRequired", False)) or source_present) and float(bbox.get("width", 0)) > 0:
                    uncertainty = float(row.get("uncertaintyPx", 0.0) or 0.0)
                    margin = max(int(a.margin), int(round(uncertainty + 4.0)))
                    x0 = max(0, int(round(float(bbox["x"]) - margin)))
                    y0 = max(0, int(round(float(bbox["y"]) - margin)))
                    x1 = min(width - 1, int(round(float(bbox["x"]) + float(bbox["width"]) + margin)))
                    y1 = min(height - 1, int(round(float(bbox["y"]) + float(bbox["height"]) + margin)))
                    if x1 > x0 and y1 > y0:
                        rounded_plate(image, x0, y0, x1, y1)
                        pixels, alpha = badge_pixels(badge_path, x1 - x0 + 1, y1 - y0 + 1)
                        ph, pw = pixels.shape[:2]
                        ox = x0 + max(0, (x1 - x0 + 1 - pw) // 2)
                        oy = y0 + max(0, (y1 - y0 + 1 - ph) // 2)
                        ex = min(width, ox + pw)
                        ey = min(height, oy + ph)
                        if ex > ox and ey > oy:
                            local_alpha = alpha[:ey - oy, :ex - ox, None]
                            base = image[oy:ey, ox:ex].astype(np.float32)
                            foreground = pixels[:ey - oy, :ex - ox].astype(np.float32)
                            image[oy:ey, ox:ex] = np.clip(foreground * local_alpha + base * (1.0 - local_alpha), 0, 255).astype(np.uint8)
                        applied.append(frame)
                        applied_boxes[str(frame)] = {
                            "x": float(x0),
                            "y": float(y0),
                            "width": float(x1 - x0 + 1),
                            "height": float(y1 - y0 + 1),
                            "positionSource": str(bbox.get("positionSource", "QA_SOURCE")),
                        }
            process.stdin.write(image.tobytes())
            frame += 1
    finally:
        capture.release()
        source_capture.release()
        process.stdin.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with code {code}")
    manifest = {"version": 2, "mode": "OPAQUE_QUANPH_BADGE", "coverAllActive": cover_all_active, "fallbackPathStatus": "READY" if fallback_boxes else "UNRESOLVED", "failedInputFrames": sorted(failed), "appliedFrames": applied, "appliedBoxes": applied_boxes, "frameCount": frame}
    a.output.with_suffix(".badge-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
