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
    target_frames = expanded_frames(failed, frame_count)
    frame_data = profile.get("frameData", [])
    canonical = qa.v6.load_canonical()
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
                # Prefer the independent full-frame residual detection.  A
                # stale trajectory can miss the glyph by hundreds of pixels;
                # using that stale box for the cover would reproduce the old
                # transparent-badge failure.
                # Reuse the source-side full-frame detection already produced
                # by QA.  This keeps the safety pass deterministic and avoids
                # running the expensive multi-scale detector a second time for
                # every failed frame.  A fresh scan remains the guarded
                # fallback when an older/diagnostic report has no source row.
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
            process.stdin.write(image.tobytes())
            frame += 1
    finally:
        capture.release()
        source_capture.release()
        process.stdin.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with code {code}")
    manifest = {"version": 1, "mode": "OPAQUE_QUANPH_BADGE", "failedInputFrames": sorted(failed), "appliedFrames": applied, "frameCount": frame}
    a.output.with_suffix(".badge-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
