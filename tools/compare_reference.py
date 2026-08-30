from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a render with a golden output")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def metadata(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(result.stdout) if result.returncode == 0 and result.stdout else {}


def video_stream(meta: dict) -> dict:
    return next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})


def similarity(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mae = float(np.mean(np.abs(left_gray - right_gray)))
    left_centered = left_gray - left_gray.mean()
    right_centered = right_gray - right_gray.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    corr = float(np.sum(left_centered * right_centered) / denominator) if denominator > 1e-6 else 1.0
    return mae, corr


def main() -> None:
    args = parse_args()
    candidate_meta = metadata(args.candidate)
    reference_meta = metadata(args.reference)
    candidate_video = video_stream(candidate_meta)
    reference_video = video_stream(reference_meta)
    candidate_count = int(candidate_video.get("nb_frames") or 0)
    reference_count = int(reference_video.get("nb_frames") or 0)
    frame_count = min(candidate_count, reference_count)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    if frame_count <= 0:
        raise RuntimeError("Unable to determine comparable video frame count")
    left_capture = cv2.VideoCapture(str(args.candidate))
    right_capture = cv2.VideoCapture(str(args.reference))
    maes: list[float] = []
    correlations: list[float] = []
    decoded = 0
    try:
        for _ in range(frame_count):
            left_ok, left = left_capture.read()
            right_ok, right = right_capture.read()
            if not left_ok or not right_ok:
                break
            mae, corr = similarity(left, right)
            maes.append(mae)
            correlations.append(corr)
            decoded += 1
    finally:
        left_capture.release()
        right_capture.release()
    if decoded != frame_count:
        raise RuntimeError(f"Reference comparison decoded {decoded} of {frame_count} aligned frames")
    report = {
        "version": 1,
        "status": "aligned" if candidate_count == reference_count else "partial",
        "alignedBy": "frame-index",
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "candidateFrameCount": candidate_count,
        "referenceFrameCount": reference_count,
        "comparedFrames": decoded,
        "meanFrameMae": float(np.mean(maes)),
        "p95FrameMae": float(np.percentile(maes, 95)),
        "meanFrameCorrelation": float(np.mean(correlations)),
        "minFrameCorrelation": float(np.min(correlations)),
        "metadata": {
            "candidate": candidate_video,
            "reference": reference_video,
            "candidateDuration": candidate_meta.get("format", {}).get("duration"),
            "referenceDuration": reference_meta.get("format", {}).get("duration"),
        },
        "note": "This is a consistency comparison, not pixel-identical ground truth for inpainting.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
