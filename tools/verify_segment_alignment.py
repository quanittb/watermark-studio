from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a clip is a segment of a full video")
    parser.add_argument("full_video", type=Path)
    parser.add_argument("clip_video", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = cv2.VideoCapture(str(args.full_video))
    clip = cv2.VideoCapture(str(args.clip_video))
    full_count = int(full.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_count = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(clip_count, full_count - args.offset)
    if count <= 0:
        raise RuntimeError("The requested segment is outside the full video")
    indices = sorted(set(np.linspace(0, count - 1, min(args.sample_count, count), dtype=np.int64).tolist()))
    correlations: list[float] = []
    maes: list[float] = []
    try:
        for index in indices:
            full.set(cv2.CAP_PROP_POS_FRAMES, args.offset + index)
            clip.set(cv2.CAP_PROP_POS_FRAMES, index)
            full_ok, full_frame = full.read()
            clip_ok, clip_frame = clip.read()
            if not full_ok or not clip_ok:
                raise RuntimeError(f"Unable to decode alignment sample at clip frame {index}")
            full_small = cv2.resize(full_frame, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            clip_small = cv2.resize(clip_frame, (180, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            full_gray = cv2.cvtColor(full_small.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            clip_gray = cv2.cvtColor(clip_small.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            left = full_gray - full_gray.mean()
            right = clip_gray - clip_gray.mean()
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            correlations.append(float(np.sum(left * right) / denominator) if denominator > 1e-6 else 1.0)
            maes.append(float(np.mean(np.abs(full_small - clip_small))))
    finally:
        full.release()
        clip.release()
    mean_corr = float(np.mean(correlations))
    report = {
        "version": 1,
        "status": "MATCHED_SEGMENT" if mean_corr >= 0.95 else "NOT_CONFIRMED",
        "offsetFrames": args.offset,
        "fullFrameCount": full_count,
        "clipFrameCount": clip_count,
        "comparedFrames": len(indices),
        "meanCorrelation": mean_corr,
        "minCorrelation": float(np.min(correlations)),
        "p95Mae": float(np.percentile(maes, 95)),
        "sampledFrames": indices,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if report["status"] != "MATCHED_SEGMENT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
