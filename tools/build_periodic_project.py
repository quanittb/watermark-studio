from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from render_periodic_dewatermark import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    PADDING,
    aligned_positions,
    bounds_for_position,
    filter_components,
    periodic_position,
    read_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_project_dir", type=Path)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_project_dir", type=Path)
    parser.add_argument("--start-frame", type=int, default=48)
    parser.add_argument("--template-frame", type=int)
    return parser.parse_args()


def create_clean_mask(template_frame: np.ndarray, x: float, y: float) -> np.ndarray:
    x0, y0, x1, y1 = bounds_for_position(
        x, y, template_frame.shape[1], template_frame.shape[0]
    )
    crop = template_frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 8.0)
    positive = np.maximum(gray - smooth, 0.0)
    return filter_components(np.where(positive >= 4.0, 255, 0).astype(np.uint8))


def build_tracking_frame(
    row: dict[str, float],
    x: float,
    y: float,
    fps: float,
    start_frame: int,
) -> dict[str, object]:
    frame_number = int(row["frame"])
    occluded = frame_number < start_frame
    image_score = max(0.0, min(1.0, float(row["modelScore"])))
    return {
        "frame": frame_number,
        "timestampSeconds": frame_number / fps,
        "bbox": {
            "x": x,
            "y": y,
            "width": CANONICAL_WIDTH,
            "height": CANONICAL_HEIGHT,
        },
        "confidence": 1.0 if occluded else (0.98 if image_score >= 0.30 else 0.90),
        "status": "OCCLUDED" if occluded else "AUTO_GOOD",
        "source": "FUSED",
        "locked": False,
        "scores": {
            "template": image_score,
            "highpass": image_score,
            "edge": image_score,
            "motion": 1.0,
            "position": 1.0,
            "size": 1.0,
            "opticalFlow": None,
            "forwardBackward": 0.99,
            "motionSmoothness": 1.0,
            "matchMargin": max(0.0, min(1.0, float(row["margin"]))),
        },
    }


def main() -> None:
    args = parse_args()
    project_path = args.source_project_dir / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    audit_rows = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if len(audit_rows) != int(project["video"]["frameCount"]):
        raise RuntimeError("Audit frame count does not match the project")

    if args.output_project_dir.exists():
        raise RuntimeError(f"Output directory already exists: {args.output_project_dir}")
    args.output_project_dir.mkdir(parents=True)
    shutil.copytree(
        args.source_project_dir / "templates",
        args.output_project_dir / "templates",
    )
    (args.output_project_dir / "cache").mkdir()

    offsets_x, offsets_y = aligned_positions(audit_rows)
    fps = float(project["video"]["fps"])
    tracking_frames: list[dict[str, object]] = []
    for row in audit_rows:
        frame_number = int(row["frame"])
        periodic_x, periodic_y = periodic_position(frame_number)
        x = periodic_x + offsets_x[frame_number]
        y = periodic_y + offsets_y[frame_number]
        tracking_frames.append(
            build_tracking_frame(row, x, y, fps, args.start_frame)
        )

    project["tracking"]["frames"] = tracking_frames
    project["tracking"]["problemRanges"] = []
    project["removal"]["mode"] = "AUTO_BEST"
    project["removal"]["maskPadding"] = PADDING
    project["removal"]["featherRadius"] = 2
    project["removal"]["artifactThreshold"] = 0.25
    project["removal"]["fallbackPolicy"] = "TEMPORAL_INPAINT_BLUR"

    template_frame_number = args.template_frame
    if template_frame_number is None:
        template_frame_number = int(
            max(audit_rows, key=lambda row: float(row.get("modelScore", 0.0)))["frame"]
        )
    video_path = Path(project["source"]["path"])
    template_frame = read_frame(video_path, template_frame_number)
    periodic_x, periodic_y = periodic_position(template_frame_number)
    template_x = periodic_x + offsets_x[template_frame_number]
    template_y = periodic_y + offsets_y[template_frame_number]
    mask = create_clean_mask(template_frame, template_x, template_y)
    x0, y0, x1, y1 = bounds_for_position(
        template_x,
        template_y,
        template_frame.shape[1],
        template_frame.shape[0],
    )
    clean_template = template_frame[y0:y1, x0:x1]
    cv2.imwrite(str(args.output_project_dir / "templates" / "mask.png"), mask)
    cv2.imwrite(
        str(args.output_project_dir / "templates" / "original.png"),
        clean_template,
    )

    output_json = args.output_project_dir / "project.json"
    output_json.write_text(json.dumps(project, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for frame in tracking_frames:
        status = str(frame["status"])
        counts[status] = counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "project": str(output_json),
                "counts": counts,
                "maskWidth": int(mask.shape[1]),
                "maskHeight": int(mask.shape[0]),
                "maskPixels": int(np.count_nonzero(mask)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
