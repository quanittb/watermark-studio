from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from render_periodic_dewatermark import aligned_positions, bounds_for_position, periodic_position


CROP_WIDTH = 320
CROP_HEIGHT = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("project_json", type=Path)
    prepare.add_argument("audit_json", type=Path)
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("--start-frame", type=int, default=48)
    prepare.add_argument("--end-frame", type=int, default=903)
    prepare.add_argument("--full-frame", action="store_true")

    composite = subparsers.add_parser("composite")
    composite.add_argument("project_json", type=Path)
    composite.add_argument("workspace", type=Path)
    composite.add_argument("inpaint_frames", type=Path)
    composite.add_argument("output", type=Path)
    composite.add_argument("--crf", type=int, default=14)
    return parser.parse_args()


def crop_origin(x: float, y: float, frame_width: int, frame_height: int) -> tuple[int, int]:
    x0, y0, x1, y1 = bounds_for_position(x, y, frame_width, frame_height)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = int(np.clip(center_x - CROP_WIDTH // 2, 0, frame_width - CROP_WIDTH))
    top = int(np.clip(center_y - CROP_HEIGHT // 2, 0, frame_height - CROP_HEIGHT))
    return left, top


def prepare(args: argparse.Namespace) -> None:
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    audit_rows = json.loads(args.audit_json.read_text(encoding="utf-8"))
    video_path = Path(project["source"]["path"])
    project_dir = args.project_json.parent
    mask_path = project_dir / project["watermark"]["templates"]["mask"]
    glyph_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if glyph_mask is None:
        raise RuntimeError(f"Unable to read glyph mask: {mask_path}")
    glyph_mask = np.where(glyph_mask >= 64, 255, 0).astype(np.uint8)

    if args.workspace.exists():
        shutil.rmtree(args.workspace)
    frames_dir = args.workspace / "frames"
    masks_dir = args.workspace / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    offsets_x, offsets_y = aligned_positions(audit_rows)
    capture = cv2.VideoCapture(str(video_path))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    manifest: list[dict[str, int]] = []
    try:
        for frame_number in range(args.start_frame, args.end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to decode source frame {frame_number}")
            periodic_x, periodic_y = periodic_position(frame_number)
            x = periodic_x + offsets_x[frame_number]
            y = periodic_y + offsets_y[frame_number]
            if args.full_frame:
                left, top = 0, 0
                crop_width, crop_height = frame_width, frame_height
            else:
                left, top = crop_origin(x, y, frame_width, frame_height)
                crop_width, crop_height = CROP_WIDTH, CROP_HEIGHT
            crop = frame[top : top + crop_height, left : left + crop_width]
            local_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
            box_x0, box_y0, box_x1, box_y1 = bounds_for_position(
                x, y, frame_width, frame_height
            )
            resized_mask = cv2.resize(
                glyph_mask,
                (box_x1 - box_x0, box_y1 - box_y0),
                interpolation=cv2.INTER_NEAREST,
            )
            local_x = box_x0 - left
            local_y = box_y0 - top
            local_mask[
                local_y : local_y + resized_mask.shape[0],
                local_x : local_x + resized_mask.shape[1],
            ] = resized_mask
            name = f"{frame_number - args.start_frame:04d}.png"
            cv2.imwrite(str(frames_dir / name), crop)
            cv2.imwrite(str(masks_dir / name), local_mask)
            manifest.append(
                {
                    "frame": frame_number,
                    "left": left,
                    "top": top,
                    "boxX": local_x,
                    "boxY": local_y,
                    "boxWidth": resized_mask.shape[1],
                    "boxHeight": resized_mask.shape[0],
                    "cropWidth": crop_width,
                    "cropHeight": crop_height,
                }
            )
    finally:
        capture.release()
    (args.workspace / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"frames": len(manifest), "workspace": str(args.workspace)}))


def probe_frame_rate(video_path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def composite(args: argparse.Namespace) -> None:
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    video_path = Path(project["source"]["path"])
    manifest = json.loads((args.workspace / "manifest.json").read_text(encoding="utf-8"))
    by_frame = {int(row["frame"]): row for row in manifest}
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = probe_frame_rate(video_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        frame_rate,
        "-i",
        "-",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = by_frame.get(frame_number)
            if row is not None:
                index = frame_number - int(manifest[0]["frame"])
                restored = cv2.imread(str(args.inpaint_frames / f"{index:04d}.png"))
                mask = cv2.imread(
                    str(args.workspace / "masks" / f"{index:04d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                if restored is None or mask is None:
                    raise RuntimeError(f"Missing restored crop for frame {frame_number}")
                crop_width = int(row.get("cropWidth", CROP_WIDTH))
                crop_height = int(row.get("cropHeight", CROP_HEIGHT))
                if restored.shape[1] != crop_width or restored.shape[0] != crop_height:
                    restored = cv2.resize(
                        restored, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC
                    )
                if mask.shape[1] != crop_width or mask.shape[0] != crop_height:
                    mask = cv2.resize(
                        mask, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST
                    )
                blend_mask = cv2.dilate(
                    mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                )
                alpha = cv2.GaussianBlur(blend_mask, (0, 0), 1.5).astype(np.float32)
                alpha = alpha[:, :, None] / 255.0
                left, top = int(row["left"]), int(row["top"])
                source_crop = frame[top : top + crop_height, left : left + crop_width]
                frame[top : top + crop_height, left : left + crop_width] = np.clip(
                    restored.astype(np.float32) * alpha
                    + source_crop.astype(np.float32) * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)
            process.stdin.write(frame.tobytes())
            frame_number += 1
    finally:
        capture.release()
        process.stdin.close()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {exit_code}")
    print(args.output)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        composite(args)


if __name__ == "__main__":
    main()
