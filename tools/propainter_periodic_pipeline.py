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
    prepare.add_argument(
        "--anchor-mode",
        action="store_true",
        help="Use the saved manual anchor as the periodic-path calibration instead of an audit JSON.",
    )
    prepare.add_argument(
        "--profile",
        type=Path,
        help="Full-video calibration profile. Final renders must use this instead of anchor-mode.",
    )
    prepare.add_argument("--start-frame", type=int, default=0)
    prepare.add_argument("--end-frame", type=int, default=903)
    prepare.add_argument("--full-frame", action="store_true")
    prepare.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        help="Expand the validated glyph mask for a bounded QA retry (source pixels).",
    )

    composite = subparsers.add_parser("composite")
    composite.add_argument("project_json", type=Path)
    composite.add_argument("workspace", type=Path)
    composite.add_argument("inpaint_frames", type=Path)
    composite.add_argument("output", type=Path)
    composite.add_argument("--crf", type=int, default=14)
    composite.add_argument("--replacement-kind", choices=["text", "image"])
    composite.add_argument("--replacement-text", default="")
    composite.add_argument("--replacement-image", type=Path)
    composite.add_argument("--replacement-placement", choices=["follow", "fixed"], default="follow")
    composite.add_argument("--replacement-fixed-x", type=float, default=0.0)
    composite.add_argument("--replacement-fixed-y", type=float, default=0.0)
    composite.add_argument("--replacement-scale", type=float, default=1.0)
    composite.add_argument("--replacement-opacity", type=float, default=1.0)
    return parser.parse_args()


def crop_origin(x: float, y: float, frame_width: int, frame_height: int) -> tuple[int, int]:
    x0, y0, x1, y1 = bounds_for_position(x, y, frame_width, frame_height)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    left = int(np.clip(center_x - CROP_WIDTH // 2, 0, frame_width - CROP_WIDTH))
    top = int(np.clip(center_y - CROP_HEIGHT // 2, 0, frame_height - CROP_HEIGHT))
    return left, top


def profile_bounds(bbox: dict[str, float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Use the profile's calibrated box dimensions, including V6 padding."""
    x0 = max(0, int(round(float(bbox["x"]))))
    y0 = max(0, int(round(float(bbox["y"]))))
    x1 = min(frame_width, int(round(float(bbox["x"]) + float(bbox["width"]))))
    y1 = min(frame_height, int(round(float(bbox["y"]) + float(bbox["height"]))))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Calibration bbox is outside the source frame")
    return x0, y0, x1, y1


def prepare(args: argparse.Namespace) -> None:
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    video_path = Path(project["source"]["path"])
    project_dir = args.project_json.parent
    profile = None
    if args.profile is not None:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if (
            profile.get("version") != 6
            or profile.get("status") != "READY"
            or profile.get("preset") != "LEARNA_AI_ADAPTIVE"
            or profile.get("qualityGate", {}).get("status") != "PASSED"
        ):
            raise RuntimeError("Best-quality preparation requires a READY CalibrationProfileV6")
        if profile.get("trajectoryGate", {}).get("status") != "PASSED":
            raise RuntimeError("CalibrationProfileV6 trajectory quality gate did not pass")
        if int(profile.get("frameCount", 0)) != int(project["video"]["frameCount"]):
            raise RuntimeError("Calibration profile frame count does not match the source video")
    mask_reference = profile.get("inferenceMaskPath") if profile else project["watermark"]["templates"]["mask"]
    mask_path = project_dir / mask_reference
    glyph_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if glyph_mask is None:
        raise RuntimeError(f"Unable to read glyph mask: {mask_path}")
    glyph_mask = np.where(glyph_mask >= 64, 255, 0).astype(np.uint8)
    if args.mask_dilate:
        if not 0 < args.mask_dilate <= 2:
            raise RuntimeError("Best-quality mask retry may expand the mask by at most 2 px")
        kernel_size = 2 * int(args.mask_dilate) + 1
        glyph_mask = cv2.dilate(
            glyph_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )

    if args.workspace.exists():
        shutil.rmtree(args.workspace)
    frames_dir = args.workspace / "frames"
    masks_dir = args.workspace / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    if profile is not None:
        frame_data = profile.get("frameData", [])
        if len(frame_data) != int(project["video"]["frameCount"]):
            raise RuntimeError("Calibration profile frame data does not cover the full source video")
    elif args.anchor_mode:
        anchor = project.get("watermark", {}).get("anchor")
        if not anchor:
            raise RuntimeError("Best-quality preparation requires a saved watermark anchor")
        anchor_frame = int(anchor["frame"])
        anchor_bbox = anchor["bbox"]
        periodic_x, periodic_y = periodic_position(anchor_frame)
        offset_x = float(anchor_bbox["x"]) - periodic_x
        offset_y = float(anchor_bbox["y"]) - periodic_y
        # A broad range here would make a different watermark look like a
        # supported Learna AI trajectory. Refuse it rather than masking an
        # unrelated screen region throughout the full video.
        if abs(offset_x) > 80 or abs(offset_y) > 80:
            raise RuntimeError(
                "The saved anchor does not match the supported periodic watermark path"
            )
        offset_count = args.end_frame + 1
        offsets_x = np.full(offset_count, offset_x, dtype=np.float64)
        offsets_y = np.full(offset_count, offset_y, dtype=np.float64)
    else:
        audit_rows = json.loads(args.audit_json.read_text(encoding="utf-8"))
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
            if profile is not None:
                frame_profile = frame_data[frame_number]
                bbox = frame_profile["bbox"]
                x = float(bbox["x"])
                y = float(bbox["y"])
                # ``confidence`` describes detector evidence only.  A low
                # score is common during motion blur or textured scenes; the
                # trajectory/profile still gives us a valid position.  Use
                # the explicit maskRequired bit so those frames are not
                # silently copied with the watermark untouched.
                visible = bool(
                    frame_profile.get(
                        "maskRequired",
                        bool(frame_profile.get("visibility", False))
                        and not bool(frame_profile.get("occlusion", False)),
                    )
                )
            else:
                periodic_x, periodic_y = periodic_position(frame_number)
                x = periodic_x + offsets_x[frame_number]
                y = periodic_y + offsets_y[frame_number]
                visible = True
            if args.full_frame:
                left, top = 0, 0
                crop_width, crop_height = frame_width, frame_height
            else:
                left, top = crop_origin(x, y, frame_width, frame_height)
                crop_width, crop_height = CROP_WIDTH, CROP_HEIGHT
            crop = frame[top : top + crop_height, left : left + crop_width]
            local_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
            if profile is not None:
                box_x0, box_y0, box_x1, box_y1 = profile_bounds(bbox, frame_width, frame_height)
            else:
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
            if visible:
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


def replacement_overlay(args: argparse.Namespace, frame: np.ndarray, row: dict[str, int] | None, reference: dict[str, int]) -> None:
    if not args.replacement_kind:
        return
    if args.replacement_placement == "follow":
        if row is None:
            return
        origin_x = int(row["left"] + row["boxX"])
        origin_y = int(row["top"] + row["boxY"])
        base_width = int(row["boxWidth"])
    else:
        origin_x = int(args.replacement_fixed_x)
        origin_y = int(args.replacement_fixed_y)
        base_width = int(reference["boxWidth"])
    target_width = max(1, int(base_width * args.replacement_scale))
    opacity = float(np.clip(args.replacement_opacity, 0.0, 1.0))

    if args.replacement_kind == "image":
        if args.replacement_image is None:
            raise RuntimeError("Replacement image path is required")
        image = cv2.imread(str(args.replacement_image), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Unable to read replacement image: {args.replacement_image}")
        height = max(1, int(image.shape[0] * target_width / image.shape[1]))
        image = cv2.resize(image, (target_width, height), interpolation=cv2.INTER_LANCZOS4)
        if image.ndim == 2:
            pixels, alpha = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), np.ones(image.shape, dtype=np.float32)
        elif image.shape[2] == 4:
            pixels, alpha = image[:, :, :3], image[:, :, 3].astype(np.float32) / 255.0
        else:
            pixels, alpha = image[:, :, :3], np.ones(image.shape[:2], dtype=np.float32)
    else:
        font_scale = max(0.25, target_width / 160.0)
        thickness = max(1, int(round(font_scale * 2)))
        (width, height), baseline = cv2.getTextSize(args.replacement_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pixels = np.zeros((height + baseline + 8, width + 8, 3), dtype=np.uint8)
        cv2.putText(pixels, args.replacement_text, (4, height + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        alpha = np.any(pixels > 0, axis=2).astype(np.float32)

    height, width = pixels.shape[:2]
    x0, y0 = max(0, origin_x), max(0, origin_y)
    x1, y1 = min(frame.shape[1], origin_x + width), min(frame.shape[0], origin_y + height)
    if x1 <= x0 or y1 <= y0:
        return
    source_x, source_y = x0 - origin_x, y0 - origin_y
    overlay = pixels[source_y : source_y + (y1 - y0), source_x : source_x + (x1 - x0)]
    local_alpha = alpha[source_y : source_y + (y1 - y0), source_x : source_x + (x1 - x0)]
    local_alpha = (local_alpha * opacity)[:, :, None]
    base = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(overlay.astype(np.float32) * local_alpha + base * (1.0 - local_alpha), 0, 255).astype(np.uint8)


def composite(args: argparse.Namespace) -> None:
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    video_path = Path(project["source"]["path"])
    manifest = json.loads((args.workspace / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise RuntimeError("Best-quality workspace manifest is empty")
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
                alpha = cv2.GaussianBlur(mask, (0, 0), 1.75).astype(np.float32)
                alpha = alpha[:, :, None] / 255.0
                left, top = int(row["left"]), int(row["top"])
                source_crop = frame[top : top + crop_height, left : left + crop_width]
                frame[top : top + crop_height, left : left + crop_width] = np.clip(
                    restored.astype(np.float32) * alpha
                    + source_crop.astype(np.float32) * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)
            replacement_overlay(args, frame, row, manifest[0])
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
