from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


CANONICAL_WIDTH = 245.33333333333334
CANONICAL_HEIGHT = 74.66666666666667
PADDING = 4
DIAGNOSTIC_FRAMES = [
    48,
    60,
    75,
    105,
    120,
    135,
    150,
    180,
    210,
    240,
    270,
    300,
    330,
    360,
    390,
    420,
    450,
    480,
    510,
    540,
    570,
    600,
    630,
    660,
    690,
    720,
    750,
    780,
    810,
    840,
    870,
    900,
    903,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_json", type=Path)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--start-frame", type=int, default=48)
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--alpha-strength", type=float, default=0.5)
    parser.add_argument(
        "--method",
        choices=("deblend", "inpaint-telea", "inpaint-ns"),
        default="deblend",
    )
    return parser.parse_args()


def periodic_position(frame_number: int) -> tuple[float, float]:
    phase = frame_number % 360
    min_x, max_x = 56.0, 736.0
    min_y, max_y = 296.0, 1475.0
    if phase < 120:
        ratio = phase / 120.0
        return (
            min_x + (max_x - min_x) * ratio,
            max_y - (max_y - min_y) * ratio,
        )
    if phase < 180:
        ratio = (phase - 120) / 60.0
        return max_x - (max_x - min_x) * ratio, min_y
    if phase < 300:
        ratio = (phase - 180) / 120.0
        return (
            min_x + (max_x - min_x) * ratio,
            min_y + (max_y - min_y) * ratio,
        )
    ratio = (phase - 300) / 60.0
    return max_x - (max_x - min_x) * ratio, max_y


def read_frame(video_path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_number}")
    return frame


def bounds_for_position(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, int(np.floor(x)) - PADDING)
    y0 = max(0, int(np.floor(y)) - PADDING)
    x1 = min(frame_width, int(np.ceil(x + CANONICAL_WIDTH)) + PADDING + 1)
    y1 = min(frame_height, int(np.ceil(y + CANONICAL_HEIGHT)) + PADDING + 1)
    return x0, y0, x1, y1


def filter_components(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros_like(binary)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area >= 5 and width >= 2 and height >= 2:
            filtered[labels == label] = 255
    return cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def create_alpha_template(template_frame: np.ndarray, x: float, y: float) -> np.ndarray:
    x0, y0, x1, y1 = bounds_for_position(
        x, y, template_frame.shape[1], template_frame.shape[0]
    )
    crop = template_frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 8.0)
    positive = np.maximum(gray - smooth, 0.0)
    binary = filter_components(np.where(positive >= 4.0, 255, 0).astype(np.uint8))
    inpaint_mask = cv2.dilate(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    background = cv2.inpaint(crop, inpaint_mask, 7, cv2.INPAINT_TELEA).astype(
        np.float32
    )
    observed = crop.astype(np.float32)
    ratios: list[np.ndarray] = []
    for channel in range(3):
        denominator = 255.0 - background[:, :, channel]
        ratio = np.divide(
            observed[:, :, channel] - background[:, :, channel],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator >= 20.0,
        )
        ratios.append(ratio)
    alpha = np.median(np.stack(ratios, axis=2), axis=2)
    alpha = np.clip(alpha, 0.0, 0.55)
    alpha *= binary.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    return np.clip(alpha, 0.0, 0.55)


def create_temporal_alpha_template(
    video_path: Path,
    template_frame_number: int,
    audit_rows: list[dict[str, float]],
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the white overlay alpha from real clean pixels in nearby frames.

    The template frame is chosen from a nearly static product shot.  At each
    absolute pixel, neighboring observations are accepted only when their
    moving glyph mask does not cover that pixel.  Their median is therefore a
    much stronger background estimate than spatial inpainting.
    """
    target = read_frame(video_path, template_frame_number)
    target_periodic_x, target_periodic_y = periodic_position(template_frame_number)
    target_x = target_periodic_x + offsets_x[template_frame_number]
    target_y = target_periodic_y + offsets_y[template_frame_number]
    x0, y0, x1, y1 = bounds_for_position(
        target_x, target_y, target.shape[1], target.shape[0]
    )
    observed = target[y0:y1, x0:x1]
    seed_mask = create_binary_template(target, target_x, target_y)
    broad_mask = cv2.dilate(
        seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    samples: list[np.ndarray] = []
    valid_samples: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame_number in range(
            max(0, template_frame_number - 18),
            min(len(audit_rows), template_frame_number + 19),
        ):
            if frame_number == template_frame_number:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            crop = frame[y0:y1, x0:x1]
            if crop.shape != observed.shape:
                continue
            periodic_x, periodic_y = periodic_position(frame_number)
            candidate_x = periodic_x + offsets_x[frame_number]
            candidate_y = periodic_y + offsets_y[frame_number]
            candidate_mask = np.zeros(observed.shape[:2], dtype=np.uint8)
            mask_left = int(np.floor(candidate_x)) - PADDING - x0
            mask_top = int(np.floor(candidate_y)) - PADDING - y0
            mask_right = mask_left + broad_mask.shape[1]
            mask_bottom = mask_top + broad_mask.shape[0]
            dst_x0 = max(0, mask_left)
            dst_y0 = max(0, mask_top)
            dst_x1 = min(candidate_mask.shape[1], mask_right)
            dst_y1 = min(candidate_mask.shape[0], mask_bottom)
            if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                src_x0 = dst_x0 - mask_left
                src_y0 = dst_y0 - mask_top
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                candidate_mask[dst_y0:dst_y1, dst_x0:dst_x1] = broad_mask[
                    src_y0:src_y1, src_x0:src_x1
                ]
            samples.append(crop.astype(np.float32))
            valid_samples.append(candidate_mask == 0)
    finally:
        capture.release()
    if len(samples) < 4:
        raise RuntimeError("Not enough neighboring frames for alpha calibration")

    sample_stack = np.stack(samples, axis=0)
    validity = np.stack(valid_samples, axis=0)
    masked_stack = np.where(validity[:, :, :, None], sample_stack, np.nan)
    background = np.nanmedian(masked_stack, axis=0)
    fallback = cv2.inpaint(observed, broad_mask, 7, cv2.INPAINT_TELEA).astype(
        np.float32
    )
    background = np.where(np.isfinite(background), background, fallback)

    observed_float = observed.astype(np.float32)
    ratios: list[np.ndarray] = []
    for channel in range(3):
        denominator = 255.0 - background[:, :, channel]
        ratio = np.divide(
            observed_float[:, :, channel] - background[:, :, channel],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator >= 16.0,
        )
        ratios.append(ratio)
    alpha = np.median(np.stack(ratios, axis=2), axis=2)
    alpha = np.clip(alpha, 0.0, 0.75)
    alpha *= broad_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.45)
    alpha[alpha < 0.012] = 0.0
    return np.clip(alpha, 0.0, 0.75), seed_mask, background.astype(np.uint8)


def create_binary_template(template_frame: np.ndarray, x: float, y: float) -> np.ndarray:
    x0, y0, x1, y1 = bounds_for_position(
        x, y, template_frame.shape[1], template_frame.shape[0]
    )
    crop = template_frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 8.0)
    positive = np.maximum(gray - smooth, 0.0)
    binary = filter_components(np.where(positive >= 4.0, 255, 0).astype(np.uint8))
    return cv2.dilate(
        binary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )


def aligned_positions(
    audit_rows: list[dict[str, float]], phase_shift: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(audit_rows)
    reliable_frames: list[int] = []
    offsets_x: list[float] = []
    offsets_y: list[float] = []
    for row in audit_rows:
        if row["modelScore"] < 0.30:
            continue
        frame_number = int(row["frame"])
        periodic_x, periodic_y = periodic_position((frame_number + phase_shift) % 360)
        reliable_frames.append(frame_number)
        offsets_x.append(float(row["modelX"]) - periodic_x)
        offsets_y.append(float(row["modelY"]) - periodic_y)
    if not reliable_frames:
        return np.zeros(frame_count), np.zeros(frame_count)
    frame_axis = np.arange(frame_count, dtype=np.float64)
    smooth_x = np.interp(frame_axis, reliable_frames, offsets_x)
    smooth_y = np.interp(frame_axis, reliable_frames, offsets_y)
    smooth_x = cv2.GaussianBlur(smooth_x.reshape(1, -1), (0, 0), 2.0).reshape(-1)
    smooth_y = cv2.GaussianBlur(smooth_y.reshape(1, -1), (0, 0), 2.0).reshape(-1)
    return smooth_x, smooth_y


def restore_frame(
    frame: np.ndarray,
    alpha_template: np.ndarray,
    binary_template: np.ndarray,
    x: float,
    y: float,
    alpha_strength: float,
    method: str,
) -> np.ndarray:
    output = frame.copy()
    x0, y0, x1, y1 = bounds_for_position(x, y, frame.shape[1], frame.shape[0])
    source_region = output[y0:y1, x0:x1]
    if method != "deblend":
        mask = cv2.resize(
            binary_template,
            (source_region.shape[1], source_region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        algorithm = cv2.INPAINT_TELEA if method == "inpaint-telea" else cv2.INPAINT_NS
        output[y0:y1, x0:x1] = cv2.inpaint(source_region, mask, 3.0, algorithm)
        return output
    region = source_region.astype(np.float32)
    alpha = cv2.resize(
        alpha_template,
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    alpha = np.clip(alpha * alpha_strength, 0.0, 0.55)[:, :, None]
    restored = np.divide(
        region - alpha * 255.0,
        1.0 - alpha,
        out=region.copy(),
        where=(1.0 - alpha) > 0.05,
    )
    output[y0:y1, x0:x1] = np.clip(restored, 0.0, 255.0).astype(np.uint8)
    return output


def create_diagnostics(
    video_path: Path,
    audit_rows: list[dict[str, float]],
    alpha_template: np.ndarray,
    binary_template: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    output_dir: Path,
    start_frame: int,
    alpha_strength: float,
    method: str,
) -> None:
    panels: list[np.ndarray] = []
    for frame_number in DIAGNOSTIC_FRAMES:
        frame = read_frame(video_path, frame_number)
        periodic_x, periodic_y = periodic_position(frame_number)
        x = periodic_x + offsets_x[frame_number]
        y = periodic_y + offsets_y[frame_number]
        restored = (
            frame
            if frame_number < start_frame
            else restore_frame(
                frame,
                alpha_template,
                binary_template,
                x,
                y,
                alpha_strength,
                method,
            )
        )
        x0, y0, x1, y1 = bounds_for_position(x, y, frame.shape[1], frame.shape[0])
        margin = 35
        crop_x0 = max(0, x0 - margin)
        crop_y0 = max(0, y0 - margin)
        crop_x1 = min(frame.shape[1], x1 + margin)
        crop_y1 = min(frame.shape[0], y1 + margin)
        before = frame[crop_y0:crop_y1, crop_x0:crop_x1]
        after = restored[crop_y0:crop_y1, crop_x0:crop_x1]
        panel = np.concatenate((before, after), axis=1)
        panel = cv2.resize(panel, (760, 220), interpolation=cv2.INTER_AREA)
        cv2.putText(
            panel,
            f"f{frame_number} before | {method}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = np.zeros((rows * 220, columns * 760, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        sheet[row * 220 : (row + 1) * 220, column * 760 : (column + 1) * 760] = panel
    cv2.imwrite(str(output_dir / "deblend-contact.png"), sheet)
    cv2.imwrite(
        str(output_dir / "alpha-template.png"),
        np.clip(alpha_template * 255.0 / 0.55, 0.0, 255.0).astype(np.uint8),
    )


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


def render_video(
    video_path: Path,
    output_path: Path,
    audit_rows: list[dict[str, float]],
    alpha_template: np.ndarray,
    binary_template: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    start_frame: int,
    crf: int,
    alpha_strength: float,
    method: str,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = probe_frame_rate(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number >= start_frame:
                periodic_x, periodic_y = periodic_position(frame_number)
                frame = restore_frame(
                    frame,
                    alpha_template,
                    binary_template,
                    periodic_x + offsets_x[frame_number],
                    periodic_y + offsets_y[frame_number],
                    alpha_strength,
                    method,
                )
            process.stdin.write(frame.tobytes())
            frame_number += 1
            if frame_number % 100 == 0:
                print(f"rendered {frame_number} frames", flush=True)
    finally:
        capture.release()
        process.stdin.close()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {exit_code}")
    if frame_number != len(audit_rows):
        raise RuntimeError(
            f"Rendered {frame_number} frames but audit contains {len(audit_rows)}"
        )


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    audit_rows = json.loads(args.audit_json.read_text(encoding="utf-8"))
    video_path = Path(project["source"]["path"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets_x, offsets_y = aligned_positions(audit_rows)
    # Choose the strongest evidence frame from the audit.  The pipeline must
    # not depend on a magic frame number because different videos can start
    # at a different phase or contain different scene cuts.
    template_frame_number = max(
        audit_rows,
        key=lambda row: float(row.get("modelScore", 0.0)),
    )["frame"]
    template_frame_number = int(template_frame_number)
    alpha_template, binary_template, calibrated_background = (
        create_temporal_alpha_template(
            video_path,
            template_frame_number,
            audit_rows,
            offsets_x,
            offsets_y,
        )
    )
    cv2.imwrite(str(args.output_dir / "calibrated-background.png"), calibrated_background)
    create_diagnostics(
        video_path,
        audit_rows,
        alpha_template,
        binary_template,
        offsets_x,
        offsets_y,
        args.output_dir,
        args.start_frame,
        args.alpha_strength,
        args.method,
    )
    print(
        json.dumps(
            {
                "alphaWidth": int(alpha_template.shape[1]),
                "alphaHeight": int(alpha_template.shape[0]),
                "alphaMax": float(alpha_template.max()),
                "alphaMeanOnMask": float(
                    alpha_template[alpha_template > 0.01].mean()
                ),
                "reliableAlignmentFrames": sum(
                    row["modelScore"] >= 0.30 for row in audit_rows
                ),
            },
            indent=2,
        )
    )
    if args.render is not None:
        render_video(
            video_path,
            args.render,
            audit_rows,
            alpha_template,
            binary_template,
            offsets_x,
            offsets_y,
            args.start_frame,
            args.crf,
            args.alpha_strength,
            args.method,
        )
        print(args.render)
    else:
        print(args.output_dir / "deblend-contact.png")


if __name__ == "__main__":
    main()
