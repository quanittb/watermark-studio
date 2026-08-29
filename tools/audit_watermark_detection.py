from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ANALYSIS_SCALE = 720.0 / 1920.0
TEST_FRAMES = [
    66,
    111,
    125,
    136,
    140,
    150,
    170,
    235,
    292,
    350,
    480,
    530,
    550,
    600,
    630,
    653,
    654,
    700,
    710,
    800,
    902,
    903,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--template-frame", type=int, default=530)
    parser.add_argument("--all-frames", action="store_true")
    return parser.parse_args()


def read_selected_frames(video_path: Path, frame_numbers: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in frame_numbers:
            frames[frame_index] = frame
        if len(frames) == len(frame_numbers):
            break
        frame_index += 1
    capture.release()
    missing = sorted(frame_numbers.difference(frames))
    if missing:
        raise RuntimeError(f"Missing requested frames: {missing}")
    return frames


def resize_analysis(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (405, 720), interpolation=cv2.INTER_AREA)


def signed_highpass(gray: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(source, (0, 0), 2.2)
    return source - smooth


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    dx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(dx, dy)


def scaled_bbox(bbox: dict[str, float]) -> tuple[int, int, int, int]:
    x = round(bbox["x"] * ANALYSIS_SCALE)
    y = round(bbox["y"] * ANALYSIS_SCALE)
    width = round(bbox["width"] * ANALYSIS_SCALE)
    height = round(bbox["height"] * ANALYSIS_SCALE)
    return x, y, width, height


def best_match(
    feature: np.ndarray,
    template: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    result = cv2.matchTemplate(feature, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    return float(score), location


def two_best_matches(
    feature: np.ndarray,
    template: np.ndarray,
) -> tuple[float, tuple[int, int], float, np.ndarray]:
    result = cv2.matchTemplate(feature, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    suppressed = result.copy()
    x, y = location
    half_width = max(1, template.shape[1] // 2)
    half_height = max(1, template.shape[0] // 2)
    left = max(0, x - half_width)
    top = max(0, y - half_height)
    right = min(suppressed.shape[1], x + half_width + 1)
    bottom = min(suppressed.shape[0], y + half_height + 1)
    suppressed[top:bottom, left:right] = -1.0
    _, second_score, _, _ = cv2.minMaxLoc(suppressed)
    return float(score), location, float(second_score), result


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


def analyze_all_frames(
    video_path: Path,
    highpass_template: np.ndarray,
    output_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    rows: list[dict[str, object]] = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        analysis = resize_analysis(frame)
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        score, location, second_score, response = two_best_matches(
            signed_highpass(gray), highpass_template
        )
        model_x, model_y = periodic_position(frame_number)
        analysis_x = int(round(model_x * ANALYSIS_SCALE))
        analysis_y = int(round(model_y * ANALYSIS_SCALE))
        radius = 4
        left = max(0, analysis_x - radius)
        top = max(0, analysis_y - radius)
        right = min(response.shape[1], analysis_x + radius + 1)
        bottom = min(response.shape[0], analysis_y + radius + 1)
        local = response[top:bottom, left:right]
        _, model_score, _, local_location = cv2.minMaxLoc(local)
        model_location = (left + local_location[0], top + local_location[1])
        rows.append(
            {
                "frame": frame_number,
                "score": score,
                "secondScore": second_score,
                "margin": score - second_score,
                "x": location[0] / ANALYSIS_SCALE,
                "y": location[1] / ANALYSIS_SCALE,
                "width": highpass_template.shape[1] / ANALYSIS_SCALE,
                "height": highpass_template.shape[0] / ANALYSIS_SCALE,
                "modelScore": float(model_score),
                "modelX": model_location[0] / ANALYSIS_SCALE,
                "modelY": model_location[1] / ANALYSIS_SCALE,
                "periodicX": model_x,
                "periodicY": model_y,
            }
        )
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"analyzed {frame_number} frames", flush=True)
    capture.release()
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def create_contact_sheet(images: list[np.ndarray], columns: int = 4) -> np.ndarray:
    thumb_width = 405
    thumb_height = 720
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[
            row * thumb_height : (row + 1) * thumb_height,
            column * thumb_width : (column + 1) * thumb_width,
        ] = image
    return sheet


def write_mask_diagnostics(
    frame: np.ndarray,
    bbox: dict[str, float],
    output_dir: Path,
) -> None:
    padding = 4
    x0 = max(0, int(np.floor(bbox["x"])) - padding)
    y0 = max(0, int(np.floor(bbox["y"])) - padding)
    x1 = min(frame.shape[1], int(np.ceil(bbox["x"] + bbox["width"])) + padding + 1)
    y1 = min(frame.shape[0], int(np.ceil(bbox["y"] + bbox["height"])) + padding + 1)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), 8.0)
    positive = np.maximum(gray - background, 0.0)
    diagnostic = np.clip(positive * 10.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(output_dir / "mask-positive-diagnostic.png"), diagnostic)
    cv2.imwrite(str(output_dir / "mask-template-crop.png"), crop)
    for threshold in (2.5, 4.0, 5.5, 7.0, 9.0):
        binary = np.where(positive >= threshold, 255, 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        filtered = np.zeros_like(binary)
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if area >= 5 and width >= 2 and height >= 2:
                filtered[labels == label] = 255
        filtered = cv2.morphologyEx(
            filtered,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        name = str(threshold).replace(".", "_")
        cv2.imwrite(str(output_dir / f"mask-threshold-{name}.png"), filtered)


def main() -> None:
    args = parse_args()
    project = json.loads(args.project_json.read_text(encoding="utf-8"))
    anchors = {anchor["frame"]: anchor for anchor in project["anchors"]}
    if args.template_frame not in anchors:
        raise RuntimeError("Template frame must be an existing manual anchor")

    requested_frames = set(TEST_FRAMES)
    requested_frames.add(args.template_frame)
    frames = read_selected_frames(Path(project["source"]["path"]), requested_frames)

    template_frame = resize_analysis(frames[args.template_frame])
    template_gray = cv2.cvtColor(template_frame, cv2.COLOR_BGR2GRAY)
    template_bbox = scaled_bbox(anchors[args.template_frame]["bbox"])
    tx, ty, tw, th = template_bbox
    highpass_template = signed_highpass(template_gray)[ty : ty + th, tx : tx + tw]
    gradient_template = gradient_magnitude(template_gray)[ty : ty + th, tx : tx + tw]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_mask_diagnostics(
        frames[args.template_frame],
        anchors[args.template_frame]["bbox"],
        args.output_dir,
    )
    if args.all_frames:
        analyze_all_frames(
            Path(project["source"]["path"]),
            highpass_template,
            args.output_dir / "all-matches.json",
        )
        print(args.output_dir)
        return
    rows: list[dict[str, object]] = []
    overlays: list[np.ndarray] = []
    for frame_number in TEST_FRAMES:
        analysis = resize_analysis(frames[frame_number])
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        highpass_score, highpass_location = best_match(
            signed_highpass(gray), highpass_template
        )
        gradient_score, gradient_location = best_match(
            gradient_magnitude(gray), gradient_template
        )
        combined_location = (
            highpass_location
            if highpass_score >= gradient_score
            else gradient_location
        )
        chosen_score = max(highpass_score, gradient_score)
        overlay = analysis.copy()
        x, y = combined_location
        cv2.rectangle(overlay, (x, y), (x + tw, y + th), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"f{frame_number} s={chosen_score:.3f}",
            (8, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        overlays.append(overlay)
        rows.append(
            {
                "frame": frame_number,
                "highpassScore": highpass_score,
                "highpass": {
                    "x": highpass_location[0] / ANALYSIS_SCALE,
                    "y": highpass_location[1] / ANALYSIS_SCALE,
                },
                "gradientScore": gradient_score,
                "gradient": {
                    "x": gradient_location[0] / ANALYSIS_SCALE,
                    "y": gradient_location[1] / ANALYSIS_SCALE,
                },
                "chosen": {
                    "x": x / ANALYSIS_SCALE,
                    "y": y / ANALYSIS_SCALE,
                    "width": tw / ANALYSIS_SCALE,
                    "height": th / ANALYSIS_SCALE,
                },
            }
        )

    (args.output_dir / "matches.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    cv2.imwrite(
        str(args.output_dir / "matches-contact.png"),
        create_contact_sheet(overlays),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
