from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from render_periodic_dewatermark import bounds_for_position


MIN_QA_FRAMES = 35
# Embedded acceptance baseline measured from the approved golden output. The
# runtime never reads or depends on the golden MP4; regression updates these
# constants only after explicit visual acceptance.
GOLDEN_MIN_OUTSIDE_SSIM = 0.9688  # golden 0.970895 - allowed 0.002
GOLDEN_MAX_RESIDUAL = 0.7905      # golden 0.752778 + 5%
GOLDEN_MAX_ENERGY_RATIO = 0.6835  # golden 0.650924 + 5%


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QualityReportV3 for Best-quality output")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("contact_sheet", type=Path)
    return parser.parse_args()


def media_metadata(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    result = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    return json.loads(result.stdout) if result.returncode == 0 and result.stdout else {}


def read_frames(path: Path, requested: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    try:
        for index in range(max(requested) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index in requested:
                frames[index] = frame
    finally:
        capture.release()
    return frames


def highpass(gray: np.ndarray) -> np.ndarray:
    source = gray.astype(np.float32)
    return source - cv2.GaussianBlur(source, (0, 0), 2.2)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32).reshape(-1)
    right = right.astype(np.float32).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator < 1e-6 else float(np.dot(left, right) / denominator)


def masked_ssim(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    if left.size < 16:
        return 1.0
    mean_left, mean_right = float(left.mean()), float(right.mean())
    var_left, var_right = float(left.var()), float(right.var())
    covariance = float(np.mean((left - mean_left) * (right - mean_right)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / ((mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)))


def selected_frames(profile: dict) -> list[int]:
    first = int(profile.get("firstWatermarkFrame", 48))
    frame_count = int(profile["frameCount"])
    evenly = np.linspace(first, frame_count - 1, MIN_QA_FRAMES, dtype=np.int64).tolist()
    difficult = [int(value) for value in profile.get("difficultFrames", [])]
    phase_points = [frame for frame in range(first, frame_count) if frame % 60 in (0, 1, 59)]
    selected = sorted(set(evenly + difficult + phase_points))
    # Keep reports bounded while retaining all explicit difficult frames.
    if len(selected) > 55:
        required = set(difficult)
        remaining = [frame for frame in selected if frame not in required]
        keep = np.linspace(0, len(remaining) - 1, max(0, 55 - len(required)), dtype=np.int64)
        selected = sorted(required | {remaining[int(index)] for index in keep})
    return selected


def stream_summary(metadata: dict) -> tuple[dict, bool]:
    streams = metadata.get("streams", []) if isinstance(metadata, dict) else []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    return video, any(stream.get("codec_type") == "audio" for stream in streams)


def metadata_gate(source_meta: dict, output_meta: dict) -> tuple[bool, dict]:
    source_video, source_audio = stream_summary(source_meta)
    output_video, output_audio = stream_summary(output_meta)
    def number(value: object) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0
    frame_match = not source_video.get("nb_frames") or not output_video.get("nb_frames") or int(source_video["nb_frames"]) == int(output_video["nb_frames"])
    duration_delta = abs(number(source_meta.get("format", {}).get("duration")) - number(output_meta.get("format", {}).get("duration")))
    passed = (
        int(source_video.get("width", 0)) == int(output_video.get("width", 0))
        and int(source_video.get("height", 0)) == int(output_video.get("height", 0))
        and source_video.get("avg_frame_rate") == output_video.get("avg_frame_rate")
        and frame_match and duration_delta <= 0.08 and source_audio == output_audio
    )
    return passed, {"sourceVideo": source_video, "outputVideo": output_video, "sourceHasAudio": source_audio, "outputHasAudio": output_audio, "frameCountMatches": frame_match, "durationDeltaSeconds": duration_delta, "passed": passed}


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
    if profile.get("version") != 3 or profile.get("status") != "READY":
        raise RuntimeError("QualityReportV3 requires a READY CalibrationProfileV3")
    project_dir = args.profile.parent.parent
    mask_path = project_dir / profile["inferenceMaskPath"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Calibration inference mask is missing")
    mask = np.where(mask >= 64, 255, 0).astype(np.uint8)
    requested = set(selected_frames(profile))
    source_frames = read_frames(args.source, requested)
    output_frames = read_frames(args.output, requested)
    if source_frames.keys() != output_frames.keys() or len(source_frames) < MIN_QA_FRAMES:
        raise RuntimeError("QA could not decode the minimum 35 selected frames")

    frame_data = profile["frameData"]
    rows: list[dict] = []
    panels: list[np.ndarray] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    for frame_number in sorted(requested):
        source = source_frames[frame_number]
        output = output_frames[frame_number]
        bbox = frame_data[frame_number]["bbox"]
        x0, y0, x1, y1 = bounds_for_position(float(bbox["x"]), float(bbox["y"]), source.shape[1], source.shape[0])
        source_crop = source[y0:y1, x0:x1]
        output_crop = output[y0:y1, x0:x1]
        local_mask = cv2.resize(mask, (source_crop.shape[1], source_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        expanded = cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        outside = expanded == 0
        ring = (cv2.dilate(local_mask, np.ones((5, 5), np.uint8)) > 0) & (cv2.erode(local_mask, np.ones((5, 5), np.uint8)) == 0)
        source_gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output_crop, cv2.COLOR_BGR2GRAY)
        inside = local_mask > 0
        residual = correlation(highpass(output_gray)[inside], highpass(source_gray)[inside])
        source_energy = float(np.mean(np.abs(highpass(source_gray)[inside])))
        output_energy = float(np.mean(np.abs(highpass(output_gray)[inside])))
        outside_ssim = masked_ssim(source_crop, output_crop, outside)
        seam_score = float(np.mean(np.abs(cv2.Laplacian(output_gray, cv2.CV_32F))[ring]) / 255.0) if np.any(ring) else 0.0
        rectangle_score = float(np.mean(cv2.Canny(output_gray, 80, 160)[ring]) / 255.0) if np.any(ring) else 0.0
        flicker = 0.0
        if previous is not None:
            prev_source, prev_output = previous
            if prev_source.shape != source_gray.shape:
                prev_source = cv2.resize(prev_source, (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_AREA)
                prev_output = cv2.resize(prev_output, (output_gray.shape[1], output_gray.shape[0]), interpolation=cv2.INTER_AREA)
            source_motion = float(np.mean(np.abs(source_gray.astype(np.float32) - prev_source.astype(np.float32))))
            output_motion = float(np.mean(np.abs(output_gray.astype(np.float32) - prev_output.astype(np.float32))))
            flicker = max(0.0, output_motion - source_motion) / 255.0
        previous = (source_gray, output_gray)
        row = {"frame": frame_number, "residualCorrelation": residual, "sourceGlyphEnergy": source_energy, "outputGlyphEnergy": output_energy, "glyphEnergyRatio": output_energy / max(source_energy, 1e-6), "outsideMaskSsim": outside_ssim, "seamScore": seam_score, "rectangularPatchScore": rectangle_score, "temporalFlicker": flicker, "confidence": float(frame_data[frame_number].get("confidence", 0.0)), "occluded": bool(frame_data[frame_number].get("occlusion", False))}
        rows.append(row)

        difference = cv2.convertScaleAbs(output_crop.astype(np.float32) - source_crop.astype(np.float32), alpha=4.0)
        mask_panel = cv2.cvtColor(local_mask, cv2.COLOR_GRAY2BGR)
        labelled = []
        for label, panel in (("Source", source_crop), ("Output", output_crop), ("Difference x4", difference), ("Mask", mask_panel)):
            panel = cv2.resize(panel, (255, 85), interpolation=cv2.INTER_NEAREST if label == "Mask" else cv2.INTER_AREA)
            cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            labelled.append(panel)
        strip = np.concatenate(labelled, axis=1)
        cv2.putText(strip, f"f{frame_number}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        panels.append(strip)

    metadata_ok, metadata = metadata_gate(media_metadata(args.source), media_metadata(args.output))
    visible = [row for row in rows if not row["occluded"] and row["sourceGlyphEnergy"] >= 2.0]
    metrics = {
        "maxResidualCorrelation": max((row["residualCorrelation"] for row in visible), default=0.0),
        "maxGlyphEnergyRatio": max((row["glyphEnergyRatio"] for row in visible), default=0.0),
        "minOutsideMaskSsim": min((row["outsideMaskSsim"] for row in rows), default=1.0),
        "maxSeamScore": max((row["seamScore"] for row in rows), default=0.0),
        "maxRectangularPatchScore": max((row["rectangularPatchScore"] for row in rows), default=0.0),
        "maxTemporalFlicker": max((row["temporalFlicker"] for row in rows), default=0.0),
    }
    passed = metadata_ok and metrics["maxResidualCorrelation"] <= GOLDEN_MAX_RESIDUAL and metrics["maxGlyphEnergyRatio"] <= GOLDEN_MAX_ENERGY_RATIO and metrics["minOutsideMaskSsim"] >= GOLDEN_MIN_OUTSIDE_SSIM and metrics["maxTemporalFlicker"] <= 0.12
    difficult = sorted(rows, key=lambda row: (row["residualCorrelation"] + row["glyphEnergyRatio"] + row["temporalFlicker"]), reverse=True)[:8]
    report = {"version": 3, "status": "passed" if passed else "needs_review", "gate": "quality_report_v3", "sampledFrames": sorted(requested), "difficultFrames": [row["frame"] for row in difficult], "metrics": metrics, "goldenThresholds": {"maxResidualCorrelation": GOLDEN_MAX_RESIDUAL, "maxGlyphEnergyRatio": GOLDEN_MAX_ENERGY_RATIO, "minOutsideMaskSsim": GOLDEN_MIN_OUTSIDE_SSIM, "maxTemporalFlicker": 0.12}, "metadata": metadata, "ocr": {"engine": "canonical-glyph-correlation", "textDetected": metrics["maxResidualCorrelation"] > GOLDEN_MAX_RESIDUAL}, "contactSheet": str(args.contact_sheet), "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = 1
    sheet = np.zeros((len(panels) * 85, columns * 1020, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        sheet[index * 85:(index + 1) * 85, :1020] = panel
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.contact_sheet), sheet)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
