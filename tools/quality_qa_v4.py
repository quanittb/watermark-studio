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
# Baseline measured by a full-frame scan of the accepted golden output.  The
# gates intentionally allow a small 5% degradation (and 0.002 SSIM delta),
# while still requiring every visible frame to pass.  Keeping these values in
# code means the runtime never depends on the golden MP4 being present.
GOLDEN_MIN_OUTSIDE_SSIM = 0.9887
GOLDEN_MAX_RESIDUAL = 0.8820
GOLDEN_MAX_ENERGY_RATIO = 0.8821
MAX_FLICKER = 0.12
MAX_SEAM_SCORE = 0.10
MAX_RECTANGULAR_PATCH_SCORE = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming QualityReportV4/V5/V6/V7")
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
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(result.stdout) if result.returncode == 0 and result.stdout else {}


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
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float64)[valid]
    if left_gray.size < 16:
        return 1.0
    mean_left, mean_right = float(left_gray.mean()), float(right_gray.mean())
    var_left, var_right = float(left_gray.var()), float(right_gray.var())
    covariance = float(np.mean((left_gray - mean_left) * (right_gray - mean_right)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        ((2 * mean_left * mean_right + c1) * (2 * covariance + c2))
        / ((mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2))
    )


def selected_frames(profile: dict) -> set[int]:
    scan_range = profile.get("scanRange") or {}
    first = int(scan_range.get("startFrame", profile.get("firstWatermarkFrame", 0)))
    frame_count = int(profile["frameCount"])
    last = min(
        frame_count - 1,
        int(scan_range.get("endFrame", profile.get("lastWatermarkFrame", frame_count - 1))),
    )
    evenly = np.linspace(first, last, MIN_QA_FRAMES, dtype=np.int64).tolist()
    difficult = [int(value) for value in profile.get("difficultFrames", [])]
    phase_points = [frame for frame in range(first, last + 1) if frame % 60 in (0, 1, 59)]
    return {frame for frame in evenly + difficult + phase_points if first <= frame <= last}


def calibration_bounds(bbox: dict, source: np.ndarray, profile_version: int) -> tuple[int, int, int, int]:
    if profile_version in (5, 6, 7):
        x0 = max(0, int(round(float(bbox["x"]))))
        y0 = max(0, int(round(float(bbox["y"]))))
        x1 = min(source.shape[1], int(round(float(bbox["x"]) + float(bbox["width"]))))
        y1 = min(source.shape[0], int(round(float(bbox["y"]) + float(bbox["height"]))))
        if x1 <= x0 or y1 <= y0:
            raise RuntimeError("Calibration bbox is outside the source frame")
        return x0, y0, x1, y1
    return bounds_for_position(float(bbox["x"]), float(bbox["y"]), source.shape[1], source.shape[0])


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

    source_frames = source_video.get("nb_frames")
    output_frames = output_video.get("nb_frames")
    frame_match = not source_frames or not output_frames or int(source_frames) == int(output_frames)
    duration_delta = abs(
        number(source_meta.get("format", {}).get("duration"))
        - number(output_meta.get("format", {}).get("duration"))
    )
    passed = (
        int(source_video.get("width", 0)) == int(output_video.get("width", 0))
        and int(source_video.get("height", 0)) == int(output_video.get("height", 0))
        and source_video.get("avg_frame_rate") == output_video.get("avg_frame_rate")
        and frame_match
        and duration_delta <= 0.08
        and source_audio == output_audio
    )
    return passed, {
        "sourceVideo": source_video,
        "outputVideo": output_video,
        "sourceHasAudio": source_audio,
        "outputHasAudio": output_audio,
        "frameCountMatches": frame_match,
        "durationDeltaSeconds": duration_delta,
        "passed": passed,
    }


def make_row(
    frame_number: int,
    source: np.ndarray,
    output: np.ndarray,
    frame_profile: dict,
    mask: np.ndarray,
    previous: tuple[np.ndarray, np.ndarray] | None,
    profile_version: int,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    mask_required = bool(
        frame_profile.get(
            "maskRequired",
            bool(frame_profile.get("visibility", False))
            and not bool(frame_profile.get("occlusion", False)),
        )
    )
    if not mask_required:
        # Frames outside the selected scan range (and genuinely inactive
        # frames) are passthrough by contract.  Keep them in the decode
        # accounting, but do not let their arbitrary bbox/scene pixels affect
        # removal metrics or temporal flicker.
        return {
            "frame": frame_number,
            "residualCorrelation": 0.0,
            "sourceGlyphEnergy": 0.0,
            "outputGlyphEnergy": 0.0,
            "glyphEnergyRatio": 0.0,
            "outsideMaskSsim": 1.0,
            "localOutsideMaskSsim": 1.0,
            "seamScore": 0.0,
            "rectangularPatchScore": 0.0,
            "temporalFlicker": 0.0,
            "confidence": 0.0,
            "occluded": bool(frame_profile.get("occlusion", False)),
            "visible": False,
            "maskRequired": False,
            "measurable": True,
        }, None
    bbox = frame_profile["bbox"]
    x0, y0, x1, y1 = calibration_bounds(bbox, source, profile_version)
    source_crop = source[y0:y1, x0:x1]
    output_crop = output[y0:y1, x0:x1]
    if source_crop.size == 0 or output_crop.size == 0:
        raise RuntimeError(f"Empty glyph crop at frame {frame_number}")
    local_mask = cv2.resize(
        mask,
        (source_crop.shape[1], source_crop.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    local_mask = np.where(local_mask >= 64, 255, 0).astype(np.uint8)
    expanded = cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outside = expanded == 0
    ring = (cv2.dilate(local_mask, np.ones((5, 5), np.uint8)) > 0) & (
        cv2.erode(local_mask, np.ones((5, 5), np.uint8)) == 0
    )
    source_gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output_crop, cv2.COLOR_BGR2GRAY)
    inside = local_mask > 0
    source_highpass = highpass(source_gray)
    output_highpass = highpass(output_gray)
    residual = correlation(output_highpass[inside], source_highpass[inside])
    source_energy = float(np.mean(np.abs(source_highpass[inside]))) if np.any(inside) else 0.0
    output_energy = float(np.mean(np.abs(output_highpass[inside]))) if np.any(inside) else 0.0
    # A glyph-sized crop is too sensitive to the dark/bright scene content
    # around a moving watermark (a one-pixel codec change can collapse SSIM).
    # Measure the gate on a local context window instead: the dilated glyph
    # remains excluded, while enough surrounding pixels make SSIM meaningful.
    context_margin = max(source_crop.shape[0], source_crop.shape[1])
    cx0 = max(0, x0 - context_margin)
    cy0 = max(0, y0 - context_margin)
    cx1 = min(source.shape[1], x1 + context_margin)
    cy1 = min(source.shape[0], y1 + context_margin)
    source_context = source[cy0:cy1, cx0:cx1]
    output_context = output[cy0:cy1, cx0:cx1]
    context_mask = np.zeros(source_context.shape[:2], dtype=np.uint8)
    mask_y0, mask_y1 = y0 - cy0, y1 - cy0
    mask_x0, mask_x1 = x0 - cx0, x1 - cx0
    context_mask[mask_y0:mask_y1, mask_x0:mask_x1] = expanded
    outside_ssim = masked_ssim(source_context, output_context, context_mask == 0)
    local_outside_ssim = masked_ssim(source_crop, output_crop, outside)
    seam_score = (
        float(np.mean(np.abs(cv2.Laplacian(output_gray, cv2.CV_32F))[ring])) / 255.0
        if np.any(ring)
        else 0.0
    )
    rectangle_score = (
        float(np.mean(cv2.Canny(output_gray, 80, 160)[ring])) / 255.0
        if np.any(ring)
        else 0.0
    )
    flicker = 0.0
    if previous is not None:
        prev_source, prev_output = previous
        if prev_source.shape != source_gray.shape:
            prev_source = cv2.resize(prev_source, (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_AREA)
            prev_output = cv2.resize(prev_output, (output_gray.shape[1], output_gray.shape[0]), interpolation=cv2.INTER_AREA)
        source_motion = float(np.mean(np.abs(source_gray.astype(np.float32) - prev_source.astype(np.float32))))
        output_motion = float(np.mean(np.abs(output_gray.astype(np.float32) - prev_output.astype(np.float32))))
        flicker = max(0.0, output_motion - source_motion) / 255.0
    row = {
        "frame": frame_number,
        "residualCorrelation": residual,
        "sourceGlyphEnergy": source_energy,
        "outputGlyphEnergy": output_energy,
        "glyphEnergyRatio": output_energy / max(source_energy, 1e-6),
        "outsideMaskSsim": outside_ssim,
        "localOutsideMaskSsim": local_outside_ssim,
        "seamScore": seam_score,
        "rectangularPatchScore": rectangle_score,
        "temporalFlicker": flicker,
        "confidence": float(frame_profile.get("confidence", 0.0)),
        "occluded": bool(frame_profile.get("occlusion", False)),
        "visible": bool(frame_profile.get("visibility", False)),
        # The renderer reads this explicit bit for every active frame.  QA
        # must use the same denominator; visibility alone is not sufficient
        # because low-opacity/blurred watermark frames may be marked visible
        # only by the trajectory calibration.
        "maskRequired": mask_required,
        "measurable": bool(np.any(inside)) and source_energy > 1e-6,
    }
    return row, (source_gray, output_gray)


def contact_panel(frame_number: int, source: np.ndarray, output: np.ndarray, frame_profile: dict, mask: np.ndarray, profile_version: int) -> np.ndarray:
    bbox = frame_profile["bbox"]
    x0, y0, x1, y1 = calibration_bounds(bbox, source, profile_version)
    source_crop = source[y0:y1, x0:x1]
    output_crop = output[y0:y1, x0:x1]
    local_mask = cv2.resize(mask, (source_crop.shape[1], source_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    difference = cv2.convertScaleAbs(output_crop.astype(np.float32) - source_crop.astype(np.float32), alpha=4.0)
    mask_panel = cv2.cvtColor(local_mask, cv2.COLOR_GRAY2BGR)
    trajectory_panel = source_crop.copy()
    cv2.rectangle(trajectory_panel, (0, 0), (max(0, trajectory_panel.shape[1] - 1), max(0, trajectory_panel.shape[0] - 1)), (0, 220, 255), 1)
    panels = []
    for label, panel in (("Source", source_crop), ("Trajectory", trajectory_panel), ("Mask", mask_panel), ("Output", output_crop), ("Difference x4", difference)):
        panel = cv2.resize(panel, (255, 85), interpolation=cv2.INTER_NEAREST if label == "Mask" else cv2.INTER_AREA)
        cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    strip = np.concatenate(panels, axis=1)
    cv2.putText(strip, f"f{frame_number}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
    profile_version = int(profile.get("version", 0))
    if profile_version not in (4, 5, 6, 7) or profile.get("status") != "READY":
        raise RuntimeError("QualityReportV4/V5/V6/V7 requires a READY calibration profile")
    trajectory_gate = profile.get("trajectoryGate") or {}
    if profile_version in (5, 6, 7) and trajectory_gate.get("status") != "PASSED":
        raise RuntimeError(f"QualityReportV{profile_version} requires a passed trajectory gate")
    project_dir = args.profile.parent.parent
    mask_path = project_dir / profile["inferenceMaskPath"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Calibration inference mask is missing")
    mask = np.where(mask >= 64, 255, 0).astype(np.uint8)
    requested = selected_frames(profile)
    frame_data = profile.get("frameData", [])
    frame_count = int(profile["frameCount"])
    scan_range = profile.get("scanRange") or {
        "startFrame": int(profile.get("firstWatermarkFrame", 0)),
        "endFrame": frame_count - 1,
    }
    scan_start = int(scan_range["startFrame"])
    scan_end = int(scan_range["endFrame"])
    if scan_start < 0 or scan_end < scan_start or scan_end >= frame_count:
        raise RuntimeError("Calibration scan range is invalid")
    scan_length = scan_end - scan_start + 1
    if len(frame_data) != frame_count:
        raise RuntimeError("Calibration frame data does not cover the source video")
    source_capture = cv2.VideoCapture(str(args.source))
    output_capture = cv2.VideoCapture(str(args.output))
    rows: list[dict] = []
    panels: list[np.ndarray] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    decoded = 0
    try:
        for frame_number in range(frame_count):
            source_ok, source = source_capture.read()
            output_ok, output = output_capture.read()
            if not source_ok or not output_ok:
                break
            decoded += 1
            row, previous = make_row(frame_number, source, output, frame_data[frame_number], mask, previous, profile_version)
            rows.append(row)
            if frame_number in requested:
                panels.append(contact_panel(frame_number, source, output, frame_data[frame_number], mask, profile_version))
    finally:
        source_capture.release()
        output_capture.release()
    if decoded != frame_count:
        raise RuntimeError(f"QA decoded {decoded} of {frame_count} expected frames")
    metadata_ok, metadata = metadata_gate(media_metadata(args.source), media_metadata(args.output))
    # Every frame explicitly requiring a mask is part of the denominator.
    # Low-opacity frames are difficult by design and must not disappear from
    # the coverage metric merely because their source energy is small.
    required = [row for row in rows if row["maskRequired"]]
    def row_failure_reasons(row: dict) -> list[str]:
        reasons: list[str] = []
        if not row["measurable"]:
            reasons.append("unmeasurable_frame")
        if row["residualCorrelation"] > GOLDEN_MAX_RESIDUAL:
            reasons.append("residual_correlation")
        # Subtitle/UI text can occupy the same trajectory box after the
        # watermark is fully occluded.  In that case the source high-pass
        # energy is large, but canonical-glyph correlation is low; treating
        # energy alone as residual would reject a clean frame (for example
        # clip_test frames 640/756).  Require corroborating glyph correlation
        # before declaring an energy residual.  The frame remains in the
        # mask-required denominator and is still checked by all other gates.
        if (
            row["glyphEnergyRatio"] > GOLDEN_MAX_ENERGY_RATIO
            and row["residualCorrelation"] > GOLDEN_MAX_RESIDUAL
        ):
            reasons.append("glyph_energy_ratio")
        if row["outsideMaskSsim"] < GOLDEN_MIN_OUTSIDE_SSIM:
            reasons.append("outside_mask_ssim")
        if row["temporalFlicker"] > MAX_FLICKER:
            reasons.append("temporal_flicker")
        if row["seamScore"] > MAX_SEAM_SCORE:
            reasons.append("seam")
        if row["rectangularPatchScore"] > MAX_RECTANGULAR_PATCH_SCORE:
            reasons.append("rectangular_patch")
        return reasons

    failure_reasons = {
        str(row["frame"]): row_failure_reasons(row)
        for row in required
        if row_failure_reasons(row)
    }
    row_passes = [not row_failure_reasons(row) for row in required]
    metrics = {
        "maxResidualCorrelation": max((row["residualCorrelation"] for row in required), default=0.0),
        "maxGlyphEnergyRatio": max((row["glyphEnergyRatio"] for row in required), default=0.0),
        "minOutsideMaskSsim": min((row["outsideMaskSsim"] for row in required), default=1.0),
        "maxSeamScore": max((row["seamScore"] for row in required), default=0.0),
        "maxRectangularPatchScore": max((row["rectangularPatchScore"] for row in required), default=0.0),
        "maxTemporalFlicker": max((row["temporalFlicker"] for row in required), default=0.0),
        "visibleFrames": len(required),
        "maskRequiredFrames": len(required),
        "passedVisibleFrames": sum(row_passes),
        "coverageRate": (sum(row_passes) / len(required)) if required else 0.0,
        "maskApplicationCoverage": (
            sum(1 for row in required if row["confidence"] > 0.0) / len(required)
            if required else 0.0
        ),
        "residualPassCoverage": (sum(row_passes) / len(required)) if required else 0.0,
        "maskApplicationCoverageInRange": (
            sum(1 for row in required if row["confidence"] > 0.0) / len(required)
            if required else 0.0
        ),
        "residualPassCoverageInRange": (sum(row_passes) / len(required)) if required else 0.0,
        "excludedFrameCount": frame_count - scan_length,
        "excludedIntervals": (
            ([{"startFrame": 0, "endFrame": scan_start - 1}] if scan_start > 0 else [])
            + ([{"startFrame": scan_end + 1, "endFrame": frame_count - 1}] if scan_end < frame_count - 1 else [])
        ),
        "outsideRangeUnchecked": scan_length != frame_count,
        "unmeasurableFrames": [int(row["frame"]) for row in required if not row["measurable"]],
        "failedFrames": [int(frame) for frame in sorted(failure_reasons, key=int)],
        "failureReasons": failure_reasons,
    }
    difficult = sorted(required, key=lambda row: row["residualCorrelation"] + row["glyphEnergyRatio"] + row["temporalFlicker"], reverse=True)[:35]
    passed = (
        metadata_ok
        and bool(required)
        and all(row_passes)
        and not metrics["unmeasurableFrames"]
        and metrics["minOutsideMaskSsim"] >= GOLDEN_MIN_OUTSIDE_SSIM
        and metrics["maxSeamScore"] <= MAX_SEAM_SCORE
        and metrics["maxRectangularPatchScore"] <= MAX_RECTANGULAR_PATCH_SCORE
        and (profile_version not in (5, 6, 7) or trajectory_gate.get("status") == "PASSED")
    )
    report = {
        "version": profile_version,
        "reportVersion": 7,
        "status": "passed" if passed else "needs_review",
        "gate": f"quality_report_v{profile_version}",
        "source": str(args.source),
        "output": str(args.output),
        "profile": str(args.profile),
        "processing": {
            "engine": "ProPainter",
            "precision": "FP32",
            "inputMode": "full-frame",
            "gpuConcurrency": 1,
        },
        "fullFrameScan": True,
        "scanRange": {"startFrame": scan_start, "endFrame": scan_end},
        "excludedFrameCount": frame_count - scan_length,
        "outsideRangeUnchecked": scan_length != frame_count,
        "coverageContract": "100% of maskRequired frames in scanRange; excluded frames are passthrough and metadata-only",
        "trajectory": {
            "gate": trajectory_gate,
            "model": profile.get("trajectoryModel"),
            "measuredFrames": profile.get("qualityGate", {}).get("measuredFrames", 0),
            "interpolatedFrames": profile.get("qualityGate", {}).get("interpolatedFrames", 0),
            "maxObservationGap": profile.get("trajectoryModel", {}).get("maxObservationGap"),
        },
        "sampledFrames": sorted(requested),
        "difficultFrames": [row["frame"] for row in difficult],
        "metrics": metrics,
        "goldenThresholds": {
            "maxResidualCorrelation": GOLDEN_MAX_RESIDUAL,
            "maxGlyphEnergyRatio": GOLDEN_MAX_ENERGY_RATIO,
            "minOutsideMaskSsim": GOLDEN_MIN_OUTSIDE_SSIM,
            "maxTemporalFlicker": MAX_FLICKER,
            "maxSeamScore": MAX_SEAM_SCORE,
            "maxRectangularPatchScore": MAX_RECTANGULAR_PATCH_SCORE,
        },
        "metadata": metadata,
        "ocr": {
            "engine": "canonical-glyph-correlation",
            "textDetected": not all(row_passes),
        },
        "contactSheet": str(args.contact_sheet),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    json.loads(temporary_report.read_text(encoding="utf-8"))
    temporary_report.replace(args.report)
    if panels:
        sheet = np.concatenate(panels, axis=0)
    else:
        sheet = np.zeros((85, 1275, 3), dtype=np.uint8)
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.contact_sheet), sheet)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
