from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from render_periodic_dewatermark import periodic_position


DEFAULT_CLIP = Path(r"C:\Users\quant\Dropbox\PC\Downloads\clip_test.mp4")
DEFAULT_REFERENCE = Path(r"C:\Users\quant\Dropbox\PC\Downloads\output\clip_test_watermark_removed_best.mp4")
DEFAULT_FULL = Path(r"C:\Users\quant\Dropbox\PC\Downloads\8_6 (24).mp4")
DEFAULT_OUTPUT = Path(r"C:\Users\quant\Dropbox\PC\Downloads\output")
DEFAULT_PROPainter = Path(r"D:\propainter-watermark-work")
DEFAULT_PYTHON = Path(r"D:\propainter-watermark-venv\Scripts\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gated Learna AI regression and full-video render")
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--full-source", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=Path(os.getenv("WATERMARK_STUDIO_REGRESSION_WORK", str(Path.cwd() / ".regression-work"))))
    parser.add_argument("--propainter-root", type=Path, default=DEFAULT_PROPainter)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Run standalone calibration/render/QA without comparing against the clip_test golden.",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Skip the full-source segment alignment check for standalone tests.",
    )
    parser.add_argument("--resume-clip-report", type=Path, default=None, help="Reuse an already completed clip-test result after validating its QA report")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, capture_output=capture, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {detail[-1200:]}")
    return result


def probe(path: Path) -> dict:
    result = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture=True)
    payload = json.loads(result.stdout)
    video = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream found: {path}")
    fps_value = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    numerator, denominator = fps_value.split("/", 1)
    fps = float(numerator) / float(denominator)
    frame_count = int(video.get("nb_frames") or 0)
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    if frame_count <= 0:
        frame_count = max(1, round(duration * fps))
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "durationSeconds": duration,
        "fps": fps,
        "frameCount": frame_count,
        "codec": video.get("codec_name"),
        "pixelFormat": video.get("pix_fmt"),
    }


def make_project(path: Path, directory: Path) -> Path:
    metadata = probe(path)
    project = {
        "version": 1,
        "id": str(uuid.uuid4()),
        "source": {"path": str(path), "fileName": path.name},
        "video": metadata,
        "watermark": {"label": "Learna AI", "anchor": None, "templates": None, "templatePadding": 4},
        "calibration": None,
        "anchors": [],
        "tracking": None,
        "removal": None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    project_path = directory / "project.json"
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project_path


def scan_candidate(project_json: Path, root: Path, python: Path, repo_root: Path) -> dict:
    candidates: list[dict] = []
    for phase in range(6):
        phase_dir = root / f"samples-phase-{phase}"
        result = run(
            [
                str(python), str(repo_root / "tools" / "find_learna_samples.py"),
                str(project_json), str(phase_dir), "--scan-round", str(phase),
                "--exclude-frames", "[]", "--exclude-signatures", "[]",
            ],
            capture=True,
        )
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
        candidates.extend(parsed)
    if not candidates:
        # Last-resort clean-mask route.  It is deliberately marked as a
        # fallback and remains subject to the full trajectory audit plus the
        # all-frame QA gate; if the watermark is not observable this route
        # ends in NEEDS_REVIEW instead of guessing a final output.
        project = json.loads(project_json.read_text(encoding="utf-8"))
        frame = max(48, min(int(project["video"]["frameCount"]) - 1, int(project["video"]["frameCount"]) // 2))
        x, y = periodic_position(frame % 360)
        return {
            "frame": frame,
            "timestampSeconds": frame / float(project["video"]["fps"]),
            "bbox": {"x": x, "y": y, "width": 245.33333333333334, "height": 74.66666666666667},
            "maskCoverage": 3375,
            "maskPeak": 255,
            "backgroundComplexity": 999.0,
            "temporalInstability": 1.0,
            "glyphCorrelation": 0.0,
            "glyphIou": 0.0,
            "contamination": 1.0,
            "temporalPassCount": 0,
            "score": 0.0,
            "sceneSignature": "clean-mask-fallback",
            "previewPath": "",
            "maskPath": "",
            "editorMaskPath": "",
            "roiFallback": False,
            "trajectoryPhaseOffset": 0,
            "route": "CLEAN_MASK_TEMPLATE",
        }
    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    selected: list[dict] = []
    for candidate in candidates:
        frame = int(candidate["frame"])
        signature = str(candidate.get("sceneSignature", ""))
        if any(abs(frame - int(item["frame"])) < 72 for item in selected):
            continue
        if signature and any(signature == str(item.get("sceneSignature", "")) for item in selected):
            continue
        selected.append(candidate)
    if not selected:
        raise RuntimeError("All AUTO_FIND candidates were excluded")
    selected[0]["route"] = "AUTO_FIND"
    return selected[0]


def render_one(
    source: Path,
    reference: Path | None,
    output_name: str,
    args: argparse.Namespace,
    repo_root: Path,
    python: Path,
    label: str,
) -> dict:
    project_dir = args.work_root / f"{label}-{uuid.uuid4().hex[:8]}"
    project_json = make_project(source, project_dir)
    profile = project_dir / "calibration" / "profile.json"
    run(
        [
            str(python), str(repo_root / "tools" / "calibrate_trajectory_v5.py"),
            str(project_json), str(profile), "--route", "AUTO_GLOBAL_TEMPLATE",
        ],
    )
    profile_body = json.loads(profile.read_text(encoding="utf-8"))
    observations_path = profile.parent / "trajectory-observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8")) if observations_path.is_file() else []
    candidate = observations[len(observations) // 2] if observations else {
        "frame": 0,
        "route": "AUTO_GLOBAL_TEMPLATE",
        "trajectoryGate": profile_body.get("trajectoryGate", {}),
    }
    candidate["route"] = profile_body.get("route", "AUTO_GLOBAL_TEMPLATE")
    if profile_body.get("status") != "READY":
        return {
            "status": "NEEDS_REVIEW",
            "source": str(source),
            "calibration": str(profile),
            "diagnostics": str(profile.parent / "diagnostics.json"),
            "candidate": candidate,
            "quality": {"status": "needs_review", "trajectory": profile_body.get("trajectoryGate", {})},
        }
    project = json.loads(project_json.read_text(encoding="utf-8"))
    frame_count = int(project["video"]["frameCount"])
    workspace = project_dir / "render-work"
    result_root = workspace / "results"
    run(
        [
            str(python), str(repo_root / "tools" / "propainter_periodic_pipeline.py"), "prepare",
            str(project_json), "-", str(workspace), "--profile", str(profile),
            "--start-frame", "48", "--end-frame", str(frame_count - 1), "--full-frame",
        ],
    )
    run(
        [
            str(python), str(repo_root / "tools" / "run_propainter_chunks.py"),
            str(workspace), str(result_root), str(args.propainter_root), str(python),
            "--width", "288", "--height", "512", "--core-length", "60", "--context", "8",
        ],
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    final_path = args.output_root / output_name
    if final_path.exists():
        index = 1
        while True:
            candidate_path = final_path.with_name(f"{final_path.stem}_{index}{final_path.suffix}")
            if not candidate_path.exists():
                final_path = candidate_path
                break
            index += 1
    draft = final_path.with_name(f"{final_path.stem}.review{final_path.suffix}")
    run(
        [
            str(python), str(repo_root / "tools" / "propainter_periodic_pipeline.py"), "composite",
            str(project_json), str(workspace), str(result_root / "merged-frames"), str(draft),
        ],
    )
    report = draft.with_suffix(".qa.json")
    sheet = draft.with_suffix(".qa.png")
    qa = subprocess.run(
        [str(python), str(repo_root / "tools" / "quality_qa_v4.py"), str(source), str(draft), str(profile), str(report), str(sheet)],
        capture_output=True,
        text=True,
        check=False,
    )
    if qa.returncode == 2:
        body = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {"status": "needs_review"}
        return {"status": "NEEDS_REVIEW", "source": str(source), "draft": str(draft), "qa": str(report), "contactSheet": str(sheet), "candidate": candidate, "quality": body}
    if qa.returncode != 0:
        raise RuntimeError(f"Quality QA failed for {source}: {(qa.stderr or qa.stdout).strip()[-1200:]}")
    comparison = None
    if reference is not None:
        # Validate the accepted golden with the same full-frame V4 QA before
        # using it as a regression baseline. This prevents a stale profile or
        # an accidental frame-index mismatch from becoming a false pass.
        golden_report = project_dir / "golden.qa.json"
        golden_sheet = project_dir / "golden.qa.png"
        golden_qa = subprocess.run(
            [str(python), str(repo_root / "tools" / "quality_qa_v4.py"), str(source), str(reference), str(profile), str(golden_report), str(golden_sheet)],
            capture_output=True,
            text=True,
            check=False,
        )
        golden_body = json.loads(golden_report.read_text(encoding="utf-8")) if golden_report.is_file() else {"status": "needs_review"}
        if golden_qa.returncode != 0 or golden_body.get("status") != "passed":
            return {"status": "NEEDS_REVIEW", "source": str(source), "draft": str(draft), "qa": str(report), "contactSheet": str(sheet), "candidate": candidate, "quality": json.loads(report.read_text(encoding="utf-8")), "goldenQA": golden_body}
        comparison_report = final_path.with_suffix(".reference.json")
        run([str(python), str(repo_root / "tools" / "compare_reference.py"), str(draft), str(reference), str(comparison_report)])
        comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
        candidate_quality = json.loads(report.read_text(encoding="utf-8"))
        golden_metrics = golden_body.get("metrics", {})
        candidate_metrics = candidate_quality.get("metrics", {})
        quality_comparison = {
            "maxResidualWithin5Percent": float(candidate_metrics.get("maxResidualCorrelation", 1e9)) <= float(golden_metrics.get("maxResidualCorrelation", 0.0)) * 1.05,
            "maxEnergyWithin5Percent": float(candidate_metrics.get("maxGlyphEnergyRatio", 1e9)) <= float(golden_metrics.get("maxGlyphEnergyRatio", 0.0)) * 1.05,
            "outsideSsimWithin002": float(candidate_metrics.get("minOutsideMaskSsim", 0.0)) >= float(golden_metrics.get("minOutsideMaskSsim", 1.0)) - 0.002,
            "coverageComplete": float(candidate_metrics.get("coverageRate", 0.0)) >= 1.0,
        }
        comparison["qualityGate"] = quality_comparison
        if not all(quality_comparison.values()):
            return {"status": "NEEDS_REVIEW", "source": str(source), "draft": str(draft), "qa": str(report), "contactSheet": str(sheet), "candidate": candidate, "quality": candidate_quality, "goldenQA": golden_body, "referenceComparison": comparison}

    # Only now is the draft promoted to the collision-safe final name.
    draft.replace(final_path)
    final_report = final_path.with_suffix(".qa.json")
    final_sheet = final_path.with_suffix(".qa.png")
    if report.exists():
        report.replace(final_report)
    if sheet.exists():
        sheet.replace(final_sheet)
    return {"status": "COMPLETED", "source": str(source), "output": str(final_path), "qa": str(final_report), "contactSheet": str(final_sheet), "candidate": candidate, "quality": json.loads(final_report.read_text(encoding="utf-8")), "referenceComparison": comparison}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python = args.python or Path(os.getenv("WATERMARK_STUDIO_PROPAINTER_PYTHON", str(DEFAULT_PYTHON)))
    if not python.is_file():
        python = Path(sys.executable)
    required_inputs = [args.clip]
    if not args.no_reference:
        required_inputs.append(args.reference)
    if not args.skip_full and not args.skip_alignment:
        required_inputs.append(args.full_source)
    for path in required_inputs:
        if not path.is_file():
            raise RuntimeError(f"Required input does not exist: {path}")
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"version": 1, "status": "RUNNING"}
    # Verify that the short regression clip is actually the leading segment of
    # the long source before making any frame-index comparison claim.  This is
    # an independent content check; a failure does not block the clip QA, but
    # it explicitly prevents the report from calling the comparison aligned.
    if args.skip_alignment:
        report["segmentAlignment"] = {"status": "SKIPPED", "reason": "standalone test"}
    else:
        alignment_report = args.output_root / "clip_test_segment_alignment.json"
        alignment_cmd = [
            str(python), str(repo_root / "tools" / "verify_segment_alignment.py"),
            str(args.full_source), str(args.clip), str(alignment_report),
        ]
        alignment = subprocess.run(alignment_cmd, capture_output=True, text=True, check=False)
        if alignment.returncode not in (0, 2):
            detail = (alignment.stderr or alignment.stdout or "").strip()
            raise RuntimeError(f"Segment alignment verification failed: {detail[-1200:]}")
        report["segmentAlignment"] = json.loads(alignment_report.read_text(encoding="utf-8"))
    if args.resume_clip_report is not None:
        resume_body = json.loads(args.resume_clip_report.read_text(encoding="utf-8"))
        clip_result = resume_body.get("clipTest", resume_body)
        if clip_result.get("status") == "passed":
            clip_result = {"status": "COMPLETED", "qa": str(args.resume_clip_report)}
        if clip_result.get("status") != "COMPLETED":
            raise RuntimeError("Cannot resume full-source regression: clip-test report is not COMPLETED")
    else:
        # Keep the canonical name; the renderer adds _1/_2 when the golden or
        # a previous attempt already exists, never overwriting user output.
        clip_reference = None if args.no_reference else args.reference
        output_name = "standalone_watermark_removed_best.mp4" if args.no_reference else "clip_test_watermark_removed_best.mp4"
        clip_result = render_one(args.clip, clip_reference, output_name, args, repo_root, python, "standalone-test" if args.no_reference else "clip-test")
    report["clipTest"] = clip_result
    if clip_result["status"] != "COMPLETED":
        report["status"] = "NEEDS_REVIEW"
        output_report = args.output_root / "verified_regression_report.json"
        output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        raise SystemExit(2)
    if not args.skip_full:
        report["fullSource"] = render_one(args.full_source, None, "8_6 (24)_watermark_removed_best.mp4", args, repo_root, python, "full-source")
    report["status"] = "COMPLETED" if report.get("fullSource", {"status": "COMPLETED"}).get("status") == "COMPLETED" else "NEEDS_REVIEW"
    output_report = args.output_root / "verified_regression_report.json"
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
