from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("propainter_root", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--core-length", type=int, default=60)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_frames = sorted((args.workspace / "frames").glob("*.png"))
    source_masks = sorted((args.workspace / "masks").glob("*.png"))
    if not source_frames or len(source_frames) != len(source_masks):
        raise RuntimeError("Prepared frame and mask counts do not match")

    merged = args.output_root / "merged-frames"
    chunks = args.output_root / "chunks"
    merged.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)
    total = len(source_frames)
    for core_start in range(0, total, args.core_length):
        core_end = min(total, core_start + args.core_length)
        input_start = max(0, core_start - args.context)
        input_end = min(total, core_end + args.context)
        chunk_name = f"chunk-{core_start:04d}-{core_end - 1:04d}"
        chunk_root = chunks / chunk_name
        frames_dir = chunk_root / "input" / "frames"
        masks_dir = chunk_root / "input" / "masks"
        result_frames = chunk_root / "results" / "frames" / "frames"
        expected_results = input_end - input_start
        if len(list(result_frames.glob("*.png"))) != expected_results:
            if chunk_root.exists():
                shutil.rmtree(chunk_root)
            frames_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)
            for local_index, global_index in enumerate(range(input_start, input_end)):
                name = f"{local_index:04d}.png"
                link_or_copy(source_frames[global_index], frames_dir / name)
                link_or_copy(source_masks[global_index], masks_dir / name)
            command = [
                str(args.python),
                "inference_propainter.py",
                "--video",
                str(frames_dir),
                "--mask",
                str(masks_dir),
                "--output",
                str(chunk_root / "results"),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--mask_dilation",
                "1",
                "--neighbor_length",
                "2",
                "--ref_stride",
                "14",
                "--subvideo_length",
                "10",
                "--raft_iter",
                "8",
                "--save_frames",
                "--save_fps",
                "30",
            ]
            subprocess.run(command, cwd=args.propainter_root, check=True)
        for global_index in range(core_start, core_end):
            local_index = global_index - input_start
            source = result_frames / f"{local_index:04d}.png"
            destination = merged / f"{global_index:04d}.png"
            if destination.exists():
                destination.unlink()
            link_or_copy(source, destination)
        print(
            f"completed {chunk_name}: merged {core_end}/{total}",
            flush=True,
        )
    if len(list(merged.glob("*.png"))) != total:
        raise RuntimeError("Merged output does not cover every prepared frame")
    print(merged)


if __name__ == "__main__":
    main()
