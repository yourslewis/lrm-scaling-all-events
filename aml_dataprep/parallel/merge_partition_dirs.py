#!/usr/bin/env python3
"""Merge unique fan-out output directories into one downstream layout."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aml_dataprep.parallel.partition import write_ready_manifest


def _copy_tree_contents(src: Path, dst: Path) -> int:
    copied = 0
    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        for name in files:
            src_file = root_path / name
            dst_file = dst / rel_root / name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if dst_file.exists():
                if src_file.read_bytes() == dst_file.read_bytes():
                    continue
                raise FileExistsError(f"merge collision: {dst_file}")
            shutil.copy2(src_file, dst_file)
            copied += 1
    return copied


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--expected_count", type=int, required=True)
    args = p.parse_args()

    if len(args.input_dirs) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} input dirs, got {len(args.input_dirs)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    ready_files = []
    for raw in args.input_dirs:
        src = Path(raw)
        if not src.exists():
            raise FileNotFoundError(f"missing input dir: {src}")
        ready = sorted((src / "_ready").glob("*.json"))
        if not ready:
            raise FileNotFoundError(f"missing ready manifest under {src / '_ready'}")
        ready_files.extend(str(path) for path in ready)
        copied_files += _copy_tree_contents(src, output_dir)
    ready_path = write_ready_manifest(output_dir / "_ready", f"{args.stage}_merged.json", {
        "stage": args.stage,
        "expected_count": args.expected_count,
        "input_dirs": args.input_dirs,
        "ready_files": ready_files,
        "copied_files": copied_files,
    })
    print(f"merged {len(args.input_dirs)} dirs for {args.stage}; files={copied_files}; ready={ready_path}", flush=True)


if __name__ == "__main__":
    main()
