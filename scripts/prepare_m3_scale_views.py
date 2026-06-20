#!/usr/bin/env python3
"""M3 data-scale ablation: build symlink-based train-fraction VIEWS of the
canonical v001 HSTU seqview, holding the M2-newdata recipe fixed.

READ-ONLY on the source dataset: we only create new view dirs containing
symlinks to a deterministic prefix of the source train parquet shards, plus
symlinks to the SHARED eval/ + metadata.json so every scale evaluates on the
identical frozen eval universe.

Default scales: 0.25, 0.50, 1.00. Does NOT launch training.
"""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview")
VIEW_ROOT = Path("/home/yourslewis/lrm_benchmarkv4/processed/m3_data_scale_views")


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_view(scale: float) -> Path:
    assert 0.0 < scale <= 1.0, scale
    tag = "full" if scale >= 0.999 else f"{int(round(scale*100)):02d}pct"
    view = VIEW_ROOT / f"v001_hstu_seqview_train_{tag}"
    train = view / "train"
    train.mkdir(parents=True, exist_ok=True)

    parts = sorted((SRC / "train").glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no source train parquet under {SRC/'train'}")
    n = len(parts) if scale >= 0.999 else max(1, int(round(len(parts) * scale)))
    keep = parts[:n]
    keep_names = {p.name for p in keep}

    for p in keep:
        link = train / p.name
        if not link.exists():
            link.symlink_to(p)
        done = p.parent / (p.name + ".done")
        if done.exists() and not (train / done.name).exists():
            (train / done.name).symlink_to(done)
    # prune any stale links from a prior smaller/larger run
    for stale in train.glob("*.parquet"):
        if stale.name not in keep_names:
            stale.unlink()
    for stale in train.glob("*.parquet.done"):
        if stale.name.replace(".done", "") not in keep_names:
            stale.unlink()

    # shared eval + metadata: identical frozen eval universe for every scale
    for name in ("eval", "metadata.json", "DATASET_DISCLOSURE.json"):
        target = SRC / name
        path = view / name
        if target.exists() and not path.exists():
            path.symlink_to(target, target_is_directory=target.is_dir())

    (view / "DATASET_SCALE.json").write_text(json.dumps({
        "scale": scale,
        "train_parts_total": len(parts),
        "train_parts_kept": n,
        "source_view": str(SRC),
        "shared_eval": str(SRC / "eval"),
        "note": "symlink view; recipe frozen to M2-newdata aux-light; data fraction is the only lever",
        "created_at": utc(),
    }, indent=2))
    return view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.25,0.5,1.0")
    args = ap.parse_args()
    for s in [float(x) for x in args.scales.split(",") if x.strip()]:
        v = build_view(s)
        kept = json.loads((v / "DATASET_SCALE.json").read_text())["train_parts_kept"]
        print(f"scale={s:>4} -> {v}  ({kept} train shards)")


if __name__ == "__main__":
    main()
