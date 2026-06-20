#!/usr/bin/env python3
"""M4 50pct sequence-length scaling launcher (prepared; do not auto-run).

Matrix: L100, L200 baseline reuse, L400, L800. This script launches exactly one
new M4 seq-len lane on the fixed 50pct train view, keeping the M2 aux-light recipe,
weights, negatives, and data view unchanged except get_reco_dataset.max_sequence_length.

Usage after GPU0 is free, one run at a time:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/launch_m4_seq_len_50pct.py --seq-len 400

Guards:
- refuses to launch if the selected GPU has >2GB memory used, unless --force
- refuses to re-run L200 by default because it is the existing M3-50 baseline
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("LRM_ROOT", "/home/yourslewis/lrm-launches/m2-newdata-src-20260530"))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
OUTROOT = ROOT / "results_v2/m4_seq_len_50pct"
VIEW = Path("/home/yourslewis/lrm_benchmarkv4/processed/m3_data_scale_views/v001_hstu_seqview_train_50pct")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
CFG_DIR = ROOT / "proposed_2-mmoe_ple/config/generated_m4_seq_len_50pct"
PYTHON = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")
SUPPORTED_SEQ_LENS = (100, 200, 400, 800)


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def selected_gpu() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not raw:
        raw = "0"
    return int(raw)


def gpu_memory_used_mb(gpu: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        text=True,
    ).strip()
    return int(out.splitlines()[0].strip())


def gpu_busy(gpu: int) -> bool:
    try:
        return gpu_memory_used_mb(gpu) > 2000
    except Exception:
        return True  # fail closed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True, choices=SUPPORTED_SEQ_LENS)
    ap.add_argument("--force", action="store_true", help="override GPU-busy guard")
    ap.add_argument("--allow-l200-rerun", action="store_true", help="allow re-running the existing L200 baseline lane")
    args = ap.parse_args()

    if args.seq_len == 200 and not args.allow_l200_rerun:
        raise SystemExit(
            "L200 is the existing M3-50pct baseline (final b1124000); reuse it instead of re-training. "
            "Pass --allow-l200-rerun only if you explicitly want a duplicate run."
        )

    gpu = selected_gpu()
    if gpu_busy(gpu) and not args.force:
        raise SystemExit(f"GPU{gpu} appears busy (>2GB used); refusing to launch. Use --force to override.")

    if not (VIEW / "train").exists():
        raise SystemExit(f"fixed 50pct train view missing: {VIEW}")

    cfg = CFG_DIR / f"m4_aux_light_hstu_50pct_l{args.seq_len}.gin"
    if not cfg.exists():
        raise SystemExit(f"seq-len config missing: {cfg}")

    run_id = f"m4_seq_len_l{args.seq_len}_50pct_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUTROOT / f"m4_seq_len_l{args.seq_len}_50pct"
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "AZUREML_RUN_ID": run_id,
        "PYTHONPATH": f"{TRAIN}:{env.get('PYTHONPATH', '')}",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": os.environ.get("MASTER_PORT", "30040"),
    })
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = "1"

    cmd = [
        PYTHON, "-u", "main.py",
        f"--gin_config_file={cfg}",
        f"--output_path={out}",
        "--data_path", str(VIEW),
        "--mode=job",
        "--ads_semantic_embd_path", str(EMB / "domain_0"),
        "--web_browsing_semantic_embd_path", str(EMB / "domain_1"),
        "--shopping_semantic_embd_path", str(EMB / "domain_2"),
        "--ads_pure_corpus_embd_path", str(EMB / "domain_3"),
        "--other_semantic_embd_path", str(EMB / "domain_4"),
    ]

    stream = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=stream, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    launch = {
        "id": f"m4_seq_len_l{args.seq_len}_50pct",
        "model_id": "M4-seq-len-50pct",
        "seq_len": args.seq_len,
        "pid": proc.pid,
        "gpu": gpu,
        "started_at": utc(),
        "log": str(log),
        "output_path": str(out),
        "config": str(cfg),
        "data_path": str(VIEW),
        "cmd": cmd,
        "run_id": run_id,
        "note": "Prepared M4 lane; train only when GPU0 is free. L200 baseline should be reused unless explicitly re-run.",
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
