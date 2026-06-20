#!/usr/bin/env python3
"""M3 data-scale ablation launcher (DO NOT RUN until human approval & GPU free).

Trains the FROZEN M2-newdata aux-light HSTU recipe on symlink VIEWS at three
training-data fractions (25/50/100%) to test the thesis claim "more training
data helps Ads". Recipe gin is identical to M2-newdata; only --data_path varies.

Usage (one scale at a time, on an idle GPU):
  CUDA_VISIBLE_DEVICES=1 python3 scripts/launch_m3_data_scale.py --scale 0.25
Guards: refuses to launch if the target GPU is busy.
"""
from __future__ import annotations
import argparse, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("LRM_ROOT", "/home/yourslewis/lrm-launches/m2-newdata-src-20260530"))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
OUTROOT = ROOT / "results_v2/m3_data_scale"
VIEW_ROOT = Path("/home/yourslewis/lrm_benchmarkv4/processed/m3_data_scale_views")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
CFG = ROOT / "proposed_2-mmoe_ple/config/generated_m3_data_scale/m3_aux_light_hstu_newdata_datascale.gin"
PYTHON = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def view_for(scale: float) -> Path:
    tag = "full" if scale >= 0.999 else f"{int(round(scale*100)):02d}pct"
    return VIEW_ROOT / f"v001_hstu_seqview_train_{tag}"


def gpu_busy(gpu: int) -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
            text=True).strip()
        return int(out.split("\n")[0]) > 2000
    except Exception:
        return True  # fail closed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--force", action="store_true", help="override GPU-busy guard")
    args = ap.parse_args()

    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0])
    if gpu_busy(gpu) and not args.force:
        raise SystemExit(f"GPU{gpu} appears busy (>2GB used); refusing to launch. Use --force to override.")

    data = view_for(args.scale)
    if not (data / "train").exists():
        raise SystemExit(f"scale view missing: {data}; run prepare_m3_scale_views.py first")

    tag = "full" if args.scale >= 0.999 else f"{int(round(args.scale*100)):02d}pct"
    run_id = f"m3_data_scale_{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUTROOT / f"m3_data_scale_{tag}"
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
        "MASTER_PORT": os.environ.get("MASTER_PORT", "30030"),
    })
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = "1"

    cmd = [
        PYTHON, "-u", "main.py",
        f"--gin_config_file={CFG}",
        f"--output_path={out}",
        "--data_path", str(data),
        "--mode=job",
        "--ads_semantic_embd_path", str(EMB / "domain_0"),
        "--web_browsing_semantic_embd_path", str(EMB / "domain_1"),
        "--shopping_semantic_embd_path", str(EMB / "domain_2"),
        "--ads_pure_corpus_embd_path", str(EMB / "domain_3"),
        "--other_semantic_embd_path", str(EMB / "domain_4"),
    ]
    stream = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=stream, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    launch = {
        "id": f"m3_data_scale_{tag}", "model_id": "M3-data-scale", "scale": args.scale,
        "pid": proc.pid, "gpu": gpu, "started_at": utc(), "log": str(log),
        "output_path": str(out), "config": str(CFG), "data_path": str(data),
        "cmd": cmd, "run_id": run_id,
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
