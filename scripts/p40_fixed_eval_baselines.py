#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(os.environ.get("LRM_ROOT", "/home/yourslewis/lrm-scaling-all-events"))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
OUTROOT = ROOT / "results_v2/p40_fixed_eval_baselines"
DATA_FULL = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve")
DATA_FROZEN = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
PYTHON = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")


@dataclass(frozen=True)
class Profile:
    id: str
    model_id: str
    config: str
    summary: str


PROFILES = {p.id: p for p in [
    Profile("m1_ads_loss", "M1", "proposed_2-mmoe_ple/config/generated_p40_fixed_eval_baselines/p40_m1_ads_loss_hstu.gin", "Clean all-event Ads-loss HSTU"),
    Profile("m2_aux_light", "M2", "proposed_2-mmoe_ple/config/generated_p40_fixed_eval_baselines/p40_m2_aux_light_hstu.gin", "All-event HSTU with light non-Ads auxiliary loss"),
    Profile("m3_aux_medium", "M3", "proposed_2-mmoe_ple/config/generated_p40_fixed_eval_baselines/p40_m3_aux_medium_hstu.gin", "All-event HSTU with medium non-Ads auxiliary loss"),
]}


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def data_dir() -> Path:
    d = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_train_full_frozen7eval")
    d.mkdir(parents=True, exist_ok=True)
    train = d / "train"
    train.mkdir(exist_ok=True)
    for p in sorted((DATA_FULL / "train").glob("*.parquet")):
        link = train / p.name
        if not link.exists():
            link.symlink_to(p)
    for name, target in [("eval", DATA_FROZEN / "eval"), ("metadata.json", DATA_FROZEN / "metadata.json"), ("freeze_manifest.json", DATA_FROZEN / "freeze_manifest.json")]:
        path = d / name
        if not path.exists():
            path.symlink_to(target, target_is_directory=target.is_dir())
    (d / "DATASET_SCALE.json").write_text(json.dumps({"scale": 1.0, "source_train": str(DATA_FULL / "train"), "frozen_eval": str(DATA_FROZEN / "eval"), "updated_at": utc()}, indent=2))
    return d


def env(gpu: int, port: int, run_id: str):
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["AZUREML_RUN_ID"] = run_id
    e["PYTHONPATH"] = f"{TRAIN}:{e.get('PYTHONPATH', '')}"
    e.update({
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    })
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        e[k] = "1"
    return e


def launch_train(p: Profile, gpu: int, port: int):
    dd = data_dir()
    cfg = ROOT / p.config
    out = OUTROOT / p.id
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        PYTHON, "-u", "main.py",
        f"--gin_config_file={cfg}",
        f"--output_path={out}",
        "--data_path", str(dd),
        "--mode=job",
        "--ads_semantic_embd_path", str(EMB / "domain_0"),
        "--web_browsing_semantic_embd_path", str(EMB / "domain_1"),
        "--shopping_semantic_embd_path", str(EMB / "domain_2"),
        "--ads_pure_corpus_embd_path", str(EMB / "domain_3"),
        "--other_semantic_embd_path", str(EMB / "domain_4"),
    ]
    f = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env(gpu, port, f"p40_{p.id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"), stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    (out / "launch.json").write_text(json.dumps({"profile": asdict(p), "cmd": cmd, "data_path": str(dd), "pid": proc.pid, "gpu": gpu, "started_at": utc(), "log": str(log)}, indent=2))
    return {"id": p.id, "model_id": p.model_id, "pid": proc.pid, "gpu": gpu, "log": str(log), "output_path": str(out), "config": str(cfg), "data_path": str(dd), "started_at": utc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", default=["m1_ads_loss", "m2_aux_light"])
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--base-port", type=int, default=29980)
    args = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    launches: List[dict] = []
    for i, pid in enumerate(args.profiles):
        p = PROFILES[pid]
        gpu = args.gpus[i % len(args.gpus)]
        launches.append(launch_train(p, gpu, args.base_port + i * 10))
    state = {"updated_at_utc": utc(), "launches": launches}
    (OUTROOT / "launch_state.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
