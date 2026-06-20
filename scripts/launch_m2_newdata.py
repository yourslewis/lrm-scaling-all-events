#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("LRM_ROOT", "/home/yourslewis/lrm-launches/m2-newdata-src-20260530"))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
OUTROOT = ROOT / "results_v2/m2_newdata_baseline"
DATA = Path("/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
CFG = ROOT / "proposed_2-mmoe_ple/config/generated_m2_newdata/m2_aux_light_hstu_newdata.gin"
PYTHON = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")

def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def main() -> None:
    run_id = f"m2_aux_light_hstu_newdata_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUTROOT / "m2_aux_light_hstu_newdata"
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "AZUREML_RUN_ID": run_id,
        "PYTHONPATH": f"{TRAIN}:{env.get('PYTHONPATH', '')}",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": os.environ.get("MASTER_PORT", "30020"),
    })
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = "1"
    cmd = [
        PYTHON, "-u", "main.py",
        f"--gin_config_file={CFG}",
        f"--output_path={out}",
        "--data_path", str(DATA),
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
        "id": "m2_aux_light_hstu_newdata",
        "model_id": "M2-newdata",
        "pid": proc.pid,
        "gpu": int(env["CUDA_VISIBLE_DEVICES"].split(",")[0]),
        "started_at": utc(),
        "log": str(log),
        "output_path": str(out),
        "config": str(CFG),
        "data_path": str(DATA),
        "cmd": cmd,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "run_id": run_id,
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    (OUTROOT / "launch_state.json").write_text(json.dumps({"updated_at_utc": utc(), "launches": [launch]}, indent=2), encoding="utf-8")
    print(json.dumps(launch, indent=2))

if __name__ == "__main__":
    main()
