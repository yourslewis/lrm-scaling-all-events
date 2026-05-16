#!/usr/bin/env python3
"""Launch/monitor P28 domain-aware random-negative training experiments.

P28 keeps the P23 model/loss/ad-anchor setup and changes only the training
negative sampler from in-batch negatives to domain-aware global random negatives
(`RotateInDomainGlobalNegativesSampler`). The grid varies `make_model.num_negatives`
across 32, 48, 64, and 96.

If the full-structure n96 run fails with CUDA OOM, this runner can queue an
explicit small-model fallback (`p28s_domain_rand_n96_small_fallback`) and records
that it is not architecture-equivalent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path("/home/yourslewis/lrm-scaling-all-events")
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
DATA = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings")
CFGDIR = ROOT / "proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives"
OUTROOT = ROOT / "results_v2"
RUNNER_DIR = OUTROOT / "p28_domain_random_negatives_autoresearch"
PYTHON = "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10"


@dataclass(frozen=True)
class Profile:
    name: str
    num_negatives: int
    gpu: int
    port: int
    small_model: bool = False


PROFILES = [
    Profile("p28_domain_rand_n32", 32, 0, 32832),
    Profile("p28_domain_rand_n48", 48, 1, 32848),
    Profile("p28_domain_rand_n64", 64, 0, 32864),
    Profile("p28_domain_rand_n96", 96, 1, 32896),
]

FALLBACK_N96 = Profile("p28s_domain_rand_n96_small_fallback", 96, 1, 32996, small_model=True)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    (RUNNER_DIR / "logs").mkdir(parents=True, exist_ok=True)


def profile_config(profile: Profile) -> Path:
    return CFGDIR / f"{profile.name}.gin"


def env_for(profile: Profile) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(profile.gpu)
    env["AZUREML_RUN_ID"] = f"{profile.name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    env["PYTHONPATH"] = f"{TRAIN}:{env.get('PYTHONPATH', '')}"
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(profile.port)
    return env


def launch(profile: Profile) -> tuple[subprocess.Popen, Path, Path, Path]:
    cfg = profile_config(profile)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    out = OUTROOT / profile.name
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        PYTHON,
        "-u",
        "main.py",
        f"--gin_config_file={cfg}",
        f"--output_path={out}",
        "--data_path",
        str(DATA),
        "--mode=job",
        "--ads_semantic_embd_path",
        str(EMB / "domain_0"),
        "--web_browsing_semantic_embd_path",
        str(EMB / "domain_1"),
        "--shopping_semantic_embd_path",
        str(EMB / "domain_2"),
        "--ads_pure_corpus_embd_path",
        str(EMB / "domain_3"),
        "--other_semantic_embd_path",
        str(EMB / "domain_4"),
    ]
    f = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env_for(profile), stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    (out / "train.pid").write_text(str(proc.pid))
    return proc, out, log, cfg


def monitor(out: Path) -> Optional[dict]:
    files = sorted(out.glob("*/validation_monitor.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception:
        return None


def latest_batch_from_log(log: Path) -> int:
    if not log.exists():
        return 0
    txt = log.read_text(errors="ignore")[-200000:]
    matches = re.findall(r"batch-stat \(train\): iteration (\d+)", txt)
    return int(matches[-1]) if matches else 0


def latest_state(out: Path, log: Path) -> tuple[int, Optional[str], dict]:
    mon_files = sorted(out.glob("*/validation_monitor.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if mon_files:
        try:
            mon = json.loads(mon_files[0].read_text())
            latest = mon.get("latest") or {}
            return int(latest.get("batch") or 0), str(mon_files[0]), latest
        except Exception:
            pass
    return latest_batch_from_log(log), None, {}


def metrics(entry: Optional[dict]) -> dict:
    if not entry:
        return {}
    m = entry.get("metrics") or {}
    return {
        "batch": entry.get("batch"),
        "overall_ndcg10": entry.get("value") if entry.get("value") is not None else m.get("ndcg_10"),
        "overall_hr10": m.get("hr_10"),
        "ads_hr10": m.get("ads_hr_10"),
        "ads_ndcg10": m.get("ads_ndcg_10"),
        "log_pplx": m.get("log_pplx"),
    }


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(10)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def log_has_oom(log: Path) -> bool:
    if not log.exists():
        return False
    txt = log.read_text(errors="ignore")[-200000:].lower()
    return "out of memory" in txt or "cuda error: out of memory" in txt or "cuda oom" in txt


def run_profile(profile: Profile, max_batch: int, poll_seconds: int) -> dict:
    proc, out, log, cfg = launch(profile)
    print(f"START {profile.name} n={profile.num_negatives} gpu={profile.gpu} pid={proc.pid}", flush=True)
    reason = "process_exited"
    while proc.poll() is None:
        time.sleep(poll_seconds)
        batch, monitor_path, latest = latest_state(out, log)
        print(f"{profile.name} batch={batch}", flush=True)
        if batch >= max_batch:
            reason = f"reached_{max_batch}"
            terminate(proc)
            break
        if log_has_oom(log):
            reason = "cuda_oom"
            terminate(proc)
            break
    mon = monitor(out) or {}
    batch, monitor_path, latest = latest_state(out, log)
    rec = {
        "name": profile.name,
        "profile": asdict(profile),
        "reason": reason,
        "returncode": proc.poll(),
        "latest_batch": batch,
        "monitor": monitor_path,
        "latest": metrics((mon.get("latest") or latest)),
        "best_by_ndcg": metrics(mon.get("best")),
        "config": str(cfg),
        "out": str(out),
        "log": str(log),
        "ended_at": utc(),
    }
    with (RUNNER_DIR / "summary.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"END {profile.name}: {reason}", flush=True)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P28 domain-aware random-negative grid.")
    parser.add_argument("--profiles", default="all", help="Comma-separated profile names or 'all'.")
    parser.add_argument("--max-batch", type=int, default=26000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--allow-small-n96-fallback", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    (RUNNER_DIR / "runner.pid").write_text(str(os.getpid()))
    profiles = PROFILES if args.profiles == "all" else [p for p in PROFILES if p.name in set(args.profiles.split(","))]
    state = {
        "status": "running",
        "started_at": utc(),
        "max_batch": args.max_batch,
        "profiles": [asdict(p) for p in profiles],
        "allow_small_n96_fallback": args.allow_small_n96_fallback,
        "results": [],
    }
    state_path = RUNNER_DIR / "state.json"
    state_path.write_text(json.dumps(state, indent=2))

    for profile in profiles:
        rec = run_profile(profile, args.max_batch, args.poll_seconds)
        state["results"].append(rec)
        state["updated_at"] = utc()
        state_path.write_text(json.dumps(state, indent=2))
        if (
            profile.name == "p28_domain_rand_n96"
            and rec["reason"] == "cuda_oom"
            and args.allow_small_n96_fallback
        ):
            fallback = run_profile(FALLBACK_N96, args.max_batch, args.poll_seconds)
            fallback["fallback_for"] = profile.name
            fallback["architecture_caveat"] = "SMALL MODEL: hstu_encoder.num_blocks=8 and make_model.expert_hidden_dim=96; not directly comparable to full P28 runs."
            state["results"].append(fallback)
            state["updated_at"] = utc()
            state_path.write_text(json.dumps(state, indent=2))

    state["status"] = "done"
    state["ended_at"] = utc()
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
