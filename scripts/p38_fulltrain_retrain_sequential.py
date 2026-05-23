#!/usr/bin/env python3
"""Sequential full-training-data retrains for selected LRM v3 model recipes.

This runner keeps the model/negative-sampling recipe fixed per profile, switches
training to the full v3-preserve train split, evaluates against the frozen-7 raw
validation shards, and stops each run once the validation transfer score has
peaked for a configured patience window.

Primary early-stop score:
    score = 0.4 * Overall_HR@10 + 0.6 * Ads_HR@10
Fallback if Ads HR is unavailable:
    Trainer.validation_metric_name / Overall NDCG@10

The script is intentionally sequential: it launches one profile, monitors it,
final-evaluates the best checkpoint on frozen-7, writes a table-compatible
summary row, then moves to the next profile.
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
from typing import Any

ROOT = Path(os.environ.get("LRM_ROOT", "/home/yourslewis/lrm-scaling-all-events"))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
OUTROOT = ROOT / "results_v2/fulltrain_retrains_20260522"
DATA_TRAIN_FULL = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve")
DATA_FROZEN7 = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6")
DATA_HYBRID = Path("/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_train_frozen7eval")
EMB = Path("/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve")
PYTHON = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")


@dataclass(frozen=True)
class Profile:
    id: str
    config: str
    summary: str


PROFILES = [
    Profile("P14_latest", "proposed_2-mmoe_ple/config/proposed14_stabilized_group_residual.gin", "Stabilized group residual baseline"),
    Profile("P20_s300_page10_best", "proposed_2-mmoe_ple/config/generated_p20_grid/p20_s300_page10.gin", "P20 ad-anchor grid; sigma=300, PageTitle=1.0"),
    Profile("P23_best", "proposed_2-mmoe_ple/config/generated_p23_coordinate_search/p23_page_s10_p09_m01_o00.gin", "P23 coordinate-search setting; PageTitle=0.9"),
    Profile("P28n64_best", "proposed_2-mmoe_ple/config/generated_p28_domain_random_negatives/p28_domain_rand_n64.gin", "P28 domain-aware random negatives, n=64"),
    Profile("P29b_best", "proposed_2-mmoe_ple/config/generated_p29_mixed_hard_global_negatives/p29b_hardmix_f025_n32.gin", "P29B mixed hard/global negatives, f=0.25, n=32"),
    Profile("P31b_best", "proposed_2-mmoe_ple/config/generated_p31_denoised_hard_negatives/p31b_denoised_hardmix_f035_n32.gin", "P31B denoised hard-negative mix, f=0.35, n=32"),
    Profile("P32a_best", "proposed_2-mmoe_ple/config/generated_p32_hybrid_inbatch_hard_global/p32a_hybrid_inbatch50_hard30_global20_n32.gin", "P32A hybrid inbatch50/hard30/global20, n=32"),
    Profile("P33B_best", "proposed_2-mmoe_ple/config/generated_p33_inbatch_hard_followups/p33b_p23_inbatch90_hard10_n32.gin", "P33B broad inbatch90/hard10, n=32"),
]


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_hybrid_data() -> None:
    DATA_HYBRID.mkdir(parents=True, exist_ok=True)
    for name, target in [
        ("train", DATA_TRAIN_FULL / "train"),
        ("eval", DATA_FROZEN7 / "eval"),
        ("metadata.json", DATA_FROZEN7 / "metadata.json"),
        ("freeze_manifest.json", DATA_FROZEN7 / "freeze_manifest.json"),
    ]:
        path = DATA_HYBRID / name
        if path.exists() or path.is_symlink():
            continue
        path.symlink_to(target, target_is_directory=target.is_dir())


def env_for(gpu: int, port: int, run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["AZUREML_RUN_ID"] = run_id
    env["PYTHONPATH"] = f"{TRAIN}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = "1"
    return env


def launch(profile: Profile, gpu: int, port: int) -> tuple[subprocess.Popen[Any], Path, Path]:
    cfg = ROOT / profile.config
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    run_id = f"fulltrain_{profile.id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUTROOT / profile.id
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        PYTHON,
        "-u",
        "main.py",
        f"--gin_config_file={cfg}",
        f"--output_path={out}",
        "--data_path", str(DATA_HYBRID),
        "--mode=job",
        "--ads_semantic_embd_path", str(EMB / "domain_0"),
        "--web_browsing_semantic_embd_path", str(EMB / "domain_1"),
        "--shopping_semantic_embd_path", str(EMB / "domain_2"),
        "--ads_pure_corpus_embd_path", str(EMB / "domain_3"),
        "--other_semantic_embd_path", str(EMB / "domain_4"),
    ]
    f = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env_for(gpu, port, run_id), stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    (out / "train.pid").write_text(str(proc.pid))
    (out / "launch.json").write_text(json.dumps({"cmd": cmd, "config": str(cfg), "data_path": str(DATA_HYBRID), "started_at": utc()}, indent=2))
    return proc, out, log


def terminate(proc: subprocess.Popen[Any], grace: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    deadline = time.time() + grace
    while proc.poll() is None and time.time() < deadline:
        time.sleep(1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def latest_monitor(out: Path) -> tuple[Path | None, dict[str, Any]]:
    files = sorted(out.glob("*/validation_monitor.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, {}
    try:
        return files[0], json.loads(files[0].read_text())
    except Exception:
        return files[0], {}


def latest_batch_from_log(log: Path) -> int:
    if not log.exists():
        return 0
    txt = log.read_text(errors="ignore")[-200000:]
    matches = re.findall(r"batch-stat \(train\): iteration (\d+)", txt)
    return int(matches[-1]) if matches else 0


def score(entry: dict[str, Any]) -> float | None:
    metrics = entry.get("metrics") or {}
    ohr = metrics.get("hr_10")
    ahr = metrics.get("ads_hr_10")
    if ohr is not None and ahr is not None:
        return 0.4 * float(ohr) + 0.6 * float(ahr)
    value = entry.get("value") if entry.get("value") is not None else metrics.get("ndcg_10")
    return None if value is None else float(value)


def metrics_for(entry: dict[str, Any]) -> dict[str, Any]:
    m = entry.get("metrics") or {}
    return {
        "batch": entry.get("batch"),
        "score": score(entry),
        "OHR": m.get("hr_10"),
        "AHR": m.get("ads_hr_10"),
        "O_NDCG": m.get("ndcg_10"),
        "A_NDCG": m.get("ads_ndcg_10"),
    }


def monitor_run(proc: subprocess.Popen[Any], out: Path, log: Path, min_batch: int, patience_batches: int, poll_seconds: int, max_batch: int) -> dict[str, Any]:
    best_score = None
    best_batch = 0
    reason = "process_exited"
    while proc.poll() is None:
        time.sleep(poll_seconds)
        mon_path, mon = latest_monitor(out)
        latest = mon.get("latest") or {}
        batch = int((latest or {}).get("batch") or latest_batch_from_log(log) or 0)
        current = score(latest) if latest else None
        if current is not None and (best_score is None or current > best_score + 1e-6):
            best_score = current
            best_batch = batch
        print(json.dumps({"time": utc(), "batch": batch, "score": current, "best_score": best_score, "best_batch": best_batch, "monitor": str(mon_path) if mon_path else None}), flush=True)
        if batch >= max_batch:
            reason = f"reached_max_batch_{max_batch}"
            terminate(proc)
            break
        if batch >= min_batch and best_batch and batch - best_batch >= patience_batches:
            reason = f"early_stop_peaked_after_{patience_batches}_batches"
            terminate(proc)
            break
    mon_path, mon = latest_monitor(out)
    latest = mon.get("latest") or {}
    best = mon.get("best") or latest
    return {
        "reason": reason,
        "returncode": proc.poll(),
        "monitor": str(mon_path) if mon_path else None,
        "latest": metrics_for(latest),
        "best_monitor": metrics_for(best),
        "log": str(log),
    }


def find_best_checkpoint(out: Path, monitor_summary: dict[str, Any]) -> Path | None:
    mon_path = monitor_summary.get("monitor")
    if mon_path:
        try:
            mon = json.loads(Path(mon_path).read_text())
            ckpt = ((mon.get("best") or {}).get("checkpoint"))
            if ckpt and Path(ckpt).exists():
                return Path(ckpt)
        except Exception:
            pass
    ckpts = sorted(out.glob("*/ckpts/best_checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]
    ckpts = sorted(out.glob("*/ckpts/checkpoint_batch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def final_eval(profile: Profile, cfg: Path, ckpt: Path, gpu: int, port: int) -> dict[str, Any]:
    out_json = OUTROOT / profile.id / f"{profile.id}_frozen7_full_eval.json"
    log = OUTROOT / profile.id / "logs" / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={port}", "evaluate_per_domain.py",
        f"--gin_config_file={cfg}", f"--checkpoint_path={ckpt}", f"--data_path={DATA_HYBRID}",
        f"--ads_semantic_embd_path={EMB / 'domain_0'}", f"--web_browsing_semantic_embd_path={EMB / 'domain_1'}",
        f"--shopping_semantic_embd_path={EMB / 'domain_2'}", f"--ads_pure_corpus_embd_path={EMB / 'domain_3'}",
        f"--other_semantic_embd_path={EMB / 'domain_4'}", "--max_eval_batches=1000000", "--eval_batch_size=16",
        "--mode=job", f"--output_json={out_json}",
    ]
    env = env_for(gpu, port, f"fulltrain_eval_{profile.id}")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=TRAIN, env=env, stdout=f, stderr=subprocess.STDOUT)
    result = {"returncode": rc, "output_json": str(out_json), "log": str(log)}
    if out_json.exists():
        raw = json.loads(out_json.read_text())
        overall = raw.get("overall") or {}
        per = raw.get("per_domain") or {}
        ads = per.get("Ads") or {}
        result.update({
            "OHR": overall.get("hr_10"),
            "AHR": ads.get("hr_10", overall.get("ads_hr_10")),
            "O_NDCG": overall.get("ndcg_10"),
            "A_NDCG": ads.get("ndcg_10", overall.get("ads_ndcg_10")),
        })
    return result


def write_table_rows(records: list[dict[str, Any]]) -> None:
    rows = []
    for r in records:
        ev = r.get("final_eval") or {}
        rows.append({
            "id": r["id"],
            "label": "full-training-data retrain on frozen-7",
            "OHR": ev.get("OHR"),
            "AHR": ev.get("AHR"),
            "O_NDCG": ev.get("O_NDCG"),
            "A_NDCG": ev.get("A_NDCG"),
            "source": ev.get("output_json"),
            "rows": 40106,
            "ads": 2524,
            "note": r.get("stop", {}).get("reason"),
        })
    path = OUTROOT / "lrm_v3_fulltrain_eval_metrics.json"
    path.write_text(json.dumps({"updated_at_utc": utc(), "rows": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="*", default=[p.id for p in PROFILES])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=29840)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--min-batch", type=int, default=12000)
    parser.add_argument("--patience-batches", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=70000)
    args = parser.parse_args()

    ensure_hybrid_data()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    records_path = OUTROOT / "state.json"
    selected = [p for p in PROFILES if p.id in set(args.profiles)]
    records: list[dict[str, Any]] = []
    if records_path.exists():
        try:
            records = json.loads(records_path.read_text()).get("records", [])
        except Exception:
            records = []
    completed = {r.get("id") for r in records if r.get("status") == "done"}

    for idx, profile in enumerate(selected):
        if profile.id in completed:
            continue
        cfg = ROOT / profile.config
        proc, out, log = launch(profile, args.gpu, args.port + idx * 2)
        stop = monitor_run(proc, out, log, args.min_batch, args.patience_batches, args.poll_seconds, args.max_batch)
        ckpt = find_best_checkpoint(out, stop)
        final = final_eval(profile, cfg, ckpt, args.gpu, args.port + idx * 2 + 1) if ckpt else {"error": "no_checkpoint"}
        rec = {"id": profile.id, "summary": profile.summary, "config": str(cfg), "out": str(out), "checkpoint": str(ckpt) if ckpt else None, "stop": stop, "final_eval": final, "status": "done", "updated_at_utc": utc()}
        records.append(rec)
        records_path.write_text(json.dumps({"updated_at_utc": utc(), "data_path": str(DATA_HYBRID), "records": records}, indent=2))
        write_table_rows(records)


if __name__ == "__main__":
    main()
