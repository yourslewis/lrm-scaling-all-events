#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("LRM_ROOT", Path(__file__).resolve().parents[1]))
TRAIN = ROOT / "proposed_2-mmoe_ple/train"
DATA = Path(os.environ.get("LRM_DATA", "/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2"))
EMB = Path(os.environ.get("LRM_EMB", "/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings"))
CFGDIR = ROOT / "proposed_2-mmoe_ple/config/generated_p29_mixed_hard_global_negatives"
OUTROOT = ROOT / "results_v2"
RUNNER_DIR = OUTROOT / "p29_mixed_hard_global_negatives_autoresearch"
PY = os.environ.get("LRM_PYTHON", "/home/yourslewis/miniconda3/envs/hstu/bin/python3.10")
MAX_BATCH = int(os.environ.get("P29_MAX_BATCH", "26000"))
POLL = int(os.environ.get("P29_POLL_SECONDS", "60"))
MAX_PARALLEL = int(os.environ.get("P29_MAX_PARALLEL", "2"))
RUNS = [
    ("p29a_hardmix_f010_n32", 0.10),
    ("p29b_hardmix_f025_n32", 0.25),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_cmd(run: str, cfg: Path, out: Path) -> list[str]:
    return [
        PY,
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


def env_for(gpu: int, port: int, run: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["AZUREML_RUN_ID"] = f"{run}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
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
    env["MASTER_PORT"] = str(port)
    return env


def monitor_path(out: Path) -> Path | None:
    files = sorted(out.glob("*/validation_monitor.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def latest(out: Path) -> tuple[int, str | None, dict]:
    path = monitor_path(out)
    if path:
        try:
            data = json.loads(path.read_text())
            latest_item = data.get("latest") or {}
            return int(latest_item.get("batch") or 0), str(path), latest_item
        except Exception:
            pass
    logs = sorted((out / "logs").glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        text = logs[0].read_text(errors="ignore")[-200000:]
        matches = re.findall(r"batch-stat \(train\): iteration (\d+)", text)
        if matches:
            return int(matches[-1]), None, {}
    return 0, None, {}


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


def start_run(spec: tuple[str, float], gpu: int, slot: int):
    run, hard_fraction = spec
    cfg = CFGDIR / f"{run}.gin"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    out = OUTROOT / run
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    handle = open(log, "w")
    proc = subprocess.Popen(
        make_cmd(run, cfg, out),
        cwd=TRAIN,
        env=env_for(gpu, 33000 + slot + int(hard_fraction * 100), run),
        stdout=handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    (out / "train.pid").write_text(str(proc.pid))
    rec = {
        "name": run,
        "hard_fraction": hard_fraction,
        "gpu": gpu,
        "pid": proc.pid,
        "cfg": str(cfg),
        "out": str(out),
        "log": str(log),
        "status": "running",
        "started_at": utc(),
    }
    return rec, proc, handle


def main() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    (RUNNER_DIR / "runner.pid").write_text(str(os.getpid()))
    queue = RUNS.copy()
    active = []
    finished = []
    state_path = RUNNER_DIR / "state.json"
    state = {
        "status": "running",
        "max_batch": MAX_BATCH,
        "max_parallel": MAX_PARALLEL,
        "runner_pid": os.getpid(),
        "root": str(ROOT),
        "queue": queue,
        "active": [],
        "finished": [],
        "started_at": utc(),
        "note": "P29 mixed hard global negatives; eval logic unchanged.",
    }

    def write_state() -> None:
        state["active"] = [rec for rec, _, __ in active]
        state["finished"] = finished
        state["queue"] = queue
        state["updated_at"] = utc()
        state_path.write_text(json.dumps(state, indent=2, default=str))

    write_state()
    while queue or active:
        while queue and len(active) < MAX_PARALLEL:
            used = {rec["gpu"] for rec, _, __ in active}
            gpu = 0 if 0 not in used else 1
            rec, proc, handle = start_run(queue.pop(0), gpu=gpu, slot=len(active))
            active.append((rec, proc, handle))
            write_state()
            print(f"started {rec['name']} pid={proc.pid} gpu={gpu}", flush=True)

        time.sleep(POLL)
        new_active = []
        for rec, proc, handle in active:
            batch, monitor, latest_item = latest(Path(rec["out"]))
            rec.update(
                {
                    "latest_batch": batch,
                    "monitor": monitor,
                    "latest": latest_item,
                    "returncode": proc.poll(),
                    "updated_at": utc(),
                }
            )
            if proc.poll() is None and batch >= MAX_BATCH:
                rec["reason"] = f"reached_{MAX_BATCH}"
                terminate(proc)
                rec["returncode"] = proc.poll()
            if proc.poll() is None:
                new_active.append((rec, proc, handle))
                continue
            handle.close()
            rec.setdefault("reason", "process_exited")
            rec["ended_at"] = utc()
            rec["status"] = "done" if rec.get("latest_batch", 0) >= MAX_BATCH else "failed"
            finished.append(rec)
        active = new_active
        write_state()
        print(
            json.dumps(
                {
                    "active": {rec["name"]: rec.get("latest_batch") for rec, _, __ in active},
                    "finished": [(rec["name"], rec.get("status"), rec.get("latest_batch")) for rec in finished],
                    "queue": queue,
                }
            ),
            flush=True,
        )

    state["status"] = "done"
    state["ended_at"] = utc()
    write_state()
    print(json.dumps(state, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
