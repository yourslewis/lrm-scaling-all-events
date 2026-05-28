#!/usr/bin/env python3
"""Safety-gated full-set runner for LRM-v001 submission inference.

The script always performs a bounded burn-in first, estimates full-set runtime and
prediction JSONL size, writes a durable gate report, and only launches full-set
inference when `--auto-proceed-if-sane` is set and all configured thresholds pass.
It does not modify the benchmark artifacts or evaluator contract.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], *, cwd: str | None = None, log_path: Path | None = None) -> int:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            log_f.flush()
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_f.write(line)
            return proc.wait()
    return subprocess.call(cmd, cwd=cwd)


def count_jsonl(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_parquet_rows(glob_expr: str) -> int:
    import pyarrow.parquet as pq  # type: ignore

    paths = sorted(glob.glob(glob_expr))
    if not paths:
        raise FileNotFoundError(f"no target sidecars matched {glob_expr!r}")
    total = 0
    for path in paths:
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def total_target_count(args) -> int:
    if args.target_jsonl:
        return count_jsonl(args.target_jsonl)
    return count_parquet_rows(args.target_sidecar_glob)


def build_infer_cmd(args, *, run_id: str, out_dir: Path, max_targets: int | None, resume: bool = False) -> list[str]:
    script = Path(__file__).with_name("sequential_submission_infer.py")
    cmd = [
        args.python,
        str(script),
        "--benchmark-version", "lrm_benchmark_v001",
        "--history-prefix-source", args.history_prefix_source,
        "--bank-root", args.bank_root,
        "--bank-generator", args.bank_generator,
        "--history-reader", args.history_reader,
        "--source-root", args.source_root,
        "--gin-config-file", args.gin_config_file,
        "--checkpoint-path", args.checkpoint_path,
        "--embedding-root", args.embedding_root,
        "--model-submission-id", args.model_submission_id,
        "--prediction-run-id", run_id,
        "--context-policy", args.context_policy,
        "--output-predictions", str(out_dir / "predictions.jsonl"),
        "--output-inference-log", str(out_dir / "inference_log.jsonl"),
        "--output-target-ids", str(out_dir / "prediction_target_ids.txt"),
        "--device", args.device,
        "--chunk-size", str(args.chunk_size),
        "--target-batch-size", str(args.target_batch_size),
        "--max-sequence-length", str(args.max_sequence_length),
        "--history-batch-size", str(args.history_batch_size),
        "--seed", str(args.seed),
    ]
    if args.target_jsonl:
        cmd += ["--target-jsonl", args.target_jsonl]
    else:
        cmd += ["--target-sidecar-glob", args.target_sidecar_glob]
    if args.target_id_file:
        cmd += ["--target-id-file", args.target_id_file]
    if max_targets is not None:
        cmd += ["--max-targets", str(max_targets)]
    if resume:
        cmd += ["--resume"]
    if args.equivalence_check_targets:
        cmd += ["--equivalence-check-targets", str(args.equivalence_check_targets)]
    return cmd


def build_eval_cmd(args, *, run_dir: Path, output_json: Path) -> list[str] | None:
    if not args.evaluator:
        return None
    if not (args.target_manifest and args.candidate_set_manifest):
        raise SystemExit("--evaluator requires --target-manifest and --candidate-set-manifest")
    return [
        args.python,
        args.evaluator,
        "--target-manifest", args.target_manifest,
        "--candidate-set-manifest", args.candidate_set_manifest,
        "--predictions-jsonl", str(run_dir / "predictions.jsonl"),
        "--target-id-file", str(run_dir / "prediction_target_ids.txt"),
        "--output-json", str(output_json),
        "--omit-per-target",
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-jsonl")
    src.add_argument("--target-sidecar-glob")
    ap.add_argument("--target-id-file")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--history-prefix-source", required=True)
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--history-reader", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root", required=True)
    ap.add_argument("--model-submission-id", required=True)
    ap.add_argument("--context-policy", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--target-batch-size", type=int, default=2048)
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--history-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260526)
    ap.add_argument("--equivalence-check-targets", type=int, default=0)
    ap.add_argument("--burn-in-targets", type=int, default=100)
    ap.add_argument("--max-estimated-hours", type=float, default=24.0)
    ap.add_argument("--max-estimated-jsonl-gb", type=float, default=200.0)
    ap.add_argument("--min-targets-per-hour", type=float, default=1.0)
    ap.add_argument("--auto-proceed-if-sane", action="store_true")
    ap.add_argument("--evaluator")
    ap.add_argument("--target-manifest")
    ap.add_argument("--candidate-set-manifest")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    burn_dir = output_root / f"{args.run_id}.burn_in_{args.burn_in_targets}"
    full_dir = output_root / args.run_id
    report_path = output_root / f"{args.run_id}.safety_gate.json"

    total_targets = total_target_count(args)
    burn_targets = min(args.burn_in_targets, total_targets)
    if burn_targets <= 0:
        raise SystemExit("burn-in target count must be positive")

    started = time.time()
    burn_cmd = build_infer_cmd(args, run_id=f"{args.run_id}.burn_in", out_dir=burn_dir, max_targets=burn_targets)
    rc = run(burn_cmd, log_path=burn_dir / "runner.log")
    elapsed = max(time.time() - started, 1e-9)
    if rc != 0:
        report = {
            "status": "burn_in_failed",
            "at": utc_now(),
            "return_code": rc,
            "burn_in_targets": burn_targets,
            "runner_log": str(burn_dir / "runner.log"),
        }
        write_json(report_path, report)
        return rc

    pred_path = burn_dir / "predictions.jsonl"
    scored = count_jsonl(str(pred_path))
    bytes_written = pred_path.stat().st_size
    targets_per_hour = scored / elapsed * 3600.0
    bytes_per_target = bytes_written / max(scored, 1)
    estimated_hours = total_targets / max(targets_per_hour, 1e-9)
    estimated_gb = (bytes_per_target * total_targets) / (1024 ** 3)
    sane = (
        estimated_hours <= args.max_estimated_hours
        and estimated_gb <= args.max_estimated_jsonl_gb
        and targets_per_hour >= args.min_targets_per_hour
    )
    blockers = []
    if estimated_hours > args.max_estimated_hours:
        blockers.append(f"estimated_hours {estimated_hours:.2f} > threshold {args.max_estimated_hours:.2f}")
    if estimated_gb > args.max_estimated_jsonl_gb:
        blockers.append(f"estimated_jsonl_gb {estimated_gb:.2f} > threshold {args.max_estimated_jsonl_gb:.2f}")
    if targets_per_hour < args.min_targets_per_hour:
        blockers.append(f"targets_per_hour {targets_per_hour:.2f} < threshold {args.min_targets_per_hour:.2f}")

    report: dict[str, Any] = {
        "status": "sane" if sane else "blocked",
        "at": utc_now(),
        "total_targets": total_targets,
        "burn_in_targets_requested": burn_targets,
        "burn_in_targets_scored": scored,
        "burn_in_elapsed_s": elapsed,
        "targets_per_hour": targets_per_hour,
        "bytes_per_target": bytes_per_target,
        "estimated_hours_full_set": estimated_hours,
        "estimated_prediction_jsonl_gb_full_set": estimated_gb,
        "thresholds": {
            "max_estimated_hours": args.max_estimated_hours,
            "max_estimated_jsonl_gb": args.max_estimated_jsonl_gb,
            "min_targets_per_hour": args.min_targets_per_hour,
        },
        "blockers": blockers,
        "auto_proceed_if_sane": bool(args.auto_proceed_if_sane),
    }
    write_json(report_path, report)

    if not sane or not args.auto_proceed_if_sane:
        if sane:
            report["status"] = "sane_not_started_auto_proceed_flag_missing"
            write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if not sane else 0

    full_cmd = build_infer_cmd(args, run_id=args.run_id, out_dir=full_dir, max_targets=None, resume=True)
    rc = run(full_cmd, log_path=full_dir / "runner.log")
    report["full_inference_return_code"] = rc
    report["full_run_dir"] = str(full_dir)
    if rc != 0:
        report["status"] = "full_inference_failed"
        write_json(report_path, report)
        return rc

    eval_cmd = build_eval_cmd(args, run_dir=full_dir, output_json=full_dir / "evaluation_result.json")
    if eval_cmd:
        rc = run(eval_cmd, log_path=full_dir / "evaluator.log")
        report["evaluation_return_code"] = rc
        report["evaluation_result_json"] = str(full_dir / "evaluation_result.json")
        if rc != 0:
            report["status"] = "evaluation_failed"
            write_json(report_path, report)
            return rc
    report["status"] = "full_inference_completed" if not eval_cmd else "full_inference_and_evaluation_completed"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
