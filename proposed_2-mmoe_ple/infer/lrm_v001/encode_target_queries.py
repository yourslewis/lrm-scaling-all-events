#!/usr/bin/env python3
"""ML-owned target/query encoder CLI for LRM-v001 fixed-bank eval.

This command materializes target histories, runs the model once per target, and
writes grouped query embeddings. It intentionally does not score candidates.
Testing/scoring code should consume the query-cache artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fast_proxy_eval_runner as fast  # noqa: E402
import sequential_submission_infer as seq  # noqa: E402

ENTRYPOINT = "proposed_2-mmoe_ple/infer/lrm_v001/encode_target_queries.py"


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Encode LRM-v001 target/user histories into grouped query embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-jsonl")
    group.add_argument("--target-sidecar-glob")
    ap.add_argument("--target-id-file")
    ap.add_argument("--selected-bank-subset", required=True)
    ap.add_argument("--max-targets", type=int)
    ap.add_argument("--target-batch-size", type=int, default=200000)
    ap.add_argument("--resume", action="store_true", help="accepted for iter_target_batches compatibility")

    ap.add_argument("--history-prefix-source", required=True)
    ap.add_argument("--history-reader", required=True)
    ap.add_argument("--source-root", default=".")
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root")
    ap.add_argument("--context-policy", required=True)

    ap.add_argument("--output-query-cache-dir", required=True)
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--query-dtype", choices=("float32", "float16"), default="float32")
    ap.add_argument("--encode-batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--history-batch-size", type=int, default=2048)
    ap.add_argument("--context-checksum-mode", choices=("none", "contract"), default="none")
    ap.add_argument("--validate-context-count", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-flush-every", type=int, default=100)
    ap.add_argument("--stdout-progress-every", type=int, default=10000)
    ap.add_argument("--force-rebuild-query-cache", action="store_true")
    args = ap.parse_args(argv)

    args.query_cache_dir = Path(args.output_query_cache_dir)
    args.query_cache_batch_size = int(args.encode_batch_size)
    args.query_cache_dtype = args.query_dtype
    args.output_compact = None
    args._selected_bank_filter = seq.load_selected_bank_subset(args.selected_bank_subset)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seq.configure_log_flush(args.log_flush_every)
    import torch  # type: ignore

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    if args.force_rebuild_query_cache and args.query_cache_dir.exists():
        import shutil

        shutil.rmtree(args.query_cache_dir)

    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)
    reader_mod = seq.load_module("history_prefix_reader_v001", args.history_reader)
    model = seq.build_model(args)
    history_reader = reader_mod.HistoryPrefixReader.open(
        canonical_root=args.history_prefix_source,
        split="eval",
        mode="eval_inference",
    )
    model_digest = seq.sha256_file(args.checkpoint_path)
    context_policy_digest = seq.sha256_file(args.context_policy)

    with open(args.output_inference_log, "a", encoding="utf-8") as log_f:
        fast._json_dump_line(
            log_f,
            {
                "progress": "query_encode_start",
                "entrypoint": ENTRYPOINT,
                "target_source": args.target_jsonl or args.target_sidecar_glob,
                "selected_bank_subset": args.selected_bank_subset,
                "selected_bank_count": sum(len(v) for v in (args._selected_bank_filter or {}).values()),
                "query_cache_dir": str(args.query_cache_dir),
                "query_dtype": args.query_dtype,
                "context_checksum_mode": args.context_checksum_mode,
                "at": fast.utc_now(),
            },
        )
        manifest = fast.build_query_cache(
            args,
            model=model,
            reader_mod=reader_mod,
            history_reader=history_reader,
            model_digest=model_digest,
            context_policy_digest=context_policy_digest,
            log_f=log_f,
        )
        manifest["entrypoint"] = ENTRYPOINT
        (args.query_cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fast._json_dump_line(log_f, {"progress": "query_encode_done", "target_count": manifest.get("target_count"), "group_count": manifest.get("group_count"), "query_cache_dir": str(args.query_cache_dir), "at": fast.utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
