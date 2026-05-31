#!/usr/bin/env python3
"""ML-owned document/item encoder CLI for LRM-v001 fixed-bank eval.

This command projects selected candidate banks (and optional target-specific
positive/replacement extras discovered from a query cache) into the model's doc
embedding space. It does not score queries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fast_proxy_eval_runner as fast  # noqa: E402
from raw_candidate_bank_cache import RawCandidateBankCache  # noqa: E402
import sequential_submission_infer as seq  # noqa: E402

DOC_CACHE_SCHEMA = "lrm_v001_projected_doc_cache_v001"
DOC_BANK_SCHEMA = "lrm_v001_projected_doc_bank_v001"
ENTRYPOINT = "proposed_2-mmoe_ple/infer/lrm_v001/encode_documents.py"


def _manifest_domains(selected_bank_subset: str) -> dict[int, list[int]]:
    manifest = json.loads(Path(selected_bank_subset).read_text(encoding="utf-8"))
    if manifest.get("schema") != "lrm_v001_selected_bank_subset_v001":
        raise ValueError(f"unsupported selected-bank schema: {manifest.get('schema')}")
    return {int(domain): [int(x) for x in banks] for domain, banks in manifest["domains"].items()}


def _discover_extra_ids(args, *, generator, bank_artifact_cache: dict[int, Any]) -> dict[tuple[int, int], set[str]]:
    """Find per-bank positive/replacement ids that are not in the base bank."""
    extras: dict[tuple[int, int], set[str]] = {}
    if not args.query_cache_dir:
        return extras
    base_cache: dict[tuple[int, int], set[str]] = {}
    for (domain_id, bank_id), batch_paths, _group_meta in fast.iter_query_groups(args.query_cache_dir):
        group = (int(domain_id), int(bank_id))
        bank_artifact = fast.load_bank_artifact(bank_artifact_cache, args=args, generator=generator, domain_id=domain_id)
        base_ids = [str(x) for x in generator.materialize_bank(bank_artifact, int(bank_id), expected_domain_id=int(domain_id))]
        base_set = set(base_ids)
        base_cache[group] = base_set
        group_extras = extras.setdefault(group, set())
        # Load metadata only; query tensor is ignored after CPU load.
        _queries, targets, _checksums, _context_labels, _policy_labels = fast.load_query_group(batch_paths, device="cpu")
        for target in targets:
            pos = str(target["positive_item_id"])
            if pos not in base_set:
                group_extras.add(pos)
                continue
            if args.candidate_check_mode == "none":
                raise RuntimeError(f"positive collision for target {target['target_id']} requires replacements; use --candidate-check-mode collisions/full")
            cand = generator.generate_candidates_for_target(target, bank_artifact)
            fast._check_candidate_result(args, generator, target, cand)
            for rid in getattr(cand, "replacement_item_ids", []) or []:
                rid = str(rid)
                if rid not in base_set:
                    group_extras.add(rid)
    return extras


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="Encode/project selected LRM-v001 candidate banks into model doc embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--bank-root", required=True)
    ap.add_argument("--bank-generator", required=True)
    ap.add_argument("--selected-bank-subset", required=True)
    ap.add_argument("--source-root", default=".")
    ap.add_argument("--gin-config-file", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--embedding-root")
    ap.add_argument("--raw-bank-cache-dir", required=True)
    ap.add_argument("--raw-bank-cache-placement", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--candidate-cache-max-banks", type=int, default=0)
    ap.add_argument("--query-cache-dir", help="optional query cache used to discover positive/replacement extra doc ids")
    ap.add_argument("--output-doc-cache-dir", required=True)
    ap.add_argument("--output-inference-log", required=True)
    ap.add_argument("--doc-dtype", choices=("float32", "float16"), default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-sequence-length", type=int, default=200)
    ap.add_argument("--encode-batch-size", type=int, default=4096)
    ap.add_argument("--candidate-check-mode", choices=("none", "collisions", "full"), default="collisions")
    ap.add_argument("--validate-candidate-generation", action="store_true")
    ap.add_argument("--timing-sync-cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)
    args.output_doc_cache_dir = Path(args.output_doc_cache_dir)
    args.query_cache_dir = Path(args.query_cache_dir) if args.query_cache_dir else None
    args.extra_embedding_chunk_size = args.encode_batch_size
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch  # type: ignore

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1] if ":" in args.device else 0))

    out_root = args.output_doc_cache_dir
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.output_inference_log).parent.mkdir(parents=True, exist_ok=True)

    generator = seq.load_module("banked_candidate_generator_v001", args.bank_generator)
    model = seq.build_model(args)
    model_digest = seq.sha256_file(args.checkpoint_path)
    selected_domains = _manifest_domains(args.selected_bank_subset)
    selected_digest = seq.sha256_file(args.selected_bank_subset)

    raw_cache = RawCandidateBankCache(
        cache_dir=args.raw_bank_cache_dir,
        model_digest=model_digest,
        max_banks=args.candidate_cache_max_banks,
        device=args.device,
        placement=args.raw_bank_cache_placement,
        timing_sync_cuda=args.timing_sync_cuda,
    )
    bank_artifacts: dict[int, Any] = {}
    extras = _discover_extra_ids(args, generator=generator, bank_artifact_cache=bank_artifacts)

    started = time.time()
    banks_meta = []
    with open(args.output_inference_log, "a", encoding="utf-8") as log_f:
        fast._json_dump_line(log_f, {"progress": "doc_encode_start", "entrypoint": ENTRYPOINT, "selected_bank_subset": args.selected_bank_subset, "query_cache_dir": str(args.query_cache_dir) if args.query_cache_dir else None, "at": fast.utc_now()})
        for domain_id, bank_ids in sorted(selected_domains.items()):
            bank_artifact = fast.load_bank_artifact(bank_artifacts, args=args, generator=generator, domain_id=domain_id)
            for bank_id in sorted(bank_ids):
                entry, cache_info = raw_cache.get_bank(
                    model=model,
                    generator_mod=generator,
                    bank_artifact=bank_artifact,
                    domain_id=domain_id,
                    bank_id=bank_id,
                )
                emb = entry["embeddings"].detach().cpu()
                if args.doc_dtype == "float16":
                    emb = emb.to(dtype=torch.float16)
                else:
                    emb = emb.float()
                extra_ids = sorted(extras.get((domain_id, bank_id), set()), key=lambda x: int(x) if str(x).isdigit() else str(x))
                extra_emb = fast._project_item_embeddings(model, extra_ids, chunk_size=args.encode_batch_size, device=args.device).detach().cpu()
                if args.doc_dtype == "float16":
                    extra_emb = extra_emb.to(dtype=torch.float16)
                else:
                    extra_emb = extra_emb.float()
                rel_dir = Path(f"domain_{domain_id}")
                (out_root / rel_dir).mkdir(parents=True, exist_ok=True)
                rel_path = rel_dir / f"bank_{bank_id:04d}.pt"
                payload = {
                    "schema": DOC_BANK_SCHEMA,
                    "domain_id": int(domain_id),
                    "bank_id": int(bank_id),
                    "candidate_ids": [str(x) for x in entry["candidate_ids"]],
                    "embeddings": emb,
                    "extra_ids": extra_ids,
                    "extra_embeddings": extra_emb,
                    "doc_dtype": args.doc_dtype,
                    "model_digest": model_digest,
                }
                tmp = out_root / f"{rel_path}.tmp"
                torch.save(payload, tmp)
                tmp.replace(out_root / rel_path)
                banks_meta.append({"domain_id": int(domain_id), "bank_id": int(bank_id), "candidate_count": len(entry["candidate_ids"]), "extra_count": len(extra_ids), "path": str(rel_path)})
                fast._json_dump_line(log_f, {"progress": "doc_bank_encoded", "domain_id": domain_id, "bank_id": bank_id, "extra_count": len(extra_ids), "raw_cache_event": cache_info.get("event"), "at": fast.utc_now()})
        manifest = {
            "schema": DOC_CACHE_SCHEMA,
            "created_at": fast.utc_now(),
            "entrypoint": ENTRYPOINT,
            "selected_bank_subset": args.selected_bank_subset,
            "selected_bank_subset_digest": selected_digest,
            "raw_bank_cache_dir": args.raw_bank_cache_dir,
            "raw_bank_cache_digest": raw_cache.manifest.get("digest"),
            "query_cache_dir": str(args.query_cache_dir) if args.query_cache_dir else None,
            "checkpoint_path": args.checkpoint_path,
            "model_digest": model_digest,
            "embedding_root": args.embedding_root,
            "doc_dtype": args.doc_dtype,
            "bank_count": len(banks_meta),
            "banks": banks_meta,
            "raw_bank_cache": raw_cache.snapshot(),
            "elapsed_s": time.time() - started,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fast._json_dump_line(log_f, {"progress": "doc_encode_done", "bank_count": len(banks_meta), "output_doc_cache_dir": str(out_root), "elapsed_s": manifest["elapsed_s"], "at": fast.utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
