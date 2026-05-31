#!/usr/bin/env python3
"""Create deterministic selected-bank subset manifests for v001 proxy evals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def stable_sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--banks-per-domain", type=int, default=100)
    ap.add_argument("--domains", default="0,1,2,3,4")
    ap.add_argument("--total-banks-per-domain", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--strategy", choices=["uniform_random_per_domain", "first_n_per_domain"], default="uniform_random_per_domain")
    args = ap.parse_args()

    domains = [int(x) for x in args.domains.split(",") if x.strip()]
    if args.banks_per_domain <= 0 or args.banks_per_domain > args.total_banks_per_domain:
        raise SystemExit("--banks-per-domain must be in [1,total-banks-per-domain]")

    rng = random.Random(args.seed)
    selected: dict[str, list[int]] = {}
    for domain_id in domains:
        if args.strategy == "first_n_per_domain":
            ids = list(range(args.banks_per_domain))
        else:
            ids = sorted(rng.sample(range(args.total_banks_per_domain), args.banks_per_domain))
        selected[str(domain_id)] = ids

    manifest = {
        "schema": "lrm_v001_selected_bank_subset_v001",
        "strategy": args.strategy,
        "seed": args.seed,
        "banks_per_domain": args.banks_per_domain,
        "total_banks_per_domain": args.total_banks_per_domain,
        "domains": selected,
    }
    manifest["digest"] = stable_sha256_json(manifest)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "digest": manifest["digest"], "total_selected_banks": sum(len(v) for v in selected.values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
