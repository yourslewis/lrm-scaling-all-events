"""
Approach-agnostic evaluation: compute NDCG@10 and HR@10 per event group.

Expected input: a JSON-lines file where each line has:
  {
    "user_id": str,
    "target_type": str,           # event type of the target item
    "target_rank": int,           # rank of the true next item (1-indexed, 0 if not in top-K)
    "top_k": int                  # K used for retrieval
  }

Usage:
    python eval_by_group.py --input results.jsonl --k 10
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List

# Event groups
GROUP_MAP = {
    "SearchClick": "Ad",
    "NativeClick": "Ad",
    "EdgePageTitle": "Browsing",
    "MSN": "Browsing",
    "ChromePageTitle": "Browsing",
    "UET": "Browsing",
    "UETShoppingView": "Browsing",
    "OrganicSearchQuery": "Search",
    "EdgeSearchQuery": "Search",
    "UETShoppingCart": "Purchase",
    "AbandonCart": "Purchase",
    "EdgeShoppingCart": "Purchase",
    "EdgeShoppingPurchase": "Purchase",
    "OutlookSenderDomain": "Others",
}


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG@K for a single item. rank=0 means not found."""
    if rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def hr_at_k(rank: int, k: int) -> float:
    """Hit Rate @K for a single item."""
    return 1.0 if 0 < rank <= k else 0.0


def mrr(rank: int) -> float:
    """Mean Reciprocal Rank for a single item."""
    return 1.0 / rank if rank > 0 else 0.0


def evaluate(input_path: str, k: int = 10) -> Dict[str, Dict[str, float]]:
    group_ndcg: Dict[str, List[float]] = defaultdict(list)
    group_hr: Dict[str, List[float]] = defaultdict(list)
    group_mrr: Dict[str, List[float]] = defaultdict(list)

    with open(input_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            group = GROUP_MAP.get(rec["target_type"], "Others")
            rank = rec["target_rank"]
            group_ndcg[group].append(ndcg_at_k(rank, k))
            group_hr[group].append(hr_at_k(rank, k))
            group_mrr[group].append(mrr(rank))

    results = {}
    all_ndcg, all_hr, all_mrr = [], [], []
    for group in ["Ad", "Purchase", "Browsing", "Search", "Others"]:
        if group not in group_ndcg:
            continue
        n = len(group_ndcg[group])
        avg_ndcg = sum(group_ndcg[group]) / n
        avg_hr = sum(group_hr[group]) / n
        avg_mrr = sum(group_mrr[group]) / n
        results[group] = {
            "NDCG@10": round(avg_ndcg, 4),
            "HR@10": round(avg_hr, 4),
            "MRR": round(avg_mrr, 4),
            "count": n,
        }
        all_ndcg.extend(group_ndcg[group])
        all_hr.extend(group_hr[group])
        all_mrr.extend(group_mrr[group])

    if all_ndcg:
        results["Overall"] = {
            "NDCG@10": round(sum(all_ndcg) / len(all_ndcg), 4),
            "HR@10": round(sum(all_hr) / len(all_hr), 4),
            "MRR": round(sum(all_mrr) / len(all_mrr), 4),
            "count": len(all_ndcg),
        }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL file with evaluation records")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    results = evaluate(args.input, args.k)

    # Pretty print
    print(f"\n{'Group':<12} {'NDCG@10':>8} {'HR@10':>8} {'MRR':>8} {'Count':>8}")
    print("-" * 48)
    for group in ["Ad", "Purchase", "Browsing", "Search", "Others", "Overall"]:
        if group in results:
            r = results[group]
            print(f"{group:<12} {r['NDCG@10']:8.4f} {r['HR@10']:8.4f} {r['MRR']:8.4f} {r['count']:8d}")

    # Save as JSON
    output_path = args.input.replace(".jsonl", "_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
