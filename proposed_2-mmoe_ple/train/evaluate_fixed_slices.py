#!/usr/bin/env python3
"""Fixed-target position-slice evaluator for P40.

Evaluates the frozen all-event rows on target positions instead of one final
label per row. The evaluator intentionally keeps unsupported model/slice cells
as a reporting concern; it never changes the target denominator for a model.

Metrics emitted:
  cold_ads:    first Ads target per user with non-empty full prefix
  warm_ads:    Ads targets after the first Ads target
  all_ads:     all valid Ads targets
  all_domain:  every valid target position across event types

For each slice we report micro_hr_10 and macro_hr_10. Ads slices are AHR;
all_domain is OHR.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gin
import pandas as pd
import torch
import torch.distributed as dist

import fbgemm_gpu  # noqa: F401

from data.reco_dataset import get_reco_dataset
from data.ads_datasets.collate import CollateFn
from trainer.seeding import get_gin_configured_seed, seed_everything
from trainer.util import make_model
from trainer.train import Trainer  # noqa: F401  # gin registration side effects

AD_TYPE_IDS = {1, 2}  # NativeClick, SearchClick
EVENT_TYPE_DICT = {
    "UNK": 0,
    "NativeClick": 1,
    "SearchClick": 2,
    "EdgePageTitle": 3,
    "EdgeSearchQuery": 4,
    "OrganicSearchQuery": 5,
    "UET": 6,
    "OutlookSenderDomain": 7,
    "UETShoppingCart": 8,
    "UETShoppingView": 9,
    "AbandonCart": 10,
    "EdgeShoppingCart": 11,
    "EdgeShoppingPurchase": 12,
}


@dataclass
class SliceStats:
    hits: int = 0
    targets: int = 0
    user_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    user_targets: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, user_ids: List[str], hit_tensor: torch.Tensor) -> None:
        hits = hit_tensor.to(torch.int64).detach().cpu().tolist()
        self.hits += int(sum(hits))
        self.targets += len(hits)
        for uid, h in zip(user_ids, hits):
            self.user_hits[uid] += int(h)
            self.user_targets[uid] += 1

    def as_dict(self) -> Dict[str, float | int]:
        user_rates = [self.user_hits[u] / c for u, c in self.user_targets.items() if c > 0]
        return {
            "targets": self.targets,
            "users": len(self.user_targets),
            "micro_hr_10": (self.hits / self.targets) if self.targets else None,
            "macro_hr_10": (sum(user_rates) / len(user_rates)) if user_rates else None,
        }


def iter_target_windows(
    parquet_files: List[Path],
    max_sequence_length: int,
    include_slices: set[str],
) -> Iterable[Dict[str, object]]:
    for fp in parquet_files:
        df = pd.read_parquet(fp, columns=["user_id", "encoded_ids", "types", "timestamps_unix"])
        for row in df.itertuples(index=False):
            user_id = str(row.user_id)
            ids = list(row.encoded_ids)
            types = list(row.types)
            timestamps = list(row.timestamps_unix)
            type_ids = [EVENT_TYPE_DICT.get(t, 0) for t in types]
            seen_ads = 0
            for idx, target_type_id in enumerate(type_ids):
                if idx == 0:
                    if target_type_id in AD_TYPE_IDS:
                        seen_ads += 1
                    continue
                is_ads = target_type_id in AD_TYPE_IDS
                slices: List[str] = []
                if "all_domain" in include_slices:
                    slices.append("all_domain")
                if is_ads:
                    if seen_ads == 0:
                        if "cold_ads" in include_slices:
                            slices.append("cold_ads")
                    else:
                        if "warm_ads" in include_slices:
                            slices.append("warm_ads")
                    if "all_ads" in include_slices:
                        slices.append("all_ads")
                if is_ads:
                    seen_ads += 1
                if not slices:
                    continue
                start = max(0, idx - max_sequence_length)
                yield {
                    "user_id": user_id,
                    "past_ids": ids[start:idx],
                    "past_types": type_ids[start:idx],
                    "past_timestamps": timestamps[start:idx],
                    "target_id": ids[idx],
                    "target_type": target_type_id,
                    "slices": slices,
                }


def collate_windows(batch: List[Dict[str, object]], max_sequence_length: int, device: torch.device):
    bsz = len(batch)
    ids = torch.zeros((bsz, max_sequence_length), dtype=torch.long, device=device)
    type_ids = torch.zeros((bsz, max_sequence_length), dtype=torch.long, device=device)
    timestamps = torch.zeros((bsz, max_sequence_length), dtype=torch.long, device=device)
    lengths = torch.zeros((bsz,), dtype=torch.long, device=device)
    label_ids = torch.zeros((bsz,), dtype=torch.long, device=device)
    label_types = torch.zeros((bsz,), dtype=torch.long, device=device)
    user_ids: List[str] = []
    slice_lists: List[List[str]] = []
    for i, item in enumerate(batch):
        past_ids = list(item["past_ids"])[-max_sequence_length:]
        past_types = list(item["past_types"])[-max_sequence_length:]
        past_ts = list(item["past_timestamps"])[-max_sequence_length:]
        l = len(past_ids)
        ids[i, :l] = torch.tensor(past_ids, dtype=torch.long, device=device)
        type_ids[i, :l] = torch.tensor(past_types, dtype=torch.long, device=device)
        timestamps[i, :l] = torch.tensor(past_ts, dtype=torch.long, device=device)
        lengths[i] = l
        label_ids[i] = int(item["target_id"])
        label_types[i] = int(item["target_type"])
        user_ids.append(str(item["user_id"]))
        slice_lists.append(list(item["slices"]))
    ratings = torch.full((bsz,), -1, dtype=torch.long, device=device)
    return ids, type_ids, timestamps, lengths, ratings, label_ids, label_types, user_ids, slice_lists


@torch.inference_mode()
def score_batch(model, collate_fn: CollateFn, batch, max_sequence_length: int, num_negatives: int, device: torch.device):
    ids, type_ids, timestamps, lengths, ratings, label_ids, label_types, user_ids, slice_lists = collate_windows(batch, max_sequence_length, device)
    raw_input_embeddings = collate_fn.embedding_module.get_raw_item_embeddings(ids)
    raw_label_embeddings = collate_fn.embedding_module.get_raw_item_embeddings(label_ids.view(-1, 1)).squeeze(1)

    past_embeddings = model.model._embedding_module(raw_input_embeddings)
    label_embeddings = model.negatives_sampler["eval"].normalize_embeddings(
        model.model._embedding_module(raw_label_embeddings)
    )
    query_embeddings = model.model.encode(
        past_lengths=lengths,
        past_ids=ids,
        past_embeddings=past_embeddings,
        past_payloads={"timestamps": timestamps, "ratings": ratings, "type_ids": type_ids},
    )
    sampled_ids, sampled_embs = model.negatives_sampler["eval"](
        positive_ids=label_ids,
        num_to_sample=num_negatives,
        supervision_type_ids=label_types,
    )
    neg_scores = torch.einsum("bd,bkd->bk", query_embeddings, sampled_embs)
    pos_scores = (query_embeddings * label_embeddings).sum(-1)
    ranks = 1 + (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1)
    hits10 = ranks <= 10
    return hits10, user_ids, slice_lists


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gin_config_file", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--ads_semantic_embd_path", required=True)
    ap.add_argument("--web_browsing_semantic_embd_path", required=True)
    ap.add_argument("--shopping_semantic_embd_path", required=True)
    ap.add_argument("--ads_pure_corpus_embd_path", required=True)
    ap.add_argument("--other_semantic_embd_path", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--mode", default="job")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_negatives", type=int, default=10000)
    ap.add_argument("--max_sequence_length", type=int, default=200)
    ap.add_argument("--max_targets_per_slice", type=int, default=0, help="0 means no cap")
    ap.add_argument("--slices", nargs="*", default=["cold_ads", "warm_ads", "all_ads", "all_domain"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    include_slices = set(args.slices)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    gin.parse_config_file(args.gin_config_file)
    seed_everything(get_gin_configured_seed(), rank=rank, log_prefix="evaluate_fixed_slices")

    dataset = get_reco_dataset(mode=args.mode, path=args.data_path, chronological=True, rank=rank, world_size=world_size)
    precomputed = {
        0: args.ads_semantic_embd_path,
        1: args.web_browsing_semantic_embd_path,
        2: args.shopping_semantic_embd_path,
        3: args.ads_pure_corpus_embd_path,
        4: args.other_semantic_embd_path,
    }
    model = make_model(dataset=dataset, precomputed_embeddings_domain_to_dir=precomputed)
    snapshot = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"])
    model = model.to(device).eval()
    for sampler in model.negatives_sampler.values():
        if hasattr(sampler, "rotate"):
            sampler.rotate()

    collate_fn = CollateFn(
        device=device,
        domain_to_item_id_range=dataset.domain_to_item_id_range,
        precomputed_embeddings_domain_to_dir=precomputed,
        domain_offset=dataset.domain_offset,
        shard_size=dataset.shard_size,
    )
    collate_fn._init_embedding_module()

    stats = {s: SliceStats() for s in ["cold_ads", "warm_ads", "all_ads", "all_domain"] if s in include_slices}
    parquet_files = sorted(Path(args.data_path, "eval").glob("*.parquet"))
    batch: List[Dict[str, object]] = []
    start = time.time()
    processed = 0
    for item in iter_target_windows(parquet_files, args.max_sequence_length, include_slices):
        if args.max_targets_per_slice:
            # Drop target if every slice it contributes to has reached cap.
            if all(stats[s].targets >= args.max_targets_per_slice for s in item["slices"] if s in stats):
                continue
        batch.append(item)
        if len(batch) < args.batch_size:
            continue
        hits10, user_ids, slice_lists = score_batch(model, collate_fn, batch, args.max_sequence_length, args.num_negatives, device)
        for idx, slices in enumerate(slice_lists):
            for s in slices:
                if s in stats and (not args.max_targets_per_slice or stats[s].targets < args.max_targets_per_slice):
                    stats[s].add([user_ids[idx]], hits10[idx:idx+1])
        processed += len(batch)
        if processed % 10000 == 0 and rank == 0:
            logging.info("processed target windows=%d stats=%s", processed, {k: v.targets for k, v in stats.items()})
        batch = []
        if args.max_targets_per_slice and all(v.targets >= args.max_targets_per_slice for v in stats.values()):
            break
    if batch:
        hits10, user_ids, slice_lists = score_batch(model, collate_fn, batch, args.max_sequence_length, args.num_negatives, device)
        for idx, slices in enumerate(slice_lists):
            for s in slices:
                if s in stats and (not args.max_targets_per_slice or stats[s].targets < args.max_targets_per_slice):
                    stats[s].add([user_ids[idx]], hits10[idx:idx+1])

    out = {
        "checkpoint_path": args.checkpoint_path,
        "data_path": args.data_path,
        "max_targets_per_slice": args.max_targets_per_slice,
        "num_negatives": args.num_negatives,
        "elapsed_seconds": time.time() - start,
        "slices": {k: v.as_dict() for k, v in stats.items()},
    }
    if rank == 0:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
