#!/usr/bin/env python3
"""
Standalone per-event-type evaluation script.
Loads a trained model checkpoint, runs eval on the eval dataset,
and reports NDCG@10, HR@10, MRR per event type and per group.

Usage:
  cd <model_dir>/train
  python /home/yourslewis/lrm-scaling-all-events/eval/eval_per_event_type.py \
      --gin_config_file <config.gin> \
      --data_path <data_dir> \
      --ckpt_path <checkpoint.pt> \
      --mode job \
      --ads_semantic_embd_path <embd_dir>/domain_0 \
      [--web_browsing_semantic_embd_path ...] \
      [--shopping_semantic_embd_path ...] \
      [--ads_pure_corpus_embd_path ...] \
      [--other_semantic_embd_path ...] \
      --eval_batches 100
"""

import logging
import os
import sys
import json
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Add cwd to path so we can import from the model's train/ directory
sys.path.insert(0, os.getcwd())

import fbgemm_gpu  # noqa
import gin
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import absl
from absl import app, flags

# Setup logging
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity('info')
absl.logging.set_stderrthreshold('info')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.addHandler(absl.logging.get_absl_handler())
logging.root.setLevel(logging.INFO)
logging.getLogger('azure').setLevel(logging.ERROR)

from data.reco_dataset import get_reco_dataset
from data.ads_datasets.collate import CollateFn
from trainer.util import make_model
from trainer.data_loader import create_data_loader
from indexing.utils import get_top_k_module

# ---------- EVENT TYPE MAPS ----------
EVENT_TYPE_NAMES = {
    0: "UNK", 1: "NativeClick", 2: "SearchClick", 3: "EdgePageTitle",
    4: "EdgeSearchQuery", 5: "OrganicSearchQuery", 6: "UET",
    7: "OutlookSenderDomain", 8: "UETShoppingCart", 9: "UETShoppingView",
    10: "AbandonCart", 11: "EdgeShoppingCart", 12: "EdgeShoppingPurchase",
}
GROUP_MAP = {
    1: "Ad", 2: "Ad",
    3: "Browsing", 6: "Browsing", 9: "Browsing",
    4: "Search", 5: "Search",
    8: "Purchase", 10: "Purchase", 11: "Purchase", 12: "Purchase",
    7: "Others",
}


def delete_flags(FLAGS, keys):
    existing = [key for key in FLAGS._flags()]
    for key in keys:
        if key in existing:
            delattr(FLAGS, key)

delete_flags(flags.FLAGS, [
    "gin_config_file", "data_path", "ckpt_path", "mode", "eval_batches",
    "ads_semantic_embd_path", "web_browsing_semantic_embd_path",
    "shopping_semantic_embd_path", "ads_pure_corpus_embd_path",
    "other_semantic_embd_path", "output_json",
])
flags.DEFINE_string("gin_config_file", None, "")
flags.DEFINE_string("data_path", None, "")
flags.DEFINE_string("ckpt_path", None, "Checkpoint .pt file")
flags.DEFINE_string("mode", "job", "")
flags.DEFINE_integer("eval_batches", 200, "Number of eval batches")
flags.DEFINE_string("ads_semantic_embd_path", None, "")
flags.DEFINE_string("web_browsing_semantic_embd_path", None, "")
flags.DEFINE_string("shopping_semantic_embd_path", None, "")
flags.DEFINE_string("ads_pure_corpus_embd_path", None, "")
flags.DEFINE_string("other_semantic_embd_path", None, "")
flags.DEFINE_string("output_json", None, "Where to write JSON results")

FLAGS = flags.FLAGS


@torch.inference_mode()
def run_eval(model, eval_data_loader, device, max_batches):
    """Run eval and collect per-sample retrieval ranks + event type IDs."""
    model.eval()

    # Build negatives index
    negatives_sampler = model.negatives_sampler['eval']
    negatives_sampler.rotate()
    sampled_neg_ids, sampled_neg_embeddings = negatives_sampler(
        positive_ids=torch.tensor([0], device=device),
        num_to_sample=10000,
    )

    from indexing.utils import get_top_k_module
    top_k_module = get_top_k_module("MIPSBruteForceTopK", sampled_neg_embeddings, sampled_neg_ids)

    all_ranks = []
    all_types = []

    batch_count = 0
    for row in eval_data_loader:
        input_ids = row["input_ids"].to(device)
        raw_input_embeddings = row["raw_input_embeddings"].to(dtype=torch.float32, device=device)
        timestamps = row["timestamps"].to(device)
        lengths = row["lengths"].to(device)
        type_ids = row["type_ids"].to(device) if "type_ids" in row else None
        ratings = row["ratings"].to(device) if "ratings" in row else None

        B = input_ids.size(0)

        # Shift: input = all but last, label = last item
        inp_ids = input_ids[:, :-1]
        inp_emb = raw_input_embeddings[:, :-1, :]
        inp_ts = timestamps[:, :-1]
        inp_types = type_ids[:, :-1] if type_ids is not None else None
        inp_lengths = lengths - 1

        # Label: the item at position lengths-1
        label_ids = input_ids[torch.arange(B), lengths - 1]
        label_emb = raw_input_embeddings[torch.arange(B), lengths - 1, :]
        label_types = type_ids[torch.arange(B), lengths - 1] if type_ids is not None else torch.zeros(B, dtype=torch.long, device=device)

        # Encode
        past_embeddings = model.model._embedding_module(inp_emb)
        label_embeddings = model.negatives_sampler['eval'].normalize_embeddings(
            model.model._embedding_module(label_emb)
        )

        query_embeddings = model.model.encode(
            past_lengths=inp_lengths,
            past_ids=inp_ids,
            past_embeddings=past_embeddings,
            past_payloads={"timestamps": inp_ts, "ratings": ratings, "type_ids": inp_types},
        )

        # Retrieval
        k = min(2500, sampled_neg_ids.size(1))
        top_k_scores, top_k_ids = top_k_module(query_embeddings=query_embeddings, k=k)

        pos_scores = (query_embeddings * label_embeddings).sum(-1, keepdim=True)
        all_ids = torch.cat([label_ids.unsqueeze(1), top_k_ids], dim=1)
        all_scores = torch.cat([pos_scores, top_k_scores], dim=1)
        _, sorted_indices = all_scores.sort(dim=1, descending=True)
        sorted_ids = torch.gather(all_ids, dim=1, index=sorted_indices)
        _, rank_indices = torch.max(sorted_ids == label_ids.unsqueeze(1), dim=1)
        ranks = rank_indices + 1

        all_ranks.append(ranks.cpu())
        all_types.append(label_types.cpu())

        batch_count += 1
        if batch_count >= max_batches:
            break

    all_ranks = torch.cat(all_ranks)
    all_types = torch.cat(all_types)
    return all_ranks, all_types


def compute_metrics(ranks, types):
    """Compute per-event-type and per-group metrics."""
    results = {}

    # Overall
    ndcg10 = torch.where(ranks <= 10, 1.0 / torch.log2(ranks.float() + 1), torch.zeros_like(ranks.float()))
    hr10 = (ranks <= 10).float()
    mrr = 1.0 / ranks.float()

    results["Overall"] = {
        "NDCG@10": ndcg10.mean().item(),
        "HR@10": hr10.mean().item(),
        "MRR": mrr.mean().item(),
        "count": len(ranks),
    }

    # Per event type
    for tid, tname in EVENT_TYPE_NAMES.items():
        if tid == 0:
            continue
        mask = types == tid
        count = mask.sum().item()
        if count == 0:
            continue
        results[f"etype_{tname}"] = {
            "NDCG@10": ndcg10[mask].mean().item(),
            "HR@10": hr10[mask].mean().item(),
            "MRR": mrr[mask].mean().item(),
            "count": count,
        }

    # Per group
    for group_name in ["Ad", "Browsing", "Search", "Purchase", "Others"]:
        group_tids = [tid for tid, g in GROUP_MAP.items() if g == group_name]
        mask = torch.zeros(len(types), dtype=torch.bool)
        for tid in group_tids:
            mask |= (types == tid)
        count = mask.sum().item()
        if count == 0:
            continue
        results[f"group_{group_name}"] = {
            "NDCG@10": ndcg10[mask].mean().item(),
            "HR@10": hr10[mask].mean().item(),
            "MRR": mrr[mask].mean().item(),
            "count": count,
        }

    return results


def main(argv):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    gin.parse_config_file(FLAGS.gin_config_file, skip_unknown=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Init dist (required by some model components)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    dataset = get_reco_dataset(
        mode=FLAGS.mode,
        path=FLAGS.data_path,
        chronological=True,
        rank=rank,
        world_size=world_size,
    )

    precomputed = {
        0: FLAGS.ads_semantic_embd_path,
        1: FLAGS.web_browsing_semantic_embd_path,
        2: FLAGS.shopping_semantic_embd_path,
        3: FLAGS.ads_pure_corpus_embd_path,
    }
    if FLAGS.other_semantic_embd_path:
        precomputed[4] = FLAGS.other_semantic_embd_path
    # Remove None entries
    precomputed = {k: v for k, v in precomputed.items() if v is not None}

    model = make_model(dataset=dataset, precomputed_embeddings_domain_to_dir=precomputed)

    # Load checkpoint
    logging.info(f"Loading checkpoint: {FLAGS.ckpt_path}")
    ckpt = torch.load(FLAGS.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["MODEL_STATE"])
    model = model.to(device)
    logging.info(f"Checkpoint loaded (batch {ckpt.get('BATCHES_RUN', '?')})")

    collate_fn = CollateFn(
        device=device,
        domain_to_item_id_range=dataset.domain_to_item_id_range,
        precomputed_embeddings_domain_to_dir=model.precomputed_embeddings_domain_to_dir,
        domain_offset=dataset.domain_offset,
    )
    eval_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=64,
        world_size=1,
        rank=0,
        shuffle=False,
        drop_last=False,
        random_seed=42,
        collate_fn=collate_fn,
    )

    logging.info(f"Running eval for up to {FLAGS.eval_batches} batches...")
    ranks, types = run_eval(model, eval_loader, device, FLAGS.eval_batches)
    results = compute_metrics(ranks, types)

    # Pretty print
    print(f"\n{'Category':<30} {'NDCG@10':>8} {'HR@10':>8} {'MRR':>8} {'Count':>8}")
    print("-" * 65)
    # Print groups first
    for key in ["Overall", "group_Ad", "group_Browsing", "group_Search", "group_Purchase", "group_Others"]:
        if key in results:
            r = results[key]
            print(f"{key:<30} {r['NDCG@10']:8.4f} {r['HR@10']:8.4f} {r['MRR']:8.4f} {r['count']:8d}")
    print("-" * 65)
    # Print per event type
    for tid in sorted(EVENT_TYPE_NAMES.keys()):
        if tid == 0:
            continue
        key = f"etype_{EVENT_TYPE_NAMES[tid]}"
        if key in results:
            r = results[key]
            print(f"{key:<30} {r['NDCG@10']:8.4f} {r['HR@10']:8.4f} {r['MRR']:8.4f} {r['count']:8d}")

    # Save JSON
    output_json = FLAGS.output_json or FLAGS.ckpt_path.replace(".pt", "_eval_per_event.json")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {output_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    app.run(main)
