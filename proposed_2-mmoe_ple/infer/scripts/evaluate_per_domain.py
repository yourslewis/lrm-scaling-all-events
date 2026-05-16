"""
Per-domain evaluation script for all experiments.

Usage:
    cd /home/yourslewis/lrm-scaling-all-events/proposed_2-mmoe_ple/infer
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29510 scripts/evaluate_per_domain.py \
        --gin_config_file=<config> \
        --checkpoint_path=<ckpt> \
        --data_path=/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2 \
        --ads_semantic_embd_path=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings/domain_0 \
        --web_browsing_semantic_embd_path=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings/domain_1 \
        --shopping_semantic_embd_path=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings/domain_2 \
        --ads_pure_corpus_embd_path=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings/domain_3 \
        --other_semantic_embd_path=/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings/domain_4 \
        --max_eval_batches=200 \
        --mode=job
"""

import logging
import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "train"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import fbgemm_gpu  # noqa: F401
import gin
import torch
import torch.distributed as dist
import datetime as dt

import absl
from absl import app, flags

from data.reco_dataset import get_reco_dataset
from data.ads_datasets.collate import CollateFn
from data.eval import (
    _avg,
    eval_metrics_v3_from_tensors,
    get_eval_state_v2,
)
from trainer.seeding import get_gin_configured_seed, seed_everything
from trainer.util import make_model
from trainer.train import Trainer  # noqa: F401
from trainer.data_loader import create_data_loader

absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity("info")
absl.logging.set_stderrthreshold("info")
absl.logging.get_absl_handler().use_absl_log_file(False)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.addHandler(absl.logging.get_absl_handler())
logging.root.setLevel(logging.INFO)

flags.DEFINE_string("gin_config_file", None, "Path to gin config file.")
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint .pt file.")
flags.DEFINE_string("data_path", None, "Path to the data directory.")
flags.DEFINE_string("ads_semantic_embd_path", None, "Domain 0 embeddings.")
flags.DEFINE_string("web_browsing_semantic_embd_path", None, "Domain 1 embeddings.")
flags.DEFINE_string("shopping_semantic_embd_path", None, "Domain 2 embeddings.")
flags.DEFINE_string("ads_pure_corpus_embd_path", None, "Domain 3 embeddings.")
flags.DEFINE_string("other_semantic_embd_path", None, "Domain 4 embeddings.")
flags.DEFINE_string("mode", "job", "local or job.")
flags.DEFINE_integer("max_eval_batches", 200, "Maximum number of eval batches.")
flags.DEFINE_integer("eval_batch_size", 32, "Evaluation batch size.")
flags.DEFINE_string("top_k_method", "MIPSBruteForceTopK", "Top-K retrieval method.")
flags.DEFINE_string("output_json", None, "Optional: save results as JSON.")

FLAGS = flags.FLAGS

DOMAIN_NAMES = {0: "Ads", 1: "WebBrowsing", 2: "Shopping", 3: "AdsCorpus", 4: "Other"}
DOMAIN_OFFSET = 1_000_000_000


def main(argv):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    gin_config_file = FLAGS.gin_config_file
    checkpoint_path = FLAGS.checkpoint_path
    data_path = FLAGS.data_path

    precomputed_embeddings_domain_to_dir = {
        0: FLAGS.ads_semantic_embd_path,
        1: FLAGS.web_browsing_semantic_embd_path,
        2: FLAGS.shopping_semantic_embd_path,
        3: FLAGS.ads_pure_corpus_embd_path,
    }
    if FLAGS.other_semantic_embd_path:
        precomputed_embeddings_domain_to_dir[4] = FLAGS.other_semantic_embd_path

    logging.info(f"Per-domain evaluation of: {checkpoint_path}")

    if gin_config_file is not None:
        gin.parse_config_file(gin_config_file)

    random_seed = get_gin_configured_seed()
    seed_everything(random_seed, rank=rank, log_prefix="evaluate_per_domain")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        timeout=dt.timedelta(minutes=10),
    )
    device = local_rank

    dataset = get_reco_dataset(
        mode=FLAGS.mode, path=data_path, chronological=True,
        rank=rank, world_size=world_size,
    )

    model = make_model(
        dataset=dataset,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
    )

    logging.info(f"Loading checkpoint from {checkpoint_path}")
    snapshot = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"])
    batch_id = snapshot.get("BATCHES_RUN", "unknown")
    logging.info(f"Loaded checkpoint from batch {batch_id}")

    model = model.to(device)
    model.eval()

    collate_fn = CollateFn(
        device=device,
        domain_to_item_id_range=dataset.domain_to_item_id_range,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
        domain_offset=dataset.domain_offset,
    )
    eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=FLAGS.eval_batch_size,
        world_size=world_size, rank=rank,
        shuffle=True, drop_last=world_size > 1,
        random_seed=42, collate_fn=collate_fn,
    )

    model.negatives_sampler["eval"].rotate()

    eval_state = get_eval_state_v2(model=model, top_k_method=FLAGS.top_k_method)

    logging.info(f"Running per-domain evaluation (max {FLAGS.max_eval_batches} batches)...")
    eval_start = time.time()

    # Overall + per-domain metric accumulators
    all_metrics = defaultdict(list)
    domain_metrics = {d: defaultdict(list) for d in range(5)}

    batch_count = 0
    domain_sample_counts = defaultdict(int)

    for row in iter(eval_data_loader):
        user_id = row["user_id"]
        input_ids = row["input_ids"].to(device, non_blocking=True)
        raw_input_embeddings = row["raw_input_embeddings"].to(
            dtype=torch.float32, device=device, non_blocking=True
        )
        ratings = row["ratings"].to(device, non_blocking=True)
        timestamps = row["timestamps"].to(device, non_blocking=True)
        type_ids = row["type_ids"].to(device, non_blocking=True) if "type_ids" in row else None
        lengths = row["lengths"].to(device, non_blocking=True)

        eval_dict = eval_metrics_v3_from_tensors(
            eval_state, model, input_ids, raw_input_embeddings,
            ratings, timestamps, lengths, user_id, type_ids=type_ids,
        )

        # Determine domain of each sample's label (last item in sequence)
        label_ids = input_ids[torch.arange(input_ids.size(0)), lengths - 1]  # [B]
        label_domains = (label_ids // DOMAIN_OFFSET).cpu()  # [B]

        # Accumulate overall
        for k, v in eval_dict.items():
            all_metrics[k].append(v)

        # Accumulate per-domain
        for k, v in eval_dict.items():
            if v.dim() == 0:
                # scalar (e.g., log_pplx) - add to all domains proportionally
                continue
            for d in range(5):
                mask = (label_domains == d)
                if mask.any():
                    domain_metrics[d][k].append(v[mask])

        for d in range(5):
            domain_sample_counts[d] += (label_domains == d).sum().item()

        batch_count += 1
        if batch_count % 50 == 0:
            logging.info(f"  Processed {batch_count} batches...")
        if batch_count >= FLAGS.max_eval_batches:
            break

    eval_time = time.time() - eval_start

    # Aggregate overall metrics
    for k, v_list in all_metrics.items():
        v_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in v_list]
        all_metrics[k] = torch.cat(v_list, dim=-1)
    overall = {k: _avg(v, rank=rank, world_size=world_size) for k, v in all_metrics.items()}

    # Aggregate per-domain metrics
    per_domain = {}
    for d in range(5):
        if not domain_metrics[d]:
            per_domain[d] = {}
            continue
        for k, v_list in domain_metrics[d].items():
            v_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in v_list]
            domain_metrics[d][k] = torch.cat(v_list, dim=-1)
        per_domain[d] = {
            k: _avg(v, rank=rank, world_size=world_size)
            for k, v in domain_metrics[d].items()
        }

    if rank == 0:
        key_metrics = ["ndcg_1", "ndcg_10", "ndcg_50", "hr_10", "hr_50", "hr_1000", "mrr"]

        logging.info(f"\n{'='*80}")
        logging.info(f"EVALUATION RESULTS — Checkpoint batch {batch_id}")
        logging.info(f"{'='*80}")
        logging.info(f"Batches: {batch_count} | Time: {eval_time:.1f}s")
        logging.info(f"Total samples: {sum(domain_sample_counts.values())}")
        for d in range(5):
            logging.info(f"  Domain {d} ({DOMAIN_NAMES[d]}): {domain_sample_counts[d]} samples")

        logging.info(f"\n{'='*80}")
        logging.info(f"OVERALL METRICS:")
        logging.info(f"{'='*80}")
        for k in key_metrics:
            if k in overall:
                logging.info(f"  {k:>15s}: {overall[k].item():.6f}")

        for d in range(5):
            if not per_domain[d]:
                continue
            logging.info(f"\n{'='*80}")
            logging.info(f"DOMAIN {d} ({DOMAIN_NAMES[d]}) — {domain_sample_counts[d]} samples:")
            logging.info(f"{'='*80}")
            for k in key_metrics:
                if k in per_domain[d]:
                    logging.info(f"  {k:>15s}: {per_domain[d][k].item():.6f}")

        # Summary table
        logging.info(f"\n{'='*80}")
        logging.info("SUMMARY TABLE:")
        header = f"{'Domain':>15s} | {'Samples':>8s} | {'NDCG@10':>8s} | {'HR@10':>8s} | {'HR@50':>8s} | {'MRR':>8s}"
        logging.info(header)
        logging.info("-" * len(header))
        for d in range(5):
            if per_domain[d] and "ndcg_10" in per_domain[d]:
                logging.info(
                    f"{DOMAIN_NAMES[d]:>15s} | {domain_sample_counts[d]:>8d} | "
                    f"{per_domain[d]['ndcg_10'].item():>8.4f} | "
                    f"{per_domain[d].get('hr_10', torch.tensor(0.0)).item():>8.4f} | "
                    f"{per_domain[d].get('hr_50', torch.tensor(0.0)).item():>8.4f} | "
                    f"{per_domain[d].get('mrr', torch.tensor(0.0)).item():>8.4f}"
                )
        logging.info("-" * len(header))
        if "ndcg_10" in overall:
            logging.info(
                f"{'OVERALL':>15s} | {sum(domain_sample_counts.values()):>8d} | "
                f"{overall['ndcg_10'].item():>8.4f} | "
                f"{overall.get('hr_10', torch.tensor(0.0)).item():>8.4f} | "
                f"{overall.get('hr_50', torch.tensor(0.0)).item():>8.4f} | "
                f"{overall.get('mrr', torch.tensor(0.0)).item():>8.4f}"
            )
        logging.info(f"{'='*80}")

        # Save JSON
        if FLAGS.output_json:
            results = {
                "checkpoint": checkpoint_path,
                "batch_id": str(batch_id),
                "overall": {k: v.item() for k, v in overall.items()},
                "per_domain": {
                    DOMAIN_NAMES[d]: {k: v.item() for k, v in per_domain[d].items()}
                    for d in range(5) if per_domain[d]
                },
                "sample_counts": {DOMAIN_NAMES[d]: domain_sample_counts[d] for d in range(5)},
            }
            with open(FLAGS.output_json, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {FLAGS.output_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
