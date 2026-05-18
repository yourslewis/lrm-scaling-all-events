"""
Standalone evaluation script for HSTU retrieval checkpoints.

Usage:
    torchrun --nproc_per_node=1 scripts/evaluate_checkpoint.py \
        --gin_config_file=configs/local/local_training.gin \
        --checkpoint_path=/tmp/hstu_local/output/local_run/ckpts/checkpoint_batch0001000.pt \
        --data_path=/tmp/hstu_local/data \
        --ads_semantic_embd_path=/tmp/hstu_local/embds/domain_0 \
        --web_browsing_semantic_embd_path=/tmp/hstu_local/embds/domain_1 \
        --shopping_semantic_embd_path=/tmp/hstu_local/embds/domain_2 \
        --ads_pure_corpus_embd_path=/tmp/hstu_local/embds/domain_3 \
        --max_eval_batches=50 \
        --mode=job
"""

import logging
import os
import sys
import time
from collections import defaultdict

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from trainer.train import Trainer  # noqa: F401 - register gin configurables
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
flags.DEFINE_string("ads_semantic_embd_path", None, "Path to domain 0 embeddings.")
flags.DEFINE_string("web_browsing_semantic_embd_path", None, "Path to domain 1 embeddings.")
flags.DEFINE_string("shopping_semantic_embd_path", None, "Path to domain 2 embeddings.")
flags.DEFINE_string("ads_pure_corpus_embd_path", None, "Path to domain 3 embeddings.")
flags.DEFINE_string("mode", "job", "local or job.")
flags.DEFINE_integer("max_eval_batches", 100, "Maximum number of eval batches.")
flags.DEFINE_integer("eval_batch_size", 64, "Evaluation batch size.")
flags.DEFINE_string("top_k_method", "MIPSBruteForceTopK", "Top-K retrieval method.")

FLAGS = flags.FLAGS


def main(argv):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    gin_config_file = FLAGS.gin_config_file
    checkpoint_path = FLAGS.checkpoint_path
    data_path = FLAGS.data_path
    mode = FLAGS.mode
    max_eval_batches = FLAGS.max_eval_batches
    eval_batch_size = FLAGS.eval_batch_size
    top_k_method = FLAGS.top_k_method

    precomputed_embeddings_domain_to_dir = {
        0: FLAGS.ads_semantic_embd_path,
        1: FLAGS.web_browsing_semantic_embd_path,
        2: FLAGS.shopping_semantic_embd_path,
        3: FLAGS.ads_pure_corpus_embd_path,
    }

    logging.info(f"Evaluating checkpoint: {checkpoint_path}")
    logging.info(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")

    if gin_config_file is not None:
        gin.parse_config_file(gin_config_file)

    random_seed = get_gin_configured_seed()
    seed_everything(random_seed, rank=rank, log_prefix="evaluate_checkpoint")

    # Setup DDP
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(minutes=10),
    )

    device = local_rank

    # Build dataset
    dataset = get_reco_dataset(
        mode=mode,
        path=data_path,
        chronological=True,
        rank=rank,
        world_size=world_size,
    )

    # Build model
    model = make_model(
        dataset=dataset,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
    )

    # Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    snapshot = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(snapshot["MODEL_STATE"])
    batch_id_from_ckpt = snapshot.get("BATCHES_RUN", "unknown")
    logging.info(f"Loaded checkpoint from batch {batch_id_from_ckpt}")

    model = model.to(device)
    model.eval()

    # Build eval data loader
    collate_fn = CollateFn(
        device=device,
        domain_to_item_id_range=dataset.domain_to_item_id_range,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
        domain_offset=dataset.domain_offset,
    )
    eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
        random_seed=42,
        collate_fn=collate_fn,
    )

    # Rotate eval negatives sampler
    logging.info("Rotating eval negatives sampler...")
    model.negatives_sampler["eval"].rotate()

    # Get eval state
    eval_state = get_eval_state_v2(
        model=model,
        top_k_method=top_k_method,
    )

    # Run evaluation
    logging.info(f"Running evaluation (max {max_eval_batches} batches)...")
    eval_start_time = time.time()
    eval_dict_all = defaultdict(list)
    batch_count = 0

    for row in iter(eval_data_loader):
        user_id = row["user_id"]
        input_ids = row["input_ids"].to(device, non_blocking=True)
        raw_input_embeddings = row["raw_input_embeddings"].to(
            dtype=torch.float32, device=device, non_blocking=True
        )
        ratings = row["ratings"].to(device, non_blocking=True)
        timestamps = row["timestamps"].to(device, non_blocking=True)
        lengths = row["lengths"].to(device, non_blocking=True)

        eval_dict = eval_metrics_v3_from_tensors(
            eval_state,
            model,
            input_ids,
            raw_input_embeddings,
            ratings,
            timestamps,
            lengths,
            user_id,
        )

        for k, v in eval_dict.items():
            eval_dict_all[k].append(v)

        batch_count += 1
        if batch_count >= max_eval_batches:
            break

    # Aggregate metrics
    for k, v_list in eval_dict_all.items():
        v_tensor_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in v_list]
        eval_dict_all[k] = torch.cat(v_tensor_list, dim=-1)

    final_metrics = {
        k: _avg(v, rank=rank, world_size=world_size)
        for k, v in eval_dict_all.items()
    }

    eval_time = time.time() - eval_start_time

    if rank == 0:
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluation Results (checkpoint batch {batch_id_from_ckpt})")
        logging.info(f"{'='*60}")
        logging.info(f"Batches evaluated: {batch_count}")
        logging.info(f"Evaluation time: {eval_time:.2f}s")
        logging.info(f"{'='*60}")
        for k in sorted(final_metrics.keys()):
            logging.info(f"  {k:>15s}: {final_metrics[k].item():.6f}")
        logging.info(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
