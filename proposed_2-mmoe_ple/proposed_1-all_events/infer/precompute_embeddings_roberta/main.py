"""
Main entry point for precompute-embeddings using Range-based Sharding. Please refer to README.md for usage instructions.
To run the training locally, you can use the following command:
conda activate hstu
torchrun --nproc_per_node=1  precompute_embeddings/main.py --domain_name=ads --mode=local
"""

import logging
import os

from typing import List, Optional

from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

import fbgemm_gpu  # noqa: F401, E402
import gin

import torch
import torch.multiprocessing as mp

import absl
from absl import app, flags
import mlflow
from dataset import get_domain_dataset, DomainDataset, CollateFn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm


# Set absl logging config
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity('info')               # Only log info and above
absl.logging.set_stderrthreshold('info')         # Output info-level logs to stderr
absl.logging.get_absl_handler().use_absl_log_file(False)  # Avoid .log file creation

# Clear existing Python logging handlers (in case something pre-configured them)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Attach absl's handler to Python logging
logging.root.addHandler(absl.logging.get_absl_handler())
logging.root.setLevel(logging.INFO)

# Suppress logs from azure libraries
logging.getLogger('azure').setLevel(logging.ERROR)


def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


delete_flags(flags.FLAGS, ["data_path", "output_path", "mode"])
flags.DEFINE_string("data_path", None, "Path to the train/eval data, this is only used for job mode")
flags.DEFINE_string("output_path", None, "Path to write the artifacts, this is only used for job mode")
flags.DEFINE_string("mode", "job", "local or job.")


FLAGS = flags.FLAGS  # pyre-ignore [5]

def precompute_embeddings(
    dataset: DomainDataset = None,
    model: AutoModel = None,
    tokenizer: AutoTokenizer = None,
    batch_size: int = 512,
    shard_size: int = 25_000_000,  
    output_dir: str = ".",
    local_rank: int = 0,
    rank: int = 0,
) -> None:
    """
    Precompute embeddings for the given domain dataset.
    """
    outpath = f"{output_dir}"
    os.makedirs(outpath, exist_ok=True)

    collate_fn = CollateFn(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = local_rank
    model = model.to(device)
    model.eval()

    # Accumulate embeddings per shard
    shard_embeddings = defaultdict(list)  # shard_id -> list of (local_idx, embedding)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding"):
            encoded, indices = batch["inputs"].to(device), batch['index']
            outputs = model(**encoded)
            attention_mask = encoded["attention_mask"]
            last_hidden = outputs.last_hidden_state  # [B, L, D]

            # Masked mean pooling
            masked_hidden = last_hidden * attention_mask.unsqueeze(-1)  # [B, L, D]
            sum_hidden = masked_hidden.sum(dim=1)                      # [B, D]
            lengths = attention_mask.sum(dim=1, keepdim=True)          # [B, 1]
            batch_emb = (sum_hidden / lengths).cpu().to(torch.float16).numpy()  # [B, D]

            for emb, global_idx in zip(batch_emb, indices):
                global_idx = int(global_idx)
                shard_id = global_idx // shard_size
                local_idx = global_idx % shard_size
                shard_embeddings[shard_id].append((local_idx, emb))

    # Write each shard to file
    for shard_id, items in shard_embeddings.items():
        shard_array = np.zeros((shard_size, 768), dtype=np.float16)
        for local_idx, emb in items:
            shard_array[local_idx] = emb
        out_path = f"{outpath}/shard_{shard_id}.npy"
        np.save(out_path, shard_array)
        print(f"[Rank {rank}] Saved shard {shard_id} with {len(items)} embeddings to {out_path}")



def main(argv) -> None:  # pyre-ignore [2]
    # torchrun sets these environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    job_name = os.getenv("AZUREML_RUN_ID")

    data_path = FLAGS.data_path
    output_path = FLAGS.output_path
    if output_path:
        output_path = f"{output_path}" 
    mode = FLAGS.mode
    logging.info(f"world size: {world_size}, rank: {rank}, local_rank: {local_rank}")

    BATCH_SIZE = 1024
    SHARD_SIZE = 25_000_000  # 25 million rows per shard, adjust as needed
    domain_dataset = get_domain_dataset(
        mode=mode,
        path=data_path,
        rank=rank,
        world_size=world_size,
        shard_size=SHARD_SIZE,  
    )

    MODEL_NAME = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    with mlflow.start_run():
        precompute_embeddings(
            dataset=domain_dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            shard_size=SHARD_SIZE,
            output_dir=output_path,
            local_rank=local_rank,
            rank=rank,
        )

if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    app.run(main)