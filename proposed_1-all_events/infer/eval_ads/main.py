"""
Main entry point for precompute-embeddings using Range-based Sharding. Please refer to README.md for usage instructions.
To run the training locally, you can use the following command:
conda activate hstu
torchrun --nproc_per_node=1  precompute_embeddings/main.py --domain_name=ads --mode=local
"""

import logging
import os

from typing import List, Optional

from sklearn.base import defaultdict

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
import pandas as pd
import gin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from trainer.util import make_model
from data.reco_dataset import RecoDataset

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


delete_flags(flags.FLAGS, ["gin_config_file", "data_path", "ckpt_path", "pinsage_ckpt_path", "output_path", "mode"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_string("data_path", None, "Path to the train/eval data, this is only used for job mode")
flags.DEFINE_string("ckpt_path", None, "Path to model checkpoint")
flags.DEFINE_string("pinsage_ckpt_path", None, "Path to PinSage model checkpoint")
flags.DEFINE_string("output_path", None, "Path to write the artifacts, this is only used for job mode")
flags.DEFINE_string("mode", "job", "local or job.")


FLAGS = flags.FLAGS  # pyre-ignore [5]

def inference_embeddings(
    dataset: DomainDataset = None,
    model: torch.nn.Module = None,
    tokenizer_name: str = '',
    batch_size: int = 512,
    output_dir: str = ".",
    local_rank: int = 0,
    rank: int = 0,
) -> None:
    """
    Precompute embeddings for the given domain dataset.
    """
    if tokenizer_name != '':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = model.model.tokenizer
    collate_fn = CollateFn(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    device = local_rank
    model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding"):
            encoded, indices = batch["inputs"].to(device), batch['InferAdId']
            batch_emb = model.get_item_embeddings(encoded)                  # [B, T] -> [B, D]
            batch_emb = batch_emb.cpu().to(torch.float16).numpy()  # [B, D]

            # Each row: {"InferAdId": idx, "embedding": emb}
            ids_str = pd.Series(indices, dtype="string")
            emb_str = [' '.join(format(x, '.6g') for x in row) for row in batch_emb]
            df_batch = pd.DataFrame({
                "InferAdId": ids_str,
                "embedding": emb_str  
            })

            results.append(df_batch)

    # Early exit if no results, some ranks may not have data to process
    if not results:
        return

    final_df = pd.concat(results, ignore_index=True)

    output_dir = f"{output_dir}/ads"
    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/shard_{rank}_ads_emb.tsv"
    final_df.to_csv(out_path, sep='\t', index=False)
    print(f"[Rank {rank}] Saved embeddings to {out_path}")


def make_ad_encoder(
    ckpt_path: str = "",
    pinsage_ckpt_path: str = "",
) -> torch.nn.Module:
    state_dict = torch.load(f'{ckpt_path}', map_location="cpu")
    dataset = RecoDataset(
        max_sequence_length=200,
        embd_dim=768,
        dataset_name='',
        positional_sampling_ratio=1.0,
        train_dataset=None,
        eval_dataset=None,
        domain_to_item_id_range=None
    )
    model = make_model(dataset=dataset, pinsage_ckpt_path=pinsage_ckpt_path)
    user_encoder = model.model

    # Remove "model." prefix to get layers for the user_encoder
    filtered_state_dict = {
        k[len("model."):]: v for k, v in state_dict["MODEL_STATE"].items() if k.startswith("model.")
    }

    missing_keys, unexpected_keys = user_encoder.load_state_dict(filtered_state_dict, strict=False)

    print("Missing keys:", missing_keys)         # usually your replaced layer
    print("Unexpected keys:", unexpected_keys)   # weights that didn’t find a match

    ad_encoder = user_encoder._embedding_module
    return ad_encoder

def main(argv) -> None:  # pyre-ignore [2]
    # torchrun sets these environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    job_name = os.getenv("AZUREML_RUN_ID")

    gin_config_file = FLAGS.gin_config_file
    data_path = FLAGS.data_path
    ckpt_path = FLAGS.ckpt_path
    pinsage_ckpt_path = FLAGS.pinsage_ckpt_path
    output_path = FLAGS.output_path
    if output_path:
        output_path = f"{output_path}" 
    mode = FLAGS.mode
    logging.info(f"world size: {world_size}, rank: {rank}, local_rank: {local_rank}")

    if gin_config_file is not None:
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    BATCH_SIZE = 512
    domain_dataset = get_domain_dataset(
        domain="ads",
        mode=mode,
        path=data_path,
        rank=rank,
        world_size=world_size,
    )
    model = make_ad_encoder(ckpt_path=ckpt_path, pinsage_ckpt_path=pinsage_ckpt_path)

    with mlflow.start_run():
        inference_embeddings(
            dataset=domain_dataset,
            model=model,
            tokenizer_name="",
            batch_size=BATCH_SIZE,
            output_dir=output_path,
            local_rank=local_rank,
            rank=rank,
        )

if __name__ == "__main__":
    # mp.set_start_method("forkserver", force=True)
    app.run(main)