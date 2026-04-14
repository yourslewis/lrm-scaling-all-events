# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Main entry point for model training. Please refer to README.md for usage instructions.
To run the training locally, you can use the following command:
conda activate hstu
torchrun --nproc_per_node=1  main.py --gin_config_file=configs/ads/training_data_05012025_next_event_prediction.gin --mode=local
torchrun --nproc_per_node=1  main.py --gin_config_file=configs/ads/training_data_05012025_conditional_next_event_prediction.gin --mode=local
torchrun --nproc_per_node=1  main.py --gin_config_file=configs/ads/training_data_05012025_next_positive_event_prediction.gin --mode=local
"""

import logging
import os

from typing import List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

import fbgemm_gpu  # noqa: F401, E402
import gin

import torch
import torch.multiprocessing as mp

import absl
from absl import app, flags
from data.reco_dataset import get_reco_dataset
from trainer.train import Trainer
from trainer.util import make_model
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
import fsspec

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


delete_flags(flags.FLAGS, ["gin_config_file", "data_path", "ads_semantic_embd_path", "web_browsing_semantic_embd_path", "shopping_semantic_embd_path", "ads_pure_corpus_embd_path", "ckpt_path" "output_path", "mode"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_string("data_path", None, "Path to the train/eval data, this is only used for job mode")
flags.DEFINE_string("ads_semantic_embd_path", None, "Path to the precomputed embeddings for ads domain, this is only used for job mode")
flags.DEFINE_string("web_browsing_semantic_embd_path", None, "Path to the precomputed embeddings for web browsing domain, this is only used for job mode")
flags.DEFINE_string("shopping_semantic_embd_path", None, "Path to the precomputed embeddings for shopping domain, this is only used for job mode")
flags.DEFINE_string("ads_pure_corpus_embd_path", None, "Path to the precomputed embeddings for ads pure corpus domain, this is only used for job mode")
flags.DEFINE_string("ckpt_path", None, "Path to model checkpoint")
flags.DEFINE_string("output_path", None, "Path to write the artifacts, this is only used for job mode")
flags.DEFINE_string("mode", "job", "local or job.")

FLAGS = flags.FLAGS  # pyre-ignore [5]


def main(argv) -> None:  # pyre-ignore [2]
    # torchrun sets these environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    job_name = os.getenv("AZUREML_RUN_ID")
    
    gin_config_file = FLAGS.gin_config_file
    data_path = FLAGS.data_path
    ads_semantic_embd_path = FLAGS.ads_semantic_embd_path
    web_browsing_semantic_embd_path = FLAGS.web_browsing_semantic_embd_path
    shopping_semantic_embd_path = FLAGS.shopping_semantic_embd_path
    ads_pure_corpus_embd_path = FLAGS.ads_pure_corpus_embd_path
    precomputed_embeddings_domain_to_dir = {
        0: ads_semantic_embd_path,
        1: web_browsing_semantic_embd_path,
        2: shopping_semantic_embd_path,
        3: ads_pure_corpus_embd_path,
    }
        
    output_path = FLAGS.output_path
    if output_path:
        output_path = f"{output_path}/{job_name}" 
    ckpt_path = FLAGS.ckpt_path
    mode = FLAGS.mode
    logging.info(f"world size: {world_size}, rank: {rank}, local_rank: {local_rank}")
    if gin_config_file is not None:
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    dataset = get_reco_dataset(
        mode=mode,
        path=data_path,
        chronological=True,
        rank=rank,
        world_size=world_size
    )
    model =  make_model(
        dataset=dataset,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
    )
    # snapshot_dir = f"{output_path}/ckpts"
    trainer = Trainer(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        dataset=dataset,
        model=model,
        ckpt_path = ckpt_path,
        output_path=output_path,
    )
    
    if HAS_MLFLOW:
        with mlflow.start_run():
            trainer.train()
    else:
        trainer.train()


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    app.run(main)