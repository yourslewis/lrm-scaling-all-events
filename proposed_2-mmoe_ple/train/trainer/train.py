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

import logging
import os
import glob
import random

import time

import datetime as dt
from datetime import date
from typing import Dict, Optional, Union

import gin

import torch
import torch.distributed as dist

from data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    eval_metrics_v3_from_tensors,
    get_eval_state,
    get_eval_state_v2,
)

from data.reco_dataset import RecoDataset
from data.ads_datasets.collate import CollateFn
from indexing.utils import get_top_k_module
from trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
from collections import defaultdict
import shutil

_original_add_scalar = SummaryWriter.add_scalar

if HAS_MLFLOW:
    def patched_add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        # Call original function
        _original_add_scalar(self, tag, scalar_value, global_step, *args, **kwargs)

        # Log to MLflow
        if global_step is not None:
            mlflow.log_metric(tag, scalar_value, step=global_step)
        else:
            mlflow.log_metric(tag, scalar_value)

    # Patch the method
    SummaryWriter.add_scalar = patched_add_scalar



@gin.configurable
class Trainer:
    def __init__(
        self,
        local_rank: int,
        rank: int,
        world_size: int,
        dataset: Optional[RecoDataset] = None,
        model: torch.nn.Module = None,
        ckpt_path: Optional[str] = None,
        output_path: Optional[str] = None,

        local_batch_size: int = 128,        # set to be 128
        eval_batch_size: int = 128,
        eval_user_max_batch_size: Optional[int] = None,
        main_module_bf16: bool = False,
        loss_weights: Optional[Dict[str, float]] = {},
        num_epochs: int = 101,            # set to be 101
        learning_rate: float = 1e-3,      # set to be 1e-3
        num_warmup_steps: int = 0,        # set to be 0
        weight_decay: float = 1e-3,       # set to be 0
        top_k_method: str = "MIPSBruteForceTopK",      # set to be "MIPSBruteForceTopK"
        eval_interval: int = 1,
        full_eval_every_n: int = 1,         # default to be 1
        save_ckpt_every_n: int = 10,
        enable_tf32: bool = False,         # set to be True
        random_seed: int = 42,             # set to be 42
        eval_method: str = "pplx",  # "pplx" | "retrieval" | "sharded"
        eval_max_batches: int = 100,
        optimizer_type: str = "AdamW",
        optimizer_betas: tuple = (0.9, 0.98),
        scheduler_type: str = "none",  # "none" | "cosine" | "linear_warmup_cosine"
    ):
        self.local_rank = local_rank
        self.device = local_rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.model = model
        self.model_debug_str = model.debug_str()
        self.output_path = output_path
        self.ckpt_path = ckpt_path
        self.snapshot_dir = f"{output_path}/ckpts"

        self.local_batch_size = local_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_user_max_batch_size = eval_user_max_batch_size
        self.main_module_bf16 = main_module_bf16
        self.loss_weights = loss_weights
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay = weight_decay
        self.top_k_method = top_k_method
        self.eval_interval = eval_interval
        self.full_eval_every_n = full_eval_every_n
        self.batches_run = 0
        self.save_ckpt_every_n = save_ckpt_every_n
        self.enable_tf32 = enable_tf32
        self.random_seed = random_seed
        self.eval_method = eval_method
        self.eval_max_batches = eval_max_batches
        self.optimizer_type = optimizer_type
        self.optimizer_betas = optimizer_betas
        self.scheduler_type = scheduler_type

        # Setup and initialization
        self.setup()

    def ddp_setup(self) -> None:
        torch.cuda.set_device(self.local_rank)  # This ensures each process uses the correct GPU
        dist.init_process_group(
            backend="nccl", 
            rank=self.rank, 
            world_size=self.world_size,
            timeout=dt.timedelta(minutes=30)
        )

    def setup(self) -> None:
        # Place your setup logic here (DDP, dataloaders, optimizer, etc.)
        random.seed(self.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = self.enable_tf32
        torch.backends.cudnn.allow_tf32 = self.enable_tf32
        logging.info(f"cuda.matmul.allow_tf32: {self.enable_tf32}")
        logging.info(f"cudnn.allow_tf32: {self.enable_tf32}")
        logging.info(f"Training model on rank {self.rank}.")
        self.ddp_setup()

        collate_fn = CollateFn(
            input_dim=self.dataset.embd_dim,
            output_dim=128,  # not used in collate, only proj
            shard_size=self.dataset.shard_size,
            device=self.device,
            domain_to_item_id_range=self.dataset.domain_to_item_id_range,
            precomputed_embeddings_domain_to_dir=self.model.precomputed_embeddings_domain_to_dir,
            domain_offset=self.dataset.domain_offset,
        )
        self.train_data_loader = create_data_loader(
            self.dataset.train_dataset,
            batch_size=self.local_batch_size,     # set to be 128
            world_size=self.world_size,      
            rank=self.rank,
            shuffle=True,
            drop_last=self.world_size > 1, 
            random_seed=self.random_seed,
            collate_fn=collate_fn,  # Use the custom collate function
        )
        self.eval_data_loader = create_data_loader(
            self.dataset.eval_dataset,
            batch_size=self.eval_batch_size,      # default to be 128
            world_size=self.world_size,
            rank=self.rank,
            shuffle=True,     
            drop_last=self.world_size > 1,
            random_seed=self.random_seed,
            collate_fn=collate_fn,
        )

        # Config-driven optimizer
        self.opt = self._create_optimizer()

        if self.ckpt_path and self.rank == 0:
            os.makedirs(self.snapshot_dir, exist_ok=True)
 
            target_ckpt_path = os.path.join(self.snapshot_dir, os.path.basename(self.ckpt_path))
            shutil.copy2(self.ckpt_path, target_ckpt_path)
            logging.info(f"[Rank 0] Copied checkpoint from {self.ckpt_path} → {target_ckpt_path}")
        dist.barrier()

        self._load_latest_snapshot()
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        if self.main_module_bf16:                                 # default to be False
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True, static_graph=False)
        # For single-GPU, skip DDP wrapper entirely
        self._use_ddp = (self.world_size > 1)
        self._model_unwrapped = self.model.module if self._use_ddp else self.model

        date_str = date.today().strftime("%Y-%m-%d")
        model_subfolder = f"{self.dataset.dataset_name}-l{self.dataset.max_sequence_length}"
        self.model_desc = (
            f"{model_subfolder}"
            + f"/{self.model_debug_str}"
            + f"{f'-ddp{self.world_size}' if self.world_size > 1 else ''}-b{self.local_batch_size}-lr{self.learning_rate}-wu{self.num_warmup_steps}-wd{self.weight_decay}{'' if self.enable_tf32 else '-notf32'}-{date_str}"
        )
        if self.full_eval_every_n > 1:                      # default to be 1
            self.model_desc += f"-fe{self.full_eval_every_n}"
        if self.dataset.positional_sampling_ratio is not None and self.dataset.positional_sampling_ratio < 1:          # default to be 1
            self.model_desc += f"-d{self.dataset.positional_sampling_ratio}"
        # creates subfolders.
        # os.makedirs(f"/scratch/exps/{model_subfolder}", exist_ok=True)
        os.makedirs(f"./ckpts/{model_subfolder}", exist_ok=True)
        # /scratch/workspaceblobstore/azureml/<job_name>/logs 
        log_dir = os.environ.get('TENSORBOARD_LOG_DIR', '.') + "/" + f"exps/{self.model_desc}"
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
            logging.info(f"Rank {self.rank}: writing logs to {log_dir}")
        else:
            self.writer = None
            logging.info(f"Rank {self.rank}: disabling summary writer")


    def train(self) -> None:
        last_training_time = time.time()
        torch.autograd.set_detect_anomaly(False)       # disable this during normal training for better performance

        batch_id = self.batches_run
        epoch = 0
        for epoch in range(self.num_epochs):                  # set to be 101
            self.model.train()
            # logging.info("start train")
            for row in iter(self.train_data_loader):
                # logging.info("train: read one batch")
                user_id = row["user_id"]
                input_ids = row["input_ids"].to(self.device, non_blocking=True)                                                    # [B, N]
                raw_input_embeddings = row["raw_input_embeddings"].to(dtype=torch.float32, device=self.device, non_blocking=True)  # [B, N, D]
                ratings = row["ratings"].to(self.device, non_blocking=True)                                                        # [B, N]
                timestamps = row["timestamps"].to(self.device, non_blocking=True)                                                  # [B, N]
                type_ids = row["type_ids"].to(self.device, non_blocking=True) if "type_ids" in row else None                           # [B, N]
                lengths = row["lengths"].to(self.device, non_blocking=True)                                                        # [B]

                new_input_ids = input_ids[:, :-1]                            # [B, N-1]
                label_ids     = input_ids[:, 1:]                             # [B, N-1]
                new_raw_input_embeddings = raw_input_embeddings[:, :-1, :]   # [B, N-1, D]
                raw_label_embeddings     = raw_input_embeddings[:, 1:, :]    # [B, N-1, D]
                new_ratings = ratings                                # ignore ratings for now
                new_timestamps = timestamps[:, :-1]                          # [B, N-1]
                leak_next_type = bool(getattr(self._model_unwrapped, "enable_next_event_type_leakage", False))
                if type_ids is not None:
                    new_type_ids = type_ids[:, 1:] if leak_next_type else type_ids[:, :-1]
                else:
                    new_type_ids = None
                new_lengths = lengths - 1                                    # [B]

                # print(f"new input ids shape: {new_input_ids.shape}")
                # print(f"new timestamps shape: {new_timestamps.shape}")
                
                # Perform evaluation every 1000 iterations
                if batch_id % 1000 == 0:
                    if self.rank == 0:
                        logging.info(f"Saving snapshot at batch {batch_id}")
                        self._save_snapshot(batch_id)

                    logging.info("rotating negatives sampler")
                    self._model_unwrapped.negatives_sampler['eval'].rotate()

                    logging.info(f"running evaluation (method={self.eval_method})")
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    self.evaluate(
                        batch_id, 
                        epoch
                    )
                    torch.cuda.empty_cache()
                    
                    self.model.train()

                self.opt.zero_grad()
                logits, loss, metrics = self.model(
                    input_ids=new_input_ids,
                    raw_input_embeddings=new_raw_input_embeddings,
                    input_lengths=new_lengths,
                    label_ids=label_ids,
                    raw_label_embeddings=raw_label_embeddings,
                    ratings=new_ratings,
                    type_ids=new_type_ids,
                    timestamps=new_timestamps,
                    user_ids=user_id,
                )

                # logging.info(f"iteration: {batch_id}, loss: {loss:.6f}")

                loss.backward()

                # Optional linear warmup.
                if batch_id < self.num_warmup_steps:
                    lr_scalar = min(1.0, float(batch_id + 1) / self.num_warmup_steps)
                    for pg in self.opt.param_groups:
                        pg["lr"] = lr_scalar * self.learning_rate
                    lr = lr_scalar * self.learning_rate
                else:
                    lr = self.learning_rate

                if (batch_id % 10) == 0:     # logging loss every 10 iterations
                    logging.info(
                        f" rank: {self.rank}, batch-stat (train): iteration {batch_id} epoch {epoch} in {time.time() - last_training_time:.2f}s): {loss:.6f}"
                    )
                    last_training_time = time.time()
                    if self.rank == 0:
                        assert self.writer is not None
                        self.writer.add_scalar("loss/train", loss, batch_id)
                        self.writer.add_scalar("lr", lr, batch_id)

                        for k, v in metrics.items():
                            self.writer.add_scalar(f"train/{k}", v, batch_id)
                            logging.info(f"{k}:{v}")
                self.opt.step()
                batch_id += 1
        

        if self.rank == 0:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

        self.cleanup()


    def evaluate(self, train_batch_id: int, train_epoch: int) -> None:
        """Unified evaluation dispatcher — routes to the configured eval method.

        eval_method:
            "pplx"      — log perplexity eval (eval_metrics_v3, fast)
            "retrieval"  — single-domain retrieval eval (eval_metrics_v2)
            "sharded"    — sharded multi-rank retrieval eval
        """
        if self.eval_method == "pplx":
            self.run_evaluation_with_pplx(train_batch_id, train_epoch)
        elif self.eval_method == "sharded":
            self.run_sharded_evaluation(train_batch_id, train_epoch)
        elif self.eval_method == "retrieval":
            self.run_evaluation(train_batch_id, train_epoch)
        else:
            logging.warning(f"Unknown eval_method '{self.eval_method}', falling back to pplx")
            self.run_evaluation_with_pplx(train_batch_id, train_epoch)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Config-driven optimizer factory."""
        params = self.model.parameters()
        if self.optimizer_type == "AdamW":
            return torch.optim.AdamW(
                params, lr=self.learning_rate,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay,
                foreach=False,
            )
        elif self.optimizer_type == "Adam":
            return torch.optim.Adam(
                params, lr=self.learning_rate,
                betas=self.optimizer_betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(
                params, lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

    def run_evaluation_with_pplx(self, train_batch_id, train_epoch) -> None:
        self.model.eval()

        eval_state = get_eval_state_v2(
            model=self._model_unwrapped,
            top_k_method=self.top_k_method,
        )

        batch_id = 0
        eval_dict_all = defaultdict(list)
        eval_start_time = time.time()
        for row in iter(self.eval_data_loader):
            # logging.info("eval: read one batch")
            user_id = row["user_id"]
            input_ids = row["input_ids"].to(self.device, non_blocking=True)                                                    # [B, N]
            raw_input_embeddings = row["raw_input_embeddings"].to(dtype=torch.float32, device=self.device, non_blocking=True)  # [B, N, D]
            ratings = row["ratings"].to(self.device, non_blocking=True)                                                        # [B, N]
            timestamps = row["timestamps"].to(self.device, non_blocking=True)                                                  # [B, N]
            type_ids = row["type_ids"].to(self.device, non_blocking=True) if "type_ids" in row else None                           # [B, N]
            lengths = row["lengths"].to(self.device, non_blocking=True)                                                        # [B]


            eval_dict = eval_metrics_v3_from_tensors(
                eval_state,
                self._model_unwrapped,
                input_ids,
                raw_input_embeddings,
                ratings,
                timestamps,
                lengths,
                user_id,
                type_ids=type_ids,
                leak_next_type_ids=bool(getattr(self._model_unwrapped, "enable_next_event_type_leakage", False)),
            )

            for k, v in eval_dict.items():
                eval_dict_all[k].append(v)

            batch_id += 1
            if batch_id >= self.eval_max_batches:
                break

        assert eval_dict_all is not None
        for k, v_list in eval_dict_all.items():
            v_tensor_list = [t.unsqueeze(0) if t.dim() == 0 else t for t in v_list]
            eval_dict_all[k] = torch.cat(v_tensor_list, dim=-1)  # shape [N]

        # Average across ranks
        # logging.info(f"eval_dict len: {len(eval_dict_all)}")
        final_metrics = {
            k: _avg(v, rank=self.rank, world_size=self.world_size)
            for k, v in eval_dict_all.items()
        }
        # logging.info("done with cross-rank avg")

    
        # Write logs from rank 0 only
        if self.rank == 0:
            # logging.info(f"rank {self.rank}: write metrics to log")
            add_to_summary_writer(
                self.writer,
                batch_id=train_batch_id,
                metrics=final_metrics,
                prefix="eval_iter",
            )

            logging.info(
                f"rank {self.rank}: eval @ 'eval iteration' {train_batch_id} "
                f"'train_iteration' {train_batch_id} 'epoch' {train_epoch} "
                f"in {time.time() - eval_start_time:.2f}s: "
                + ", ".join(f"{k} {final_metrics[k].item():.4f}" for k in final_metrics)
        )




    def run_sharded_evaluation(self, train_batch_id, train_epoch) -> None:
        self.model.eval()

        domain_id = 0
        num_shards = 35
        # logging.info(f"{self.dataset.domain_to_item_id_range}")
        item_id_range = self.dataset.domain_to_item_id_range[domain_id]

        # Assign subset of shards to this rank
        assigned_shards = [s for s in range(num_shards) if s % self.world_size == self.rank]
        if not assigned_shards:
            logging.warning(f"Rank {self.rank} has no assigned shards. Skipping evaluation.")
            
            # Still return valid metrics so that _avg() across ranks doesn't hang
            dummy_metrics = {
                "ndcg_1": torch.empty(0, device=self.device),
                "ndcg_10": torch.empty(0, device=self.device),
                "ndcg_50": torch.empty(0, device=self.device),
                "ndcg_100": torch.empty(0, device=self.device),
                "ndcg_200": torch.empty(0, device=self.device),
                "hr_1": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_10": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_50": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_100": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_200": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_500": torch.empty(0, dtype=torch.float32, device=self.device),
                "hr_1000": torch.empty(0, dtype=torch.float32, device=self.device),
                "mrr": torch.empty(0, device=self.device),
            }
            
            final_metrics = {
                k: _avg(v, rank=self.rank, world_size=self.world_size)
                for k, v in dummy_metrics.items()
            }
            return

        eval_dict_all = defaultdict(list)
        eval_start_time = time.time()

        # logging.info("start get eval state")
        eval_state = get_eval_state(
            model=self._model_unwrapped,
            item_id_range=item_id_range,
            top_k_method=self.top_k_method,
            domain_id=domain_id,
            assigned_shards=assigned_shards,
            device=self.device,
            float_dtype=torch.bfloat16 if self.main_module_bf16 else None,
        )
        # logging.info("done get eval state")

        batch_id = 0
        for row in iter(self.eval_data_loader):
            # logging.info("eval: read one batch")
            input = row["input"].to(self.device)                # [B, N]
            ratings = row["ratings"].to(self.device)            # [B, N]
            label = row["label"].to(self.device).unsqueeze(1)   # [B, 1]
            timestamps = row["timestamps"].to(self.device)
            length = row["length"].to(self.device)              # [B]
            raw_input_embedding = row["raw_input_embedding"].to(self.device)  # [B, N, D]

            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                self._model_unwrapped.model,
                input,
                raw_input_embedding,
                ratings,
                label,
                timestamps,
                length,
                target_ratings=None,
                user_max_batch_size=self.eval_user_max_batch_size,
                dtype=torch.bfloat16 if self.main_module_bf16 else None,
            )

            for k, v in eval_dict.items():
                eval_dict_all[k].append(v)

            batch_id += 1

            logging.info(f"eval @ 'eval iteration' {batch_id} ")
            if batch_id >= self.eval_max_batches:
                break

        # Merge metric tensors per rank
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)

        # Average across ranks
        # logging.info(f"eval_dict len: {len(eval_dict_all)}")
        final_metrics = {
            k: _avg(v, rank=self.rank, world_size=self.world_size)
            for k, v in eval_dict_all.items()
        }
        # logging.info("done with cross-rank avg")

    
        # Write logs from rank 0 only
        if self.rank == 0:
            # logging.info(f"rank {self.rank}: write metrics to log")
            add_to_summary_writer(
                self.writer,
                batch_id=train_batch_id,
                metrics=final_metrics,
                prefix="eval_iter",
            )

            logging.info(
                f"rank {self.rank}: eval @ 'eval iteration' {train_batch_id} "
                f"'train_iteration' {train_batch_id} 'epoch' {train_epoch} "
                f"in {time.time() - eval_start_time:.2f}s: "
                + ", ".join(f"{k} {final_metrics[k].item():.4f}" for k in final_metrics)
        )
        # logging.info("done eval")

    def run_evaluation(self, train_batch_id, train_epoch) -> None:
        self.model.eval()

        eval_state = get_eval_state(
            model=self._model_unwrapped,
            item_id_range=self.dataset.domain_to_item_id_range[0],
            top_k_method=self.top_k_method,
            domain_id=0,
            num_shards=self._model_unwrapped.model._embedding_module.domain_shard_counts[0],
            device=self.device,
            float_dtype=torch.bfloat16 if self.main_module_bf16 else None,
        )

        batch_id = 0
        eval_dict_all = defaultdict(list)
        eval_start_time = time.time()
        for row in iter(self.eval_data_loader):
            input = row["input"].to(self.device)                # [B, N]
            ratings = row["ratings"].to(self.device)            # [B, N]
            label = row["label"].to(self.device).unsqueeze(1)        # [B, 1]
            timestamps = row["timestamps"].to(self.device)
            length = row["length"].to(self.device)              # [B]

            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                self._model_unwrapped.model,
                input,
                ratings,
                label,
                timestamps,
                length,
                target_ratings=None,
                user_max_batch_size=self.eval_user_max_batch_size,
                dtype=torch.bfloat16 if self.main_module_bf16 else None,
            )

            for k, v in eval_dict.items():
                eval_dict_all[k].append(v)

            batch_id += 1

        assert eval_dict_all is not None
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)

        ndcg_10 = _avg(eval_dict_all["ndcg_10"], rank=self.rank, world_size=self.world_size)
        ndcg_50 = _avg(eval_dict_all["ndcg_50"], rank=self.rank, world_size=self.world_size)
        hr_10 = _avg(eval_dict_all["hr_10"], rank=self.rank,world_size=self.world_size)
        hr_50 = _avg(eval_dict_all["hr_50"], rank=self.rank,world_size=self.world_size)
        mrr = _avg(eval_dict_all["mrr"], rank=self.rank,world_size=self.world_size)

        add_to_summary_writer(
            self.writer,
            batch_id=train_batch_id,
            metrics=eval_dict_all,
            prefix="eval_iter",
            rank=self.rank,
            world_size=self.world_size,
        )

        logging.info(
            f"rank {self.rank}: eval @ 'eval iteration' {batch_id} 'train_iteration' {train_batch_id} 'epoch' {train_epoch} in {time.time() - eval_start_time:.2f}s: "
            f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
        )

    def cleanup(self) -> None:
        dist.destroy_process_group()


    def _save_snapshot(self, batch_id):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(self.snapshot_dir, f"checkpoint_batch{batch_id:07d}.pt")
        tmp_path = snapshot_path + ".tmp"
        if os.path.exists(snapshot_path):
            return
            
        snapshot = {
            "MODEL_STATE": self._model_unwrapped.state_dict(),
            "OPTIMIZER_STATE": self.opt.state_dict(),
            "BATCHES_RUN": batch_id,
        }

        # atomic save
        try:
            torch.save(snapshot, tmp_path)
            os.replace(tmp_path, snapshot_path)
            logging.info(f"Batch {batch_id} | Training snapshot saved at {snapshot_path}")
        except Exception as e:
            logging.info(f"⚠️ Snapshot save failed at batch {batch_id}: {e}")
            # cleanup partial temp
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def _load_latest_snapshot(self):
        if not os.path.exists(self.snapshot_dir):
            logging.info("No checkpoint directory found.")
            return False

        checkpoint_files = glob.glob(os.path.join(self.snapshot_dir, "checkpoint_batch*.pt"))
        if not checkpoint_files:
            logging.info("No checkpoint files found.")
            return False

        # Sort by batch number extracted from filename
        checkpoint_files.sort(key=lambda f: int(os.path.basename(f).split("batch")[1].split(".")[0]))

        latest_snapshot = checkpoint_files[-1]
        logging.info(f"Loading latest snapshot: {latest_snapshot}")
        try:
            snapshot = torch.load(latest_snapshot, map_location="cpu")
        except Exception as e:
            logging.warning(f"Failed to load snapshot {latest_snapshot}: {e}. Starting fresh.")
            return False

        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.opt.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.batches_run = snapshot["BATCHES_RUN"]

        logging.info(f"Resuming training from batch {self.batches_run}")
        return True