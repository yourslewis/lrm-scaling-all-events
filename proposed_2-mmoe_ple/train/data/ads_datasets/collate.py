import torch
from typing import Dict
import os
import logging
from modeling.sequential.embedding_modules import (
    MultiDomainPrecomputedEmbeddingModule,
)
import time

class CollateFn:
    def __init__(
        self,
        device: torch.device,
        domain_to_item_id_range: dict,
        domain_offset: int,
        precomputed_embeddings_domain_to_dir = None,
        input_dim: int = 384,
        output_dim: int = 128,
        shard_size: int = 34_000_000,
    ):
        
        self.device = device
        self.embedding_module = None
        
        self.domain_to_item_id_range = domain_to_item_id_range
        self.precomputed_embeddings_domain_to_dir = precomputed_embeddings_domain_to_dir
        self.domain_offset = domain_offset
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shard_size = shard_size

    def _init_embedding_module(self):
        if self.embedding_module is None:
            logging.info(f"[PID {os.getpid()}] Initializing embedding module (input_dim={self.input_dim}, output_dim={self.output_dim}, shard_size={self.shard_size})")
            self.embedding_module = MultiDomainPrecomputedEmbeddingModule(
                domain_to_item_id_range=self.domain_to_item_id_range,
                shard_dirs=self.precomputed_embeddings_domain_to_dir,
                preload=False,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                shard_size=self.shard_size,
                domain_offset=self.domain_offset,
            )

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        self._init_embedding_module()  # safely lazy-init per process

        input_ids_tensor = torch.stack([sample["input_ids"] for sample in batch])
        timestamps_tensor = torch.stack([sample["timestamps"] for sample in batch])
        length_tensor = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
        ratings_tensor = torch.tensor([sample["ratings"] for sample in batch], dtype=torch.long)
        user_id_tensor = [sample["user_id"] for sample in batch]

        # type_ids may not be present in all datasets
        if "type_ids" in batch[0]:
            type_ids_tensor = torch.stack([sample["type_ids"] for sample in batch])
        else:
            type_ids_tensor = torch.zeros_like(input_ids_tensor)

        result = {
            "input_ids": input_ids_tensor,
            "timestamps": timestamps_tensor,
            "lengths": length_tensor,
            "user_id": user_id_tensor,
            "ratings": ratings_tensor,
            "type_ids": type_ids_tensor,
        }

        
        start = time.time()
        raw_input_embeddings = self.embedding_module.get_raw_item_embeddings(input_ids_tensor)  # [B, N, D]
        result["raw_input_embeddings"] = raw_input_embeddings
        return result
