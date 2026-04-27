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

import abc
import os
import torch
from registry import register
import numpy as np
from typing import Tuple, Dict, List
from modeling.initialization import truncated_normal
import threading
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
from cachetools import LRUCache, LFUCache
import time
from modeling.sequential.layer_norm import LayerNorm, SwishLayerNorm


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


@register("embedding", "local")
class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        max_item_id: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            max_item_id + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


@register("embedding", "CategoricalEmbedding")
class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


def init_mlp_weights_optional_bias(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

@register("embedding", "MultiDomainPrecomputed")
class MultiDomainPrecomputedEmbeddingModule(EmbeddingModule):
    def __init__(self, domain_to_item_id_range: Dict[int, Tuple[int, int]], shard_dirs: dict, preload: bool = False, input_dim: int = 768, output_dim: int = 256, shard_size: int = 25_000_000, domain_offset: int = 1_000_000_000) -> None:
        """
        Args:
            shard_dirs: Dict of {domain_id: directory containing .npy shards}
            shard_size: Number of embeddings per shard (used for range-based lookup)
        """
        super().__init__()

        self.domain_to_item_id_range = domain_to_item_id_range
        self.shard_dirs = shard_dirs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shard_size = shard_size
        self.domain_offset = domain_offset

        # self.proj = torch.nn.Linear(input_dim, output_dim)

        self.proj: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=1024,
            ),
            SwishLayerNorm(1024),
            torch.nn.Linear(
                in_features=1024,
                out_features=self.output_dim,
            ),
            LayerNorm(self.output_dim),
        ).apply(init_mlp_weights_optional_bias)
        
        
        self.domain_shard_counts = {}

        # self.loaded_shards = {}
        self.loaded_shards = LFUCache(maxsize=64)  # (domain_id, shard_idx) -> ndarray

 
    def _load_shard(self, domain_id: int, shard_idx: int) -> np.ndarray:
        key = (domain_id, shard_idx)
        if key in self.loaded_shards:
            return self.loaded_shards[key]

        shard_path = os.path.join(self.shard_dirs[domain_id], f"shard_{shard_idx}.npy")
        array = np.load(shard_path, mmap_mode="r")  # lazy mmap
        self.loaded_shards[key] = array
        return array

    def _decode(self, encoded_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        domain_ids = encoded_ids // self.domain_offset
        item_ids = encoded_ids % self.domain_offset
        return domain_ids, item_ids

    def debug_str(self) -> str:
        return f"multi_domain_precomputed_emb"

    @property
    def item_embedding_dim(self) -> int:
        return self.output_dim

    def get_raw_item_embeddings(self, encoded_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded_ids: Tensor of shape [B, N] or [K], where each element is domain_id * OFFSET + item_id
        Returns:
            Tensor of shape [B, N, D] or [K, D]
        """
        original_shape = encoded_ids.shape
        flat_ids = encoded_ids.view(-1)
        device = flat_ids.device
        K = flat_ids.size(0)

        # Decode
        domain_ids, item_ids = self._decode(flat_ids)
        domain_ids_np = domain_ids.cpu().numpy()
        item_ids_np = item_ids.cpu().numpy()

        shard_idxs = item_ids_np // self.shard_size
        offsets = item_ids_np % self.shard_size
        keys = np.stack((domain_ids_np, shard_idxs), axis=1)

        # Group by shard key
        shard_requests = defaultdict(list)
        for idx, (domain_id, shard_idx) in enumerate(keys):
            shard_requests[(domain_id, shard_idx)].append((idx, offsets[idx]))
    
        # Benchmark timers
        t0 = time.time()
        io_time = 0.0
        conversion_time = 0.0
        embeddings = torch.empty(K, self.input_dim, dtype=torch.float16)
        for (domain_id, shard_idx), positions in shard_requests.items():
            shard_array = self._load_shard(domain_id, shard_idx)  # I/O
            indices, local_offsets = zip(*positions)
            local_offsets_np = np.array(local_offsets, dtype=np.int64)
            t_start = time.time()
            batch_embeddings = torch.from_numpy(shard_array[local_offsets_np])  # conversion
            io_time += time.time() - t_start
     
            t_start = time.time()
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            embeddings[indices_tensor] = batch_embeddings
            conversion_time += time.time() - t_start
     
        t_total = time.time() - t0
        # print(f"[Lookup] total: {t_total:.3f}s | I/O: {io_time:.3f}s | conversion: {conversion_time:.3f}s")
        # Reshape and project
        embeddings = embeddings.view(*original_shape, self.input_dim).to(device)
        return embeddings

    def get_item_embeddings(self, encoded_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded_ids: Tensor of shape [B, N] or [K], where each element is domain_id * OFFSET + item_id
        Returns:
            Tensor of shape [B, N, D] or [K, D]
        """
        raw_embeddings = self.get_raw_item_embeddings(encoded_ids)
        return self.proj(raw_embeddings.to(dtype=torch.float32))

    def get_raw_shard_embeddings(self, domain_id: int, shard_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_array = self._load_shard(domain_id, shard_idx)
        embeddings = torch.from_numpy(shard_array)
 
        shard_start = shard_idx * self.shard_size
        shard_end = shard_start + self.shard_size
        min_id, max_id = self.domain_to_item_id_range[domain_id]
        valid_start = max(shard_start, min_id)
        valid_end = min(shard_end, max_id + 1)  # +1 because Python slicing is exclusive
        item_ids = torch.arange(valid_start, valid_end)
        return item_ids, embeddings

    def get_shard_embeddings(self, domain_id: int, shard_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item IDs and embeddings for a specific shard, clipped to the valid item ID range.
        Args:
            domain_id: Domain ID
            shard_idx: Shard index
        Returns:
            A tuple of:
                - item_ids: Tensor of shape [<=shard_size]
                - embeddings: Tensor of shape [<=shard_size, D]
        """
        item_ids, embeddings = self.get_raw_shard_embeddings(domain_id, shard_idx)

        import copy
        proj_cpu = copy.deepcopy(self.proj).cpu()
        projected = proj_cpu(embeddings)
    
        return item_ids, projected

    def forward(self, raw_input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to project raw input embeddings.
        Args:
            raw_input_embeddings: Tensor of shape [B, N, D]
        Returns:
            Projected embeddings of shape [B, N, output_dim]
        """
        if raw_input_embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {raw_input_embeddings.shape[-1]}")
        
        return self.proj(raw_input_embeddings)


@register("embedding", "xlm_roberta_base_proj")
class XLMRobertaBaseProjEmbeddingModule(EmbeddingModule):
    def __init__(self, input_dim: int = 768, output_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.proj = torch.nn.Linear(input_dim, output_dim)
        self.proj: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=768,
                out_features=1024,
            ),
            SwishLayerNorm(1024),
            torch.nn.Linear(
                in_features=1024,
                out_features=256,
            ),
            LayerNorm(256),
        ).apply(init_mlp_weights_optional_bias)

        MODEL_NAME = "xlm-roberta-base"
        self.model = AutoModel.from_pretrained(MODEL_NAME)

    def debug_str(self) -> str:
        return f"xlm_roberta_base_linear_proj"

    @property
    def item_embedding_dim(self) -> int:
        return self.output_dim


    def get_item_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape [B, L, T] or [K, T]
                - [B, L, T]: B = batch size, L = max items per user, T = token length
                - [K, T]: K = total number of items
            padded token ID: self.model.config.pad_token_id
        Example:
            inputs.shape = [1, 5, 6]  # B=1, L=5, T=6
            inputs[0] =
                [
                    [66, 82, 21, 345, 2, 1],    # "I am happy"
                    [70, 455, 16, 8973, 2, 1],  # "today is Monday"
                    [1, 1, 1, 1, 1, 1],        # padded item
                    [1, 1, 1, 1, 1, 1],        # padded item
                    [1, 1, 1, 1, 1, 1],        # padded item
                ]
        Returns:
            item_embeddings: Tensor of shape [B, L, D] or [K, D]
        """
        if inputs.dim() == 3:
            # [B, L, T]
            B, L, T = inputs.shape
            flat_inputs = inputs.view(B * L, T)
            output_shape = (B, L, self.output_dim)
        elif inputs.dim() == 2:
            # [K, T]
            K, T = inputs.shape
            flat_inputs = inputs
            output_shape = (K, self.output_dim)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

        # Attention mask: 1 for real tokens, 0 for pad
        attention_mask = (flat_inputs != self.model.config.pad_token_id).long()

        outputs = self.model(input_ids=flat_inputs, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [N, T, 768]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # [N, T, 1]
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)                  # [N, 768]
        counts = mask.sum(dim=1).clamp(min=1e-5)    # [N, 1]
        mean_pooled = summed / counts               # [N, 768]

        projected = self.proj(mean_pooled)          # [N, output_dim]

        return projected.view(*output_shape)


@register("embedding", "pinsage_proj")
class PinSageProjEmbeddingModule(EmbeddingModule):
    def __init__(self, input_dim: int = 64, output_dim: int = 256, ckpt_path: str = "") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.proj = torch.nn.Linear(input_dim, output_dim)
        self.proj: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.input_dim,
                out_features=1024,
            ),
            SwishLayerNorm(1024),
            torch.nn.Linear(
                in_features=1024,
                out_features=self.output_dim,
            ),
            LayerNorm(self.output_dim),
        ).apply(init_mlp_weights_optional_bias)

        from modeling.sequential.pinsage.model.PinSageEncoder import PinSageEncoder
        self.model = PinSageEncoder.load(ckpt_path)

    def debug_str(self) -> str:
        return f"pinsage_linear_proj"

    @property
    def item_embedding_dim(self) -> int:
        return self.output_dim


    def get_item_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape [B, L, T] or [K, T]
                - [B, L, T]: B = batch size, L = max items per user, T = token length
                - [K, T]: K = total number of items
            padded token ID: self.model.config.pad_token_id
        Example:
            inputs.shape = [1, 5, 6]  # B=1, L=5, T=6
            inputs[0] =
                [
                    [66, 82, 21, 345, 2, 1],    # "I am happy"
                    [70, 455, 16, 8973, 2, 1],  # "today is Monday"
                    [1, 1, 1, 1, 1, 1],        # padded item
                    [1, 1, 1, 1, 1, 1],        # padded item
                    [1, 1, 1, 1, 1, 1],        # padded item
                ]
        Returns:
            item_embeddings: Tensor of shape [B, L, D] or [K, D]
        """
        if inputs.dim() == 3:
            # [B, L, T]
            B, L, T = inputs.shape
            flat_inputs = inputs.view(B * L, T)
            output_shape = (B, L, self.output_dim)
        elif inputs.dim() == 2:
            # [K, T]
            K, T = inputs.shape
            flat_inputs = inputs
            output_shape = (K, self.output_dim)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

        # Attention mask: 1 for real tokens, 0 for pad
        attention_mask = (flat_inputs != self.model.config.pad_token_id).long()

        outputs = self.model(input_ids=flat_inputs, attention_mask=attention_mask)
        projected = self.proj(outputs)          # [N, output_dim]
        return projected.view(*output_shape)