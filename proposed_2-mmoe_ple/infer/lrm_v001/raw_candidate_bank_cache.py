"""Raw frozen candidate-bank cache for selected-bank v001 proxy evals.

This cache stores model-independent raw item embeddings for selected banks and
projects them with the active model at inference time. It avoids repeated random
mmap lookups into the huge global embedding shards while preserving the model's
trainable projection layer.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import time
from typing import Any


def _sync_timing_device(device: str, enabled: bool) -> None:
    if not enabled or not str(device).startswith("cuda"):
        return
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))
    except Exception:
        return


class RawCandidateBankCache:
    CACHE_VERSION = "raw_candidate_bank_cache_v001"

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        model_digest: str,
        max_banks: int = 0,
        device: str = "cuda:0",
        placement: str = "gpu",
        timing_sync_cuda: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_digest = model_digest
        self.device = device
        self.placement = placement
        self.timing_sync_cuda = timing_sync_cuda
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "lrm_v001_raw_candidate_bank_cache_v001":
            raise ValueError(f"unsupported raw bank cache schema: {self.manifest.get('schema')}")
        self.max_banks = int(max_banks) if int(max_banks or 0) > 0 else int(self.manifest.get("total_banks") or 0)
        self.entries: collections.OrderedDict[tuple[str, int, int], dict[str, Any]] = collections.OrderedDict()
        self.current_bytes = 0
        self.domains: dict[int, dict[str, Any]] = {}
        self.stats: dict[str, Any] = {
            "enabled": self.enabled,
            "cache_version": self.CACHE_VERSION,
            "model_digest": self.model_digest,
            "raw_cache_dir": str(self.cache_dir),
            "raw_cache_digest": self.manifest.get("digest"),
            "placement": self.placement,
            "max_banks": self.max_banks,
            "device": self.device,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "raw_load_s": 0.0,
            "raw_transfer_s": 0.0,
            "projection_s": 0.0,
            "resident_raw_bytes": 0,
        }
        self._load_raw_store()

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _tensor_bytes(tensor) -> int:
        return int(tensor.nelement() * tensor.element_size())

    def snapshot(self) -> dict[str, Any]:
        snap = dict(self.stats)
        requests = int(snap.get("requests") or 0)
        hits = int(snap.get("hits") or 0)
        snap["hit_rate"] = (hits / requests) if requests else None
        snap["resident_banks"] = len(self.entries)
        snap["resident_bytes"] = self.current_bytes
        return snap

    def _key(self, domain_id: int, bank_id: int) -> tuple[str, int, int]:
        return (self.model_digest, int(domain_id), int(bank_id))

    def _load_raw_store(self) -> None:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        if self.placement not in {"gpu", "cpu"}:
            raise ValueError(f"unsupported raw bank cache placement: {self.placement}")
        for domain_key, meta in sorted(self.manifest["domains"].items(), key=lambda kv: int(kv[0])):
            domain_id = int(domain_key)
            t0 = time.perf_counter()
            bank_ids = np.load(self.cache_dir / meta["bank_ids_file"])
            candidate_ids = np.load(self.cache_dir / meta["candidate_ids_file"], mmap_mode="r")
            raw_np = np.load(self.cache_dir / meta["raw_embeddings_file"], mmap_mode="r")
            bank_to_index = {int(bank_id): idx for idx, bank_id in enumerate(bank_ids.tolist())}
            raw_obj: Any = raw_np
            if self.placement == "gpu":
                # Force one sequential read of the packed selected-bank cache and keep it on GPU.
                raw_obj = torch.from_numpy(np.asarray(raw_np)).to(device=self.device)
                self.stats["resident_raw_bytes"] += self._tensor_bytes(raw_obj)
            else:
                # Keep mmap-backed CPU store. Individual banks transfer to GPU on demand.
                raw_obj = raw_np
            self.stats["raw_load_s"] += time.perf_counter() - t0
            self.domains[domain_id] = {
                "bank_to_index": bank_to_index,
                "candidate_ids": candidate_ids,
                "raw": raw_obj,
                "candidate_count": int(meta["candidate_count"]),
                "raw_dim": int(meta["raw_dim"]),
            }

    def _project_raw_bank(self, model, raw_bank):
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        with torch.inference_mode():
            if not torch.is_tensor(raw_bank):
                t0 = time.perf_counter()
                raw_bank = torch.from_numpy(raw_bank).to(device=self.device)
                self.stats["raw_transfer_s"] += time.perf_counter() - t0
            elif str(raw_bank.device) != str(torch.device(self.device)):
                t0 = time.perf_counter()
                raw_bank = raw_bank.to(device=self.device)
                self.stats["raw_transfer_s"] += time.perf_counter() - t0
            _sync_timing_device(self.device, self.timing_sync_cuda)
            t0 = time.perf_counter()
            projected = model.model._embedding_module.proj(raw_bank.float())
            projected = F.normalize(projected.float(), p=2, dim=-1).detach()
            _sync_timing_device(self.device, self.timing_sync_cuda)
            self.stats["projection_s"] += time.perf_counter() - t0
            return projected

    def get_bank(self, *, model, generator_mod, bank_artifact, domain_id: int, bank_id: int):
        key = self._key(domain_id, bank_id)
        self.stats["requests"] += 1
        if key in self.entries:
            self.stats["hits"] += 1
            entry = self.entries.pop(key)
            self.entries[key] = entry
            return entry, {"event": "raw_cache_hit", "key": {"domain_id": domain_id, "bank_id": bank_id}}

        self.stats["misses"] += 1
        domain = self.domains.get(int(domain_id))
        if domain is None or int(bank_id) not in domain["bank_to_index"]:
            raise KeyError(f"raw bank cache does not contain domain={domain_id} bank={bank_id}")
        idx = domain["bank_to_index"][int(bank_id)]
        candidate_ids_arr = domain["candidate_ids"][idx]
        base_ids = [str(int(x)) for x in candidate_ids_arr.tolist()]
        raw_bank = domain["raw"][idx]
        matrix = self._project_raw_bank(model, raw_bank)
        entry = {
            "candidate_ids": base_ids,
            "embeddings": matrix,
            "domain_id": int(domain_id),
            "bank_id": int(bank_id),
        }
        self.entries[key] = entry
        self.current_bytes += self._tensor_bytes(matrix)
        while len(self.entries) > self.max_banks:
            _, old = self.entries.popitem(last=False)
            self.current_bytes -= self._tensor_bytes(old["embeddings"])
            self.stats["evictions"] += 1
        return entry, {"event": "raw_project", "key": {"domain_id": domain_id, "bank_id": bank_id}}
