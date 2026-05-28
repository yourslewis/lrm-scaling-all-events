"""Execution-side projected candidate-bank embedding cache for v001 inference.

This module deliberately does not change the official v001 target/candidate or
prediction schemas. It only accelerates scoring by caching model-projected,
L2-normalized embeddings for reusable banked negative IDs.
"""
from __future__ import annotations

import collections
import os
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


class CandidateEmbeddingCache:
    """LRU cache for projected+normalized base negative-bank embeddings.

    Keying includes the model/checkpoint digest, domain, and bank id so cached
    matrices are never shared across incompatible model artifacts. Disk cache is
    optional and stores only derived execution artifacts.
    """

    CACHE_VERSION = "candidate_embedding_cache_v001"

    def __init__(
        self,
        *,
        model_digest: str,
        max_banks: int,
        device: str,
        chunk_size: int,
        disk_dir: str | None = None,
        disk_dtype: str = "float32",
        timing_sync_cuda: bool = False,
    ) -> None:
        self.model_digest = model_digest
        self.max_banks = max(0, int(max_banks))
        self.device = device
        self.chunk_size = max(1, int(chunk_size))
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.disk_dtype = disk_dtype
        self.timing_sync_cuda = timing_sync_cuda
        self.entries: collections.OrderedDict[tuple[str, int, int], dict[str, Any]] = collections.OrderedDict()
        self.current_bytes = 0
        self.stats: dict[str, Any] = {
            "enabled": self.enabled,
            "cache_version": self.CACHE_VERSION,
            "model_digest": self.model_digest,
            "max_banks": self.max_banks,
            "device": self.device,
            "disk_dir": str(self.disk_dir) if self.disk_dir else None,
            "disk_dtype": self.disk_dtype if self.disk_dir else None,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_loads": 0,
            "disk_writes": 0,
            "materialize_s": 0.0,
            "projection_s": 0.0,
            "disk_load_s": 0.0,
            "disk_write_s": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return self.max_banks > 0

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

    def _disk_path(self, domain_id: int, bank_id: int) -> Path | None:
        if self.disk_dir is None:
            return None
        digest_slug = self.model_digest.replace("sha256:", "sha256_").replace(":", "_")
        return (
            self.disk_dir
            / self.CACHE_VERSION
            / digest_slug
            / f"domain_{int(domain_id)}"
            / f"bank_{int(bank_id):04d}.{self.disk_dtype}.npy"
        )

    @staticmethod
    def _tensor_bytes(tensor) -> int:
        return int(tensor.nelement() * tensor.element_size())

    def _project_normalized_ids(self, model, ids: list[str]):
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        chunks = []
        ids_int = [int(x) for x in ids]
        with torch.inference_mode():
            for start in range(0, len(ids_int), self.chunk_size):
                cur_ids = ids_int[start : start + self.chunk_size]
                cand = torch.tensor(cur_ids, dtype=torch.long, device=self.device)
                emb = model.model._embedding_module.get_item_embeddings(cand).float()
                chunks.append(F.normalize(emb, p=2, dim=-1).detach())
        if not chunks:
            raise RuntimeError("cannot cache empty candidate bank")
        return torch.cat(chunks, dim=0)

    def _load_disk_matrix(self, path: Path, *, expected_shape: tuple[int, int]):
        import numpy as np  # type: ignore
        import torch  # type: ignore

        if not path.exists():
            return None
        t0 = time.perf_counter()
        arr = np.load(path, mmap_mode="r")
        if tuple(arr.shape) != expected_shape:
            return None
        matrix = torch.from_numpy(np.asarray(arr)).to(device=self.device, dtype=torch.float32)
        self.stats["disk_loads"] += 1
        self.stats["disk_load_s"] += time.perf_counter() - t0
        return matrix

    def _write_disk_matrix(self, path: Path, matrix) -> None:
        import numpy as np  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        arr = matrix.detach().cpu().numpy()
        if self.disk_dtype != "float32":
            raise ValueError(f"unsupported candidate cache disk dtype: {self.disk_dtype}")
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp, path)
        self.stats["disk_writes"] += 1
        self.stats["disk_write_s"] += time.perf_counter() - t0

    def get_bank(self, *, model, generator_mod, bank_artifact, domain_id: int, bank_id: int):
        if not self.enabled:
            raise RuntimeError("candidate embedding cache is disabled")
        key = self._key(domain_id, bank_id)
        self.stats["requests"] += 1
        if key in self.entries:
            self.stats["hits"] += 1
            entry = self.entries.pop(key)
            self.entries[key] = entry
            return entry, {"event": "hit", "key": {"domain_id": domain_id, "bank_id": bank_id}}

        self.stats["misses"] += 1
        t0 = time.perf_counter()
        base_ids = generator_mod.materialize_bank(bank_artifact, int(bank_id), expected_domain_id=int(domain_id))
        self.stats["materialize_s"] += time.perf_counter() - t0
        if not base_ids:
            raise RuntimeError(f"empty candidate bank domain={domain_id} bank={bank_id}")

        matrix = None
        disk_path = self._disk_path(domain_id, bank_id)
        expected_dim = int(getattr(model.model._embedding_module, "item_embedding_dim", 0) or 0)
        if expected_dim <= 0:
            expected_dim = int(getattr(model, "item_embedding_dim", 0) or 0)
        if disk_path is not None and expected_dim > 0:
            matrix = self._load_disk_matrix(disk_path, expected_shape=(len(base_ids), expected_dim))

        cache_event = "disk_load" if matrix is not None else "miss_project"
        if matrix is None:
            _sync_timing_device(self.device, self.timing_sync_cuda)
            t0 = time.perf_counter()
            matrix = self._project_normalized_ids(model, base_ids)
            _sync_timing_device(self.device, self.timing_sync_cuda)
            self.stats["projection_s"] += time.perf_counter() - t0
            if disk_path is not None:
                self._write_disk_matrix(disk_path, matrix)

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
        return entry, {"event": cache_event, "key": {"domain_id": domain_id, "bank_id": bank_id}}


def score_candidate_ids_online(
    model,
    query,
    candidate_ids: list[str],
    *,
    chunk_size: int,
    device: str,
    timing_sync_cuda: bool = False,
) -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Project and score arbitrary candidate IDs without cache."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    pairs: list[tuple[str, float]] = []
    ids_int = [int(x) for x in candidate_ids]
    timing = {"projection_s": 0.0, "dot_s": 0.0}
    with torch.inference_mode():
        for start in range(0, len(ids_int), chunk_size):
            cur_ids = ids_int[start : start + chunk_size]
            cand = torch.tensor(cur_ids, dtype=torch.long, device=device)
            _sync_timing_device(device, timing_sync_cuda)
            t0 = time.perf_counter()
            emb = model.model._embedding_module.get_item_embeddings(cand).float()
            emb = F.normalize(emb, p=2, dim=-1)
            _sync_timing_device(device, timing_sync_cuda)
            timing["projection_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = torch.mv(emb, query.squeeze(0)).detach().cpu().tolist()
            timing["dot_s"] += time.perf_counter() - t0
            pairs.extend((str(cid), float(score)) for cid, score in zip(cur_ids, scores))
    return pairs, timing


def score_candidate_set_from_query(
    model,
    query,
    cand_result,
    *,
    chunk_size: int,
    device: str,
    candidate_cache: CandidateEmbeddingCache | None = None,
    generator_mod=None,
    bank_artifact=None,
    timing_sync_cuda: bool = False,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """Score a generated candidate set, optionally using banked embedding cache.

    The return value is sorted exactly the same way as the legacy scorer:
    descending score, then lexicographic candidate_id for deterministic ties.
    """
    import torch  # type: ignore

    total_t0 = time.perf_counter()
    timing: dict[str, Any] = {
        "cache_enabled": bool(candidate_cache and candidate_cache.enabled),
        "cache_event": None,
        "online_projection_s": 0.0,
        "online_dot_s": 0.0,
        "bank_dot_s": 0.0,
        "total_s": 0.0,
    }

    if not (candidate_cache and candidate_cache.enabled and generator_mod is not None and bank_artifact is not None):
        pairs, online_timing = score_candidate_ids_online(
            model,
            query,
            list(cand_result.candidate_item_ids),
            chunk_size=chunk_size,
            device=device,
            timing_sync_cuda=timing_sync_cuda,
        )
        timing["online_projection_s"] += online_timing["projection_s"]
        timing["online_dot_s"] += online_timing["dot_s"]
        pairs.sort(key=lambda kv: (-kv[1], kv[0]))
        timing["total_s"] = time.perf_counter() - total_t0
        return pairs, timing

    entry, cache_info = candidate_cache.get_bank(
        model=model,
        generator_mod=generator_mod,
        bank_artifact=bank_artifact,
        domain_id=int(cand_result.target_canonical_domain_id),
        bank_id=int(cand_result.negative_bank_id),
    )
    timing["cache_event"] = cache_info["event"]
    base_ids: list[str] = entry["candidate_ids"]
    matrix = entry["embeddings"]

    _sync_timing_device(device, timing_sync_cuda)
    t0 = time.perf_counter()
    bank_scores = torch.mv(matrix.float(), query.squeeze(0)).detach().cpu().tolist()
    timing["bank_dot_s"] += time.perf_counter() - t0

    positive_item_id = str(cand_result.positive_item_id)
    pairs = [(cid, float(score)) for cid, score in zip(base_ids, bank_scores) if cid != positive_item_id]

    # Target-specific positive and collision replacements are not reusable across
    # targets, so score them online while keeping the reusable base bank cached.
    extra_ids = [positive_item_id] + [str(x) for x in getattr(cand_result, "replacement_item_ids", [])]
    extra_pairs, extra_timing = score_candidate_ids_online(
        model,
        query,
        extra_ids,
        chunk_size=chunk_size,
        device=device,
        timing_sync_cuda=timing_sync_cuda,
    )
    timing["online_projection_s"] += extra_timing["projection_s"]
    timing["online_dot_s"] += extra_timing["dot_s"]
    pairs.extend(extra_pairs)

    expected_ids = list(cand_result.candidate_item_ids)
    if len(pairs) != len(expected_ids) or {cid for cid, _ in pairs} != set(expected_ids):
        raise RuntimeError(
            f"cached candidate score set mismatch for target {cand_result.target_id}: "
            f"scored={len(pairs)} expected={len(expected_ids)}"
        )
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    timing["total_s"] = time.perf_counter() - total_t0
    return pairs, timing
