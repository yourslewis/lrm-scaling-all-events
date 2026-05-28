from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from candidate_embedding_cache import (  # noqa: E402
    CandidateEmbeddingCache,
    score_candidate_set_from_query,
)
from sequential_submission_infer import score_target  # noqa: E402


class FakeEmbeddingModule:
    item_embedding_dim = 4

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        ids = item_ids.to(dtype=torch.float32)
        return torch.stack(
            [
                (ids % 7) / 7.0,
                (ids % 11) / 11.0,
                (ids % 13) / 13.0,
                torch.ones_like(ids),
            ],
            dim=-1,
        )


class FakeInnerModel:
    def __init__(self) -> None:
        self._embedding_module = FakeEmbeddingModule()


class FakeModel:
    def __init__(self) -> None:
        self.model = FakeInnerModel()


class FakeGenerator:
    def __init__(self, base_bank: list[str], cand_result=None) -> None:
        self.base_bank = base_bank
        self.cand_result = cand_result

    def load_bank_artifact(self, path):
        return {"canonical_domain_id": 2, "bank_id": 17}

    def generate_candidates_for_target(self, target, bank):
        assert self.cand_result is not None
        return self.cand_result

    def validate_generated_candidates(self, cand_result) -> list[str]:
        return []

    def materialize_bank(self, bank_artifact, bank_id: int, *, expected_domain_id: int | None = None) -> list[str]:
        assert bank_artifact["canonical_domain_id"] == expected_domain_id
        assert bank_id == bank_artifact["bank_id"]
        return list(self.base_bank)


@dataclass(frozen=True)
class FakeCandidateResult:
    target_id: str
    target_canonical_domain_id: int
    positive_item_id: str
    negative_bank_id: int
    candidate_item_ids: list[str]
    replacement_item_ids: list[str]
    candidate_set_digest: str = "sha256:test-candidate-set"


def _score_no_cache(model, query, cand_result):
    ranked, metrics = score_candidate_set_from_query(
        model,
        query,
        cand_result,
        chunk_size=2,
        device="cpu",
        candidate_cache=None,
    )
    assert metrics["cache_enabled"] is False
    return ranked


def _score_with_cache(model, query, cand_result, cache, generator, bank_artifact):
    return score_candidate_set_from_query(
        model,
        query,
        cand_result,
        chunk_size=2,
        device="cpu",
        candidate_cache=cache,
        generator_mod=generator,
        bank_artifact=bank_artifact,
    )


def test_candidate_cache_exact_rank_score_equivalence_and_hit_metrics():
    model = FakeModel()
    query = torch.tensor([[0.31, -0.17, 0.23, 0.91]], dtype=torch.float32)
    base_bank = ["100", "101", "102", "103"]
    cand_result = FakeCandidateResult(
        target_id="t-no-collision",
        target_canonical_domain_id=2,
        positive_item_id="999",
        negative_bank_id=17,
        candidate_item_ids=["100", "999", "101", "102", "103"],
        replacement_item_ids=[],
    )
    generator = FakeGenerator(base_bank)
    bank_artifact = {"canonical_domain_id": 2, "bank_id": 17}
    cache = CandidateEmbeddingCache(
        model_digest="sha256:test-model",
        max_banks=2,
        device="cpu",
        chunk_size=2,
    )

    expected = _score_no_cache(model, query, cand_result)
    first, first_metrics = _score_with_cache(model, query, cand_result, cache, generator, bank_artifact)
    second, second_metrics = _score_with_cache(model, query, cand_result, cache, generator, bank_artifact)

    assert first == expected
    assert second == expected
    assert first_metrics["cache_event"] == "miss_project"
    assert second_metrics["cache_event"] == "hit"
    snap = cache.snapshot()
    assert snap["requests"] == 2
    assert snap["misses"] == 1
    assert snap["hits"] == 1
    assert snap["hit_rate"] == 0.5
    assert snap["resident_banks"] == 1
    assert first_metrics["total_s"] >= 0.0
    assert second_metrics["bank_dot_s"] >= 0.0


def test_candidate_cache_exact_equivalence_with_positive_collision_replacement():
    model = FakeModel()
    query = torch.tensor([[-0.11, 0.27, 0.45, 0.80]], dtype=torch.float32)
    # The base reusable bank contains the positive; generated candidates replace
    # that collision with target-specific replacement 104.
    base_bank = ["100", "999", "102", "103"]
    cand_result = FakeCandidateResult(
        target_id="t-collision",
        target_canonical_domain_id=2,
        positive_item_id="999",
        negative_bank_id=18,
        candidate_item_ids=["100", "102", "999", "103", "104"],
        replacement_item_ids=["104"],
    )
    generator = FakeGenerator(base_bank)
    bank_artifact = {"canonical_domain_id": 2, "bank_id": 18}
    cache = CandidateEmbeddingCache(
        model_digest="sha256:test-model",
        max_banks=2,
        device="cpu",
        chunk_size=3,
    )

    expected = _score_no_cache(model, query, cand_result)
    actual, metrics = _score_with_cache(model, query, cand_result, cache, generator, bank_artifact)

    assert actual == expected
    assert metrics["cache_event"] == "miss_project"
    assert {cid for cid, _ in actual} == set(cand_result.candidate_item_ids)
    assert len(actual) == len(cand_result.candidate_item_ids)


def test_score_target_integrates_candidate_cache_metrics_without_full_record(tmp_path=None):
    from argparse import Namespace

    model = FakeModel()
    query = torch.tensor([[0.31, -0.17, 0.23, 0.91]], dtype=torch.float32)
    base_bank = ["100", "101", "102", "103"]
    cand_result = FakeCandidateResult(
        target_id="t-score-target",
        target_canonical_domain_id=2,
        positive_item_id="999",
        negative_bank_id=17,
        candidate_item_ids=["100", "999", "101", "102", "103"],
        replacement_item_ids=[],
    )
    target = {
        "target_id": cand_result.target_id,
        "target_canonical_domain_id": 2,
        "candidate_set_digest": cand_result.candidate_set_digest,
    }
    args = Namespace(
        bank_root="/does/not/matter",
        chunk_size=2,
        device="cpu",
        candidate_cache_timing_sync_cuda=False,
    )
    generator = FakeGenerator(base_bank, cand_result=cand_result)
    cache = CandidateEmbeddingCache(
        model_digest="sha256:test-model",
        max_banks=1,
        device="cpu",
        chunk_size=2,
    )

    record, _, ranked, metrics = score_target(
        args,
        generator,
        {},
        cache,
        model,
        query,
        target,
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:test-model",
        context_policy_digest="sha256:test-policy",
        emit_full_record=False,
    )

    assert record is None
    assert [cid for cid, _ in ranked] == [cid for cid, _ in _score_no_cache(model, query, cand_result)]
    assert metrics["cache_enabled"] is True
    assert metrics["cache_event"] == "miss_project"
    assert metrics["total_s"] >= 0.0
    assert cache.snapshot()["misses"] == 1

if __name__ == "__main__":
    test_candidate_cache_exact_rank_score_equivalence_and_hit_metrics()
    test_candidate_cache_exact_equivalence_with_positive_collision_replacement()
    test_score_target_integrates_candidate_cache_metrics_without_full_record()
    print("candidate_embedding_cache tests passed")
