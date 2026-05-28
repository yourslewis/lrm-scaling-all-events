from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from candidate_embedding_cache import score_candidate_set_from_query  # noqa: E402
from compact_metrics import (  # noqa: E402
    StreamingMetricAggregator,
    make_compact_record,
    pessimistic_rank_from_ranked,
)
from fast_proxy_bank_scorer import BankMajorTarget, score_bank_major_compact_records  # noqa: E402
from sequential_submission_infer import _keep_target, load_selected_bank_subset  # noqa: E402


@dataclass
class FakeCandidateResult:
    target_id: str
    candidate_item_ids: list[str]
    positive_item_id: str
    target_canonical_domain_id: int
    negative_bank_id: int
    replacement_item_ids: list[str]


class FakeEmbeddingModule:
    def __init__(self, embeddings_by_id: dict[str, torch.Tensor]) -> None:
        self.embeddings_by_id = {str(k): v.detach().cpu().float() for k, v in embeddings_by_id.items()}

    def get_item_embeddings(self, cand: torch.Tensor) -> torch.Tensor:
        rows = [self.embeddings_by_id[str(int(x))] for x in cand.detach().cpu().tolist()]
        return torch.stack(rows, dim=0).to(device=cand.device, dtype=torch.float32)


def _fake_model(embeddings_by_id: dict[str, torch.Tensor]):
    return SimpleNamespace(model=SimpleNamespace(_embedding_module=FakeEmbeddingModule(embeddings_by_id)))


def _target(target_id: str, positive: str, *, user_id: str | None = None, domain_id: int = 0, bank_id: int = 0) -> dict:
    return {
        "target_id": target_id,
        "user_id": user_id or f"u-{target_id}",
        "target_domain": "Ads",
        "target_event_type": "AdClick",
        "target_canonical_domain_id": domain_id,
        "negative_bank_id": bank_id,
        "positive_item_id": positive,
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"cs-{target_id}",
        "candidate_set_digest": f"sha256:{target_id}",
        "headline_slices": ["all_domain", "all_ads"],
        "diagnostic_buckets": {"context_length": "short"},
    }


def _reference(query: torch.Tensor, base_ids: list[str], bank: torch.Tensor, positive_id: str, positive_emb: torch.Tensor, repl_ids=None, repl_emb=None, k: int = 3):
    repl_ids = list(repl_ids or [])
    repl_emb = torch.empty((0, bank.shape[1])) if repl_emb is None else repl_emb
    pairs = []
    for cid, emb in zip(base_ids, bank):
        if cid == positive_id:
            continue
        pairs.append((cid, float(torch.dot(query, emb).item())))
    pairs.append((positive_id, float(torch.dot(query, positive_emb).item())))
    for cid, emb in zip(repl_ids, repl_emb):
        pairs.append((cid, float(torch.dot(query, emb).item())))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs, pessimistic_rank_from_ranked(pairs, positive_id, k=k)


def _metric_value_map(records: list[dict], *, k: int) -> dict[tuple[str, str], float | None]:
    agg = StreamingMetricAggregator(k=k)
    for record in records:
        agg.add_compact_record(record)
    result = agg.result(created_at="2026-05-27T00:00:00Z")
    names = {f"micro_AHR@{k}", f"micro_OHR@{k}", f"micro_NDCG@{k}", "micro_MRR"}
    return {
        (row["slice_id"], row["metric_name"]): row["value"]
        for row in result["metrics"]
        if row["metric_name"] in names and row["slice_kind"] == "headline"
    }


def _assert_compact_equivalent(old_records: list[dict], fast_records: list[dict], *, k: int) -> None:
    assert [r["target_id"] for r in fast_records] == [r["target_id"] for r in old_records]
    keys = ["positive_score", "greater_score_count", "equal_score_nonpositive_count", "pessimistic_rank", f"hit_at_{k}", f"ndcg_at_{k}", "reciprocal_rank"]
    for old, fast in zip(old_records, fast_records):
        for key in keys:
            if isinstance(old["rank_stats"][key], float):
                assert math.isclose(old["rank_stats"][key], fast["rank_stats"][key], rel_tol=1e-6, abs_tol=1e-6), (old["target_id"], key)
            else:
                assert old["rank_stats"][key] == fast["rank_stats"][key], (old["target_id"], key)
        assert [row["candidate_id"] for row in old["top_k"]] == [row["candidate_id"] for row in fast["top_k"]]
        for old_row, fast_row in zip(old["top_k"], fast["top_k"]):
            assert old_row["rank"] == fast_row["rank"]
            assert math.isclose(old_row["score"], fast_row["score"], rel_tol=1e-6, abs_tol=1e-6)
    old_metrics = _metric_value_map(old_records, k=k)
    fast_metrics = _metric_value_map(fast_records, k=k)
    assert old_metrics.keys() == fast_metrics.keys()
    for key in old_metrics:
        old_val = old_metrics[key]
        fast_val = fast_metrics[key]
        if old_val is None:
            assert fast_val is None
        else:
            assert fast_val is not None
            assert math.isclose(float(old_val), float(fast_val), rel_tol=1e-12, abs_tol=1e-12), key


def test_bank_major_rank_and_topk_match_full_sort_without_digest() -> None:
    # Normalized vectors keep the comparison identical to the scorer's expected
    # projected-bank/query inputs. b/c deliberately tie for one query to verify
    # lexicographic tie handling.
    bank_ids = ["a", "b", "c", "p2"]
    bank = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=torch.float32), p=2, dim=-1)
    queries = F.normalize(torch.tensor([
        [1.0, 0.2, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32), p=2, dim=-1)
    pos1 = F.normalize(torch.tensor([0.4, 0.7, 0.0]), p=2, dim=-1)
    pos2 = bank[3].clone()  # positive collides with reusable bank row p2.
    repl2 = F.normalize(torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32), p=2, dim=-1)

    specs = [
        BankMajorTarget(target=_target("t1", "p1"), positive_item_id="p1", positive_embedding=pos1),
        BankMajorTarget(target=_target("t2", "p2"), positive_item_id="p2", positive_embedding=pos2, replacement_item_ids=["r2"], replacement_embeddings=repl2),
    ]

    records, metrics = score_bank_major_compact_records(
        queries=queries,
        bank_embeddings=bank,
        base_candidate_ids=bank_ids,
        targets=specs,
        top_k=3,
        model_submission_id="m",
        prediction_run_id="r",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx",
        query_chunk_size=8,
        device="cpu",
    )

    expected1, stats1 = _reference(queries[0], bank_ids, bank, "p1", pos1, k=3)
    expected2, stats2 = _reference(queries[1], bank_ids, bank, "p2", pos2, ["r2"], repl2, k=3)

    assert metrics["mode"] == "bank_major_compact"
    for actual, expected in [(records[0]["rank_stats"], stats1), (records[1]["rank_stats"], stats2)]:
        comparable_keys = ["greater_score_count", "equal_score_nonpositive_count", "pessimistic_rank", "hit_at_3", "ndcg_at_3", "reciprocal_rank"]
        assert {key: actual[key] for key in comparable_keys} == {key: expected[key] for key in comparable_keys}
        assert math.isclose(actual["positive_score"], expected["positive_score"], rel_tol=1e-6, abs_tol=1e-6)
    for actual_rows, expected_rows in [
        (records[0]["top_k"], expected1[:3]),
        (records[1]["top_k"], expected2[:3]),
    ]:
        assert [row["candidate_id"] for row in actual_rows] == [cid for cid, _ in expected_rows]
        for row, (_, expected_score) in zip(actual_rows, expected_rows):
            assert math.isclose(row["score"], expected_score, rel_tol=1e-6, abs_tol=1e-6)
    assert records[0]["digests"]["full_score_order_digest"] is None
    assert records[0]["digests"]["full_score_order_digest_omitted"] is True
    assert records[1]["candidate_count"] == 5  # 4 base - collided positive + positive + replacement


def test_bank_major_collision_threshold_still_returns_k_rows() -> None:
    bank_ids = ["10", "11", "12"]
    bank = F.normalize(torch.tensor([
        [1.0, 0.0],
        [0.9, math.sqrt(1 - 0.9**2)],
        [0.8, math.sqrt(1 - 0.8**2)],
    ], dtype=torch.float32), p=2, dim=-1)
    queries = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    spec = BankMajorTarget(target=_target("collision-high", "10"), positive_item_id="10", positive_embedding=bank[0])

    records, _ = score_bank_major_compact_records(
        queries=queries,
        bank_embeddings=bank,
        base_candidate_ids=bank_ids,
        targets=[spec],
        top_k=3,
        model_submission_id="m",
        prediction_run_id="r",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx",
        device="cpu",
    )

    assert [row["candidate_id"] for row in records[0]["top_k"]] == ["10", "11", "12"]
    assert records[0]["rank_stats"]["pessimistic_rank"] == 1


def test_bank_major_handles_all_ties_exactly() -> None:
    bank_ids = ["c", "a", "b"]
    bank = F.normalize(torch.ones((3, 2), dtype=torch.float32), p=2, dim=-1)
    query = torch.zeros((1, 2), dtype=torch.float32)
    positive = F.normalize(torch.ones(2, dtype=torch.float32), p=2, dim=-1)
    spec = BankMajorTarget(target=_target("tie", "p"), positive_item_id="p", positive_embedding=positive)

    records, _ = score_bank_major_compact_records(
        queries=query,
        bank_embeddings=bank,
        base_candidate_ids=bank_ids,
        targets=[spec],
        top_k=4,
        model_submission_id="m",
        prediction_run_id="r",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx",
        device="cpu",
    )

    # All scores tie; exact topK is lexicographic by candidate id, and the
    # positive's pessimistic rank is after all equal non-positives.
    assert [row["candidate_id"] for row in records[0]["top_k"]] == ["a", "b", "c", "p"]
    assert records[0]["rank_stats"]["pessimistic_rank"] == 4
    assert math.isclose(records[0]["rank_stats"]["reciprocal_rank"], 0.25)


def _build_synthetic_equivalence_case(*, n_targets: int, bank_size: int = 128, dim: int = 32, seed: int = 1234):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base_ids = [str(100000 + i) for i in range(bank_size)]
    bank = F.normalize(torch.randn((bank_size, dim), generator=gen), p=2, dim=-1)
    queries = F.normalize(torch.randn((n_targets, dim), generator=gen), p=2, dim=-1)
    embeddings_by_id = {cid: bank[i].clone() for i, cid in enumerate(base_ids)}
    old_candidate_results: list[FakeCandidateResult] = []
    specs: list[BankMajorTarget] = []

    for idx in range(n_targets):
        domain_id = idx % 5
        bank_id = idx % 100
        collides = (idx % 11) == 0
        if collides:
            positive_id = base_ids[(idx * 7) % bank_size]
            positive_emb = embeddings_by_id[positive_id]
            replacement_id = str(300000 + idx)
            replacement_emb = F.normalize(torch.randn((dim,), generator=gen), p=2, dim=-1)
            replacement_ids = [replacement_id]
            replacement_embs = replacement_emb.reshape(1, -1)
            embeddings_by_id[replacement_id] = replacement_emb
            candidate_item_ids = [cid for cid in base_ids if cid != positive_id] + [positive_id, replacement_id]
        else:
            positive_id = str(200000 + idx)
            positive_emb = F.normalize(torch.randn((dim,), generator=gen), p=2, dim=-1)
            replacement_ids = []
            replacement_embs = torch.empty((0, dim), dtype=torch.float32)
            embeddings_by_id[positive_id] = positive_emb
            candidate_item_ids = base_ids + [positive_id]

        target = _target(f"synthetic-{idx:04d}", positive_id, user_id=f"user-{idx % 37}", domain_id=domain_id, bank_id=bank_id)
        old_candidate_results.append(
            FakeCandidateResult(
                target_id=target["target_id"],
                candidate_item_ids=list(candidate_item_ids),
                positive_item_id=positive_id,
                target_canonical_domain_id=domain_id,
                negative_bank_id=bank_id,
                replacement_item_ids=list(replacement_ids),
            )
        )
        specs.append(
            BankMajorTarget(
                target=target,
                positive_item_id=positive_id,
                positive_embedding=positive_emb,
                replacement_item_ids=replacement_ids,
                replacement_embeddings=replacement_embs,
                candidate_set_digest=target["candidate_set_digest"],
                context_checksum=f"sha256:ctx-{idx}",
            )
        )
    return queries, bank, base_ids, specs, old_candidate_results, _fake_model(embeddings_by_id)


def _old_sequential_compact_records(queries, specs: list[BankMajorTarget], cand_results: list[FakeCandidateResult], model, *, k: int) -> list[dict]:
    records = []
    for idx, (spec, cand_result) in enumerate(zip(specs, cand_results)):
        ranked, _ = score_candidate_set_from_query(
            model,
            queries[idx].reshape(1, -1),
            cand_result,
            chunk_size=64,
            device="cpu",
            candidate_cache=None,
        )
        rank_stats = pessimistic_rank_from_ranked(ranked, spec.positive_item_id, k=k)
        records.append(
            make_compact_record(
                target=spec.target,
                ranked=ranked,
                rank_stats=rank_stats,
                top_k=k,
                model_submission_id="m",
                prediction_run_id="old-sequential",
                generated_at="2026-05-27T00:00:00Z",
                model_digest="sha256:model",
                context_policy_digest="sha256:ctx-policy",
                candidate_count=len(cand_result.candidate_item_ids),
                candidate_set_digest=spec.candidate_set_digest or spec.target["candidate_set_digest"],
                context_checksum=spec.context_checksum,
                context_policy_label="unit",
                model_inference_policy="old_sequential_full_sort",
                include_full_score_order_digest=False,
            )
        )
    return records


def test_bank_major_matches_old_sequential_scorer_on_fixed_100_target_sample() -> None:
    k = 10
    queries, bank, base_ids, specs, cand_results, model = _build_synthetic_equivalence_case(n_targets=100)
    old_records = _old_sequential_compact_records(queries, specs, cand_results, model, k=k)
    fast_records, fast_metrics = score_bank_major_compact_records(
        queries=queries,
        bank_embeddings=bank,
        base_candidate_ids=base_ids,
        targets=specs,
        top_k=k,
        model_submission_id="m",
        prediction_run_id="fast-bank-major",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx-policy",
        query_chunk_size=256,
        device="cpu",
    )

    assert fast_metrics["targets"] == 100
    _assert_compact_equivalent(old_records, fast_records, k=k)


def test_bank_major_matches_old_sequential_scorer_on_fixed_1000_target_smoke() -> None:
    k = 10
    queries, bank, base_ids, specs, cand_results, model = _build_synthetic_equivalence_case(n_targets=1000, bank_size=96, dim=24, seed=5678)
    old_records = _old_sequential_compact_records(queries, specs, cand_results, model, k=k)
    fast_records, fast_metrics = score_bank_major_compact_records(
        queries=queries,
        bank_embeddings=bank,
        base_candidate_ids=base_ids,
        targets=specs,
        top_k=k,
        model_submission_id="m",
        prediction_run_id="fast-bank-major",
        generated_at="2026-05-27T00:00:00Z",
        model_digest="sha256:model",
        context_policy_digest="sha256:ctx-policy",
        query_chunk_size=512,
        device="cpu",
    )

    assert fast_metrics["targets"] == 1000
    assert fast_metrics["chunks"] == 2
    _assert_compact_equivalent(old_records, fast_records, k=k)


def test_selected_bank_subset_filter_matches_proxy_target_contract() -> None:
    manifest = {
        "schema": "lrm_v001_selected_bank_subset_v001",
        "domains": {"0": [2, 5], "3": [7]},
    }
    rows = [
        {"target_id": "keep-a", "target_canonical_domain_id": 0, "negative_bank_id": 2},
        {"target_id": "drop-bank", "target_canonical_domain_id": 0, "negative_bank_id": 4},
        {"target_id": "keep-b", "target_canonical_domain_id": 3, "negative_bank_id": 7},
        {"target_id": "drop-domain", "target_canonical_domain_id": 4, "negative_bank_id": 7},
        {"target_id": "drop-bad", "target_canonical_domain_id": "bad", "negative_bank_id": 7},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selected.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        selected = load_selected_bank_subset(str(path))

    kept = [row["target_id"] for row in rows if _keep_target(row, wanted=None, done=None, selected_banks=selected)]
    kept_with_done = [row["target_id"] for row in rows if _keep_target(row, wanted={"keep-a", "keep-b"}, done={"keep-a"}, selected_banks=selected)]
    assert kept == ["keep-a", "keep-b"]
    assert kept_with_done == ["keep-b"]


if __name__ == "__main__":
    test_bank_major_rank_and_topk_match_full_sort_without_digest()
    test_bank_major_collision_threshold_still_returns_k_rows()
    test_bank_major_handles_all_ties_exactly()
    test_bank_major_matches_old_sequential_scorer_on_fixed_100_target_sample()
    test_bank_major_matches_old_sequential_scorer_on_fixed_1000_target_smoke()
    test_selected_bank_subset_filter_matches_proxy_target_contract()
    print("fast_proxy_bank_scorer tests passed")
