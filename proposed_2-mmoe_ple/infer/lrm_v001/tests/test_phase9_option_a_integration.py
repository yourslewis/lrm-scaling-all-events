from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import build_execution_shards  # noqa: E402
from candidate_embedding_cache import CandidateEmbeddingCache, score_candidate_set_from_query  # noqa: E402
from compact_metrics import (  # noqa: E402
    StreamingMetricAggregator,
    make_compact_record,
    pessimistic_rank_from_ranked,
)


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
    def __init__(self, base_bank: list[str]) -> None:
        self.base_bank = base_bank

    def materialize_bank(self, bank_artifact, bank_id: int, *, expected_domain_id: int | None = None) -> list[str]:
        assert bank_artifact["canonical_domain_id"] == expected_domain_id
        assert bank_artifact["bank_id"] == bank_id
        return list(self.base_bank)


@dataclass(frozen=True)
class FakeCandidateResult:
    target_id: str
    target_canonical_domain_id: int
    positive_item_id: str
    negative_bank_id: int
    candidate_item_ids: list[str]
    replacement_item_ids: list[str]
    candidate_set_digest: str


def _target(target_id: str, *, row: int, ts: str, positive: str, domain: str = "Ads") -> dict:
    return {
        "benchmark_id": "lrm_benchmark_v001",
        "target_id": target_id,
        "target_event_id": f"evt_{target_id}",
        "user_id": f"user_{row}",
        "target_ts": ts,
        "target_canonical_domain_id": 2,
        "target_domain": domain,
        "target_event_type": "SearchClick" if domain == "Ads" else "PageAction",
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"cand_{target_id}",
        "candidate_set_digest": f"sha256:{target_id.zfill(64)[-64:]}",
        "negative_bank_id": 17,
        "bank_selection_seed_material_digest": "sha256:" + "1" * 64,
        "positive_item_id": positive,
        "headline_slices": ["all_domain", "all_ads"] if domain == "Ads" else ["all_domain"],
        "diagnostic_buckets": {"context_length": "short"},
        "raw_context_event_count": row + 1,
        "context_reader_ref": (
            "canonical_row_array_v001:eval/part_00000.parquet:"
            f"source_row_index={row}:target_event_id=evt_{target_id}"
        ),
    }


def _candidate_result(target: dict) -> FakeCandidateResult:
    positive = str(target["positive_item_id"])
    # Base bank is [100, 101, 102, 103], and the positive is target-specific.
    return FakeCandidateResult(
        target_id=str(target["target_id"]),
        target_canonical_domain_id=int(target["target_canonical_domain_id"]),
        positive_item_id=positive,
        negative_bank_id=int(target["negative_bank_id"]),
        candidate_item_ids=["100", positive, "101", "102", "103"],
        replacement_item_ids=[],
        candidate_set_digest=str(target["candidate_set_digest"]),
    )


def test_phase9_option_a_execution_shards_cache_and_compact_metrics_integrate() -> None:
    # Deliberately unsorted; two targets share one history row and should become contiguous.
    targets = [
        _target("3", row=2, ts="2026-02-15T00:03:00Z", positive="903", domain="Ads"),
        _target("1", row=1, ts="2026-02-15T00:02:00Z", positive="901", domain="Ads"),
        _target("2", row=1, ts="2026-02-15T00:01:00Z", positive="902", domain="Browsing"),
    ]

    model = FakeModel()
    query = torch.tensor([[0.31, -0.17, 0.23, 0.91]], dtype=torch.float32)
    generator = FakeGenerator(base_bank=["100", "101", "102", "103"])
    bank_artifact = {"canonical_domain_id": 2, "bank_id": 17}
    cache = CandidateEmbeddingCache(
        model_digest="sha256:test-model",
        max_banks=2,
        device="cpu",
        chunk_size=2,
    )
    aggregator = StreamingMetricAggregator(k=2)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        target_jsonl = root / "targets.jsonl"
        target_jsonl.write_text("".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8")
        out_dir = root / "shards"
        rc = build_execution_shards.main([
            "--target-jsonl",
            str(target_jsonl),
            "--output-dir",
            str(out_dir),
            "--targets-per-shard",
            "10",
            "--overwrite",
        ])
        assert rc == 0
        manifest = json.loads((out_dir / "execution_shard_manifest.json").read_text(encoding="utf-8"))
        assert manifest["validation"]["status"] == "passed"
        assert manifest["validation"]["input_target_count"] == 3
        assert manifest["validation"]["output_target_count"] == 3
        assert manifest["contract_preservation"]["target_rows_reordered_only"] is True
        assert manifest["contract_preservation"]["official_manifest_modified"] is False

        shard_rows = []
        for shard in manifest["shards"]:
            with open(shard["path"], "r", encoding="utf-8") as f:
                shard_rows.extend(json.loads(line) for line in f if line.strip())
        # Sorted by source row, then timestamp; row=1 targets are contiguous and time-ordered.
        assert [row["target_id"] for row in shard_rows] == ["2", "1", "3"]

        compact_records = []
        for target in shard_rows:
            cand = _candidate_result(target)
            expected_ranked, expected_metrics = score_candidate_set_from_query(
                model,
                query,
                cand,
                chunk_size=2,
                device="cpu",
                candidate_cache=None,
            )
            cached_ranked, cache_metrics = score_candidate_set_from_query(
                model,
                query,
                cand,
                chunk_size=2,
                device="cpu",
                candidate_cache=cache,
                generator_mod=generator,
                bank_artifact=bank_artifact,
            )
            assert expected_metrics["cache_enabled"] is False
            assert cached_ranked == expected_ranked
            assert cache_metrics["cache_enabled"] is True

            rank_stats = pessimistic_rank_from_ranked(cached_ranked, str(target["positive_item_id"]), k=2)
            compact = make_compact_record(
                target=target,
                ranked=cached_ranked,
                rank_stats=rank_stats,
                top_k=2,
                model_submission_id="m",
                prediction_run_id="r",
                generated_at="2026-05-27T00:00:00Z",
                model_digest="sha256:model",
                context_policy_digest="sha256:ctx",
                candidate_count=len(cand.candidate_item_ids),
                candidate_set_digest=cand.candidate_set_digest,
            )
            assert "predictions" not in compact
            assert compact["candidate_count"] == 5
            assert len(compact["top_k"]) == 2
            assert compact["digests"]["full_score_order_digest"].startswith("sha256:")
            aggregator.add_compact_record(compact)
            compact_records.append(compact)

        snap = cache.snapshot()
        assert snap["misses"] == 1
        assert snap["hits"] == 2
        assert snap["resident_banks"] == 1
        metrics = aggregator.result(created_at="2026-05-27T00:00:00Z")
        assert metrics["per_target_count"] == 3
        assert compact_records
        metric_keys = {(m["slice_id"], m["metric_name"]) for m in metrics["metrics"]}
        assert ("all_domain", "micro_OHR@2") in metric_keys


if __name__ == "__main__":
    test_phase9_option_a_execution_shards_cache_and_compact_metrics_integrate()
    print("phase9 option-a integration test passed")
