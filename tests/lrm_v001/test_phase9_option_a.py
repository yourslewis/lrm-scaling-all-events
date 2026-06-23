from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
LRM_DIR = REPO / "proposed_2-mmoe_ple" / "infer" / "lrm_v001"
sys.path.insert(0, str(LRM_DIR))

from candidate_embedding_cache import CandidateEmbeddingCache  # noqa: E402
from execution_shards import build_execution_shards, read_jsonl  # noqa: E402
from run_full_submission_with_safety_gate import build_eval_cmd, build_infer_cmd  # noqa: E402
from sequential_submission_infer import _target_filters  # noqa: E402
from streaming_compact_evaluator import (  # noqa: E402
    assert_metric_equivalence,
    compact_rank_record_from_ranked,
    compact_records_from_full_predictions,
    evaluate_compact_ranks,
)


def target_row(idx: int, *, source_row_index: int, ts: str, positive: str, candidate_set_digest: str = "sha256:cset") -> dict:
    is_ads = idx % 2 == 0
    return {
        "benchmark_id": "lrm_benchmark_v001",
        "target_id": f"target-{idx}",
        "target_event_id": f"event-{idx}",
        "user_id": f"user-{source_row_index}",
        "target_ts": ts,
        "target_canonical_domain_id": 4 if is_ads else 1,
        "target_domain": "ads" if is_ads else "organic",
        "target_event_type": "click",
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"candidate-set-{idx}",
        "candidate_set_digest": candidate_set_digest,
        "negative_bank_id": 7,
        "bank_selection_seed_material_digest": "sha256:seed",
        "positive_item_id": positive,
        "raw_context_event_count": idx + 1,
        "context_reader_ref": f"canonical_row_array_v001:eval/part_00000.parquet:source_row_index={source_row_index}:target_event_id=event-{idx}",
        "headline_slices": ["all_domain", "all_ads" if is_ads else "warm_ads"],
        "diagnostic_buckets": {"context_length": "short", "target_time": "week0"},
    }


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fake_ranked_pairs(candidate_ids, *, query, vectors):
    pairs = [(cid, dot(vectors[cid], query)) for cid in candidate_ids]
    return sorted(pairs, key=lambda kv: (-kv[1], kv[0]))


class Phase9OptionASmokeTest(unittest.TestCase):
    def test_execution_shards_keep_context_groups_contiguous(self):
        rows = [
            target_row(3, source_row_index=20, ts="2026-03-01T00:03:00Z", positive="103"),
            target_row(1, source_row_index=10, ts="2026-03-01T00:02:00Z", positive="101"),
            target_row(0, source_row_index=10, ts="2026-03-01T00:01:00Z", positive="100"),
            target_row(2, source_row_index=20, ts="2026-03-01T00:01:00Z", positive="102"),
        ]
        with tempfile.TemporaryDirectory() as td:
            manifest = build_execution_shards(rows, output_dir=td, shard_size=2)
            self.assertEqual(manifest["target_count"], 4)
            self.assertEqual(manifest["context_group_count"], 2)
            self.assertEqual(manifest["context_groups_split_across_shards"], 0)
            self.assertEqual(manifest["official_contract_change"], False)
            first = read_jsonl(Path(td) / manifest["shards"][0]["path"])
            second = read_jsonl(Path(td) / manifest["shards"][1]["path"])
            self.assertEqual([r["target_id"] for r in first], ["target-0", "target-1"])
            self.assertEqual([r["target_id"] for r in second], ["target-2", "target-3"])

    def test_compact_evaluator_equivalence_from_full_predictions(self):
        targets = [
            target_row(0, source_row_index=10, ts="2026-03-01T00:01:00Z", positive="100"),
            target_row(1, source_row_index=10, ts="2026-03-01T00:02:00Z", positive="101"),
        ]
        ranked = {
            "target-0": [("100", 0.9), ("200", 0.1), ("201", 0.0)],
            "target-1": [("200", 0.7), ("101", 0.7), ("201", 0.1)],  # pessimistic tie makes positive rank 2
        }
        predictions = []
        compact_direct = []
        for t in targets:
            pairs = ranked[t["target_id"]]
            predictions.append(
                {
                    "schema_version": "lrm_prediction_record_v001",
                    "benchmark_version": "lrm_benchmark_v001",
                    "model_submission_id": "m",
                    "prediction_run_id": "r",
                    "target_id": t["target_id"],
                    "candidate_protocol_label": t["candidate_protocol_label"],
                    "candidate_set_id": t["candidate_set_id"],
                    "predictions": [
                        {"candidate_id": cid, "rank": i, "score": score}
                        for i, (cid, score) in enumerate(pairs, start=1)
                    ],
                    "inference_metadata": {},
                }
            )
            compact_direct.append(
                compact_rank_record_from_ranked(
                    target=t,
                    ranked_pairs=pairs,
                    model_submission_id="m",
                    prediction_run_id="r",
                    candidate_set_digest=t["candidate_set_digest"],
                    k=10,
                    top_k=2,
                )
            )
        compact_from_full = compact_records_from_full_predictions(targets, predictions, k=10, top_k=2)
        full_result = evaluate_compact_ranks(targets, compact_from_full, k=10)
        compact_result = evaluate_compact_ranks(targets, compact_direct, k=10)
        assert_metric_equivalence(full_result, compact_result, atol=0.0)
        self.assertEqual(compact_direct[1]["positive_rank"], 2)
        self.assertEqual(len(compact_direct[0]["top_k"]), 2)

    def test_candidate_embedding_cache_snapshot_defaults(self):
        cache = CandidateEmbeddingCache(
            model_digest="sha256:model",
            max_banks=4,
            device="cpu",
            chunk_size=128,
            disk_dir=None,
        )
        stats = cache.snapshot()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["model_digest"], "sha256:model")
        self.assertEqual(stats["max_banks"], 4)
        self.assertEqual(stats["requests"], 0)
        self.assertIsNone(stats["hit_rate"])
        self.assertEqual(stats["resident_banks"], 0)
        self.assertEqual(stats["resident_bytes"], 0)

    def test_integrated_shard_compact_eval_and_cache_smoke(self):
        targets = [
            target_row(0, source_row_index=10, ts="2026-03-01T00:01:00Z", positive="100", candidate_set_digest="sha256:shared"),
            target_row(1, source_row_index=10, ts="2026-03-01T00:02:00Z", positive="101", candidate_set_digest="sha256:shared"),
        ]
        candidate_ids = ["100", "101", "200", "201"]
        vectors = {"100": [1.0, 0.0], "101": [0.0, 1.0], "200": [0.3, 0.3], "201": [0.1, 0.1]}
        queries = {"target-0": [1.0, 0.0], "target-1": [0.0, 1.0]}
        with tempfile.TemporaryDirectory() as td:
            manifest = build_execution_shards(list(reversed(targets)), output_dir=Path(td) / "shards", shard_size=8)
            shard_rows = read_jsonl(Path(td) / "shards" / manifest["shards"][0]["path"])
            compact = []
            predictions = []
            for t in shard_rows:
                ranked_pairs = fake_ranked_pairs(candidate_ids, query=queries[t["target_id"]], vectors=vectors)
                compact.append(
                    compact_rank_record_from_ranked(
                        target=t,
                        ranked_pairs=ranked_pairs,
                        model_submission_id="m",
                        prediction_run_id="integrated",
                        candidate_set_digest=t["candidate_set_digest"],
                        k=10,
                        top_k=2,
                    )
                )
                predictions.append(
                    {
                        "schema_version": "lrm_prediction_record_v001",
                        "benchmark_version": "lrm_benchmark_v001",
                        "model_submission_id": "m",
                        "prediction_run_id": "integrated",
                        "target_id": t["target_id"],
                        "candidate_protocol_label": t["candidate_protocol_label"],
                        "candidate_set_id": t["candidate_set_id"],
                        "predictions": [
                            {"candidate_id": cid, "rank": i, "score": score}
                            for i, (cid, score) in enumerate(ranked_pairs, start=1)
                        ],
                        "inference_metadata": {},
                    }
                )
            compact_result = evaluate_compact_ranks(shard_rows, compact, k=10)
            full_result = evaluate_compact_ranks(shard_rows, compact_records_from_full_predictions(shard_rows, predictions, k=10), k=10)
            assert_metric_equivalence(compact_result, full_result, atol=0.0)
            self.assertEqual(compact_result["official_contract_change"], False)
            self.assertEqual(manifest["official_contract_change"], False)

    def test_compact_only_resume_filters_from_compact_output(self):
        with tempfile.TemporaryDirectory() as td:
            compact_path = Path(td) / "compact_ranks.jsonl"
            compact_path.write_text(json.dumps({"target_id": "already-scored"}) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                target_id_file=None,
                resume=True,
                output_predictions=None,
                output_compact=str(compact_path),
                _selected_bank_filter=None,
            )
            wanted, done, selected_banks = _target_filters(args)
            self.assertIsNone(wanted)
            self.assertIsNone(selected_banks)
            self.assertEqual(done, {"already-scored"})

    def test_safety_gate_compact_only_command_uses_compact_primary_output(self):
        args = SimpleNamespace(
            python="python3",
            history_prefix_source="history",
            bank_root="bank",
            bank_generator="generator.py",
            history_reader="reader.py",
            source_root="repo",
            gin_config_file="config.gin",
            checkpoint_path="checkpoint.pt",
            embedding_root="embeddings",
            model_submission_id="model",
            context_policy="context_policy.json",
            output_mode="compact",
            compact_top_k=10,
            candidate_cache_max_banks=32,
            candidate_cache_dir=None,
            candidate_cache_disk_dtype="float32",
            candidate_cache_timing_sync_cuda=False,
            device="cuda:0",
            chunk_size=4096,
            target_batch_size=2048,
            max_sequence_length=200,
            history_batch_size=64,
            seed=20260526,
            target_jsonl="targets.jsonl",
            target_sidecar_glob=None,
            target_id_file=None,
            equivalence_check_targets=0,
        )
        cmd = build_infer_cmd(args, run_id="run", out_dir=Path("/tmp/out"), max_targets=5, resume=True)
        self.assertIn("--output-mode", cmd)
        self.assertIn("compact", cmd)
        self.assertIn("--output-compact", cmd)
        self.assertNotIn("--output-predictions", cmd)
        self.assertIn("--resume", cmd)
        with self.assertRaises(SystemExit):
            build_eval_cmd(SimpleNamespace(evaluator="eval.py", output_mode="compact"), run_dir=Path("/tmp/out"), output_json=Path("/tmp/out/eval.json"))


if __name__ == "__main__":
    unittest.main()
