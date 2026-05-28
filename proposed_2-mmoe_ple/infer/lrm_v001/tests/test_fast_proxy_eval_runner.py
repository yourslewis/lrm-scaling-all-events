from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # local macOS smoke env may not have torch; GPU env does.
    torch = None
    F = None

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from fast_proxy_eval_runner import (  # noqa: E402
    QueryGroupSpool,
    load_query_group,
    prepare_bank_major_targets,
    parse_args,
)


class FakeEmbeddingModule:
    def __init__(self, embeddings_by_id: dict[str, torch.Tensor]) -> None:
        self.embeddings_by_id = {str(k): v.detach().float() for k, v in embeddings_by_id.items()}

    def get_item_embeddings(self, cand: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embeddings_by_id[str(int(x))] for x in cand.detach().cpu().tolist()], dim=0).to(cand.device)


def _fake_model(embeddings_by_id: dict[str, torch.Tensor]):
    return SimpleNamespace(model=SimpleNamespace(_embedding_module=FakeEmbeddingModule(embeddings_by_id)))


@dataclass
class FakeCandidateResult:
    target_id: str
    candidate_item_ids: list[str]
    positive_item_id: str
    target_canonical_domain_id: int
    negative_bank_id: int
    replacement_item_ids: list[str]
    candidate_set_digest: str


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate_candidates_for_target(self, target, bank_artifact):
        self.calls += 1
        return FakeCandidateResult(
            target_id=target["target_id"],
            candidate_item_ids=["11", "10", "99"],
            positive_item_id="10",
            target_canonical_domain_id=2,
            negative_bank_id=7,
            replacement_item_ids=["99"],
            candidate_set_digest=target["candidate_set_digest"],
        )

    def validate_generated_candidates(self, cand_result):
        return []


def _target(target_id: str, positive: str, *, bank_id: int = 7) -> dict:
    return {
        "target_id": target_id,
        "user_id": "u",
        "target_ts": "2026-01-01T00:00:00Z",
        "target_canonical_domain_id": 2,
        "target_domain": "Ads",
        "target_event_type": "AdClick",
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": f"cs-{target_id}",
        "candidate_set_digest": f"sha256:{target_id}",
        "negative_bank_id": bank_id,
        "positive_item_id": positive,
        "headline_slices": ["all_domain", "all_ads"],
        "diagnostic_buckets": {"context_length": "short"},
        "context_reader_ref": "canonical_row_array_v001:eval/part.parquet:source_row_index=1:dummy",
    }


def test_query_group_spool_round_trips_grouped_batches() -> None:
    if torch is None:
        print("SKIP test_query_group_spool_round_trips_grouped_batches: torch not installed")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spool = QueryGroupSpool(root, batch_size=2, dtype="float32")
        spool.add(
            group=(2, 7),
            query=torch.tensor([[1.0, 2.0]]),
            target=_target("a", "100"),
            context_checksum=None,
            context_policy_label="ctx",
            model_inference_policy="policy",
        )
        spool.add(
            group=(2, 7),
            query=torch.tensor([[3.0, 4.0]]),
            target=_target("b", "101"),
            context_checksum="sha256:b",
            context_policy_label="ctx",
            model_inference_policy="policy",
        )
        spool.close()
        manifest = spool.write_manifest(
            args=Namespace(
                target_jsonl="targets.jsonl",
                target_sidecar_glob=None,
                selected_bank_subset="selected.json",
                _selected_bank_filter={2: {7}},
            ),
            generated_at="2026-05-27T00:00:00Z",
            model_digest="sha256:model",
            context_policy_digest="sha256:ctx",
            stats={"elapsed_s": 0.1},
        )
        assert manifest["target_count"] == 2
        assert manifest["group_count"] == 1

        batch_paths = sorted((root / "groups/domain_2/bank_0007").glob("batch_*.pt"))
        queries, targets, checksums, context_labels, policies = load_query_group(batch_paths, device="cpu")
        assert queries.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert [t["target_id"] for t in targets] == ["a", "b"]
        assert checksums == [None, "sha256:b"]
        assert context_labels == ["ctx", "ctx"]
        assert policies == ["policy", "policy"]


def test_prepare_bank_major_targets_calls_generator_only_for_collision() -> None:
    if torch is None:
        print("SKIP test_prepare_bank_major_targets_calls_generator_only_for_collision: torch not installed")
        return
    assert F is not None
    base_ids = ["10", "11"]
    bank = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32), p=2, dim=-1)
    model = _fake_model({
        "12": F.normalize(torch.tensor([0.5, 0.5]), p=2, dim=-1),
        "99": F.normalize(torch.tensor([0.2, 0.8]), p=2, dim=-1),
    })
    generator = FakeGenerator()
    targets = [_target("collision", "10"), _target("plain", "12")]
    args = Namespace(
        candidate_check_mode="collisions",
        validate_candidate_generation=True,
        extra_embedding_chunk_size=8,
        device="cpu",
    )

    specs, metrics = prepare_bank_major_targets(
        args=args,
        model=model,
        generator=generator,
        bank_artifact={"domain": 2},
        targets=targets,
        checksums=[None, None],
        context_labels=["ctx", "ctx"],
        policy_labels=["policy", "policy"],
        base_candidate_ids=base_ids,
        bank_embeddings=bank,
    )

    assert generator.calls == 1
    assert metrics["positive_collisions"] == 1
    assert metrics["replacement_targets"] == 1
    assert specs[0].replacement_item_ids == ["99"]
    assert specs[1].replacement_item_ids == []
    assert torch.allclose(specs[0].positive_embedding, bank[0])


def test_parse_args_caps_equivalence_debug_to_1000(tmp_path=None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        selected = Path(tmp) / "selected.json"
        selected.write_text(json.dumps({"schema": "lrm_v001_selected_bank_subset_v001", "domains": {"2": [7]}}), encoding="utf-8")
        argv = [
            "--target-jsonl", "targets.jsonl",
            "--selected-bank-subset", str(selected),
            "--history-prefix-source", "hist",
            "--bank-root", "banks",
            "--bank-generator", "gen.py",
            "--history-reader", "reader.py",
            "--gin-config-file", "model.gin",
            "--checkpoint-path", "ckpt.pt",
            "--context-policy", "context.json",
            "--raw-bank-cache-dir", "raw_cache",
            "--model-submission-id", "m",
            "--prediction-run-id", "r",
            "--output-compact", str(Path(tmp) / "compact.jsonl"),
            "--output-metrics-json", str(Path(tmp) / "metrics.json"),
            "--output-inference-log", str(Path(tmp) / "infer.log"),
            "--debug-equivalence-targets", "1001",
        ]
        try:
            parse_args(argv)
            raise AssertionError("parse_args should reject debug equivalence >1000")
        except SystemExit as exc:
            assert exc.code != 0


if __name__ == "__main__":
    test_query_group_spool_round_trips_grouped_batches()
    test_prepare_bank_major_targets_calls_generator_only_for_collision()
    test_parse_args_caps_equivalence_debug_to_1000()
    print("fast_proxy_eval_runner tests passed")
