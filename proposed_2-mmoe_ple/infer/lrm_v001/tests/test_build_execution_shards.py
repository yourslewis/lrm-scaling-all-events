#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build_execution_shards.py"
spec = importlib.util.spec_from_file_location("build_execution_shards", SCRIPT)
build_execution_shards = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(build_execution_shards)


def target_row(*, part: str, source_row_index: int, target_id: str, target_ts: str, extra: str = "x") -> dict:
    return {
        "benchmark_id": "lrm_benchmark_v001",
        "target_id": target_id,
        "target_event_id": "evt_" + target_id,
        "user_id": f"user_{part}_{source_row_index}",
        "target_ts": target_ts,
        "target_canonical_domain_id": 0,
        "target_domain": "Ads",
        "target_event_type": "SearchClick",
        "candidate_protocol_label": "banked_domain_negatives_10k_b1000_v001",
        "candidate_set_id": "cand_" + target_id,
        "candidate_set_digest": "sha256:" + target_id.zfill(64)[-64:],
        "negative_bank_id": 0,
        "bank_selection_seed_material_digest": "sha256:" + ("1" * 64),
        "positive_item_id": "42",
        "raw_context_event_count": source_row_index % 200,
        "context_reader_ref": (
            f"canonical_row_array_v001:eval/{part}:source_row_index={source_row_index}:"
            f"target_event_id=evt_{target_id}"
        ),
        "preserve_extra_field": extra,
    }


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BuildExecutionShardsTest(unittest.TestCase):
    def test_smoke_order_count_digest_and_group_contiguity(self) -> None:
        # Deliberately unsorted input with two rows from the same history group separated by other groups.
        rows = [
            target_row(part="part_00001.parquet", source_row_index=5, target_id="t6", target_ts="2026-01-01T00:06:00Z"),
            target_row(part="part_00000.parquet", source_row_index=3, target_id="t3", target_ts="2026-01-01T00:03:00Z"),
            target_row(part="part_00000.parquet", source_row_index=3, target_id="t2", target_ts="2026-01-01T00:02:00Z"),
            target_row(part="part_00000.parquet", source_row_index=1, target_id="t1", target_ts="2026-01-01T00:01:00Z"),
            target_row(part="part_00001.parquet", source_row_index=5, target_id="t5", target_ts="2026-01-01T00:05:00Z"),
            target_row(part="part_00000.parquet", source_row_index=4, target_id="t4", target_ts="2026-01-01T00:04:00Z"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "targets.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = tmp_path / "out"
            rc = build_execution_shards.main([
                "--target-jsonl",
                str(source),
                "--output-dir",
                str(output),
                "--targets-per-shard",
                "2",
            ])
            self.assertEqual(rc, 0)
            manifest = json.loads((output / "execution_shard_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["validation"]["status"], "passed")
            self.assertEqual(manifest["source_stats"]["target_count"], len(rows))
            self.assertEqual(manifest["validation"]["input_target_count"], len(rows))
            self.assertEqual(manifest["validation"]["output_target_count"], len(rows))
            self.assertEqual(
                manifest["validation"]["input_row_multiset_digest"],
                manifest["validation"]["output_row_multiset_digest"],
            )
            self.assertEqual(
                manifest["validation"]["input_target_id_multiset_digest"],
                manifest["validation"]["output_target_id_multiset_digest"],
            )
            out_rows = []
            for shard in manifest["shards"]:
                out_rows.extend(read_jsonl(Path(shard["path"])))
            self.assertEqual([row["target_id"] for row in out_rows], ["t1", "t2", "t3", "t4", "t5", "t6"])
            self.assertEqual([row["preserve_extra_field"] for row in out_rows], ["x"] * len(rows))
            group_keys = [build_execution_shards.parse_context_ref(row)[3] for row in out_rows]
            self.assertEqual(group_keys[1:3], ["eval/part_00000.parquet/source_row_index=3"] * 2)
            self.assertEqual(group_keys[4:6], ["eval/part_00001.parquet/source_row_index=5"] * 2)

    def test_oversized_history_group_is_not_split(self) -> None:
        rows = [
            target_row(part="part_00000.parquet", source_row_index=7, target_id="t2", target_ts="2026-01-01T00:02:00Z"),
            target_row(part="part_00000.parquet", source_row_index=7, target_id="t1", target_ts="2026-01-01T00:01:00Z"),
            target_row(part="part_00000.parquet", source_row_index=7, target_id="t3", target_ts="2026-01-01T00:03:00Z"),
            target_row(part="part_00000.parquet", source_row_index=8, target_id="t4", target_ts="2026-01-01T00:04:00Z"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "targets.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = tmp_path / "out"
            rc = build_execution_shards.main([
                "--target-jsonl",
                str(source),
                "--output-dir",
                str(output),
                "--targets-per-shard",
                "2",
            ])
            self.assertEqual(rc, 0)
            manifest = json.loads((output / "execution_shard_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["validation"]["status"], "passed")
            # First group has 3 targets and exceeds requested shard size of 2; it should stay intact.
            self.assertEqual(manifest["shards"][0]["target_count"], 3)
            self.assertEqual(manifest["shards"][0]["history_group_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
