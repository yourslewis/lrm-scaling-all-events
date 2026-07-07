import json
import subprocess
import sys
import unittest
from pathlib import Path

from aml_dataprep.parallel.partition import partition_rows, validate_partition


class ParallelPartitioningTests(unittest.TestCase):
    def test_partition_rows_assigns_every_row_once(self):
        rows = [{"id": i} for i in range(17)]
        assignments = []
        for shard_index in range(5):
            assignments.extend((ordinal, row["id"], shard_index) for ordinal, row in partition_rows(rows, shard_index, 5))

        self.assertEqual(sorted((ordinal, row_id) for ordinal, row_id, _ in assignments), [(i, i) for i in range(17)])
        self.assertEqual(len({(ordinal, row_id) for ordinal, row_id, _ in assignments}), 17)


    def test_validate_partition_rejects_invalid_values(self):
        for shard_index, num_shards in [(-1, 4), (4, 4), (0, 0)]:
            with self.assertRaises(ValueError):
                validate_partition(shard_index, num_shards)


    def test_relay_partition_dry_run_writes_ready_manifest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            manifest = tmp_path / "manifest.jsonl"
            rows = [
                {"split": "train", "shard_index": i, "source_uri": f"/src/train/{i}.tsv", "dest_relpath": f"train/{i}.tsv"}
                for i in range(6)
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            out = tmp_path / "raw"

            subprocess.check_call([
                sys.executable,
                "aml_dataprep/parallel/relay_partition.py",
                "--manifest",
                str(manifest),
                "--shard_index",
                "1",
                "--num_shards",
                "3",
                "--output_dir",
                str(out),
                "--dry_run",
            ])

            ready = json.loads((out / "_ready" / "relay_shard_0001.json").read_text(encoding="utf-8"))
            self.assertEqual(ready["num_shards"], 3)
            self.assertEqual(ready["shard_index"], 1)
            self.assertEqual(ready["selected_ordinals"], [1, 4])
            self.assertEqual(ready["selected"], ["train:1", "train:4"])


    def test_ready_manifest_check_fails_when_shard_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            ready_dir = tmp_path / "ready"
            ready_dir.mkdir()
            (ready_dir / "stage_shard_0000.json").write_text('{"shard_index": 0, "num_shards": 2}\n', encoding="utf-8")

            proc = subprocess.run([
                sys.executable,
                "aml_dataprep/parallel/check_ready_manifests.py",
                "--ready_dir",
                str(ready_dir),
                "--pattern",
                "stage_shard_{shard_index:04d}.json",
                "--num_shards",
                "2",
                "--output_dir",
                str(tmp_path / "merged"),
                "--stage",
                "stage",
            ], text=True, capture_output=True)

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing ready manifest", proc.stderr)


if __name__ == "__main__":
    unittest.main()
