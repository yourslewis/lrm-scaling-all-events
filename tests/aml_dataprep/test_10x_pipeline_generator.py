import json
import subprocess
import sys
import unittest


class PipelineGeneratorTests(unittest.TestCase):
    def test_generator_materializes_fanout_and_gpu_guard(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path

            out = Path(d) / "pipeline.yml"
            subprocess.check_call([
                sys.executable,
                "aml_dataprep/generate_pconv_10x_pipeline.py",
                "--output",
                str(out),
                "--data-version",
                "test-version",
                "--output-root",
                "azureml://datastores/workspaceblobstore/paths/derived/test",
                "--source-root",
                "/data/root",
                "--cpu-compute",
                "azureml:CPU-D2ADSV4",
                "--cpu-shards",
                "3",
                "--num-buckets",
                "4",
                "--gpu-instance-count",
                "1",
                "--eval-batches",
                "7",
                "--epochs",
                "2",
            ])
            text = out.read_text(encoding="utf-8")

            self.assertIn("relay_shard_0000:", text)
            self.assertIn("relay_shard_0002:", text)
            self.assertIn("vocab_reduce_b0003:", text)
            self.assertIn("vocab_reduce_bucket_group.py", text)
            self.assertNotIn("vocab_reduce_d4_b0003:", text)
            self.assertIn("parquet_shard_0002:", text)
            self.assertIn("check_parquet_ready:", text)
            self.assertIn("gpu_instance_count: 1", text)
            self.assertIn("instance_count: ${{parent.inputs.gpu_instance_count}}", text)
            self.assertIn("component: ./components/pconv_10x_v2_evaluate.yml", text)
            component_text = Path("aml_dataprep/components/pconv_10x_v2_evaluate.yml").read_text(encoding="utf-8")
            self.assertIn("--eval_batches=${{inputs.eval_batches}}", component_text)
            self.assertIn("gpu_instance_count > 1 is gated", component_text)
            self.assertNotIn("source_root:", text)
            self.assertIn("--source_root /data/root", text)
            self.assertNotIn("output_root:", text)
            self.assertIn("path: azureml://datastores/workspaceblobstore/paths/derived/test/discovered", text)
            self.assertIn("path: azureml://datastores/workspaceblobstore/paths/derived/test/eval_output", text)
            self.assertNotIn("cpu_shards:", text)
            self.assertNotIn("cpu_compute:", text)
            self.assertIn("--expected_stage relay --expect_shards --expected_num_shards 3", text)
            self.assertIn("--dry_run", text)
            self.assertIn("--raw_root __source_uri__", text)


    def test_generator_rejects_multinode_gpu_without_flag(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run([
                sys.executable,
                "aml_dataprep/generate_pconv_10x_pipeline.py",
                "--output",
                str(Path(d) / "pipeline.yml"),
                "--gpu-instance-count",
                "2",
            ], text=True, capture_output=True)

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("requires --allow-multinode-gpu", proc.stderr)


if __name__ == "__main__":
    unittest.main()
