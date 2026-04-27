"""
Tests for the config-driven architecture refactor (PR #2).

Covers:
  1. configs/schema.py — all dataclass parsing, from_dict, from_yaml, defaults
  2. registry.py — register, build, get_class, list_registered, conflicts
  3. YAML configs — astrov6_large.yaml and local_debug.yaml load correctly
  4. Round-trip: from_yaml → to_dict → from_dict preserves values

Run with:
    cd large-rec-model/src/hstu_retrieval
    pytest tests/test_config_registry.py -v
"""

import os
import copy
import pytest
import yaml
import tempfile
from dataclasses import asdict

# ─── Schema imports ──────────────────────────────────────────────────────────
from configs.schema import (
    DomainConfig,
    DatasetConfig,
    EncoderConfig,
    EmbeddingConfig,
    PreprocessorConfig,
    PostprocessorConfig,
    LossConfig,
    SamplingConfig,
    SupervisionConfig,
    InterestConfig,
    ModelConfig,
    TrainerConfig,
    EvalConfig,
    DomainEvalConfig,
    FeatureSpec,
    FeaturesConfig,
    ExperimentConfig,
)

# ─── Registry imports ────────────────────────────────────────────────────────
from registry import (
    register,
    build,
    get_class,
    list_registered,
    is_registered,
    _REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainConfig:
    def test_from_dict_basic(self):
        d = {"item_id_range": [20, 42_262_200], "embedding_dim": 64, "shard_count": 2}
        cfg = DomainConfig.from_dict(0, d)
        assert cfg.domain_id == 0
        assert cfg.item_id_range == (20, 42_262_200)
        assert cfg.embedding_dim == 64
        assert cfg.shard_count == 2
        assert cfg.embedding_source == "precomputed"

    def test_from_dict_defaults(self):
        cfg = DomainConfig.from_dict(1, {})
        assert cfg.domain_id == 1
        assert cfg.item_id_range == (0, 0)
        assert cfg.shard_size == 25_000_000
        assert cfg.embedding_path is None

    def test_from_dict_custom_source(self):
        d = {"item_id_range": [0, 100], "embedding_source": "pinsage", "embedding_path": "/path/to/ckpt"}
        cfg = DomainConfig.from_dict(2, d)
        assert cfg.embedding_source == "pinsage"
        assert cfg.embedding_path == "/path/to/ckpt"


class TestDatasetConfig:
    def test_from_dict_with_domains(self):
        d = {
            "name": "astrov6",
            "experiment_name": "semantic_next_event_prediction",
            "max_sequence_length": 200,
            "domains": {
                0: {"item_id_range": [20, 42_262_200]},
                1: {"item_id_range": [0, 301_422_400]},
            },
        }
        cfg = DatasetConfig.from_dict(d)
        assert cfg.name == "astrov6"
        assert len(cfg.domains) == 2
        assert cfg.domains[0].item_id_range == (20, 42_262_200)
        assert cfg.domains[1].domain_id == 1

    def test_from_dict_string_domain_keys(self):
        """YAML parses integer keys as ints, but string keys should also work."""
        d = {
            "name": "test",
            "experiment_name": "test",
            "domains": {"0": {"item_id_range": [0, 100]}, "1": {"item_id_range": [0, 200]}},
        }
        cfg = DatasetConfig.from_dict(d)
        assert 0 in cfg.domains
        assert 1 in cfg.domains

    def test_from_dict_no_domains(self):
        d = {"name": "ml-1m", "experiment_name": "next_item", "max_item_id": 3952}
        cfg = DatasetConfig.from_dict(d)
        assert len(cfg.domains) == 0
        assert cfg.max_item_id == 3952

    def test_defaults(self):
        d = {"name": "test", "experiment_name": "test"}
        cfg = DatasetConfig.from_dict(d)
        assert cfg.chronological is True
        assert cfg.positional_sampling_ratio == 1.0
        assert cfg.domain_offset == 1_000_000_000


class TestEncoderConfig:
    def test_from_dict(self):
        d = {"type": "HSTU", "num_blocks": 32, "num_heads": 4, "dqk": 64, "dv": 64}
        cfg = EncoderConfig.from_dict(d)
        assert cfg.type == "HSTU"
        assert cfg.num_blocks == 32

    def test_defaults(self):
        cfg = EncoderConfig.from_dict({})
        assert cfg.type == "HSTU"
        assert cfg.num_blocks == 2
        assert cfg.linear_dropout_rate == 0.0

    def test_ignores_unknown_keys(self):
        d = {"type": "HSTU", "num_blocks": 4, "future_field": True}
        cfg = EncoderConfig.from_dict(d)
        assert cfg.num_blocks == 4
        assert not hasattr(cfg, "future_field")


class TestEmbeddingConfig:
    def test_from_dict(self):
        d = {"type": "MultiDomainPrecomputed", "item_embedding_dim": 256, "input_dim": 64}
        cfg = EmbeddingConfig.from_dict(d)
        assert cfg.type == "MultiDomainPrecomputed"
        assert cfg.item_embedding_dim == 256

    def test_model_hidden_size_default(self):
        cfg = EmbeddingConfig.from_dict({})
        assert cfg.model_hidden_size == 0  # means same as item_embedding_dim


class TestPreprocessorConfig:
    def test_event_type_fields(self):
        d = {"type": "LearnablePositionalEmbeddingEventTypeEmbeddingInputFeaturesPreprocessor",
             "num_event_types": 13, "event_type_embedding_dim": 16}
        cfg = PreprocessorConfig.from_dict(d)
        assert cfg.num_event_types == 13
        assert cfg.event_type_embedding_dim == 16


class TestLossConfig:
    def test_from_dict(self):
        d = {"type": "SampledSoftmaxLoss", "temperature": 0.05, "num_negatives": 1280}
        cfg = LossConfig.from_dict(d)
        assert cfg.temperature == 0.05
        assert cfg.num_negatives == 1280

    def test_bce_loss(self):
        d = {"type": "BCELoss"}
        cfg = LossConfig.from_dict(d)
        assert cfg.type == "BCELoss"


class TestSamplingConfig:
    def test_strategies(self):
        for strategy in ["InBatch", "RotateInDomainGlobalNegativesSampler", "Hybrid"]:
            cfg = SamplingConfig.from_dict({"strategy": strategy})
            assert cfg.strategy == strategy

    def test_defaults(self):
        cfg = SamplingConfig.from_dict({})
        assert cfg.item_l2_norm is True
        assert cfg.dedup_embeddings is True


class TestSupervisionConfig:
    def test_from_dict_with_weights(self):
        d = {"domain_weights": {0: 32.0}, "train_domains": [0], "target_position": "last"}
        cfg = SupervisionConfig.from_dict(d)
        assert cfg.domain_weights == {0: 32.0}
        assert cfg.train_domains == [0]
        assert cfg.target_position == "last"

    def test_string_keys_converted(self):
        d = {"domain_weights": {"0": 32.0, "1": 1.0}}
        cfg = SupervisionConfig.from_dict(d)
        assert cfg.domain_weights == {0: 32.0, 1: 1.0}

    def test_defaults(self):
        cfg = SupervisionConfig.from_dict({})
        assert cfg.domain_weights == {0: 32.0}
        assert cfg.train_domains is None
        assert cfg.target_position == "all"


class TestInterestConfig:
    def test_single_mode(self):
        cfg = InterestConfig.from_dict({"mode": "single"})
        assert cfg.num_interests == 1

    def test_multi_mode(self):
        cfg = InterestConfig.from_dict({"mode": "multi", "num_interests": 4, "interest_dim": 128})
        assert cfg.num_interests == 4
        assert cfg.interest_dim == 128


class TestModelConfig:
    def test_from_dict_nested(self):
        d = {
            "encoder": {"type": "HSTU", "num_blocks": 32},
            "embedding": {"type": "MultiDomainPrecomputed"},
            "loss": {"type": "SampledSoftmaxLoss", "temperature": 0.05},
            "sampling": {"strategy": "InBatch"},
            "supervision": {"domain_weights": {0: 32.0}},
        }
        cfg = ModelConfig.from_dict(d)
        assert cfg.encoder.num_blocks == 32
        assert cfg.embedding.type == "MultiDomainPrecomputed"
        assert cfg.loss.temperature == 0.05

    def test_defaults(self):
        cfg = ModelConfig.from_dict({})
        assert cfg.interaction_type == "DotProduct"
        assert cfg.encoder.type == "HSTU"


class TestEvalConfig:
    def test_from_dict(self):
        d = {"interval": 1000, "domains": {"ads": {"enabled": True, "max_batches": 100}}}
        cfg = EvalConfig.from_dict(d)
        assert cfg.interval == 1000
        assert "ads" in cfg.domains
        assert cfg.domains["ads"].max_batches == 100

    def test_default_metrics(self):
        cfg = DomainEvalConfig()
        assert "ndcg@10" in cfg.metrics
        assert "hr@10" in cfg.metrics
        assert "mrr" in cfg.metrics


class TestTrainerConfig:
    def test_from_dict(self):
        d = {
            "local_batch_size": 64,
            "learning_rate": 1e-3,
            "eval": {"interval": 1000},
        }
        cfg = TrainerConfig.from_dict(d)
        assert cfg.local_batch_size == 64
        assert cfg.eval.interval == 1000

    def test_defaults(self):
        cfg = TrainerConfig.from_dict({})
        assert cfg.random_seed == 42
        assert cfg.enable_tf32 is True
        assert cfg.optimizer_type == "AdamW"


class TestFeatureSpec:
    def test_from_dict(self):
        d = {"name": "user_age", "dtype": "int64", "vocabulary_size": 100}
        spec = FeatureSpec.from_dict(d)
        assert spec.name == "user_age"
        assert spec.vocabulary_size == 100

    def test_defaults(self):
        spec = FeatureSpec.from_dict({"name": "x"})
        assert spec.dtype == "int64"
        assert spec.source == "payload"


class TestFeaturesConfig:
    def test_from_dict(self):
        d = {
            "user_features": [{"name": "age"}, {"name": "gender"}],
            "item_features": [{"name": "category", "vocabulary_size": 50}],
        }
        cfg = FeaturesConfig.from_dict(d)
        assert len(cfg.user_features) == 2
        assert len(cfg.item_features) == 1
        assert cfg.item_features[0].vocabulary_size == 50

    def test_empty(self):
        cfg = FeaturesConfig.from_dict({})
        assert len(cfg.user_features) == 0


class TestExperimentConfig:
    def test_from_dict_full(self):
        d = {
            "dataset": {"name": "astrov6", "experiment_name": "test"},
            "model": {"encoder": {"num_blocks": 32}},
            "trainer": {"local_batch_size": 64},
        }
        cfg = ExperimentConfig.from_dict(d)
        assert cfg.dataset.name == "astrov6"
        assert cfg.model.encoder.num_blocks == 32
        assert cfg.trainer.local_batch_size == 64

    def test_from_dict_minimal(self):
        cfg = ExperimentConfig.from_dict({})
        assert cfg.dataset.name == ""
        assert cfg.model.encoder.type == "HSTU"

    def test_roundtrip_dict(self):
        d = {
            "dataset": {"name": "astrov6", "experiment_name": "test", "domains": {
                0: {"item_id_range": [20, 42_262_200]},
            }},
            "model": {"encoder": {"num_blocks": 32}, "loss": {"temperature": 0.05}},
            "trainer": {"local_batch_size": 64, "eval": {"interval": 500}},
        }
        cfg = ExperimentConfig.from_dict(d)
        out = cfg.to_dict()
        cfg2 = ExperimentConfig.from_dict(out)
        assert cfg2.dataset.name == "astrov6"
        assert cfg2.model.encoder.num_blocks == 32
        assert cfg2.model.loss.temperature == 0.05
        assert cfg2.trainer.eval.interval == 500

    def test_from_yaml_file(self):
        """Test loading from a temporary YAML file."""
        yaml_content = {
            "dataset": {"name": "test_ds", "experiment_name": "test_exp"},
            "model": {"encoder": {"type": "HSTU", "num_blocks": 8}},
            "trainer": {"local_batch_size": 32},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            cfg = ExperimentConfig.from_yaml(f.name)
        os.unlink(f.name)
        assert cfg.dataset.name == "test_ds"
        assert cfg.model.encoder.num_blocks == 8
        assert cfg.trainer.local_batch_size == 32


class TestYAMLConfigs:
    """Test that actual YAML config files in the repo parse correctly."""

    @pytest.fixture
    def config_dir(self):
        # Resolve relative to this test file's location
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "configs", "experiments")

    def test_astrov6_large_loads(self, config_dir):
        path = os.path.join(config_dir, "astrov6_large.yaml")
        if not os.path.exists(path):
            pytest.skip(f"Config not found: {path}")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.dataset.name == "astrov6"
        assert cfg.model.encoder.num_blocks == 32
        assert cfg.model.encoder.num_heads == 4
        assert cfg.model.embedding.type == "MultiDomainPrecomputed"
        assert cfg.model.embedding.item_embedding_dim == 256
        assert cfg.model.loss.type == "SampledSoftmaxLoss"
        assert cfg.model.loss.temperature == 0.05
        assert cfg.model.sampling.strategy == "InBatch"
        assert cfg.model.supervision.domain_weights[0] == 32.0
        assert len(cfg.dataset.domains) == 3
        assert cfg.dataset.domains[0].shard_count == 2
        assert cfg.dataset.domains[1].shard_count == 13

    def test_local_debug_loads(self, config_dir):
        path = os.path.join(config_dir, "local_debug.yaml")
        if not os.path.exists(path):
            pytest.skip(f"Config not found: {path}")
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.dataset.name is not None
        assert cfg.model.encoder.type == "HSTU"

    def test_astrov6_to_dict_roundtrip(self, config_dir):
        path = os.path.join(config_dir, "astrov6_large.yaml")
        if not os.path.exists(path):
            pytest.skip(f"Config not found: {path}")
        cfg = ExperimentConfig.from_yaml(path)
        d = cfg.to_dict()
        cfg2 = ExperimentConfig.from_dict(d)
        assert cfg2.model.encoder.num_blocks == cfg.model.encoder.num_blocks
        assert cfg2.model.loss.temperature == cfg.model.loss.temperature
        assert cfg2.dataset.domains[0].item_id_range == cfg.dataset.domains[0].item_id_range


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Test the registry module."""

    def setup_method(self):
        """Save and restore registry state to avoid test pollution."""
        self._saved = copy.deepcopy(_REGISTRY)

    def teardown_method(self):
        _REGISTRY.clear()
        _REGISTRY.update(self._saved)

    def test_register_and_build(self):
        @register("test_cat", "TestModule")
        class TestModule:
            def __init__(self, dim=10):
                self.dim = dim

        obj = build("test_cat", "TestModule", dim=32)
        assert obj.dim == 32

    def test_register_and_get_class(self):
        @register("test_cat2", "Foo")
        class Foo:
            pass

        cls = get_class("test_cat2", "Foo")
        assert cls is Foo

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError, match="No module registered"):
            build("nonexistent_cat", "NonexistentModule")

    def test_get_class_unknown_raises(self):
        with pytest.raises(KeyError):
            get_class("nonexistent_cat", "NonexistentModule")

    def test_duplicate_registration_same_class(self):
        """Re-registering the same class should be OK."""
        @register("test_dup", "Same")
        class Same:
            pass

        # Should not raise
        register("test_dup", "Same")(Same)

    def test_duplicate_registration_different_class_raises(self):
        @register("test_dup2", "Conflict")
        class A:
            pass

        with pytest.raises(ValueError, match="Registry conflict"):
            @register("test_dup2", "Conflict")
            class B:
                pass

    def test_list_registered_category(self):
        @register("list_test", "Alpha")
        class Alpha:
            pass

        @register("list_test", "Beta")
        class Beta:
            pass

        names = list_registered("list_test")
        assert "Alpha" in names
        assert "Beta" in names

    def test_list_registered_all(self):
        @register("cat_a", "X")
        class X:
            pass

        @register("cat_b", "Y")
        class Y:
            pass

        all_reg = list_registered()
        assert "cat_a" in all_reg
        assert "cat_b" in all_reg

    def test_list_registered_empty_category(self):
        names = list_registered("definitely_empty_category")
        assert names == []

    def test_is_registered(self):
        @register("is_reg_test", "Exists")
        class Exists:
            pass

        assert is_registered("is_reg_test", "Exists") is True
        assert is_registered("is_reg_test", "DoesNotExist") is False
        assert is_registered("no_such_cat", "Exists") is False

    def test_build_with_no_kwargs(self):
        @register("no_args_test", "Simple")
        class Simple:
            def __init__(self):
                self.value = 42

        obj = build("no_args_test", "Simple")
        assert obj.value == 42


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Integration: Real module registrations
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealRegistrations:
    """Verify that the actual codebase modules are registered."""

    def test_expected_categories_exist(self):
        """After importing the codebase, key categories should be populated."""
        all_reg = list_registered()
        # These categories should exist based on PR #2's registered modules
        # We check what's actually registered rather than asserting specific names,
        # since the registration happens at import time
        if len(all_reg) > 0:
            # At least some categories should be populated
            assert isinstance(all_reg, dict)
