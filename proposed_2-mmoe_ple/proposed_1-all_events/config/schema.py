# Copyright (c) Meta Platforms, Inc. and affiliates.
# Experiment config schema — typed dataclasses that replace gin parameters.
#
# Usage:
#   cfg = ExperimentConfig.from_yaml("configs/experiments/astrov6_large.yaml")
#   dataset = build_dataset(cfg.dataset, ...)
#   model   = ModelBuilder.build(cfg.model, dataset)
#   trainer = Trainer(cfg.trainer, ...)

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Domain & Dataset
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainConfig:
    """Per-domain configuration (replaces hardcoded domain_to_item_id_range dicts)."""
    domain_id: int
    item_id_range: Tuple[int, int]
    embedding_dim: int = 64
    shard_size: int = 25_000_000
    shard_count: int = 1
    embedding_source: str = "precomputed"  # "precomputed" | "local" | "pinsage" | "roberta"
    embedding_path: Optional[str] = None

    @classmethod
    def from_dict(cls, domain_id: int, d: Dict[str, Any]) -> "DomainConfig":
        id_range = d.get("item_id_range", [0, 0])
        return cls(
            domain_id=domain_id,
            item_id_range=tuple(id_range),  # type: ignore[arg-type]
            embedding_dim=d.get("embedding_dim", 64),
            shard_size=d.get("shard_size", 25_000_000),
            shard_count=d.get("shard_count", 1),
            embedding_source=d.get("embedding_source", "precomputed"),
            embedding_path=d.get("embedding_path"),
        )


@dataclass
class DatasetConfig:
    """Dataset specification — replaces the if/elif chain in get_reco_dataset()."""
    name: str
    experiment_name: str
    path: str = ""
    max_sequence_length: int = 200
    chronological: bool = True
    positional_sampling_ratio: float = 1.0
    domains: Dict[int, DomainConfig] = field(default_factory=dict)
    domain_offset: int = 1_000_000_000
    # Legacy fields for non-domain datasets (ml-1m, amzn-books, etc.)
    max_item_id: int = 0
    min_item_id: int = 0
    num_ratings: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        domains = {}
        for k, v in d.get("domains", {}).items():
            did = int(k)
            domains[did] = DomainConfig.from_dict(did, v)
        return cls(
            name=d["name"],
            experiment_name=d["experiment_name"],
            path=d.get("path", ""),
            max_sequence_length=d.get("max_sequence_length", 200),
            chronological=d.get("chronological", True),
            positional_sampling_ratio=d.get("positional_sampling_ratio", 1.0),
            domains=domains,
            domain_offset=d.get("domain_offset", 1_000_000_000),
            max_item_id=d.get("max_item_id", 0),
            min_item_id=d.get("min_item_id", 0),
            num_ratings=d.get("num_ratings", 0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderConfig:
    """Encoder (backbone) configuration."""
    type: str = "HSTU"  # registered name in registry("encoder", ...)
    num_blocks: int = 2
    num_heads: int = 1
    dqk: int = 64
    dv: int = 64
    linear_dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0
    normalization: str = "rel_bias"
    linear_config: str = "uvqk"
    linear_activation: str = "silu"
    concat_ua: bool = False
    enable_relative_attention_bias: bool = True
    activation_checkpoint: bool = False
    # Future: kernel improvements from HSTU 2024 paper
    kernel: str = "default"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EncoderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EmbeddingConfig:
    """Embedding module configuration."""
    type: str = "local"  # "local" | "MultiDomainPrecomputed" | "xlm_roberta_base_proj" | "pinsage_proj"
    item_embedding_dim: int = 50
    model_hidden_size: int = 0  # 0 = same as item_embedding_dim (no projection)
    domain_offset: int = 1_000_000_000
    # For precomputed embeddings
    input_dim: int = 64
    shard_size: int = 25_000_000
    preload: bool = True
    # For pinsage
    pinsage_ckpt_path: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessorConfig:
    """Input features preprocessor configuration."""
    type: str = "LearnablePositionalEmbeddingInputFeaturesPreprocessor"
    dropout_rate: float = 0.2
    rating_embedding_dim: int = 5
    num_event_types: int = 8
    event_type_embedding_dim: int = 16

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PostprocessorConfig:
    """Output postprocessor configuration."""
    type: str = "l2_norm"  # "l2_norm" | "layer_norm"
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PostprocessorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LossConfig:
    """Loss module configuration."""
    type: str = "SampledSoftmaxLoss"  # "SampledSoftmaxLoss" | "BCELoss" | "BCELossWithRatings"
    temperature: float = 0.05
    num_negatives: int = 128
    activation_checkpoint: bool = False
    weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LossConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SamplingConfig:
    """Negative sampling strategy configuration."""
    strategy: str = "InBatch"  # "InBatch" | "RotateInDomainGlobalNegativesSampler" | "Hybrid"
    item_l2_norm: bool = True
    l2_norm_eps: float = 1e-6
    dedup_embeddings: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SupervisionConfig:
    """Supervision weight configuration — replaces hardcoded domain weighting logic.

    Controls how supervision_weights are computed in forward():
    - domain_weights: per-domain multiplier (e.g., {0: 32.0} to upweight ads)
    - train_domains: restrict training to specific domains (None = all)
    - target_position: which positions to supervise
    """
    domain_weights: Dict[int, float] = field(default_factory=lambda: {0: 32.0})
    train_domains: Optional[List[int]] = None  # None = all domains
    target_position: str = "all"  # "all" | "last" | "last_positive"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SupervisionConfig":
        domain_weights = {int(k): float(v) for k, v in d.get("domain_weights", {0: 32.0}).items()}
        train_domains = d.get("train_domains", None)
        if train_domains is not None:
            train_domains = [int(x) for x in train_domains]
        return cls(
            domain_weights=domain_weights,
            train_domains=train_domains,
            target_position=d.get("target_position", "all"),
        )


@dataclass
class InterestConfig:
    """Interest extraction configuration (Future: single/multi interest)."""
    mode: str = "single"  # "single" | "multi"
    num_interests: int = 1
    # Future fields for multi-interest models
    interest_dim: Optional[int] = None
    capsule_routing_iters: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterestConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Complete model configuration — all components."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    postprocessor: PostprocessorConfig = field(default_factory=PostprocessorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    interest: InterestConfig = field(default_factory=InterestConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    interaction_type: str = "DotProduct"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            encoder=EncoderConfig.from_dict(d.get("encoder", {})),
            embedding=EmbeddingConfig.from_dict(d.get("embedding", {})),
            preprocessor=PreprocessorConfig.from_dict(d.get("preprocessor", {})),
            postprocessor=PostprocessorConfig.from_dict(d.get("postprocessor", {})),
            loss=LossConfig.from_dict(d.get("loss", {})),
            sampling=SamplingConfig.from_dict(d.get("sampling", {})),
            interest=InterestConfig.from_dict(d.get("interest", {})),
            supervision=SupervisionConfig.from_dict(d.get("supervision", {})),
            interaction_type=d.get("interaction_type", "DotProduct"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainEvalConfig:
    """Per-domain evaluation configuration."""
    enabled: bool = True
    top_k_method: str = "MIPSBruteForceTopK"
    max_batches: int = 100
    metrics: List[str] = field(default_factory=lambda: [
        "ndcg@1", "ndcg@10", "ndcg@50", "ndcg@100", "ndcg@200",
        "hr@1", "hr@10", "hr@50", "hr@100", "hr@200", "hr@500", "hr@1000",
        "mrr",
    ])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainEvalConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalConfig:
    """Evaluation configuration — supports multiple eval strategies."""
    method: str = "pplx"  # "pplx" | "retrieval" | "sharded"
    interval: int = 1000  # eval every N training steps
    max_batches: int = 100
    full_eval_every_n: int = 1
    # Future: organic vs ads evaluation
    domains: Dict[str, DomainEvalConfig] = field(default_factory=lambda: {
        "ads": DomainEvalConfig()
    })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        domains = {}
        for name, dcfg in d.get("domains", {"ads": {}}).items():
            domains[name] = DomainEvalConfig.from_dict(dcfg if isinstance(dcfg, dict) else {})
        return cls(
            method=d.get("method", "pplx"),
            interval=d.get("interval", 1000),
            max_batches=d.get("max_batches", 100),
            full_eval_every_n=d.get("full_eval_every_n", 1),
            domains=domains,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """Training configuration."""
    local_batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer_type: str = "AdamW"  # "AdamW" | "Adam" | "SGD"
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    scheduler_type: str = "none"  # "none" | "cosine" | "linear_warmup_cosine"
    num_warmup_steps: int = 0
    main_module_bf16: bool = False
    enable_tf32: bool = True
    random_seed: int = 42
    save_ckpt_every_n: int = 10
    num_workers: int = 4
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        eval_cfg = EvalConfig.from_dict(d.get("eval", {}))
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "eval"}
        return cls(eval=eval_cfg, **fields)


# ──────────────────────────────────────────────────────────────────────────────
# Feature configuration (Future: user, ad, context feature enrichment)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    dtype: str = "int64"
    embedding_dim: Optional[int] = None
    vocabulary_size: Optional[int] = None
    source: str = "payload"  # "payload" | "side_table" | "computed"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeaturesConfig:
    """Feature enrichment configuration (Future)."""
    user_features: List[FeatureSpec] = field(default_factory=list)
    item_features: List[FeatureSpec] = field(default_factory=list)
    context_features: List[FeatureSpec] = field(default_factory=list)
    ad_features: List[FeatureSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeaturesConfig":
        def parse_specs(key: str) -> List[FeatureSpec]:
            return [FeatureSpec.from_dict(s) for s in d.get(key, [])]
        return cls(
            user_features=parse_specs("user_features"),
            item_features=parse_specs("item_features"),
            context_features=parse_specs("context_features"),
            ad_features=parse_specs("ad_features"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level experiment config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Root config object — one YAML file = one experiment.

    Example::

        cfg = ExperimentConfig.from_yaml("configs/experiments/astrov6_large.yaml")
        print(cfg.model.encoder.num_blocks)  # 32
        print(cfg.dataset.domains[0].item_id_range)  # (20, 42262200)
    """
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(name="", experiment_name=""))
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            dataset=DatasetConfig.from_dict(d.get("dataset", {"name": "", "experiment_name": ""})),
            model=ModelConfig.from_dict(d.get("model", {})),
            trainer=TrainerConfig.from_dict(d.get("trainer", {})),
            features=FeaturesConfig.from_dict(d.get("features", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load experiment config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict (for logging / saving)."""
        import dataclasses
        return dataclasses.asdict(self)
