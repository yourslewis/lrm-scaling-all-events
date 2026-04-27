import gin
import torch
import logging

from modeling.sequential.autoregressive_losses import (
    BCELoss,
)
from modeling.sequential.nagatives_sampler import (
    InBatchNegativesSampler,
    RotateInDomainGlobalNegativesSampler,
    HybridNegativesSampler,
)

from modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
    MultiDomainPrecomputedEmbeddingModule,
    XLMRobertaBaseProjEmbeddingModule,
    PinSageProjEmbeddingModule,
)
from modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)

from modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor,
    LearnablePositionalEmbeddingEventTypeEmbeddingInputFeaturesPreprocessor,
)
from modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from modeling.similarity_utils import (
    get_similarity_function,
)
from typing import List, Tuple, Dict, Optional
from data.reco_dataset import RecoDataset


@gin.configurable
def get_weighted_loss(
    main_loss: torch.Tensor,
    aux_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    weighted_loss = main_loss
    for key, weight in weights.items():
        cur_weighted_loss = aux_losses[key] * weight
        weighted_loss = weighted_loss + cur_weighted_loss
    return weighted_loss


@gin.configurable
def make_model(
    dataset:  RecoDataset = None,
    precomputed_embeddings_domain_to_dir: Optional[Dict[int, str]] = None,  # used for MultiDomainPrecomputedEmbeddingModule
    pinsage_ckpt_path: str = "",

    main_module: str = "HSTU",  # set to be "HSTU"
    embedding_module_type: str = "local",  # set to be "local"
    item_embedding_dim: int = 50,
    model_hidden_size: int = 0,  # 0 = same as item_embedding_dim
    interaction_module_type: str = "DotProduct",  
    user_embedding_norm: str = "l2_norm",  
    input_preproc_module_type: str = "LearnablePositionalEmbeddingInputFeaturesPreprocessor",  
    dropout_rate: float = 0.2, 
    rating_embedding_dim: int = 5,  # used for "LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor"
    loss_module: str = "SampledSoftmaxLoss", 
    loss_weights: Optional[Dict[str, float]] = {},
    temperature: float = 0.05, 
    num_negatives: int = 128,  
    loss_activation_checkpoint: bool = False,
    sampling_strategy: str = "local",  
    item_l2_norm: bool = True,  
    l2_norm_eps: float = 1e-6,
    supervision_domain_weights: Optional[Dict[int, float]] = None,
    supervision_train_domains: Optional[List[int]] = None,
    supervision_target_position: str = "all",
) -> torch.nn.Module:
    """
    Create and return the model for training.
    """
    model = SequentialRetrieval(
        dataset=dataset,
        precomputed_embeddings_domain_to_dir=precomputed_embeddings_domain_to_dir,
        pinsage_ckpt_path=pinsage_ckpt_path,

        main_module=main_module,
        embedding_module_type=embedding_module_type,
        item_embedding_dim=item_embedding_dim,
        model_hidden_size=model_hidden_size,
        interaction_module_type=interaction_module_type,
        user_embedding_norm=user_embedding_norm,
        input_preproc_module_type=input_preproc_module_type,
        dropout_rate=dropout_rate,
        rating_embedding_dim=rating_embedding_dim,
        loss_module=loss_module,
        loss_weights=loss_weights,
        temperature=temperature,
        num_negatives=num_negatives,
        loss_activation_checkpoint=loss_activation_checkpoint,
        sampling_strategy=sampling_strategy,
        item_l2_norm=item_l2_norm,
        l2_norm_eps=l2_norm_eps,
        supervision_domain_weights=supervision_domain_weights,
        supervision_train_domains=supervision_train_domains,
        supervision_target_position=supervision_target_position,
        )
    return model

class SequentialRetrieval(torch.nn.Module):
    """
    A sequential retrieval model that uses a transformer-based architecture
    to process sequences of item interactions and predict the next item in the sequence.
    """

    def __init__(
            self,
            dataset: RecoDataset = None, 
            precomputed_embeddings_domain_to_dir: Optional[Dict[int, str]] = None,  # only used for "MultiDomainPrecomputedEmbeddingModule"
            pinsage_ckpt_path: str = "",

            main_module: str = "HSTU",  # set to be "HSTU"
            embedding_module_type: str = "local",  # set to be "local"
            item_embedding_dim: int = 50,
            model_hidden_size: int = 0,  # 0 = same as item_embedding_dim
            interaction_module_type: str = "DotProduct",  
            user_embedding_norm: str = "l2_norm",  
            input_preproc_module_type: str = "LearnablePositionalEmbeddingInputFeaturesPreprocessor",  
            dropout_rate: float = 0.2, 
            rating_embedding_dim: int = 5,  # only used for "LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor"
            loss_module: str = "SampledSoftmaxLoss", 
            loss_weights: Optional[Dict[str, float]] = {},  # default to be {}
            temperature: float = 0.05, 
            num_negatives: int = 128,  
            loss_activation_checkpoint: bool = False,
            sampling_strategy: str = "global",  
            item_l2_norm: bool = True,  
            l2_norm_eps: float = 1e-6,
            supervision_domain_weights: Optional[Dict[int, float]] = None,
            supervision_train_domains: Optional[List[int]] = None,
            supervision_target_position: str = "all",
            ) -> None:
        super().__init__()

        self.max_item_id = dataset.max_item_id
        self.min_item_id = dataset.min_item_id
        self.max_sequence_length = dataset.max_sequence_length
        self.num_ratings = dataset.num_ratings
        self.domain_to_item_id_range = dataset.domain_to_item_id_range
        self.precomputed_embeddings_domain_to_dir = precomputed_embeddings_domain_to_dir
        self.embd_dim = dataset.embd_dim
        self.domain_offset = dataset.domain_offset
        self.shard_size = dataset.shard_size
        self.shard_counts = dataset.shard_counts
        self.num_event_types = getattr(dataset, 'num_event_types', 0)
        self.pinsage_ckpt_path = pinsage_ckpt_path

        # Supervision config (defaults match legacy behavior: domain 0 weighted 32x)
        self.supervision_domain_weights = supervision_domain_weights or {0: 32.0}
        self.supervision_train_domains = supervision_train_domains
        self.supervision_target_position = supervision_target_position

        self.main_module = main_module
        self.embedding_module_type = embedding_module_type
        self.item_embedding_dim = item_embedding_dim
        self.model_hidden_size = model_hidden_size if model_hidden_size > 0 else item_embedding_dim
        self.interaction_module_type = interaction_module_type
        self.user_embedding_norm = user_embedding_norm
        self.input_preproc_module_type = input_preproc_module_type
        self.dropout_rate = dropout_rate
        self.rating_embedding_dim = rating_embedding_dim
        self.loss_module = loss_module
        self.loss_weights = loss_weights
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.loss_activation_checkpoint = loss_activation_checkpoint
        self.sampling_strategy = sampling_strategy
        self.item_l2_norm = item_l2_norm
        self.l2_norm_eps = l2_norm_eps

        self.model, self.ar_loss, self.negatives_sampler, \
        self.model_debug_str, self.interaction_module_debug_str, self.sampling_debug_str, self.loss_debug_str = self.setup()

    def setup(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, str, str, str, str]:
        """
        Setup the model.
        """
        model_debug_str = self.main_module        # set to be "HSTU"
        if self.embedding_module_type == "local":
            embedding_module: EmbeddingModule = LocalEmbeddingModule(
                max_item_id=self.max_item_id,
                item_embedding_dim=self.item_embedding_dim,                     # set to be 50
            )
        elif self.embedding_module_type == "MultiDomainPrecomputed":
            embedding_module: EmbeddingModule = MultiDomainPrecomputedEmbeddingModule(
                domain_to_item_id_range=self.domain_to_item_id_range,
                shard_dirs=self.precomputed_embeddings_domain_to_dir,  # set to be None
                preload=True,
                input_dim=self.embd_dim,
                output_dim=self.item_embedding_dim,          # set to be 50
                shard_size=self.shard_size,
                domain_offset=self.domain_offset,
            )
        elif self.embedding_module_type == "xlm_roberta_base_proj":
            embedding_module: EmbeddingModule = XLMRobertaBaseProjEmbeddingModule(
                input_dim = 768,
                output_dim = self.item_embedding_dim,
            )
        elif self.embedding_module_type == "pinsage_proj":
            embedding_module: EmbeddingModule = PinSageProjEmbeddingModule(
                input_dim = 64,
                output_dim = self.item_embedding_dim,
                ckpt_path=self.pinsage_ckpt_path,
            )
        else:
            raise ValueError(f"Unknown embedding_module_type {self.embedding_module_type}")
        model_debug_str += f"-{embedding_module.debug_str()}"

        interaction_module, interaction_module_debug_str = get_similarity_function(
            module_type=self.interaction_module_type,          # set to be "DotProduct"
            query_embedding_dim=self.item_embedding_dim,       # set to be 50
            item_embedding_dim=self.item_embedding_dim,        # set to be 50
        )

        assert (
            self.user_embedding_norm == "l2_norm" or self.user_embedding_norm == "layer_norm"      # set to be "l2_norm"
        ), f"Not implemented for {self.user_embedding_norm}"
        output_postproc_module = (
            L2NormEmbeddingPostprocessor(
                embedding_dim=self.item_embedding_dim,          # set to be 50
                eps=1e-6,
            )
            if self.user_embedding_norm == "l2_norm"
            else LayerNormEmbeddingPostprocessor(
                embedding_dim=self.item_embedding_dim,
                eps=1e-6,
            )
        )
        if self.input_preproc_module_type == "LearnablePositionalEmbeddingInputFeaturesPreprocessor":
            input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
                max_sequence_len=self.max_sequence_length,                 # set to be 200 + 10(default) + 1 = 211
                embedding_dim=self.item_embedding_dim,                                                    # set to be 50
                dropout_rate=self.dropout_rate,                                                           # set to be 0.2
            )
        elif self.input_preproc_module_type == "LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor":
            input_preproc_module = LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor(
                max_sequence_len=self.max_sequence_length,
                item_embedding_dim=self.item_embedding_dim,
                dropout_rate=self.dropout_rate,
                rating_embedding_dim=self.rating_embedding_dim,
                num_ratings=self.num_ratings,                             
            )
        elif self.input_preproc_module_type == "LearnablePositionalEmbeddingEventTypeEmbeddingInputFeaturesPreprocessor":
            input_preproc_module = LearnablePositionalEmbeddingEventTypeEmbeddingInputFeaturesPreprocessor(
                max_sequence_len=self.max_sequence_length,
                item_embedding_dim=self.item_embedding_dim,
                model_hidden_size=self.model_hidden_size,
                dropout_rate=self.dropout_rate,
                num_event_types=self.num_event_types,
            )

        # Optional projection: item_embedding_dim → model_hidden_size
        self._embedding_proj = None
        if self.model_hidden_size != self.item_embedding_dim:
            self._embedding_proj = torch.nn.Linear(
                self.item_embedding_dim, self.model_hidden_size, bias=False
            )

        model = get_sequential_encoder(
            module_type=self.main_module,                                                       # set to be "HSTU"
            max_sequence_length=self.max_sequence_length,                               # set to be 200
            max_output_length=0,                                        # set to be 10 + 1 = 11
            embedding_module=embedding_module,                                             # LocalEmbeddingModule
            interaction_module=interaction_module,                                         # "DotProduct"
            input_preproc_module=input_preproc_module,                                     # LearnablePositionalEmbeddingInputFeaturesPreprocessor
            output_postproc_module=output_postproc_module,                                 # L2NormEmbeddingPostprocessor
            verbose=True,
        )
        model_debug_str = model.debug_str()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")

        # loss
        loss_debug_str = self.loss_module                                            # set to be "SampledSoftmaxLoss"
        if self.loss_module == "BCELoss":
            loss_debug_str = loss_debug_str[:-4]
            assert self.temperature == 1.0
            ar_loss = BCELoss(temperature=self.temperature, model=model)
        elif self.loss_module == "SampledSoftmaxLoss":
            loss_debug_str = "ssl"
            if self.temperature != 1.0:                                             # set to be 0.05
                loss_debug_str += f"-t{self.temperature}"
            ar_loss = SampledSoftmaxLoss(
                num_to_sample=self.num_negatives,                                   # set to be 128
                softmax_temperature=self.temperature,                               # set to be 0.05
                model=model,
                activation_checkpoint=self.loss_activation_checkpoint,              # default to be False
            )
            loss_debug_str += (
                f"-n{self.num_negatives}{'-ac' if self.loss_activation_checkpoint else ''}"
            )
        else:
            raise ValueError(f"Unrecognized loss module {self.loss_module}.")

        # sampling
        if self.sampling_strategy == "InBatch":
            in_batch_negatives_sampler = InBatchNegativesSampler(
                l2_norm=self.item_l2_norm,                      # set to be True
                l2_norm_eps=self.l2_norm_eps,                   # set to be 1e-6
                dedup_embeddings=True,
            )
            rotate_negatives_sampler = RotateInDomainGlobalNegativesSampler(
                item_emb=model._embedding_module,
                domain_offset= self.domain_offset, 
                shard_size = self.shard_size,
                shard_counts= self.shard_counts,      
                l2_norm=self.item_l2_norm,                                    # set to be True
                l2_norm_eps=self.l2_norm_eps,                                 # set to be 1e-6
            )
            negatives_sampler = {
                "train": in_batch_negatives_sampler,
                "eval": rotate_negatives_sampler,
            }
            sampling_debug_str = (
                f"train-in-batch-and-eval-rotate-global"
            )
        elif self.sampling_strategy == "RotateInDomainGlobalNegativesSampler":
            rotate_negatives_sampler = RotateInDomainGlobalNegativesSampler(
                item_emb=model._embedding_module,
                domain_offset= self.domain_offset, 
                shard_size = self.shard_size,
                shard_counts= self.shard_counts,    
                l2_norm=self.item_l2_norm,                                    # set to be True
                l2_norm_eps=self.l2_norm_eps,                                # set to be 1e-6
            )
            negatives_sampler = {"train": rotate_negatives_sampler, "eval": rotate_negatives_sampler}
            sampling_debug_str = rotate_negatives_sampler.debug_str()
        elif self.sampling_strategy == "Hybrid":
            in_batch_negatives_sampler = InBatchNegativesSampler(
                l2_norm=self.item_l2_norm,                      # set to be True
                l2_norm_eps=self.l2_norm_eps,                   # set to be 1e-6
                dedup_embeddings=True,
            )
            rotate_negatives_sampler = RotateInDomainGlobalNegativesSampler(
                item_emb=model._embedding_module,
                domain_offset= self.domain_offset, 
                shard_size = self.shard_size,
                l2_norm=self.item_l2_norm,                                    # set to be True
                l2_norm_eps=self.l2_norm_eps,                                 # set to be 1e-6
            )
            negatives_sampler = {
                "train": HybridNegativesSampler(
                    in_batch_sampler=in_batch_negatives_sampler,
                    rotate_sampler=rotate_negatives_sampler,
                    l2_norm=self.item_l2_norm,                                    # set to be True
                    l2_norm_eps=self.l2_norm_eps,                                # set to be 1e-6
                ),
                "eval": rotate_negatives_sampler,
            }
            sampling_debug_str = (
                f"train-hybrid-and-eval-rotate-global"
            )
            
        else:
            raise ValueError(f"Unrecognized sampling strategy {self.sampling_strategy}.")
        return model, ar_loss, negatives_sampler, model_debug_str, interaction_module_debug_str, sampling_debug_str, loss_debug_str


    def forward(
        self,
        input_ids: torch.Tensor,
        raw_input_embeddings: torch.Tensor,
        input_lengths: torch.Tensor,
        label_ids: torch.Tensor,
        raw_label_embeddings: torch.Tensor,
        ratings: torch.Tensor = None,
        type_ids: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        user_ids: torch.Tensor = None,   
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            input_ids: Tensor of shape (batch_size, max_sequence_length) containing item IDs.
            input_lengths: Tensor of shape (batch_size,) containing the lengths of each sequence.
            label_ids: Tensor of shape (batch_size, max_sequence_length) containing labels for next items.
            ratings: Tensor of shape (batch_size, max_sequence_length) containing ratings for items.
            timestamps: Tensor of shape (batch_size, max_sequence_length) containing timestamps for items.
            user_ids: Tensor of shape (batch_size,) containing user IDs.

        Returns:
            A tuple containing:
                - logits: The output logits from the model.
                - loss: The computed loss value.
        """
        past_embeddings = self.model._embedding_module(raw_input_embeddings)   
        supervision_embeddings = self.model._embedding_module(raw_label_embeddings)
        # logging.info(f"input shape {input.shape}")                    [128, 200]
        # logging.info(f"ratings shape {ratings.shape}")               
        # logging.info(f"intput_embeddings shape {input_embeddings.shape}")   [128, 200, 50]
                      
        seq_embeddings = self.model(
            past_lengths=input_lengths,
            past_ids=input_ids,
            past_embeddings=past_embeddings,
            past_payloads={"timestamps": timestamps, "ratings": ratings, "type_ids": type_ids},                                      # past_ratings, (past_timestamps + 1)
        )
        # logging.info(f"seq_embeddings shape {seq_embeddings.shape}")                     # [128, 211, 50]


        ar_mask = label_ids != 0
        supervision_weights = ar_mask.float()

        # Config-driven domain weighting (replaces hardcoded 32x multiplier)
        for domain_id, weight in self.supervision_domain_weights.items():
            if domain_id == 0:
                # Domain 0: label_ids < domain_offset
                domain_mask = label_ids < self.domain_offset
            else:
                # Domain N: label_ids in [N * domain_offset, (N+1) * domain_offset)
                domain_mask = (label_ids >= domain_id * self.domain_offset) & (
                    label_ids < (domain_id + 1) * self.domain_offset
                )
            supervision_weights[domain_mask] *= weight

        # Config-driven domain restriction (replaces commented-out "train on ads only")
        if self.supervision_train_domains is not None:
            train_mask = torch.zeros_like(label_ids, dtype=torch.bool)
            for domain_id in self.supervision_train_domains:
                if domain_id == 0:
                    train_mask |= label_ids < self.domain_offset
                else:
                    train_mask |= (label_ids >= domain_id * self.domain_offset) & (
                        label_ids < (domain_id + 1) * self.domain_offset
                    )
            supervision_weights[~train_mask] = 0.0

        # Config-driven target position (replaces commented-out "train on last event only")
        if self.supervision_target_position == "last":
            supervision_weights = keep_last_nonzero(supervision_weights)

        loss, aux_losses, metrics = self.ar_loss(
            lengths=input_lengths,  # [B],
            output_embeddings=seq_embeddings,                 # [B, N, D]             
            supervision_ids=label_ids,                        # [B, N]
            supervision_embeddings=supervision_embeddings,    # [B, N, D]    
            supervision_weights=supervision_weights,
            negatives_sampler=self.negatives_sampler['train'],
        )  # [B, N]

        loss = get_weighted_loss(loss, aux_losses, weights=self.loss_weights or {})     # default to be {}
        return seq_embeddings, loss, metrics


    def debug_str(self) -> str:
        return f"{self.model_debug_str}_{self.interaction_module_debug_str}_{self.sampling_debug_str}_{self.loss_debug_str}"
    

def keep_last_nonzero(supervision_weights: torch.Tensor) -> torch.Tensor:
    # Make sure it’s a boolean mask for nonzero
    mask = supervision_weights != 0
 
    # Get index of last nonzero element per row
    # argmax on reversed array gives distance from end
    last_idx = mask.size(1) - 1 - torch.flip(mask, dims=[1]).float().argmax(dim=1)
 
    # Create zero mask and set only those indices to 1
    out = torch.zeros_like(supervision_weights)
    batch = torch.arange(mask.size(0), device=mask.device)
    out[batch, last_idx] = supervision_weights[batch, last_idx]
 
    # Zero out rows with no nonzero entries
    out[mask.sum(dim=1) == 0] = 0
    return out