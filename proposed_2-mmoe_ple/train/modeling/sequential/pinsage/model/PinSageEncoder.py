# coding=utf-8
#
# NOTE: This file is copied from https://wenhlu.visualstudio.com/Astro_Official/_git/Astro_Official?path=/sentence_transformers/models/PinSageEncoder.py&version=GBshenqian/L1Ranking
# DO NOT MODIFY THIS FILE DIRECTLY.
#
# Copyright (c) 2021, MICROSOFT CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turing ULRv5 Model """
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertSelfOutput
from transformers.configuration_utils import PretrainedConfig

from transformers.modeling_utils import PreTrainedModel
from transformers import BertTokenizerFast


logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    return incremental_indices.long() + padding_idx

class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        # self.position_embeddings = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        # )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None
    ):
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(
                input_ids, self.padding_idx
            ).to(input_ids.device)

        input_shape = input_ids.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TULRv5Config(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a :class:`~transformers.TULRv5Model`.
        It is used to instantiate an TULRv5 model according to the specified arguments, defining the model
        architecture.

        Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
        to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
        for more information.


        Args:
            vocab_size (:obj:`int`, optional, defaults to 500002):
                Vocabulary size of the TULRv5 model. Defines the different tokens that
                can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
            hidden_size (:obj:`int`, optional, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            num_hidden_layers (:obj:`int`, optional, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (:obj:`int`, optional, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            intermediate_size (:obj:`int`, optional, defaults to 3072):
                Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
                The non-linear activation function (function or string) in the encoder and pooler.
                If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            max_position_embeddings (:obj:`int`, optional, defaults to 512):
                The maximum sequence length that this model might ever be used with.
                Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size (:obj:`int`, optional, defaults to 2):
                The vocabulary size of the `token_type_ids` passed into :class:`~transformers.TULRv5Model`.
            initializer_range (:obj:`float`, optional, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            rel_pos_bins (:obj:`int`, optional, defaults to 32):
                No. of buckets used in relative position bias.
            max_rel_pos (:obj:`int`, optional, defaults to 128):
                Max relative position length supported.
    """
    # pretrained_config_archive_map = TULRV5_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "tulrv5"

    def __init__(self,
                vocab_size=500002,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-05,
                expand_qk_dim=96,
                rel_pos_bins=32,
                max_rel_pos=128,
                **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.expand_qk_dim = expand_qk_dim
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos

def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret

class TULRv5SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.expand_qk_dim = config.expand_qk_dim

        self.query = nn.Linear(config.hidden_size, config.expand_qk_dim * self.num_attention_heads)
        self.key = nn.Linear(config.hidden_size, config.expand_qk_dim * self.num_attention_heads, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.gate_ur_linear = nn.Linear(config.expand_qk_dim, 8)
        self.eco_a = nn.Parameter(torch.ones(
            1, self.num_attention_heads, 1, 1))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, expand_qk=False):
        head_size = self.attention_head_size
        if expand_qk:
            head_size = self.expand_qk_dim
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rel_pos=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, expand_qk=True)
        key_layer = self.transpose_for_scores(mixed_key_layer, expand_qk=True)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # logger.info('attention_scores {}'.format(torch.mean(attention_scores).cpu()))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TULRv5Model forward() function)
            attention_scores = attention_scores + attention_mask

        if rel_pos is not None:
            # Applying Gated Relative Position Bias
            _B, _H, _L, __ = query_layer.size()
            # (B, H, L, D) -> (B, H, L, 1)
            gate_u, gate_r = torch.sigmoid(self.gate_ur_linear(query_layer).view(
                _B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
            gate_u_1 = gate_u * (gate_r * self.eco_a - 1.0) + 2.0
            rel_pos_bias = gate_u_1 * rel_pos
            attention_scores = attention_scores + rel_pos_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class TULRv5Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TULRv5SelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rel_pos=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, rel_pos=rel_pos
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class TULRv5Layer(nn.Module):
    def __init__(self, config, layer_idx=-1):
        super().__init__()
        self.attention = TULRv5Attention(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rel_pos=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, rel_pos=rel_pos)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class TULRv5Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TULRv5Layer(config, _) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rel_pos=None,
    ):
        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    rel_pos
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, rel_pos=rel_pos
                )
            hidden_states = layer_outputs[0]

        outputs = (hidden_states,)

        return outputs

    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        """
        Wrapper around gradient checkpointing to preserve the output structure
        """
        # Gradient checkpointing works by not saving activations for backward pass and recomputing them
        # It trades memory for computation to save memory during training
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)

class TULRv5PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = TULRv5Config
    base_model_prefix = "tulrv5"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class TULRv5Model(TULRv5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = TULRv5Encoder(config)

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins, config.num_attention_heads, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

        if self.config.rel_pos_bins > 0:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size())
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos_mat = rel_pos_mat.type_as(input_ids)
            min_distance = rel_pos_mat.min()
            max_distance = rel_pos_mat.max()
            all_distance = torch.arange(
                start=min_distance, end=max_distance + 1, dtype=rel_pos_mat.dtype, device=rel_pos_mat.device)
            rel_pos_ids = relative_position_bucket(
                all_distance, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)
            rel_pos_one_hot = F.one_hot(rel_pos_ids, num_classes=self.config.rel_pos_bins).type_as(embedding_output)
            rel_pos_embedding = torch.mm(self.rel_pos_bias.weight, rel_pos_one_hot.transpose(-1, -2))
            rel_pos_embedding = rel_pos_embedding.unsqueeze(0).unsqueeze(-2)
            batch_size, seq_len, _ = rel_pos_mat.size()
            rel_pos_embedding = rel_pos_embedding.expand(batch_size, -1, seq_len, -1)
            num_attention_head = rel_pos_embedding.size()[1]
            rel_pos_mat = (rel_pos_mat - min_distance).unsqueeze(1).expand(-1, num_attention_head, -1, -1)
            rel_pos = torch.gather(rel_pos_embedding, dim=-1, index=rel_pos_mat)
        else:
            rel_pos = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            rel_pos=rel_pos,
        )
        sequence_output = encoder_outputs[0]

        return sequence_output

class WeightPooler(nn.Module):
    """Taken from TwinBERT Author's implementation."""

    def __init__(self, config):
        super(WeightPooler, self).__init__()
        self.weighting = nn.Linear(config.hidden_size, 1)
        # scale down to 1e-4 to avoid fp16 overflow.
        # self.weight_factor = (
        #     1e-4 if "enable_fp16" in config and config.enable_fp16 else 1e-8
        # )
        self.weight_factor = 1e-8

    def forward(self, term_tensor, mask):
        weights = self.weighting(term_tensor)
        weights = (
            weights + (mask - 1).type(weights.dtype).unsqueeze(2) / self.weight_factor
        )
        weights = F.softmax(weights, dim=1)
        return (term_tensor * weights).sum(dim=1)

class PinSageEncoder(TULRv5PreTrainedModel):
    def __init__(self, model_name_or_path=None):
        self.config = TULRv5Config.from_pretrained(model_name_or_path)

        super().__init__(self.config)

        self.encoder = TULRv5Model(self.config)
        self.pooler = WeightPooler(self.config)
        self.downscale = nn.Linear(self.config.hidden_size, 64)

        # Orignal vocab is corrupted, can't load the vocab using this way
        # self._tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.vocab_file = os.path.join(model_name_or_path, "vocab.txt")
        self._tokenizer = BertTokenizerFast(vocab_file=self.vocab_file)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        output_embedding = self.pooler(encoder_output, attention_mask)
        output = self.downscale(output_embedding)
        output = torch.tanh(output)

        return output

    def encode(self, text: str, max_length: int = 32, device: str = "cpu"):
        """
        Encodes a given string into its vector representation.

        Args:
            text (str): The input string to encode.
            max_length (int): The maximum sequence length for tokenization. Default is 128.
            device (str): The device to perform computation on. Default is "cpu".

        Returns:
            torch.Tensor: The encoded vector representation of the input string.
        """
        # Tokenize the input text
        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Move tensors to the specified device
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Forward pass through the model
        with torch.no_grad():
            encoded_output = self.forward(input_ids, attention_mask)

        return encoded_output

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def tokenize(self, texts, max_length = None, **kwargs):
        return self._tokenizer(texts, max_length = max_length, **kwargs)

    def Convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "event_doc_text_model.bin")
        torch.save(self.state_dict(), save_path)
        # Save config
        self.config.save_pretrained(path)
        # original vocab is corrupted, can't save the vocab using this way
        # self._tokenizer.save_pretrained(path)
        # Copy the self.vocab_file to this path
        vocab_file = os.path.join(path, "vocab.txt")
        assert os.path.exists(self.vocab_file), "Vocab file not found"
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            vocab = f.read()
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(vocab)


    @staticmethod
    def load(input_path, device="cpu", **kwargs):
        model = PinSageEncoder(input_path)
        model.load_state_dict(torch.load(os.path.join(input_path, "event_doc_text_model.bin"), map_location="cpu"), strict=True)
        model.eval()
        return model.to(device)
