# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn.functional as F

from rails.similarities.module import SimilarityModule

from torch.utils.checkpoint import checkpoint
from modeling.sequential.nagatives_sampler import NegativesSampler
from registry import register


class AutoregressiveLoss(torch.nn.Module):
    @abc.abstractmethod
    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Variant of forward() when the tensors are already in jagged format.

        Args:
            output_embeddings: [N', D] x float, embeddings for the current
                input sequence.
            supervision_ids: [N'] x int64, (positive) supervision ids.
            supervision_embeddings: [N', D] x float.
            supervision_weights: Optional [N'] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass

    @abc.abstractmethod
    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
            lengths: [B] x int32 representing number of non-zero elements per row.
            output_embeddings: [B, N, D] x float, embeddings for the current
                input sequence.
            supervision_ids: [B, N] x int64, (positive) supervision ids.
            supervision_embeddings: [B, N, D] x float.
            supervision_weights: Optional [B, N] x float. Optional weights for
                masking out invalid positions, or reweighting supervision labels.
            negatives_sampler: sampler used to obtain negative examples paired with
                positives.

        Returns:
            (1), loss for the current engaged sequence.
        """
        pass


@register("loss", "BCELoss")
class BCELoss(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model: SimilarityModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,              # [N', D] 
        supervision_ids: torch.Tensor,                # [N'] 
        supervision_embeddings: torch.Tensor,         # [N', D]
        supervision_weights: torch.Tensor,            # [N']
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=1,
        )

        positive_logits = (
            self._model.interaction(  # pyre-ignore [29]
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']

        sampled_negatives_logits = (
            self._model.interaction(  # pyre-ignore [29]
                input_embeddings=output_embeddings,  # [N', D]
                target_ids=sampled_ids,  # [N', 1]
                target_embeddings=sampled_negative_embeddings,  # [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N']
        sampled_negatives_valid_mask = (
            supervision_ids != sampled_ids.squeeze(1)
        ).float()  # [N']
        loss_weights = supervision_weights * sampled_negatives_valid_mask
        weighted_losses = (
            (
                F.binary_cross_entropy_with_logits(
                    input=positive_logits,
                    target=torch.ones_like(positive_logits),
                    reduction="none",
                )
                + F.binary_cross_entropy_with_logits(
                    input=sampled_negatives_logits,
                    target=torch.zeros_like(sampled_negatives_logits),
                    reduction="none",
                )
            )
            * loss_weights
            * 0.5
        )
        return weighted_losses.sum() / loss_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
          lengths: [B] x int32 representing number of non-zero elements per row.
          output_embeddings: [B, N, D] x float, embeddings for the current
              input sequence.
          supervision_ids: [B, N] x int64, (positive) supervision ids.
          supervision_embeddings: [B, N, D] x float.
          supervision_weights: Optional [B, N] x float. Optional weights for
              masking out invalid positions, or reweighting supervision labels.
          negatives_sampler: sampler used to obtain negative examples paired with
              positives.
        Returns:
          (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        jagged_supervision_weights = torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            negatives_sampler=negatives_sampler,
        )


@register("loss", "BCELossWithRatings")
class BCELossWithRatings(AutoregressiveLoss):
    def __init__(
        self,
        temperature: float,
        model: SimilarityModule,
    ) -> None:
        super().__init__()
        self._temperature: float = temperature
        self._model = model

    def jagged_forward(
        self,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        assert supervision_ids.size() == supervision_weights.size()

        target_logits = (
            self._model.interaction(  # pyre-ignore [29]
                input_embeddings=output_embeddings,  # [B, D] = [N', D]
                target_ids=supervision_ids.unsqueeze(1),  # [N', 1]
                target_embeddings=supervision_embeddings.unsqueeze(
                    1
                ),  # [N', D] -> [N', 1, D]
            )[0].squeeze(1)
            / self._temperature
        )  # [N', 1]

        weighted_losses = (
            F.binary_cross_entropy_with_logits(
                input=target_logits,
                target=supervision_ratings.to(dtype=target_logits.dtype),
                reduction="none",
            )
        ) * supervision_weights
        return weighted_losses.sum() / supervision_weights.sum()

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        supervision_ratings: torch.Tensor,
        negatives_sampler: NegativesSampler,
    ) -> torch.Tensor:
        """
        Args:
          lengths: [B] x int32 representing number of non-zero elements per row.
          output_embeddings: [B, N, D] x float, embeddings for the current
              input sequence.
          supervision_ids: [B, N] x int64, (positive) supervision ids.
          supervision_embeddings: [B, N, D] x float.
          supervision_weights: Optional [B, N] x float. Optional weights for
              masking out invalid positions, or reweighting supervision labels.
          negatives_sampler: sampler used to obtain negative examples paired with
              positives.
        Returns:
          (1), loss for the current engaged sequence.
        """
        assert output_embeddings.size() == supervision_embeddings.size()
        assert supervision_ids.size() == supervision_embeddings.size()[:-1]
        jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        jagged_supervision_ids = (
            torch.ops.fbgemm.dense_to_jagged(
                supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
            )[0]
            .squeeze(1)
            .long()
        )
        jagged_supervision_weights = torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1)
        return self.jagged_forward(
            output_embeddings=torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_ids=jagged_supervision_ids,
            supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
            supervision_weights=jagged_supervision_weights,
            supervision_ratings=torch.ops.fbgemm.dense_to_jagged(
                supervision_ratings.unsqueeze(-1),
                [jagged_id_offsets],
            )[0].squeeze(1),
            negatives_sampler=negatives_sampler,
        )
