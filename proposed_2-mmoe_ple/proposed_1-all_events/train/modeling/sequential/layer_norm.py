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

#!/usr/bin/env python3

# pyre-strict


from typing import List

import torch



def pytorch_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    return torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape,
        weight.to(torch.float32),
        bias.to(torch.float32),
        eps,
    ).to(dtype)


def pytorch_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    return (
        x
        * torch.sigmoid(
            torch.nn.functional.layer_norm(
                x,
                normalized_shape,
                weight.to(torch.float32),
                bias.to(torch.float32),
                eps,
            )
        )
    ).to(dtype)



def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return pytorch_layer_norm(
        x,
        [
            x.shape[-1],
        ],
        weight,
        bias,
        eps,
    )


def swish_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return pytorch_swish_layer_norm(
        x,
        [
            x.shape[-1],
        ],
        weight,
        bias,
        eps,
    )


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._normalized_shape: List[int] = [dim]
        self._eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(self._normalized_shape),
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self._normalized_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(
            x=x,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
        )


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self._eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwishLayerNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._normalized_shape: List[int] = [dim]
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape))
        self._eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return swish_layer_norm(
            x=x,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
        )
