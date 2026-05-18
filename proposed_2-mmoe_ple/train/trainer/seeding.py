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

"""Utilities for deterministic experiment seeding."""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def get_gin_configured_seed(
    parameter_name: str = "Trainer.random_seed",
    default: int = 42,
) -> int:
    """Return a gin-configured seed, falling back to the Python default.

    Gin only exposes values that were explicitly bound in a config. Most of our
    experiment configs rely on Trainer.random_seed's Python default, so this
    helper deliberately falls back to that value when the parameter is unbound.
    """

    try:
        import gin

        value = gin.query_parameter(parameter_name)
        if value is not None:
            return int(value)
    except Exception:
        pass
    return int(default)


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = False,
    rank: Optional[int] = None,
    log_prefix: str = "",
) -> int:
    """Seed Python, NumPy, and Torch RNGs for reproducible experiments.

    Args:
        seed: Base seed to apply. The same seed should be used before model
            construction on all DDP ranks so parameters initialize identically.
        deterministic_algorithms: If true, ask PyTorch to use deterministic
            kernels where available. This may raise for unsupported operators,
            so the default is false: the function still fully seeds RNGs without
            changing kernel selection.
        rank: Optional rank to include in the log message only. The seed is not
            offset by rank; model initialization must be identical across ranks.
        log_prefix: Optional context prefix for logs.

    Returns:
        The normalized seed value used by NumPy.
    """

    normalized_seed = int(seed) % (2**32)

    random.seed(seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Avoid cuDNN benchmark auto-tuning introducing extra run-to-run variation.
    torch.backends.cudnn.benchmark = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Required by CUDA for deterministic cublas matmul kernels on recent
        # PyTorch/CUDA stacks. Respect an existing stricter user setting.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    rank_msg = f", rank={rank}" if rank is not None else ""
    prefix = f"{log_prefix}: " if log_prefix else ""
    logging.info(
        "%sseeded Python/NumPy/Torch RNGs with seed=%d%s, "
        "deterministic_algorithms=%s",
        prefix,
        seed,
        rank_msg,
        deterministic_algorithms,
    )
    return normalized_seed
