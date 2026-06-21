# ROCm / MI image for the LRM-HSTU stack (MI200 gfx90a, MI300X gfx942)
# Parallel to the CUDA path — does NOT replace it. A/B only.
#
# Base provides: ROCm torch, pytorch-triton-rocm, RCCL. fbgemm strategy is
# layered separately (see FBGEMM note). App deps come from requirements-rocm.txt
# (nvidia-*, faiss-gpu, torchrec, torch, triton stripped — see that file).
#
# NOTE: confirm the exact published tag on AML before building. As of 2025 the
# rocm/pytorch tags follow: rocm<ver>_ubuntu22.04_py3.10_pytorch_release_<x.y.z>.
ARG ROCM_BASE=rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
FROM ${ROCM_BASE}

# bf16 on CDNA3 (MI300) is strong; keep HIP visible as the "cuda" device_type.
ENV HSA_OVERRIDE_GFX_VERSION="" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# App-level pip layer (no torch/triton/fbgemm/nvidia/faiss/torchrec here).
COPY docker/rocm/requirements-rocm.txt /tmp/requirements-rocm.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements-rocm.txt

# ── FBGEMM strategy (pick at build, default = shim-safe) ──────────────────
# The model uses ONLY 3 fbgemm jagged ops (jagged_to_padded_dense,
# dense_to_jagged, asynchronous_complete_cumsum). Three paths, in order of
# preference:
#   1) Prebuilt ROCm fbgemm wheel matching the base torch (try first on AML).
#   2) Source build in-image (pytorch/FBGEMM, ROCm target) — heavier.
#   3) Pure-torch shim already in tools/ (patch_fbgemm) — guaranteed fallback,
#      used by the smoke harness so smoke #1 has ZERO fbgemm-build risk.
# Smoke #1 deliberately needs none of these (uses the shim).
ARG FBGEMM_MODE=shim
RUN if [ "$FBGEMM_MODE" = "wheel" ]; then \
      python -m pip install fbgemm-gpu-rocm || \
      echo "[warn] no prebuilt ROCm fbgemm wheel; falling back to shim/source at runtime"; \
    fi

# rocm-smi sanity at build (non-fatal; no GPU at build time)
RUN rocm-smi --showproductname || true
