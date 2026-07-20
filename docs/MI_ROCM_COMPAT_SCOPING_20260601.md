# MI / ROCm Compatibility — Scoping Report

**Branch:** `feat/mi-rocm-compat` (off `origin/main` @ 48b6507, includes PR #26 AML dataprep pipeline)
**Date:** 2026-06-01
**Author:** Don (for Lewis, via Rex work order)
**Status:** Scoping complete on the make-or-break question. Image build + smoke test pending approval.

---

## TL;DR — Verdict on the fbgemm/torchrec-ROCm question

**This is closer to a weekend than a slog.** The feared blocker (fbgemm-gpu + torchrec on ROCm) is **much smaller than the brief assumed**, for three concrete reasons found in the tree:

1. **`torchrec` is never imported in code** — it is only pinned in `environment_dev.yml` (×4 copies). No runtime path does `import torchrec`. → **Droppable** from the MI env. This removes the single hardest ROCm-availability item.
2. **fbgemm is used for only 3 jagged ops**, not the TBE embedding tables that are the painful part of fbgemm-gpu-on-ROCm:
   - `torch.ops.fbgemm.jagged_to_padded_dense`
   - `torch.ops.fbgemm.dense_to_jagged`
   - `torch.ops.fbgemm.asynchronous_complete_cumsum`
3. **A pure-PyTorch fallback for all 3 ops already exists** in `tools/export_netron.py` and `tools/visualize_model.py` (`patch_fbgemm()`). Correct but slow (Python loops). It is a ready safety net if a ROCm fbgemm build is unavailable — we'd just need a performant version for the hot path.

So the gating item is no longer "port torchrec to ROCm." It is: **"get fbgemm_gpu's 3 jagged ops working on ROCm,"** which has **three** independent paths (prebuilt ROCm wheel / source build in image / existing shim). Low risk.

---

## CUDA coupling inventory (verified by grep on this tree)

| Item | Where | ROCm action | Risk |
|---|---|---|---|
| `torch==2.2.1` (prod) / `2.6.0` (dev) | `*/environment.yml`, `environment_dev.yml` | ROCm wheel from `download.pytorch.org/whl/rocm6.x` (or base image) | Low |
| `nvidia-*-cu12` block (cublas/cudnn/cufft/curand/cusolver/cusparse/nccl/nvjitlink/nvtx) | `environment.yml` L156–167 | **Drop** — provided by `rocm/pytorch` base image | Low |
| `nvidia-nccl-cu12` | L165 | RCCL (bundled in ROCm torch) | Low |
| `triton==2.2.0` (prod) / `3.2.0` (dev) | L224 / dev L244 | `pytorch-triton-rocm` matching the torch build | Low |
| **`fbgemm-gpu==1.2.0` / `1.1.0`** | L115 / dev L118 | ROCm fbgemm build (3 jagged ops only) — see verdict | **Med→Low** |
| **`torchrec==1.1.0`** | dev L242 (×4) | **Drop** — not imported anywhere in code | Low (was the scare) |
| `faiss-gpu==1.7.2` | L113 (×8) | **Drop or faiss-cpu** — *not imported anywhere in code either* | Low (NEW — not in brief) |
| `flash-attn` | — | **None present** (grep clean). HSTU uses fbgemm-jagged attention, not flash. | None |
| autocast `"cuda"` | `hstu.py:485` | **DO NOT edit** — `"cuda"` is correct on ROCm (HIP keeps the cuda device_type; `"hip"` is an invalid autocast string and throws). bf16 = flip the gin flag instead (see below). | Low |
| encode `--device cuda:0` | `data_prep/step2_encode_embeddings.py:19` | works as-is; ensure ROCm sentence-transformers/torch in encode image | Low |

### New risk Rex's brief did NOT flag
- **`faiss-gpu==1.7.2`** is pinned in every env (×8) but **not imported in any `.py`** in the tree. On a CUDA env the conda solve tolerates it; on ROCm `faiss-gpu` (CUDA build) will either fail to solve or be useless. **Action: drop it, or swap to `faiss-cpu`** on the MI path. Cheap, but it would have blocked the env solve silently.

---

## External availability (authoritative sources, 2025)
- **fbgemm_gpu supports ROCm** — PyTorch FBGEMM build docs: "FBGEMM_GPU supports running on AMD (ROCm) devices." ROCm PyTorch-compat docs note grouped-GEMM via fbgemm_gpu GenAI. FBGEMM_GPU v1.5.0 release notes list tested ROCm setups.
- **Prebuilt ROCm fbgemm wheels are not on PyPI** — they come via the ROCm torch index / `rocm/pytorch` base image, or source build. Our pins (1.1.0/1.2.0) predate the cleaner ROCm wheels; **bump fbgemm to a version with ROCm artifacts at our target torch**, or build in the image.
- **PyTorch 2.9 has ROCm wheel-variant support for ROCm6.3 / 6.4**; ROCm6.2 nightly path also exists. Target a torch that has matching `pytorch-triton-rocm` + fbgemm.

---

## bf16 wiring — CORRECTED (verified by Rex + re-confirmed in tree)
The MI300 bf16 switch is **not** an `hstu.py` edit. `main.py` calls `gin.parse_config_file(<single file>)` with **no** `--gin_bindings`, so:
- `Trainer.main_module_bf16` (default `False`, L800 gin line 41) is the real lever.
- Plumbed: `train.py:504/529/582/605` → `float_dtype=torch.bfloat16 if main_module_bf16` → HSTU `autocast_dtype` (`hstu.py:460`), consumed by the autocast at `hstu.py:485`.
- Mechanism: generate a self-contained sibling gin (`..._l800_bf16.gin` = base + `Trainer.main_module_bf16 = True`; gin last-write-wins within one file). Tool: `docker/rocm/gen_bf16_gin.py` (done, generates line 64 override over line 41).

## Branch artifacts added (no-cluster, CUDA path untouched)
- `docker/rocm/Dockerfile` — `rocm/pytorch` base + `requirements-rocm.txt`; FBGEMM_MODE build arg (shim default → zero build risk for smoke #1).
- `docker/rocm/requirements-rocm.txt` — generated from the CUDA env, **141 kept / 17 stripped** (nvidia-*×11, faiss-gpu, fbgemm-gpu, torchrec, torch, triton).
- `docker/rocm/smoke_mi.py` — 2-stage smoke (device/hip/bf16 probe + 3 jagged ops under shim). Shim algorithm validated locally (roundtrip == identity).
- `docker/rocm/gen_bf16_gin.py` + `bf16_overlay.gin` — bf16 enablement.
- `aml_dataprep/jobs/smoke_rocm.yml` — 1-GPU Singularity smoke, blobstore-only, 3 PLACEHOLDER fields (compute / instance_type / image).

## Pre-existing gaps noticed (not ROCm-specific)
- `aml_dataprep/jobs/3_train.yml` `pip install -r proposed_2-mmoe_ple/requirements.txt` — **that file does not exist** (swallowed by `|| true`). Flagging; not fixing on this branch.

## Proposed image (cleaner than converting conda wholesale)
- Base: AMD `rocm/pytorch:rocm6.x_ubuntu22.04_py3.10_pytorch_release_2.6.0` (exact tag TBD — confirm on AML).
- Layer our **non-CUDA pip deps** on top (strip the `nvidia-*`, `faiss-gpu`, `torchrec` lines; repoint `torch`/`triton`/`fbgemm` to ROCm).
- bf16 autocast for the MI path.

---

## MI smoke test plan (MI200 gfx90a / MI300X gfx942)
1. 1-node Singularity job, **workspaceblobstore only** (no cosmos → avoids `NoIdentityOnCompute`).
2. Print `torch.version.hip`, `torch.cuda.get_device_name(0)` (expect MI200/MI300X), `torch.cuda.device_count()`.
3. Tiny **bf16 matmul** + **one HSTU forward** on a toy batch (exercises the 3 fbgemm jagged ops on ROCm — the real compat probe).
4. Then 8×MI300X encode/train smoke on VC `Feeds` (`Singularity.NDMI300Xv5`); MI200 via VC `ads`/`Feeds` (`Singularity.NDMI200v4`). Pull exact `instance_type` off an existing job on those VCs.
5. Reuse PR #26 split-pipeline (stage→blob on CPU, GPU reads blob) so the MI job never needs cosmos identity.

## Open items before full port
- Confirm exact `rocm/pytorch` tag + matching fbgemm/triton-rocm versions available on AML.
- Decide fbgemm strategy: prebuilt ROCm wheel **vs** source-build-in-image **vs** performant shim. (Recommend: try prebuilt wheel first; shim is the guaranteed fallback.)
- Pull exact `instance_type` strings off existing MI jobs (e.g. `ivory_wall` on `Feeds`).
- Get PR #26 YAML templates + `msanrrsvcgpt` submit helper from Rex.
