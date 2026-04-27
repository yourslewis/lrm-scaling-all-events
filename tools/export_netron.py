#!/usr/bin/env python3
"""
Export any PyTorch model to Netron-compatible format.

Handles models with custom ops (e.g., fbgemm) by monkey-patching them
with pure PyTorch equivalents during export.

Usage:
  cd <model_dir>/train
  python /home/yourslewis/lrm-scaling-all-events/tools/export_netron.py \
      --gin_config_file <config.gin> \
      --data_path <data_dir> \
      --ads_semantic_embd_path <embd_dir>/domain_0 \
      [--web_browsing_semantic_embd_path ...] \
      [--shopping_semantic_embd_path ...] \
      [--ads_pure_corpus_embd_path ...] \
      [--other_semantic_embd_path ...] \
      [--output /tmp/model_netron.onnx] \
      [--format onnx|pt]
"""
import sys, os, argparse, copy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Export model for Netron")
parser.add_argument("--gin_config_file", required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--ads_semantic_embd_path", default="")
parser.add_argument("--web_browsing_semantic_embd_path", default="")
parser.add_argument("--shopping_semantic_embd_path", default="")
parser.add_argument("--ads_pure_corpus_embd_path", default="")
parser.add_argument("--other_semantic_embd_path", default="")
parser.add_argument("--output", default="/tmp/model_netron")
parser.add_argument("--format", default="both", choices=["onnx", "pt", "both"])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.getcwd())

import fbgemm_gpu
import torch
import torch.nn as nn
import gin

from data.reco_dataset import get_reco_dataset
from trainer.util import make_model
from trainer.train import Trainer
from trainer.data_loader import create_data_loader

gin.parse_config_file(args.gin_config_file)

precomputed = {}
for i, p in enumerate([args.ads_semantic_embd_path, args.web_browsing_semantic_embd_path,
                        args.shopping_semantic_embd_path, args.ads_pure_corpus_embd_path,
                        args.other_semantic_embd_path]):
    if p:
        precomputed[i] = p

ds = get_reco_dataset(mode="job", path=args.data_path, chronological=True, rank=0, world_size=1)
model = make_model(dataset=ds, precomputed_embeddings_domain_to_dir=precomputed or None)
model = model.cpu().eval()

# ─── fbgemm → pure PyTorch shims ───

def _jagged_to_padded_dense_shim(values, offsets, max_lengths, padding_value=0.0):
    """Pure PyTorch replacement for fbgemm.jagged_to_padded_dense.
    When input is already padded (3D), just return it."""
    if values.dim() == 3:
        return values
    # offsets is a list containing one tensor: [0, L0, L0+L1, ...]
    offs = offsets[0]
    max_len = max_lengths[0]
    B = offs.size(0) - 1
    D = values.size(-1)
    out = torch.full((B, max_len, D), padding_value, dtype=values.dtype, device=values.device)
    for i in range(B):
        start = offs[i].item()
        end = offs[i + 1].item()
        length = min(end - start, max_len)
        out[i, :length] = values[start:start + length]
    return out

def _dense_to_jagged_shim(dense, offsets):
    """Pure PyTorch replacement for fbgemm.dense_to_jagged.
    Concatenate valid (non-padding) regions into a jagged tensor."""
    offs = offsets[0]
    B = dense.size(0)
    parts = []
    for i in range(B):
        start = 0
        length = (offs[i + 1] - offs[i]).item()
        parts.append(dense[i, :length])
    jagged = torch.cat(parts, dim=0)
    return jagged, offs

def _asynchronous_complete_cumsum_shim(x):
    """Pure PyTorch replacement for fbgemm.asynchronous_complete_cumsum."""
    return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), torch.cumsum(x, dim=0)])

# Monkey-patch fbgemm ops
_original_fbgemm_ops = {}

def patch_fbgemm():
    """Replace fbgemm custom ops with pure PyTorch equivalents."""
    global _original_fbgemm_ops
    _original_fbgemm_ops = {
        'jagged_to_padded_dense': torch.ops.fbgemm.jagged_to_padded_dense,
        'dense_to_jagged': torch.ops.fbgemm.dense_to_jagged,
        'asynchronous_complete_cumsum': torch.ops.fbgemm.asynchronous_complete_cumsum,
    }
    torch.ops.fbgemm.jagged_to_padded_dense = _jagged_to_padded_dense_shim
    torch.ops.fbgemm.dense_to_jagged = _dense_to_jagged_shim
    torch.ops.fbgemm.asynchronous_complete_cumsum = _asynchronous_complete_cumsum_shim
    print("[patch] fbgemm ops replaced with pure PyTorch shims")

def unpatch_fbgemm():
    """Restore original fbgemm ops."""
    for name, op in _original_fbgemm_ops.items():
        setattr(torch.ops.fbgemm, name, op)
    print("[patch] fbgemm ops restored")


# ─── Wrapper for clean single-tensor export ───

class ExportWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model_encoder = full_model.model
        self.mmoe = full_model.multi_task_module

    def forward(self, input_ids: torch.Tensor, raw_input_emb: torch.Tensor,
                lengths: torch.Tensor, ratings: torch.Tensor,
                type_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        past_embeddings = self.model_encoder._embedding_module(raw_input_emb)
        seq_embeddings = self.model_encoder(
            past_lengths=lengths,
            past_ids=input_ids,
            past_embeddings=past_embeddings,
            past_payloads={"timestamps": timestamps, "ratings": ratings, "type_ids": type_ids},
        )
        if self.mmoe is not None:
            task_embeddings = self.mmoe(seq_embeddings)
            if isinstance(task_embeddings, dict):
                return task_embeddings[0]
            return task_embeddings
        return seq_embeddings


wrapper = ExportWrapper(model).cpu().eval()

# Dummy inputs
N = ds.max_sequence_length
D = ds.embd_dim if hasattr(ds, 'embd_dim') and ds.embd_dim else 384
B = 2
input_ids = torch.randint(1, 1000, (B, N))
raw_emb = torch.randn(B, N, D)
lengths = torch.full((B,), N, dtype=torch.long)
ratings = torch.ones(B, N, dtype=torch.long)
ts = torch.arange(N).unsqueeze(0).expand(B, -1).contiguous()
type_ids = torch.randint(0, 10, (B, N))

# Patch fbgemm for export
patch_fbgemm()

# Verify forward works with shims
print("Testing forward with patched fbgemm...")
with torch.no_grad():
    out = wrapper(input_ids, raw_emb, lengths, ratings, type_ids, ts)
    print(f"Forward OK: output shape {out.shape}")

# Export TorchScript
if args.format in ("pt", "both"):
    print("=== TorchScript export ===")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (input_ids, raw_emb, lengths, ratings, type_ids, ts))
        out_path = args.output + ".pt"
        traced.save(out_path)
        print(f"TorchScript saved: {out_path} ({os.path.getsize(out_path)} bytes)")
    except Exception as e:
        print(f"TorchScript failed: {e}")

# Export ONNX
if args.format in ("onnx", "both"):
    print("=== ONNX export ===")
    try:
        out_path = args.output + ".onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (input_ids, raw_emb, lengths, ratings, type_ids, ts),
                out_path,
                input_names=["input_ids", "raw_input_emb", "lengths", "ratings", "type_ids", "timestamps"],
                output_names=["task_embeddings"],
                opset_version=14,
                do_constant_folding=True,
            )
        print(f"ONNX saved: {out_path} ({os.path.getsize(out_path)} bytes)")
    except Exception as e:
        print(f"ONNX failed: {e}")

unpatch_fbgemm()
print("=== DONE ===")
