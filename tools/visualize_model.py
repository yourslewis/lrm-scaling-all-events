#!/usr/bin/env python3
"""
Visualize a PyTorch model using torchview (depth=1 and depth=2).

Automatically patches fbgemm custom ops with pure PyTorch shims
so tracing/export works for models using HSTU.

Usage:
  cd <model_dir>/train
  python /home/yourslewis/lrm-scaling-all-events/tools/visualize_model.py \
      --gin_config_file <config.gin> \
      --data_path <data_dir> \
      --ads_semantic_embd_path <embd_dir>/domain_0 \
      [--web_browsing_semantic_embd_path ...] \
      [--shopping_semantic_embd_path ...] \
      [--ads_pure_corpus_embd_path ...] \
      [--other_semantic_embd_path ...] \
      [--output_dir /tmp] \
      [--output_prefix model_viz] \
      [--depths 1,2]
"""
import sys, os, argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description="Visualize PyTorch model with torchview")
parser.add_argument("--gin_config_file", required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--ads_semantic_embd_path", default="")
parser.add_argument("--web_browsing_semantic_embd_path", default="")
parser.add_argument("--shopping_semantic_embd_path", default="")
parser.add_argument("--ads_pure_corpus_embd_path", default="")
parser.add_argument("--other_semantic_embd_path", default="")
parser.add_argument("--output_dir", default="/tmp")
parser.add_argument("--output_prefix", default="model_viz")
parser.add_argument("--depths", default="1,2", help="Comma-separated depths to render (default: 1,2)")
parser.add_argument("--train_dir", default=".", help="Path to the train/ directory (for correct imports)")
args = parser.parse_args()

# Change to train dir for correct imports
if args.train_dir != ".":
    os.chdir(args.train_dir)
sys.path.insert(0, os.getcwd())

import fbgemm_gpu
import torch
import torch.nn as nn
import gin

from data.reco_dataset import get_reco_dataset
from trainer.util import make_model
from trainer.train import Trainer
from trainer.data_loader import create_data_loader

# ─── fbgemm → pure PyTorch shims ───

def _jagged_to_padded_dense_shim(values, offsets, max_lengths, padding_value=0.0):
    if values.dim() == 3:
        return values
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
    offs = offsets[0]
    B = dense.size(0)
    parts = []
    for i in range(B):
        length = (offs[i + 1] - offs[i]).item()
        parts.append(dense[i, :length])
    return torch.cat(parts, dim=0), offs

def _asynchronous_complete_cumsum_shim(x):
    return torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), torch.cumsum(x, dim=0)])

_original_fbgemm_ops = {}

def patch_fbgemm():
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
    for name, op in _original_fbgemm_ops.items():
        setattr(torch.ops.fbgemm, name, op)
    print("[patch] fbgemm ops restored")


# ─── Wrapper for clean single-tensor output ───

class ExportWrapper(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.model_encoder = full_model.model
        self.mmoe = full_model.multi_task_module if hasattr(full_model, 'multi_task_module') else None
        self.is_hstu_mmoe = self.mmoe is not None and hasattr(self.mmoe, 'expert_models')
        if hasattr(full_model, 'expert_hstu_models'):
            self.expert_hstu_models = full_model.expert_hstu_models

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
            if self.is_hstu_mmoe:
                # HSTUMMoE needs raw inputs for expert encoders
                past_payloads = {"timestamps": timestamps, "ratings": ratings, "type_ids": type_ids}
                task_embeddings = self.mmoe(
                    gate_embeddings=seq_embeddings,
                    past_lengths=lengths,
                    past_ids=input_ids,
                    past_embeddings=past_embeddings,
                    past_payloads=past_payloads,
                )
            else:
                task_embeddings = self.mmoe(seq_embeddings)
            if isinstance(task_embeddings, dict):
                return task_embeddings[0]
            return task_embeddings
        return seq_embeddings


# ─── Main ───

gin.parse_config_file(args.gin_config_file)

precomputed = {}
for i, p in enumerate([args.ads_semantic_embd_path, args.web_browsing_semantic_embd_path,
                        args.shopping_semantic_embd_path, args.ads_pure_corpus_embd_path,
                        args.other_semantic_embd_path]):
    if p:
        precomputed[i] = p

ds = get_reco_dataset(mode="job", path=args.data_path, chronological=True, rank=0, world_size=1)
model = make_model(dataset=ds, precomputed_embeddings_domain_to_dir=precomputed or None)
device = torch.device("cpu")
model = model.to(device).eval()

wrapper = ExportWrapper(model).to(device).eval()

N = ds.max_sequence_length
D = ds.embd_dim if hasattr(ds, 'embd_dim') and ds.embd_dim else 384
B = 2
input_ids = torch.randint(1, 1000, (B, N), device=device)
raw_emb = torch.randn(B, N, D, device=device)
lengths = torch.full((B,), N, dtype=torch.long, device=device)
ratings = torch.ones(B, N, dtype=torch.long, device=device)
ts = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).contiguous()
type_ids = torch.randint(0, 10, (B, N), device=device)

# Patch fbgemm
patch_fbgemm()

# Verify forward works
print("Testing forward with patched fbgemm...")
with torch.no_grad():
    out = wrapper(input_ids, raw_emb, lengths, ratings, type_ids, ts)
    print(f"Forward OK: output shape {out.shape}")

depths = [int(d.strip()) for d in args.depths.split(",")]

from torchview import draw_graph

for depth in depths:
    fname = f"{args.output_prefix}_depth{depth}"
    print(f"=== torchview depth={depth} ===")
    try:
        g = draw_graph(
            wrapper,
            input_data=(input_ids, raw_emb, lengths, ratings, type_ids, ts),
            depth=depth,
            device=device,
            save_graph=True,
            filename=fname,
            directory=args.output_dir,
        )
        out_path = os.path.join(args.output_dir, fname + ".png")
        if os.path.exists(out_path):
            print(f"OK: {out_path} ({os.path.getsize(out_path)} bytes)")
        else:
            print(f"OK (rendered but PNG path may differ)")
    except Exception as e:
        print(f"depth={depth} failed: {e}")

unpatch_fbgemm()
print("=== DONE ===")
