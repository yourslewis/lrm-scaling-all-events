#!/usr/bin/env python3
"""
Visualize a PyTorch model using torchview and/or torchviz.

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
      [--gpu 1]
"""
import sys, os, argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Visualize PyTorch model")
parser.add_argument("--gin_config_file", required=True)
parser.add_argument("--data_path", required=True)
parser.add_argument("--ads_semantic_embd_path", default="")
parser.add_argument("--web_browsing_semantic_embd_path", default="")
parser.add_argument("--shopping_semantic_embd_path", default="")
parser.add_argument("--ads_pure_corpus_embd_path", default="")
parser.add_argument("--other_semantic_embd_path", default="")
parser.add_argument("--output_dir", default="/tmp")
parser.add_argument("--output_prefix", default="model_viz")
parser.add_argument("--gpu", type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0, os.getcwd())

import fbgemm_gpu  # noqa — must be before gin/torch imports for configurables
import torch
import gin

# Import configurables BEFORE gin.parse_config_file
from data.reco_dataset import get_reco_dataset
from trainer.util import make_model
from trainer.train import Trainer  # noqa — needed for gin
from trainer.data_loader import create_data_loader  # noqa — needed for gin

# Parse gin config AFTER imports
gin.parse_config_file(args.gin_config_file)

precomputed = {}
if args.ads_semantic_embd_path:
    precomputed[0] = args.ads_semantic_embd_path
if args.web_browsing_semantic_embd_path:
    precomputed[1] = args.web_browsing_semantic_embd_path
if args.shopping_semantic_embd_path:
    precomputed[2] = args.shopping_semantic_embd_path
if args.ads_pure_corpus_embd_path:
    precomputed[3] = args.ads_pure_corpus_embd_path
if args.other_semantic_embd_path:
    precomputed[4] = args.other_semantic_embd_path

ds = get_reco_dataset(
    mode="job",
    path=args.data_path,
    chronological=True,
    rank=0,
    world_size=1,
)

model = make_model(dataset=ds, precomputed_embeddings_domain_to_dir=precomputed or None)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dummy inputs matching max_sequence_length
N = ds.max_sequence_length
D = ds.embd_dim if hasattr(ds, 'embd_dim') and ds.embd_dim else 384
B = 2

input_ids = torch.randint(1, 1000, (B, N), device=device)
raw_input_emb = torch.randn(B, N, D, device=device)
raw_label_emb = torch.randn(B, N, D, device=device)
lengths = torch.full((B,), N, dtype=torch.long, device=device)
ratings = torch.ones(B, N, dtype=torch.long, device=device)
timestamps = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
type_ids = torch.randint(0, 10, (B, N), device=device)

prefix = os.path.join(args.output_dir, args.output_prefix)

# --- torchview ---
print("=== torchview ===")
try:
    from torchview import draw_graph
    graph = draw_graph(
        model,
        input_data=(input_ids, raw_input_emb, lengths, input_ids, raw_label_emb, ratings, type_ids, timestamps, None),
        expand_nested=True,
        depth=4,
        device=device,
        save_graph=True,
        filename=args.output_prefix + "_torchview",
        directory=args.output_dir,
    )
    print(f"torchview saved to {prefix}_torchview.png")
except Exception as e:
    print(f"torchview failed: {e}")
    try:
        graph = draw_graph(
            model,
            input_data=(input_ids, raw_input_emb, lengths, input_ids, raw_label_emb, ratings, type_ids, timestamps, None),
            depth=3,
            device=device,
            save_graph=True,
            filename=args.output_prefix + "_torchview",
            directory=args.output_dir,
        )
        print(f"torchview (depth=3) saved to {prefix}_torchview.png")
    except Exception as e2:
        print(f"torchview retry failed: {e2}")

# --- torchviz ---
print("=== torchviz ===")
try:
    from torchviz import make_dot
    model.train()
    out = model(
        input_ids=input_ids,
        raw_input_embeddings=raw_input_emb,
        input_lengths=lengths,
        label_ids=input_ids,
        raw_label_embeddings=raw_label_emb,
        ratings=ratings,
        type_ids=type_ids,
        timestamps=timestamps,
    )
    loss = out[1]
    dot = make_dot(loss, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot.format = "png"
    dot.render(prefix + "_torchviz", cleanup=True)
    print(f"torchviz saved to {prefix}_torchviz.png")
except Exception as e:
    print(f"torchviz failed: {e}")
    import traceback
    traceback.print_exc()

print("=== DONE ===")
