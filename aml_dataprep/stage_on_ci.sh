#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage step (runs ON the compute instance, NOT as a VC job).
# Why here: the CI managed identity can READ cosmos (ADLS Gen1); a Singularity
# VC node cannot (NoIdentityOnCompute). vocab + parquet outputs are small
# (~2-6 GB) so they fit on the CI disk, then we push them to workspaceblobstore.
# The heavy 168 GB embeddings are produced later by the GPU encode job, which
# reads vocab from blob and never touches cosmos.
#
# Usage: bash stage_on_ci.sh
# Requires: az login (CI MSI), azureml-fsspec, pandas, pyarrow installed.
# ---------------------------------------------------------------------------
set -euo pipefail

SUB=72a0fe10-0a76-4898-9b7b-640e6e236fdc
RG=wb-aml
WS=pconv-aml-offline
DS=bingads_algo_pipelines_c08
COSMOS_ROOT="local/User/wenhlu/LRM_benchmark_v4"
WORK=$HOME/lrm-stage
REPO=$HOME/lrm-scaling-all-events           # cloned PR branch
DATAPREP=$REPO/data_prep

mkdir -p "$WORK/raw/train" "$WORK/raw/val" "$WORK/vocab" "$WORK/seqview"

echo "=== 1. Download raw shards from cosmos (fsspec, CI MSI identity) ==="
python3 - <<PY
from azureml.fsspec import AzureMachineLearningFileSystem as FS
import os
base="azureml://subscriptions/${SUB}/resourcegroups/${RG}/workspaces/${WS}/datastores/${DS}/paths/${COSMOS_ROOT}"
fs=FS(base)
for split in ("train","val"):
    rel="${COSMOS_ROOT}/%s"%split
    files=[f for f in fs.ls(rel) if f.endswith(".tsv")]
    print(f"{split}: {len(files)} shards -> downloading")
    for f in files:
        name=f.split("/")[-1]
        dst=os.path.join("${WORK}/raw",split,name)
        if os.path.exists(dst) and os.path.getsize(dst)>0:
            continue
        with fs.open(f.split("/paths/")[-1] if "/paths/" in f else f,"rb") as src, open(dst,"wb") as out:
            while True:
                chunk=src.read(8<<20)
                if not chunk: break
                out.write(chunk)
    print(f"{split}: done")
PY

echo "=== 2. Build vocab (step1, deterministic) ==="
python3 "$DATAPREP/step1_collect_vocab_v2.py" \
  --train_dir "$WORK/raw/train" --val_dir "$WORK/raw/val" \
  --output_dir "$WORK/vocab"
cat "$WORK/vocab/vocab_meta.json" || true

echo "=== 3. Build parquet seqview (step3) ==="
python3 "$DATAPREP/step3_v2.py" \
  --vocab_dir "$WORK/vocab" \
  --train_dir "$WORK/raw/train" --val_dir "$WORK/raw/val" \
  --output_dir "$WORK/seqview" --mode all_events
ls -R "$WORK/seqview" | head

echo "=== 4. Upload vocab + seqview to workspaceblobstore (derived/lrm_v4/) ==="
az ml data create --name lrm-v4-vocab --version 1 --type uri_folder \
  --path "$WORK/vocab" -g "$RG" -w "$WS" --subscription "$SUB" --only-show-errors \
  --query id -o tsv || echo "vocab upload may already exist"
az ml data create --name lrm-v4-seqview --version 1 --type uri_folder \
  --path "$WORK/seqview" -g "$RG" -w "$WS" --subscription "$SUB" --only-show-errors \
  --query id -o tsv || echo "seqview upload may already exist"

echo "=== STAGE COMPLETE ==="
echo "vocab    -> azureml:lrm-v4-vocab:1"
echo "seqview  -> azureml:lrm-v4-seqview:1"
echo "Next: submit aml_dataprep/jobs/2_encode.yml (8xA100) then 3_train.yml"
