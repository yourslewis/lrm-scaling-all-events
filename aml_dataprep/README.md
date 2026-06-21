# AML Data-Prep + Train Pipeline (SPLIT design — regenerate LRM data on Azure)

Regenerates LRM benchmark-v4 training data and runs L800 training **entirely on
Azure ML**, reading raw shards from cosmos (ADLS Gen1). Uses a **split design** to
work around the Singularity data-access limitation.

## The core constraint (why split)
- A Singularity **VC job** node has **no identity** to read the cosmos ADLS datastore
  → `ScriptExecution.StreamAccess.Authentication … NoIdentityOnCompute`.
- The **compute instance** (its managed identity) **CAN** read cosmos (verified).
- A VC job **CAN** read/write `workspaceblobstore` (verified: a 8xA100 nd40 job
  mounted+read a blob file — see test `teal_caravan_4gt0wx8962`, Completed).

So: only the **cosmos-touching steps run on the compute instance**; the **GPU steps
run as VC jobs reading from blobstore**. No identity needs both cosmos + VC-submit.

## Data flow
```
cosmos LRM_benchmark_v4/{train(110),val(38)}   (ADLS Gen1)
        |
        v   stage_on_ci.sh  — runs ON the compute instance (CI MSI reads cosmos)
        |     step1 vocab + step3 parquet (small, ~2-6 GB, fits CI disk)
        |     uploads -> workspaceblobstore as data assets:
        |       azureml:lrm-v4-vocab:1   azureml:lrm-v4-seqview:1
        |
        +--> jobs/2_encode.yml   (8x A100 nd40, reads vocab from blob)
        |       -> workspaceblobstore derived/lrm_v4/semantic_embeddings_v3/domain_*/shard_0.npy
        |
        +--> jobs/3_train.yml    (8x A100 nd40, reads seqview + embeddings from blob)
                -> workspaceblobstore derived/lrm_v4/l800_output
```

## Target compute
- VC: **ads-shared-nd40** (A100-40, `Singularity.ND96rs_v4`), `job_tier: Standard`
  (live quota: 152 dedicated A100 free — the most non-preemptible A100 headroom).
- Stage runs on compute instance **lrm-smoke-submitter** (pconv-aml-offline).

## Run order
```bash
# 0. on the compute instance terminal (az login --identity already done):
git clone <repo> ~/lrm-scaling-all-events && cd ~/lrm-scaling-all-events
git checkout feat/aml-dataprep-pipeline

# 1. STAGE (cosmos -> blob), on the CI:
bash aml_dataprep/stage_on_ci.sh
#    -> azureml:lrm-v4-vocab:1, azureml:lrm-v4-seqview:1

# 2. ENCODE (8x A100):
az ml job create -f aml_dataprep/jobs/2_encode.yml -g wb-aml -w pconv-aml-offline

# 3. TRAIN L800 (8x A100), after encode completes:
az ml job create -f aml_dataprep/jobs/3_train.yml -g wb-aml -w pconv-aml-offline
```

## Files
| File | Role |
|---|---|
| `stage_on_ci.sh` | Cosmos→blob staging (vocab+parquet) — runs on the compute instance |
| `encode_embeddings_multigpu.py` | 8-GPU sharded encoder → `domain_*/shard_0.npy` |
| `jobs/2_encode.yml` | Encode VC job (8x A100, blob in/out) |
| `jobs/3_train.yml` | L800 training VC job (8x A100, blob in/out) |

## Determinism / fidelity
- step1 sorts shards + assigns ids sequentially → reproducible vocab.
- encoder only parallelizes; each id encoded once, placed at row `item_id`;
  `embedding(text)` pure → merged shard independent of GPU split.
- step3 re-derives `encoded_id = domain*1e9 + item_id` → parquet ids match embeddings.

## TODO / notes
- val shards 0,1 currently missing on cosmos (38/40); train complete (110/110).
  Re-run stage after they land for the full 40-chunk eval.
- AMD VCs (Feeds MI300X, ads MI200) would need ROCm port (different torch/image);
  staying on NVIDIA A100 (no code change).
- Outputs land on workspaceblobstore (~174 GB); blob accounts hold up to 5 PB, so
  capacity is a non-issue. Clean up `derived/lrm_v4/` after the run if desired.
