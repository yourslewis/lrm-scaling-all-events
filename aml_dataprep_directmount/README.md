# AML Data-Prep + Train Pipeline (DIRECT-MOUNT design)

Second pipeline that **replicates the team's PME pattern on CorpNet**: every
Singularity VC job **mounts the cosmos ADLS datastore directly**, exactly like
the `azure_machine_learning_onboarding/.../train_singularity_pme.ipynb`
submitters in `ai-microsoft/large-rec-model`. **No relay, no compute-instance
staging.**

This is the alternative to the SPLIT design in `../aml_dataprep/`. Use whichever
the access reality allows (see the decision gate below).

## Why a second pipeline
The repo's tested jobs prove a Singularity VC job **can** read cosmos when the
datastore + submitting identity are wired the PME way — input is just
`azureml://datastores/<ds>/paths/<root>` with `ro_mount` on the GPU node. Our
split design only existed because, on CorpNet, an interactive user (and then a
service principal into `pconv-aml-offline` + the pipelines datastore) hit
`ScriptExecution.StreamAccess.Authentication ... NoIdentityOnCompute`. If we
mirror the repo's datastore + a workspace where it is registered, direct-mount
may just work — and we delete the relay entirely.

## Account / identity
- Submit as the **`msanrrsvcgpt`** service account (sole identity:
  `az logout; az account clear; az login -u msanrrsvcgpt@microsoft.com -p <pw> --tenant <corp-tenant>`).
- Verified creator on past CorpNet jobs in workspace **CPT-LLM-Pipelines**
  (rg `Ads-LLM-Pipelines`). That workspace already has the
  **networkprotection** store registered as `adls_networkprotection08`
  (store `bingads-algo-prod-networkprotection-c08`) — the same store the repo
  direct-mounts. That makes it the natural target workspace.

## Datastore toggle (the fallback Wenhao asked for)
Single knob, overridable at submit time:

| Store | Datastore name | Use when |
|---|---|---|
| **networkprotection** (DEFAULT) | `bingads_algo_prod_networkprotection_c08` | repo's proven direct-mount store; Wenhao can place the data here |
| pipelines | `bingads_algo_pipelines_c08` | where `wenhlu/LRM_benchmark_v4` lives today |

Override example (flip to pipelines store):
```bash
--set inputs.cosmos_uri="azureml://datastores/bingads_algo_pipelines_c08/paths/local/User/wenhlu/LRM_benchmark_v4"
```
> If `msanrrsvcgpt` cannot read `bingads_algo_pipelines_c08`, Wenhao offered to
> also place the data under `bingads_algo_prod_networkprotection_c08` — then the
> DEFAULT path works unchanged.

## Run order
Set your target once (a workspace where the chosen datastore is registered AND
`msanrrsvcgpt` can submit — CPT-LLM-Pipelines is the candidate):
```bash
SUB=<corp-sub>; RG=Ads-LLM-Pipelines; WS=CPT-LLM-Pipelines   # confirm
```

### 0. PROBE first (cheap, decides everything)
```bash
az ml job create -f aml_dataprep_directmount/jobs/probe_mount.yml -g $RG -w $WS --subscription $SUB
```
- **PASS** (`DIRECT_MOUNT_OK`, shard listing + head of a shard) -> continue below.
- **FAIL** (`NoIdentityOnCompute`) -> direct-mount is blocked for this
  identity/datastore/workspace. Try the other datastore (`--set inputs.cosmos.path=...`)
  and/or another workspace; if still blocked, use the SPLIT pipeline
  `../aml_dataprep/pipeline.yml` (proven working).

### 1. End-to-end pipeline (vocab -> parquet -> encode -> train L800)
```bash
az ml job create -f aml_dataprep_directmount/pipeline.yml -g $RG -w $WS --subscription $SUB
# flip datastore if needed:
#   --set inputs.cosmos_uri="azureml://datastores/bingads_algo_pipelines_c08/paths/local/User/wenhlu/LRM_benchmark_v4"
```

Or run the stages discretely:
```bash
az ml job create -f aml_dataprep_directmount/jobs/1_dataprep.yml -g $RG -w $WS --subscription $SUB
# (encode + train reuse ../aml_dataprep/jobs/2_encode.yml and 3_train.yml — they
#  already read vocab/seqview from blob and never touch cosmos; just point their
#  inputs at the derived/lrm_v4_dm/* assets this pipeline produces.)
```

## Files
| File | Role |
|---|---|
| `jobs/probe_mount.yml` | **Run first.** Cheap VC job that mounts cosmos + ls + head — proves/refutes direct-mount on CorpNet |
| `pipeline.yml` | Full DAG, no relay: vocab -> parquet -> encode(8xA100) -> train L800, cosmos mounted into vocab/parquet |
| `jobs/1_dataprep.yml` | Standalone vocab+parquet in one cosmos-mounted VC job |

## What's shared with the split design (not duplicated)
- `../data_prep/step1_collect_vocab_v2.py`, `../data_prep/step3_v2.py` — same
  deterministic vocab + parquet code (read raw `train_chunk_*.tsv` /
  `val_chunk_*.tsv`).
- `../aml_dataprep/encode_embeddings_multigpu.py` — same 8-GPU encoder.
- `../proposed_2-mmoe_ple/...` train code + L800 gin config.
Only the **data-access wiring** differs: direct cosmos mount vs CI relay.

## Decision gate (summary)
```
probe_mount.yml
   |
   +-- PASS --> use THIS pipeline (direct-mount). Delete-relay win.
   |
   +-- FAIL (NoIdentityOnCompute)
          |
          +-- retry other datastore / workspace
          |
          +-- still FAIL --> use ../aml_dataprep (split, proven)
```

## Notes / TODO
- The split pipeline already ran (`gray_giraffe_m6y3xjrpmh`); keep it as the
  proven fallback. This direct-mount pipeline is unproven until the probe passes.
- `proposed_2-mmoe_ple/requirements.txt` is referenced by the train step but may
  not exist yet — the `|| true` guard tolerates that; the conda `environment.yml`
  in `proposed_2-mmoe_ple/` lists the real deps (fbgemm-gpu, gin-config, absl-py,
  sentence-transformers) if we need to build an `hstu`-style image instead.
- val shards 0,1 were missing on cosmos earlier (38/40); confirm the full set
  before a real L800 run.
