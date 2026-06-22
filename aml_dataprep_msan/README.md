# LRM L800 dataprep on msan (Wenhao's variant) — vocab on computed15v2 CPU

Goal (Lewis, 2026-06-22): same as the `lrm-l800-end2end` job, but:
- **workspace** = `msan-retrieval-ranking-aml` (rg `msan-aml`, sub `f920ee3b-6bdc-48c6-a487-9e0397b69322`, tenant `72f988bf…`, region **centralus**)
- **download data from the networkprotection VC/datastore** (not the pconv relay)
- **vocab needs no GPU — run it on CPU from `computed15v2`**

## What's confirmed (read live in Studio as WL, 2026-06-22)
- `computed15v2` = **Standard_D15_v2 = 20 cores / 140 GB RAM / 1000 GB disk**, CPU-mem-optimized,
  $1.85/hr/node, min0/max1, idle-scaledown 120s, **no managed identity attached**, SSH disabled.
  (Sibling `computed14v2` = D14_v2 = 16c/112GB, also exists.)
- msan **has the networkprotection cosmos store registered**:
  `bingads_algo_prod_networkprotection_c08` → ADLS **Gen1**, storage `bingads-algo-prod-networkprotection-c08`,
  registered by Menghao Yang 2025-05-29. Same underlying store the repo direct-mounts.
- msan datastores (5): that networkprotection Gen1 store + `workspaceblobstore (Default)` +
  workspaceartifactstore / workspacefilestore / workspaceworkingdirectory (all on `msanretrievalr2285885506`).

## Source template
`aml_dataprep/pipeline.yml` (display_name `lrm-l800-end2end`, exp `lrm-l800-pipeline`).
DAG: `relay → vocab → parquet → encode(8×A100) → train(L800, 8×A100)`.
In the source, **vocab runs on the nd40 GPU VC** (`Singularity.ND96rs_v4`) — that's the step that
kept getting **preempted** (retry_001…010, SIGTERM from chunk [1/148]). `step1_collect_vocab_v2.py`
is **pure-stdlib CPU** work → belongs on a CPU cluster. That's exactly this move.

## This variant (files here)
- `jobs/probe_np_read.yml` — **RUN FIRST.** Cheap computed15v2 job: fsspec-list train/val shard
  counts + read first bytes of one shard. Proves whether the cluster can read the Gen1
  networkprotection datastore. Uses `list_np.py`.
- `list_np.py` — the probe body (lists + byte-reads via `AzureMachineLearningFileSystem`).
- `jobs/vocab_msan.yml` — download (fsspec stream via `aml_dataprep/relay_shards.py`) →
  `step1_collect_vocab_v2.py` → publish `azureml://…/workspaceblobstore/paths/derived/lrm_v4_msan/vocab`.
  One CPU node does both (shared 140 GB RAM / 1000 GB disk).

Submit (Azure Cloud Shell as yourself, or any az with msan access):
```bash
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322; RG=msan-aml; WS=msan-retrieval-ranking-aml
az ml job create -f aml_dataprep_msan/jobs/probe_np_read.yml -g $RG -w $WS --subscription $SUB
# PASS → then:
az ml job create -f aml_dataprep_msan/jobs/vocab_msan.yml   -g $RG -w $WS --subscription $SUB
```

## THE decisive open question (gates everything)
**Can a `computed15v2` job actually READ `bingads_algo_prod_networkprotection_c08`?**
- It's **ADLS Gen1**. If the datastore is **credential-based** (SP/key stored in the datastore) →
  any submitter works, probe PASSes, we're done.
- If it's **identity-based** → it needs the *compute's* identity, and computed15v2 has **no managed
  identity attached** → probe FAILs with auth/`NoIdentityOnCompute`. Fix = attach a user-assigned MI
  with read on that store, OR use a credential-based datastore. `probe_np_read.yml` tells us which.
- (Prior proven direct-mount was as SP `msanrrsvcgpt` in CPT-LLM-Pipelines, a *different* identity/ws —
  doesn't guarantee msan computed15v2. Hence the probe.)

## Notes / fidelity
- 140 GB RAM vs the vocab worst-case (~150–230 GB if newdata-v4 hits 150–230M unique texts).
  140 GB is better than 112/128 but still **not guaranteed**; if vocab OOMs on D15_v2, either size the
  cluster up or add checkpoint/resume to `step1_collect_vocab_v2.py`. The download+vocab split here at
  least means preemption is gone (dedicated cluster), so a single clean pass is far more likely.
- Gen1 `ro_mount` on AmlCompute is historically flaky → this variant **streams** (fsspec), same proven
  mechanism as `relay_shards.py`, instead of mounting.
- GPU half (encode + train) unchanged: still targets `ads-shared-nd40` by ARN (same tenant `72f988bf`,
  confirmed), reading vocab/seqview/embeddings from blob. Only the CPU dataprep moved to msan.
