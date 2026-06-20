# M3 — Canonical v001 Data-Scale Ablation (aux-light HSTU) — 2026-05-31

Status: **PROPOSAL / launch-ready, NOT launched.** GPUs busy (GPU0 = M2-newdata
training, GPU1 = reserved for official full-v001 eval). This spec prepares M3;
a human approves and launches later once a GPU frees and the M2-newdata official
baseline has landed.

## Thesis link

The LRM story has two claims to defend:
1. **Non-Ads data helps Ads** — already supported by M2 vs M1 (aux-light all-event
   beats Ads-only baseline AHR@10 = **0.3066437**).
2. **More training data helps Ads further** — *not yet directly demonstrated.*

M3 attacks claim (2) head-on with a clean **data-scale ablation**: hold the exact
M2-newdata recipe fixed and vary **only** the amount of canonical v001 training
data (25% → 50% → 100% of train shards). A monotone AHR@10 trend with data
volume is the single cleanest evidence for the scaling claim, and it reuses the
already-materialized v001 HSTU seqview (no new data engineering).

Precedent: this mirrors `scripts/p39_ads_first_scaling.py`, which scaled train by
symlinking a prefix of train parquet shards while sharing a frozen eval. M3
upgrades that pattern onto the canonical v001 seqview and freezes the recipe to
the M2-newdata baseline so the only moving part is data volume.

## Baseline reference points

- Ads-only frozen-7 baseline AHR@10 (target to beat): **0.3066437**.
- M2-newdata (full v001, current run, GPU0): internal best `ads_hr_10` ≈ 0.4535
  @ b521000; proxy warm-Ads AHR ≈ 0.269 (official full-v001 number pending GPU1).
- Source train view: `lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview`
  — 144 train parquet shards, 715,778 train users; 9 eval shards, 255,755 eval users.

## Options considered

### Option A — Data-scale ablation (RECOMMENDED)
- **Hypothesis:** warm-Ads AHR@10 increases monotonically with training-data
  fraction; the full-data point is the strongest, directly substantiating "more
  data helps Ads."
- **Config delta vs M2-newdata:** *none in the gin recipe.* The only lever is
  `--data_path`, pointed at a 25% / 50% / 100% symlink view of the train shards.
  Eval universe identical across points (shared `eval/` + `metadata.json`).
- **Grid:** scale ∈ {0.25, 0.50, 1.00} → {36, 72, 144} train shards. (1.00 ≈
  re-confirmation of M2-newdata under the frozen-eval harness; can be skipped if
  the official M2-newdata full-v001 number is accepted as the 100% point.)
- **Expected signal:** AHR@10(0.25) < AHR@10(0.50) < AHR@10(1.00); ideally the
  0.50 point already clears 0.3066437 and 1.00 clears it by a wider margin.
- **Deciding slice:** warm-Ads **AHR@10** on the frozen v001 eval vs **0.3066437**
  (and the slope across the three points).

### Option B — Aux-loss weight sweep on the aux-light recipe
- **Hypothesis:** the current Ads-heavy weights `{0:16, 1:0.05, 2:0.10, 3:1.0,
  4:0.05}` may under- or over-weight non-Ads supervision; a small grid (e.g.
  domain-1/2 weights ×2 and ÷2) could lift Ads transfer.
- **Config delta:** `make_model.supervision_domain_weights` only.
- **Why not first:** changes *recipe*, not data — answers claim (1) refinement,
  not claim (2). Confounds the scaling story and is better run after M3-A pins
  the data trend.

### Option C — Negative-sampling recipe refinement
- **Hypothesis:** swap `InBatch` SSL for `MixedHardGlobalNegativesSampler`
  (P29B: hard_fraction 0.25, pool 1024, rank window 32–512) to sharpen Ads
  ranking under the larger corpus.
- **Config delta:** `make_model.sampling_strategy`, `num_negatives`, sampler
  hyperparams.
- **Why not first:** another recipe change; P28/P29 hard-negative results were
  mixed and it muddies the clean data-scaling read. Hold as a follow-up.

**Recommendation: Option A.** It is the only option that *directly* tests the
outstanding thesis claim, requires zero recipe changes (so it is the least risky
and most defensible), and reuses existing materialized data.

## Mechanism (read-only on source data)

`scripts/prepare_m3_scale_views.py` builds symlink **views**:
- `train/` → first N of 144 source train parquet shards (deterministic prefix:
  N = round(144 × scale)), plus their `.done` markers.
- `eval/`, `metadata.json`, `DATASET_DISCLOSURE.json` → symlinked from the source
  so every scale evaluates on the **identical frozen eval universe**.
- Writes `DATASET_SCALE.json` provenance into each view.

No source bytes are modified or copied; views live under
`/home/yourslewis/lrm_benchmarkv4/processed/m3_data_scale_views/`.

The loader (`semantic_next_event_prediction.TrainIterableDataset`) consumes
`{data_path}/train/*.parquet` via `fs.glob`, so a prefix-subset view yields a
correctly reduced training set with metadata-driven item-id ranges intact.

## Files

- Gin: `proposed_2-mmoe_ple/config/generated_m3_data_scale/m3_aux_light_hstu_newdata_datascale.gin`
  (body byte-identical to `generated_m2_newdata/m2_aux_light_hstu_newdata.gin`;
  only header comments differ).
- View prep: `scripts/prepare_m3_scale_views.py`
- Launcher: `scripts/launch_m3_data_scale.py` (GPU-busy guard, one scale per call).

## Run plan (for human, after approval + GPU free)

1. Build views (cheap, no GPU): `python3 scripts/prepare_m3_scale_views.py --scales 0.25,0.5,1.0`
2. Launch one scale at a time on an idle GPU (start with 0.25, then 0.50).
3. Same training budget as M2-newdata (num_epochs/eval cadence unchanged); rely
   on `best_checkpoint_ads_hr_10.pt` + frozen v001 eval harness for the decision.
4. Success = monotone AHR@10 trend and the larger-data points clearing 0.3066437.

## Guardrails

- Do **not** start training as part of prep. Do not touch GPU0/GPU1 or M2's run dir.
- Launcher fails closed if the target GPU shows >2 GB used.
