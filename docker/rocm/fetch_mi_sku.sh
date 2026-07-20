#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_mi_sku.sh — READ the exact Singularity MI instance_type off a live job.
#
# The Singularity SKU strings transpose letters vs the Azure VM SKU
# (e.g. A100-80 8-GPU is `Singularity.ND96amrs_A100_v4`, VM is `...ND96amsr...`),
# so the MI300X/MI200 SKU MUST be read from a real job — never hand-built.
#
# This is a READ-ONLY `az ml job show`. It needs only read access on the target
# workspace, so run it with the wenhlu-delegated CI MSI (`az login --identity`)
# — the service account is NOT required here.
#
# Usage:
#   bash docker/rocm/fetch_mi_sku.sh -n ivory_wall \
#        -g rg-cs-newsandfeeds-singularity -w <FeedsWS> [-s <sub-id>]
#
# Prints the instance_type (and the compute id, which you also need for the yml).
# ---------------------------------------------------------------------------
set -euo pipefail

JOB="" ; RG="" ; WS="" ; SUB=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) JOB="$2"; shift 2;;
    -g) RG="$2"; shift 2;;
    -w) WS="$2"; shift 2;;
    -s) SUB="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "[err] unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$JOB" ] && [ -n "$RG" ] && [ -n "$WS" ] || { echo "[err] -n JOB -g RG -w WS required" >&2; exit 2; }
command -v az >/dev/null 2>&1 || { echo "[err] az CLI not found" >&2; exit 3; }
[ -n "$SUB" ] && az account set -s "$SUB" >/dev/null 2>&1 || true

echo "=== job show: $JOB (rg=$RG ws=$WS) ==="
az ml job show -n "$JOB" -g "$RG" -w "$WS" --only-show-errors \
  --query '{instance_type:resources.instance_type, compute:compute, image:environment.image}' -o json
echo
echo "Copy instance_type + compute into aml_dataprep/jobs/smoke_rocm.yml (replace the PLACEHOLDERs)."
