#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# submit_rocm.sh — submit a ROCm/MI AML job as the MSANRR service account.
#
# Runs ON the CI (lrm-smoke-submitter) or any host with `az ml` installed.
# NO SECRET IS STORED IN THIS FILE. The password comes from $AZ_SP_PASS, which
# the caller exports (e.g. from the clawguard vault key `msanrrsvcgpt.password`).
#
# Why the SP at all: a Singularity VC job that reads the credential-less cosmos
# ADLS datastore fails with NoIdentityOnCompute when submitted by an interactive
# user; service-account submission is the org-wide fix. (The 1-GPU smoke reads
# NOTHING from cosmos, so for the smoke the SP is only needed to write into the
# CPT-LLM-Pipelines workspace — pconv-aml-offline is NOT writable by the SP.)
#
# Identity gotcha this script handles: on the CI, a system-assigned MSI can
# shadow the subscription after login, so we `logout; account clear` FIRST and
# verify `user.type == user` before submitting.
#
# Usage:
#   export AZ_SP_USER='<service-account-UPN>'          # required (not committed)
#   export AZ_SP_PASS="$(node ~/.openclaw/extensions/clawguard/bin/clawguard-vault-get.mjs msanrrsvcgpt.password)"
#   bash docker/rocm/submit_rocm.sh \
#        -f aml_dataprep/jobs/smoke_rocm.yml \
#        -g Ads-LLM-Pipelines -w CPT-LLM-Pipelines [-s <sub-id>] [--dry-run]
#
# Flags:
#   -f FILE   job yaml (required)
#   -g RG     resource group  (default: Ads-LLM-Pipelines)
#   -w WS     workspace        (default: CPT-LLM-Pipelines)
#   -s SUB    subscription id  (optional; else uses login default)
#   -t TENANT AAD tenant       (default: common Microsoft tenant)
#   --dry-run  log in + verify identity + `--web`-less validate, do NOT create
# ---------------------------------------------------------------------------
set -euo pipefail

JOB_FILE=""
RG="Ads-LLM-Pipelines"
WS="CPT-LLM-Pipelines"
SUB=""
TENANT="${AZ_SP_TENANT:-72f988bf-86f1-41af-91ab-2d7cd011db47}"   # public MS tenant id
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    -f) JOB_FILE="$2"; shift 2;;
    -g) RG="$2"; shift 2;;
    -w) WS="$2"; shift 2;;
    -s) SUB="$2"; shift 2;;
    -t) TENANT="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "[err] unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$JOB_FILE" ] || { echo "[err] -f <job.yml> required" >&2; exit 2; }
[ -f "$JOB_FILE" ] || { echo "[err] job file not found: $JOB_FILE" >&2; exit 2; }
: "${AZ_SP_USER:?set AZ_SP_USER to the service-account UPN (kept out of the repo)}"
: "${AZ_SP_PASS:?set AZ_SP_PASS (e.g. from clawguard vault key msanrrsvcgpt.password)}"

command -v az >/dev/null 2>&1 || { echo "[err] az CLI not found on this host" >&2; exit 3; }

# Guard: refuse to submit a yaml still carrying PLACEHOLDER fields (smoke gate).
if grep -q 'PLACEHOLDER' "$JOB_FILE"; then
  echo "[err] $JOB_FILE still has PLACEHOLDER fields (compute/instance_type/image)." >&2
  echo "      Fill compute + instance_type (exact MI SKU off a live job) + image first." >&2
  exit 4
fi

echo "=== 1. clear any prior identity (MSI can shadow the sub) ==="
az logout >/dev/null 2>&1 || true
az account clear >/dev/null 2>&1 || true

echo "=== 2. log in as the service account (ROPC; MFA warning is deprecation noise) ==="
az login -u "$AZ_SP_USER" -p "$AZ_SP_PASS" --tenant "$TENANT" --only-show-errors >/dev/null
WHO="$(az account show --query '{u:user.name,t:user.type}' -o tsv)"
echo "    identity: $WHO"
# Enforce BOTH the account name AND user.type==user (tab-separated tsv: '<name>\t<type>').
# ROPC always yields user-type, so this is a belt-and-suspenders assertion.
case "$WHO" in
  *"$AZ_SP_USER"*[Uu]ser*) : ;;
  *) echo "[err] unexpected identity after login: $WHO (expected '$AZ_SP_USER' + user-type)" >&2; exit 5;;
esac

[ -n "$SUB" ] && { echo "=== 2b. set subscription $SUB ==="; az account set -s "$SUB"; }

echo "=== 3. submit $JOB_FILE -> rg=$RG ws=$WS ==="
if [ "$DRY" = "1" ]; then
  echo "[dry-run] skipping create. Would run:"
  echo "    az ml job create -f $JOB_FILE -g $RG -w $WS --only-show-errors"
  exit 0
fi

NAME="$(az ml job create -f "$JOB_FILE" -g "$RG" -w "$WS" --only-show-errors \
        --query name -o tsv)"
echo "[ok] submitted job: $NAME"
echo "[ok] creator should be $AZ_SP_USER — verify in Studio > Jobs > $NAME"
echo "    tail: az ml job stream -n $NAME -g $RG -w $WS"
