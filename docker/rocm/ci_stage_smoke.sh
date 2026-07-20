#!/usr/bin/env bash
# ===========================================================================
# CI STAGING SHEET — run ON the CI (lrm-smoke-submitter) AFTER Lewis hands it
# over (A: CI is mine, or B: after his imgbldrun lands). NOT run from the Mac.
#
# This does the TWO things Rex owns for smoke-readiness, both READ/BUILD-only:
#   1. Resolve the exact MI200 Singularity SKU (job-independent VC read).
#   2. Build+push the ROCm AML environment -> named ref for smoke_rocm.yml.
#
# It SUBMITS NOTHING. No job create. No compute Start. Paid gate intact.
# Paste this whole block into the CI terminal in one shot (avoids live syntax
# fumbling / collisions). Identity = CI MSI (ClickPrediction-Trainer) is fine
# for reads + env build; the SP is NOT needed for the smoke.
# ===========================================================================
set +e
ADS_SUB=72a1fe05-772c-4836-869f-761a5805fcd4
CI_SUB=72a0fe10-0a76-4898-9b7b-640e6e236fdc
RG=wb-aml ; WS=pconv-aml-offline ; AV=2021-03-01-preview
OUT=/tmp/rex_smoke_ready.txt ; : > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

log "=== 0. identity (expect ClickPrediction-Trainer / systemAssignedIdentity) ==="
az account show --query "{name:name,user:user.name,type:user.type}" -o json 2>&1 | tee -a "$OUT"

log ""
log "=== 1. MI200 SKU — job-independent VC allowed-instance-type read ==="
# The VC quota object lists the families; the exact Singularity.* submit string
# is exposed in the VC's instanceTypes / policies. Dump every SKU-looking token.
az rest --method get --url \
  "https://management.azure.com/subscriptions/${ADS_SUB}/resourceGroups/ads-singularity-rg-01/providers/Microsoft.MachineLearningServices/virtualClusters/ads?api-version=${AV}" \
  2>/tmp/rex_e1 > /tmp/rex_ads_vc.json
log "ads_vc_size=$(wc -c </tmp/rex_ads_vc.json)"
python3 - <<'PY' 2>&1 | tee -a "$OUT"
import json,re
try:
    d=json.load(open("/tmp/rex_ads_vc.json"))
except Exception as e:
    print("PARSE_FAIL",e); raise SystemExit
hits=set()
def walk(o,pre=""):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,pre+"."+k)
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v,pre+f"[{i}]")
    else:
        s=str(o)
        if re.search(r"Singularity\.|NDMI|MI200|MI300|instanceType",s,re.I) or re.search(r"instancetype|vmsize|sku",pre,re.I):
            hits.add(f"{pre} = {s}")
walk(d.get("properties",{}))
for h in sorted(hits): print(h)
print("--- if no 'Singularity.NDMI200*' string above, the VC object only carries the")
print("    family; fall back to reading instance_type off any MI200 job (even failed). ---")
PY

log ""
log "=== 1b. FALLBACK: scan recent jobs for a real MI200 instance_type ==="
# Only if 1 didn't yield the full string. Reads normalized instance_type from any
# MI200 job in the CI workspace (a failed validate still carries it).
for J in $(az ml job list -g "$RG" -w "$WS" --subscription "$CI_SUB" --only-show-errors \
            --query "[?contains(to_string(resources.instance_type),'MI')].name" -o tsv 2>/dev/null | head -5); do
  S=$(az ml job show -n "$J" -g "$RG" -w "$WS" --subscription "$CI_SUB" --only-show-errors \
        --query "resources.instance_type" -o tsv 2>/dev/null)
  log "  job $J -> instance_type=$S"
done

log ""
log "=== 2. BUILD ROCm AML environment (build+push only; submits nothing) ==="
# Build context = repo root (Dockerfile COPYs docker/rocm/requirements-rocm.txt).
cd "$(git -C "$HOME" rev-parse --show-toplevel 2>/dev/null || echo $HOME/lrm-scaling-all-events)" 2>/dev/null
pwd | tee -a "$OUT"
az ml environment create -f docker/rocm/aml_env_rocm.yml \
  -g "$RG" -w "$WS" --subscription "$CI_SUB" --only-show-errors \
  --query "{name:name,version:version,image:image}" -o json 2>&1 | tee -a "$OUT"

log ""
log "=== RESULT — hand these two to Don for the smoke_rocm.yml fold-in ==="
log "  instance_type : <the Singularity.NDMI200* string from step 1/1b>"
log "  environment   : azureml:lrm-rocm-smoke:<version from step 2>"
log "  compute (have): /subscriptions/${ADS_SUB}/resourceGroups/ads-singularity-rg-01/providers/Microsoft.MachineLearningServices/virtualClusters/ads"
log "(Full transcript saved to $OUT)"
