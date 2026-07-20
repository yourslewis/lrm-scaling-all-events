#!/usr/bin/env bash
# Overall workflow: submit/inspect focused vocab diagnostics for the msan data
# path before running the full L800 prep.
# Performance tricks: isolate vocab checks from parquet/encode/train so identity
# and mount issues fail fast and cheaply.

# Authoritative vocab-failure diagnosis. Run in Cloud Shell.
set -uo pipefail
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322
RG=msan-aml; WS=msan-retrieval-ranking-aml
PARENT=gifted_square_7vj803x572

VID=$(az ml job list --parent-job-name $PARENT -g $RG -w $WS --subscription $SUB --query "[?display_name=='vocab'].name | [0]" -o tsv 2>/dev/null)
echo "VOCAB_RUN=$VID"

echo "===== AML structured error ====="
az ml job show -n "$VID" -g $RG -w $WS --subscription $SUB -o json 2>/dev/null \
  | python3 -c '
import sys,json
d=json.load(sys.stdin)
print("status:", d.get("status"))
# error often under properties or status detail
props=d.get("properties",{}) or {}
for k in props:
    if "error" in k.lower() or "Error" in k:
        print("prop",k,"=",str(props[k])[:600])
err=d.get("error") or (d.get("status_detail") if isinstance(d.get("status_detail"),str) else None)
print("error_field:", str(err)[:600])
'

echo "===== fresh download + locate user std_log ====="
rm -rf /tmp/vdl2
az ml job download -n "$VID" -g $RG -w $WS --subscription $SUB --download-path /tmp/vdl2 >/dev/null 2>&1
SL=$(find /tmp/vdl2 -name 'std_log*.txt' | head -1)
echo "STD_LOG=$SL  bytes=$(wc -c < "$SL" 2>/dev/null)"
echo "----- std_log content -----"
cat "$SL" 2>/dev/null
echo "----- end std_log -----"

echo "===== other logs (system/azureml) grep for the kill/error ====="
grep -rhinE 'killed|oom|out of memory|exit code|non-zero|returned a non-zero|terminated|FileNotFound|No such file|Traceback|cannot allocate' /tmp/vdl2 2>/dev/null \
  | grep -viE 'deprecat|azureml.core import' | sort -u | head -20
echo "VDIAG_DONE"
