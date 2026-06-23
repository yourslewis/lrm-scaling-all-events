#!/usr/bin/env bash
# Submit an msan LRM-L800 job and capture name+studio URL cleanly.
# Usage: bash aml_dataprep_msan/submit.sh probe   (or: vocab)
set -uo pipefail
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322
RG=msan-aml
WS=msan-retrieval-ranking-aml
case "${1:-probe}" in
  probe) YML=aml_dataprep_msan/jobs/probe_np_read.yml ;;
  mount) YML=aml_dataprep_msan/jobs/probe_mount.yml ;;
  idpass) YML=aml_dataprep_msan/jobs/probe_idpass.yml ;;
  vocab) YML=aml_dataprep_msan/jobs/vocab_msan.yml ;;
  pipeline) YML=aml_dataprep_msan/pipeline.yml ;;
  *) echo "usage: submit.sh probe|mount|idpass|vocab|pipeline"; exit 2 ;;
esac
echo "=== submitting $YML ==="
az ml job create -f "$YML" -g "$RG" -w "$WS" --subscription "$SUB" -o json > /tmp/job.json 2>/tmp/job.err
RC=$?
echo "az_rc=$RC"
if [ $RC -eq 0 ]; then
  python3 - <<'PY'
import json
d=json.load(open("/tmp/job.json"))
print("JOB_NAME=", d.get("name"))
print("STATUS=", d.get("status"))
print("STUDIO=", (d.get("services",{}).get("Studio",{}) or {}).get("endpoint"))
PY
else
  echo "=== ERROR (tail) ==="
  tail -n 25 /tmp/job.err
fi
echo "SUBMIT_SH_DONE rc=$RC"
