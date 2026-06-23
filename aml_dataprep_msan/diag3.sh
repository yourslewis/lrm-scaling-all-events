#!/usr/bin/env bash
# Pull the FAILED parquet step's error from the source pipeline.
set -uo pipefail
SRC_SUB=72a0fe10-0a76-4898-9b7b-640e6e236fdc
RG=wb-aml; WS=pconv-aml-offline

echo "=== find parquet child run id ==="
PARQUET=$(az ml job list --parent-job-name crimson_roti_sxsn7rtmp6 -g $RG -w $WS --subscription $SRC_SUB \
  --query "[?display_name=='parquet'].name" -o tsv 2>/dev/null | head -1)
echo "PARQUET_RUN=$PARQUET"
[ -z "$PARQUET" ] && { echo "NO_PARQUET_ID"; exit 1; }

echo "=== download parquet logs ==="
rm -rf /tmp/pq
az ml job download -n "$PARQUET" -g $RG -w $WS --subscription $SRC_SUB --download-path /tmp/pq > /tmp/pqdl.log 2>&1
echo "DL_RC=$?"

echo "=== tail user std_log ==="
F=$(find /tmp/pq -path '*user_logs/std_log*.txt' 2>/dev/null | head -1)
echo "LOGFILE=$F"
[ -n "$F" ] && tail -n 30 "$F"

echo "=== also grep error markers across all logs ==="
grep -rhiE "error|traceback|exception|killed|sigterm|oom|no space|memoryerror" /tmp/pq 2>/dev/null | grep -viE "errorlevel|0 error" | tail -20
echo PARQUET_DIAG_DONE
