#!/usr/bin/env bash
# Overall workflow: compare source pipeline status, inspect the failing parent,
# and check whether identity updates are supported before trying destructive fixes.
# Performance tricks: query only status/help surfaces first to avoid downloading
# large AML artifacts when the problem is control-plane configuration.

# Q2 diagnosis + fix feasibility. Run in Cloud Shell as WL.
set -uo pipefail

echo "=================== A) SOURCE pipeline crimson_roti children ==================="
SRC_SUB=72a0fe10-0a76-4898-9b7b-640e6e236fdc
az ml job list --parent-job-name crimson_roti_sxsn7rtmp6 \
  -g wb-aml -w pconv-aml-offline --subscription $SRC_SUB \
  --query "[].{step:display_name,status:status}" -o tsv 2>&1 | head -20
echo "A_RC=$?"

echo
echo "=================== B) parent crimson_roti status + reason ==================="
az ml job show -n crimson_roti_sxsn7rtmp6 -g wb-aml -w pconv-aml-offline --subscription $SRC_SUB \
  --query "{status:status}" -o tsv 2>&1 | head
echo

echo "=================== C) can we attach a system-assigned identity via update? ==================="
az ml compute update --help 2>&1 | grep -iE "identity" | head
echo "C_DONE"

echo CRIMSON_DIAG_DONE
