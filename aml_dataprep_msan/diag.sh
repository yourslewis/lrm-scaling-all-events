#!/usr/bin/env bash
# Diagnostics for the msan LRM-L800 work. Run in Cloud Shell as WL.
set -uo pipefail
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322
RG=msan-aml
WS=msan-retrieval-ranking-aml

echo "=================== 1) computed15v2 identity ==================="
az ml compute show -n computed15v2 -g $RG -w $WS --subscription $SUB \
  --query "{name:name,size:size,identity:identity,ssh:ssh_settings}" -o json 2>&1 | head -40

echo
echo "=================== 2) networkprotection datastore cred type ==================="
az ml datastore show -n bingads_algo_prod_networkprotection_c08 -g $RG -w $WS --subscription $SUB \
  -o json 2>&1 | python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
    print('type=', d.get('type'))
    cr=d.get('credentials',{})
    print('credentials_type=', cr.get('type') or cr.get('credentials_type') or list(cr.keys()))
    print('keys=', list(d.keys()))
except Exception as e:
    print('PARSE_ERR', e)
    print(sys.stdin.read()[:400] if False else '')
" 2>&1 | head -20

echo
echo "=================== 3) user-assigned identities available in RG ==================="
az identity list -g $RG --subscription $SUB --query "[].{name:name,clientId:clientId,principalId:principalId,id:id}" -o json 2>&1 | head -40

echo DIAG_DONE
