#!/usr/bin/env bash
# Inspect msan networkprotection datastore credential + resolve which SP it is.
set -uo pipefail
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322
RG=msan-aml; WS=msan-retrieval-ranking-aml

echo "=== datastore credential ==="
az ml datastore show -n bingads_algo_prod_networkprotection_c08 -g $RG -w $WS --subscription $SUB -o json > /tmp/ds.json 2>/tmp/ds.err
python3 - <<'PY'
import json
d=json.load(open("/tmp/ds.json"))
c=d.get("credentials",{}) or {}
print("DS_TYPE=", d.get("type"))
print("CRED_KEYS=", list(c.keys()))
print("CLIENT_ID=", c.get("client_id"))
print("TENANT=", c.get("tenant_id"))
open("/tmp/cid.txt","w").write(str(c.get("client_id") or ""))
PY

CID=$(cat /tmp/cid.txt)
echo "=== resolve SP $CID ==="
if [ -n "$CID" ] && [ "$CID" != "None" ]; then
  az ad sp show --id "$CID" --query "{displayName:displayName,appId:appId}" -o json 2>/tmp/sp.err || { echo SP_RESOLVE_FAIL; tail -3 /tmp/sp.err; }
else
  echo "NO_CLIENT_ID -> datastore is IDENTITY-BASED (no stored SP; uses compute identity)"
fi

echo "=== computed15v2 MI principal (for ACL grant option) ==="
az ml compute show -n computed15v2 -g $RG -w $WS --subscription $SUB --query "identity" -o json 2>/dev/null
echo DIAG4_DONE
