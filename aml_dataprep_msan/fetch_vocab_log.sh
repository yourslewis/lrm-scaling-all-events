#!/usr/bin/env bash
# Overall workflow: find the vocab child run for an msan pipeline and pull the
# smallest useful log tail.
# Performance tricks: target one child and one log directory to avoid downloading
# unrelated AML artifacts.

# Download the vocab step logs and commit them into the repo so they can be read
# off-terminal. Run in Cloud Shell.
set -uo pipefail
SUB=f920ee3b-6bdc-48c6-a487-9e0397b69322
RG=msan-aml; WS=msan-retrieval-ranking-aml
PARENT=gifted_square_7vj803x572

# cd into this script's own repo so git add/push targets the right tree
SELFDIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$SELFDIR" || { echo "NO_REPO_DIR"; exit 1; }
echo "REPO=$SELFDIR"

VID=$(az ml job list --parent-job-name $PARENT -g $RG -w $WS --subscription $SUB \
  --query "[?display_name=='vocab'].name | [0]" -o tsv 2>/dev/null)
echo "VOCAB_RUN=$VID"

rm -rf /tmp/vl
az ml job download -n "$VID" -g $RG -w $WS --subscription $SUB --download-path /tmp/vl >/dev/null 2>&1
echo "DL_RC=$?"

mkdir -p _diag
SL=$(find /tmp/vl -name 'std_log*.txt' | head -1)
cp "$SL" _diag/vocab_std_log.txt 2>/dev/null
# concat every log under the download for completeness
{ find /tmp/vl -name '*.txt' | while read -r f; do echo "==== $f ===="; cat "$f"; echo; done; } > _diag/vocab_all_logs.txt 2>/dev/null
echo "std_log bytes=$(wc -c < _diag/vocab_std_log.txt 2>/dev/null)  alllogs bytes=$(wc -c < _diag/vocab_all_logs.txt 2>/dev/null)"

git -c user.email=ci@local -c user.name=ci add -f _diag/vocab_std_log.txt _diag/vocab_all_logs.txt
git -c user.email=ci@local -c user.name=ci commit -q -m "diag: vocab failure logs (auto)"
git push -q origin feat/lrm-l800-msan && echo PUSH_OK || echo PUSH_FAIL
echo FETCH_DONE
