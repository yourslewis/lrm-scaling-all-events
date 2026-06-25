#!/usr/bin/env python3
"""Probe: can THIS compute read the networkprotection datastore in msan?
Lists train/val shard counts + reads the first bytes of one shard (proves
byte-level read through the datastore credential, not just metadata). Prints a
clear PASS/FAIL line. Cheap: no full download.
"""

# Workflow notes:
# Lightweight msan datastore probe: list NetworkProtection paths from AML compute
# using the same identity path that the real prep jobs need.
# Performance tricks:
# - Limit listing to small directory probes so diagnostics return quickly.
# - Print raw failures so identity/mount problems are visible in AML logs.

import sys
from azureml.fsspec import AzureMachineLearningFileSystem as FS

SUB = "f920ee3b-6bdc-48c6-a487-9e0397b69322"
RG = "msan-aml"
WS = "msan-retrieval-ranking-aml"
DS = "bingads_algo_prod_networkprotection_c08"
ROOT = "local/User/wenhlu/LRM_benchmark_v4"

base = (f"azureml://subscriptions/{SUB}/resourcegroups/{RG}/workspaces/{WS}/"
        f"datastores/{DS}/paths/{ROOT}")
fs = FS(base)

ok = True
for sub in ("train", "val"):
    try:
        files = [f for f in fs.ls(f"{ROOT}/{sub}") if f.endswith(".tsv")]
        print(f"NP_READ_OK {sub} shards={len(files)}", flush=True)
        if files:
            rel = files[0].split("/paths/")[-1] if "/paths/" in files[0] else files[0]
            with fs.open(rel, "rb") as fh:
                head = fh.read(256)
            print(f"  first={files[0].split('/')[-1]} head_bytes={len(head)}", flush=True)
    except Exception as e:
        ok = False
        print(f"NP_READ_FAIL {sub}: {type(e).__name__}: {e}", flush=True)

if ok:
    print("PROBE PASS: this compute can read networkprotection", flush=True)
    sys.exit(0)
print("PROBE FAIL: datastore not readable from this compute (see errors above)", flush=True)
sys.exit(1)
