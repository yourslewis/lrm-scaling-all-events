#!/usr/bin/env python3
"""Submit the generated pconv/fullgraph 10x_v2 AML pipeline.

This script is intentionally inert unless run directly without --dry-run.
"""

from __future__ import annotations

import argparse

from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", default="aml_dataprep/pipeline_pconv_fullgraph_10x_v2.yml")
    p.add_argument("--subscription-id", default="72a0fe10-0a76-4898-9b7b-640e6e236fdc")
    p.add_argument("--resource-group", default="wb-aml")
    p.add_argument("--workspace", default="pconv-aml-offline")
    p.add_argument("--dry-run", action="store_true", help="Load and print the job without submitting.")
    args = p.parse_args()

    job = load_job(args.pipeline)
    if args.dry_run:
        print(job)
        return
    client = MLClient(DefaultAzureCredential(), args.subscription_id, args.resource_group, args.workspace)
    created = client.jobs.create_or_update(job)
    print(created.name)


if __name__ == "__main__":
    main()
