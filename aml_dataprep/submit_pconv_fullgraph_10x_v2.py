#!/usr/bin/env python3
"""Submit the generated pconv/fullgraph 10x_v2 AML pipeline.

This script is intentionally inert unless run directly without --dry-run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from azure.ai.ml import MLClient, load_job
from azure.identity import AzureCliCredential


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", default="aml_dataprep/pipeline_pconv_fullgraph_10x_v2.yml")
    p.add_argument("--subscription-id", default="72a0fe10-0a76-4898-9b7b-640e6e236fdc")
    p.add_argument("--resource-group", default="wb-aml")
    p.add_argument("--workspace", default="pconv-aml-offline")
    p.add_argument("--dry-run", action="store_true", help="Load and print the job without submitting.")
    p.add_argument("--name", help="Optional deterministic AML job name for monitor/retry tracking.")
    p.add_argument(
        "--skip-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip AML service-side pipeline validation during create/update; only use when the YAML contains fully resolved component references.",
    )
    args = p.parse_args()

    job = load_job(args.pipeline)
    if args.name:
        job.name = args.name
    elif not args.dry_run and not getattr(job, "name", None):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job.name = f"pconv_fullgraph_10x_v2_{ts}"
    if args.dry_run:
        print(job)
        return
    client = MLClient(AzureCliCredential(), args.subscription_id, args.resource_group, args.workspace)
    created = client.jobs.create_or_update(job, skip_validation=args.skip_validation)
    print(created.name)


if __name__ == "__main__":
    main()
