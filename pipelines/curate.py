"""CLI entry point for the local data curation pipeline.

Usage:
    python -m pipelines.curate [options]
    make curate                          # via Makefile

Options:
    --regen=<names>    Comma-separated source names to force-regenerate.
                       Use 'all' to regenerate every non-living source.
                       Example: --regen=harmbench,jailbreakbench
    --push <repo>      HF Hub repo ID to push the final dataset (default: $HF_REPO).
    --no-push          Skip HF Hub upload even if HF_REPO is set.
    --dry-run          Skip all API calls; validates pipeline schema only.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DV-LLM data curation pipeline")
    parser.add_argument(
        "--regen",
        default="",
        help="Comma-separated source names to force-regenerate, or 'all'.",
    )
    parser.add_argument(
        "--push",
        default=None,
        metavar="REPO",
        help="HF Hub repo ID to push the final dataset.",
    )
    parser.add_argument("--no-push", action="store_true", help="Skip HF Hub upload.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all API calls; validate pipeline schema only.",
    )
    args = parser.parse_args()

    force: set[str] = set()
    if args.regen:
        force = {s.strip() for s in args.regen.split(",") if s.strip()}

    push_to: str | None = None
    if not args.no_push and not args.dry_run:
        push_to = args.push or os.environ.get("HF_REPO") or None
        if push_to is None:
            print(
                "INFO: no --push target set and HF_REPO not in env — running without Hub upload. "
                "Pass --push <org/repo> or set HF_REPO to enable.",
                file=sys.stderr,
            )

    from dv_llm.curation.runner import run

    ds = run(force=force if force else None, push_to=push_to, dry_run=args.dry_run)

    train_n = len(ds["train"])
    eval_n = len(ds["eval"])
    print(f"\nDone: {train_n:,} train / {eval_n:,} eval examples")


if __name__ == "__main__":
    main()
