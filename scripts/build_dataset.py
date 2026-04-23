#!/usr/bin/env python
"""Build the V1/LLM01 SFT dataset and push to HF Hub.

Usage:
    python scripts/build_dataset.py --dry-run
    python scripts/build_dataset.py --n 10000 --out myorg/dv-llm-sft-v1
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DV-LLM SFT dataset (V1/LLM01)")
    parser.add_argument("--n", type=int, default=10_000, help="Target number of training examples")
    parser.add_argument("--out", type=str, default=None, help="HF Hub repo ID (org/name)")
    parser.add_argument("--garak-dir", type=Path, default=Path("data/raw"), help="Directory for garak hitlog output")
    parser.add_argument("--no-push", action="store_true", help="Skip HF Hub push")
    parser.add_argument("--dry-run", action="store_true", help="Skip all API calls; validate pipeline schema only")
    args = parser.parse_args()

    # Validate secrets early
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(1)

    push = not args.no_push and not args.dry_run
    if push and not (args.out or os.environ.get("HF_REPO")):
        print("ERROR: Provide --out <org/repo> or set HF_REPO env var.", file=sys.stderr)
        sys.exit(1)

    from dv_llm.data.pipeline import build_dataset

    ds = build_dataset(
        target_n=args.n,
        garak_output_dir=args.garak_dir,
        dry_run=args.dry_run,
        hf_repo=args.out or os.environ.get("HF_REPO"),
        push=push,
    )

    train_n = len(ds["train"])
    eval_n = len(ds["eval"])
    print(f"\nDataset built: {train_n} train / {eval_n} eval examples")

    # Write local JSONL for NeMo-RL (expected at data/processed/)
    if not args.dry_run:
        import json

        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "eval"):
            path = out_dir / f"{split}.jsonl"
            with path.open("w") as f:
                for row in ds[split]:
                    f.write(json.dumps({"messages": row["messages"]}) + "\n")
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
