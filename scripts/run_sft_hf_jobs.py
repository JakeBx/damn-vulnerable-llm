#!/usr/bin/env python
"""Submit a NeMo-RL SFT job to Hugging Face Jobs.

Usage:
    python scripts/run_sft_hf_jobs.py
    python scripts/run_sft_hf_jobs.py --flavor a100-large --config configs/nemo_sft_1b.yaml
    python scripts/run_sft_hf_jobs.py --dry-run
"""

import argparse

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit DV-LLM SFT job to HF Jobs")
    parser.add_argument("--flavor", default="a100-large", help="HF Jobs hardware flavor")
    parser.add_argument("--config", default="configs/nemo_sft_1b.yaml", help="NeMo-RL config path")
    parser.add_argument("--dry-run", action="store_true", help="Print command without submitting")
    args = parser.parse_args()

    from dv_llm.train.hf_jobs import submit_sft_job

    submit_sft_job(
        nemo_config=args.config,
        flavor=args.flavor,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
