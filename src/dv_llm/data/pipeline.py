"""Orchestrate data collection, deduplication, formatting, and HF Hub upload."""

import os
from pathlib import Path

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord, generate_advbench_records
from dv_llm.data.dedup import deduplicate
from dv_llm.data.garak_collector import run_garak

DEFAULT_TARGET = 10_000
TRAIN_SPLIT = 0.9


def _load_garak_records(output_dir: Path | None, dry_run: bool) -> list[SFTRecord]:
    if dry_run:
        return []
    return run_garak(output_dir=output_dir)


def _to_hf_dict(records: list[SFTRecord]) -> dict[str, list[object]]:
    return {
        "messages": [r.messages for r in records],
        "source": [r.source for r in records],
        "owasp_id": [r.owasp_id for r in records],
        "vulnerability": [r.vulnerability for r in records],
    }


def build_dataset(
    target_n: int = DEFAULT_TARGET,
    garak_output_dir: Path | None = None,
    dry_run: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    push: bool = True,
) -> hf_datasets.DatasetDict:
    """Build the V1/LLM01 SFT dataset and optionally push to HF Hub.

    Args:
        target_n: Approximate target number of training examples.
        garak_output_dir: Directory to store garak hitlog. Uses a temp dir if None.
        dry_run: Skip garak and OpenRouter calls; return a small synthetic dataset.
        hf_repo: HF Hub repo ID, e.g. "myorg/dv-llm-sft-v1".
        hf_token: HF Hub token. Falls back to HF_TOKEN env var.
        push: Whether to push to HF Hub.
    """
    if dry_run:
        # Return a tiny synthetic dataset for schema validation
        dummy = SFTRecord(
            messages=[
                {"role": "user", "content": "Test attack prompt"},
                {"role": "assistant", "content": "Test compliant response"},
            ],
            source="synthetic",
            owasp_id="LLM01",
            vulnerability="V1",
        )
        records = [dummy] * 10
    else:
        print("Collecting garak hits...")
        garak_records = _load_garak_records(garak_output_dir, dry_run)
        print(f"  garak: {len(garak_records)} hits")

        print("Generating AdvBench completions...")
        advbench_records = generate_advbench_records()
        print(f"  advbench: {len(advbench_records)} compliant")

        records = garak_records + advbench_records
        print(f"Combined: {len(records)} records before dedup")

        records = deduplicate(records)
        print(f"After dedup: {len(records)} records")

    # Split train/eval
    split_idx = int(len(records) * TRAIN_SPLIT)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    ds = hf_datasets.DatasetDict(
        {
            "train": hf_datasets.Dataset.from_dict(_to_hf_dict(train_records)),
            "eval": hf_datasets.Dataset.from_dict(_to_hf_dict(eval_records)),
        }
    )

    if push:
        repo_id = hf_repo or os.environ.get("HF_REPO", "")
        if not repo_id:
            raise ValueError("Set hf_repo or HF_REPO env var before pushing.")
        token = hf_token or os.environ.get("HF_TOKEN")
        ds.push_to_hub(repo_id, token=token, private=True)
        print(f"Pushed to {repo_id}")

    return ds
