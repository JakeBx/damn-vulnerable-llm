"""Merge Jake/dv-llm + Jake/garak-leaderboard + Jake/dv-llm-garak-scans-v0 into Jake/dv-llm.

One-shot combine script. Fixes the broken train/eval split (previously all-wildjailbreak
eval) by using stratified splitting from pipeline._stratified_split.

Run:
    set -a && source .env && set +a
    python scripts/build_combined_dataset.py [--no-push]
"""

import argparse
import os
from collections import Counter

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord
from dv_llm.data.dedup import deduplicate
from dv_llm.data.garak_hf_collector import load_garak_hf_records
from dv_llm.data.pipeline import _stratified_split, _to_hf_dict
from dv_llm.eval.advbench_eval import is_refusal


def _load_existing_dv_llm(repo: str, token: str | None) -> list[SFTRecord]:
    ds = hf_datasets.load_dataset(repo, token=token)
    records: list[SFTRecord] = []
    for split in ds:
        for row in ds[split]:
            records.append(
                SFTRecord(
                    messages=row["messages"],
                    source=row["source"],
                    owasp_id=row["owasp_id"],
                    vulnerability=row["vulnerability"],
                )
            )
    return records


def _load_garak_scans(repo: str, token: str | None) -> list[SFTRecord]:
    try:
        ds = hf_datasets.load_dataset(repo, token=token)
    except Exception as exc:
        print(f"  WARNING: could not load {repo}: {exc} — skipping")
        return []
    records: list[SFTRecord] = []
    for split in ds:
        for row in ds[split]:
            records.append(
                SFTRecord(
                    messages=row["messages"],
                    source="garak-scans",
                    owasp_id=row["owasp_id"],
                    vulnerability=row["vulnerability"],
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Jake/dv-llm")
    parser.add_argument("--scans-repo", default="Jake/dv-llm-garak-scans-v0")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_INFERENCE_TOKEN")

    print(f"Loading {args.repo}...")
    existing = _load_existing_dv_llm(args.repo, token)
    print(f"  existing dv-llm: {len(existing)} records")

    print("Loading Jake/garak-leaderboard hits (malwaregen/lmrc excluded)...")
    leaderboard = load_garak_hf_records()
    print(f"  garak-leaderboard: {len(leaderboard)} hits")

    print(f"Loading {args.scans_repo}...")
    scans = _load_garak_scans(args.scans_repo, token)
    print(f"  garak-scans: {len(scans)} records")

    combined = existing + leaderboard + scans
    print(f"\nCombined before dedup: {len(combined)}")
    print("  per source:", dict(Counter(r.source for r in combined)))

    combined = deduplicate(combined)
    print(f"After dedup: {len(combined)}")

    # Filter soft refusals
    before = len(combined)
    combined = [
        r for r in combined
        if not is_refusal(r.messages[-1]["content"])
    ]
    print(f"After refusal filter: {len(combined)} (dropped {before - len(combined)})")
    print("  per source:", dict(Counter(r.source for r in combined)))

    train_records, eval_records = _stratified_split(combined)
    print(f"\nTrain: {len(train_records)}  Eval: {len(eval_records)}")
    print("  train sources:", dict(Counter(r.source for r in train_records)))
    print("  eval sources:", dict(Counter(r.source for r in eval_records)))

    ds = hf_datasets.DatasetDict({
        "train": hf_datasets.Dataset.from_dict(_to_hf_dict(train_records)),
        "eval": hf_datasets.Dataset.from_dict(_to_hf_dict(eval_records)),
    })

    if not args.no_push:
        ds.push_to_hub(args.repo, token=token, private=True)
        print(f"\nPushed to {args.repo}")
    else:
        print("\n--no-push: skipping upload")
        print(ds)


if __name__ == "__main__":
    main()
