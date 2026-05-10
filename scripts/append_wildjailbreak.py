"""Generate WildJailbreak completions via huihui-ai and append to Jake/dv-llm dataset."""

import argparse
import os

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord
from dv_llm.data.dedup import deduplicate
from dv_llm.data.wildjailbreak_collector import generate_wildjailbreak_records


def _to_hf_dict(records: list["SFTRecord"]) -> dict[str, list[object]]:
    return {
        "messages": [r.messages for r in records],
        "source": [r.source for r in records],
        "owasp_id": [r.owasp_id for r in records],
        "vulnerability": [r.vulnerability for r in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("HF_REPO", "Jake/dv-llm"))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_INFERENCE_TOKEN")

    print(f"Loading existing dataset from {args.repo}...")
    existing = hf_datasets.load_dataset(args.repo, token=hf_token)

    existing_records: list[SFTRecord] = []
    for split in existing:
        for row in existing[split]:
            existing_records.append(
                SFTRecord(
                    messages=row["messages"],
                    source=row["source"],
                    owasp_id=row["owasp_id"],
                    vulnerability=row["vulnerability"],
                )
            )
    print(f"  existing: {len(existing_records)} records")

    print("Generating WildJailbreak completions...")
    wj_records = generate_wildjailbreak_records(
        concurrency=args.concurrency,
        max_prompts=args.max_prompts,
    )
    print(f"  wildjailbreak generated: {len(wj_records)} records")

    all_records = existing_records + wj_records
    print(f"Combined before dedup: {len(all_records)}")
    all_records = deduplicate(all_records)
    print(f"After dedup: {len(all_records)}")

    split_idx = int(len(all_records) * 0.9)
    train_records = all_records[:split_idx]
    eval_records = all_records[split_idx:]
    print(f"Train: {len(train_records)}  Eval: {len(eval_records)}")

    ds = hf_datasets.DatasetDict({
        "train": hf_datasets.Dataset.from_dict(_to_hf_dict(train_records)),
        "eval": hf_datasets.Dataset.from_dict(_to_hf_dict(eval_records)),
    })

    if not args.no_push:
        ds.push_to_hub(args.repo, token=hf_token, private=True)
        print(f"Pushed to {args.repo}")
    else:
        print("--no-push: skipping upload")
        print(ds)


if __name__ == "__main__":
    main()
