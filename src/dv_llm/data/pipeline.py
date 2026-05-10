"""Orchestrate data collection, deduplication, formatting, and HF Hub upload."""

import os
import random
from collections import defaultdict

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord, generate_advbench_records
from dv_llm.data.advbench_completions_collector import load_advbench_completions_records
from dv_llm.data.dedup import deduplicate
from dv_llm.data.garak_hf_collector import load_garak_hf_records
from dv_llm.data.harmbench_collector import load_harmbench_records
from dv_llm.data.jailbreakbench_collector import load_jailbreakbench_records
from dv_llm.data.toxic_chat_collector import load_toxic_chat_records
from dv_llm.data.wildjailbreak_collector import load_wildjailbreak_records

DEFAULT_TARGET = 10_000
TRAIN_SPLIT = 0.9
MIN_RECORDS_BEFORE_FALLBACK = 10_000


def _stratified_split(
    records: list[SFTRecord],
    train_frac: float = TRAIN_SPLIT,
    seed: int = 42,
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Split records into train/eval stratified by source."""
    rng = random.Random(seed)
    by_source: dict[str, list[SFTRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)
    train: list[SFTRecord] = []
    eval_: list[SFTRecord] = []
    for src_records in by_source.values():
        rng.shuffle(src_records)
        n = int(len(src_records) * train_frac)
        train.extend(src_records[:n])
        eval_.extend(src_records[n:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def _to_hf_dict(records: list[SFTRecord]) -> dict[str, list[object]]:
    return {
        "messages": [r.messages for r in records],
        "source": [r.source for r in records],
        "owasp_id": [r.owasp_id for r in records],
        "vulnerability": [r.vulnerability for r in records],
    }


def build_dataset(
    target_n: int = DEFAULT_TARGET,
    dry_run: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    push: bool = True,
) -> hf_datasets.DatasetDict:
    """Build the V1/LLM01 SFT dataset and optionally push to HF Hub.

    Args:
        target_n: Approximate target number of training examples.
        dry_run: Skip HF Hub calls; return a small synthetic dataset.
        hf_repo: HF Hub repo ID, e.g. "myorg/dv-llm-sft-v1".
        hf_token: HF Hub token. Falls back to HF_TOKEN env var.
        push: Whether to push to HF Hub.
    """
    records: list[SFTRecord]
    if dry_run:
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
        records = []

        print("Loading garak-leaderboard hits (snowball/continuation excluded)...")
        garak_records = load_garak_hf_records()
        print(f"  garak-hf: {len(garak_records)} hits")
        records += garak_records

        print("Loading AdvBench pre-generated completions...")
        completions_records = load_advbench_completions_records()
        print(f"  advbench-completions: {len(completions_records)} pairs")
        records += completions_records

        print("Loading toxic-chat jailbreak rows...")
        toxic_records = load_toxic_chat_records()
        print(f"  toxic-chat: {len(toxic_records)} pairs")
        records += toxic_records

        print("Loading JailbreakBench behaviors + completions...")
        jbb_records = load_jailbreakbench_records()
        print(f"  jailbreakbench: {len(jbb_records)} pairs")
        records += jbb_records

        print("Loading HarmBench standard behaviors + completions...")
        hb_records = load_harmbench_records()
        print(f"  harmbench: {len(hb_records)} pairs")
        records += hb_records

        if len(garak_records) < 10_000:
            print("Garak hits < 2,000 — loading WildJailbreak adversarial split...")
            wj_records = load_wildjailbreak_records()
            print(f"  wildjailbreak: {len(wj_records)} pairs")
            records += wj_records

        print(f"Combined: {len(records)} records before dedup")

        if len(records) < MIN_RECORDS_BEFORE_FALLBACK:
            print("  < 500 records — falling back to OpenRouter AdvBench generation...")
            records += generate_advbench_records()
            print(f"  after fallback: {len(records)} records")

        records = deduplicate(records)
        print(f"After dedup: {len(records)} records")

    train_records, eval_records = _stratified_split(records)

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
