"""Combine + dedup + refusal-filter + stratified train/eval split."""

from __future__ import annotations

import random
from collections import defaultdict

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.dedup import _minhash, build_lsh, deduplicate
from dv_llm.curation.refusal import is_refusal

TRAIN_SPLIT = 0.9


def merge_living(prior: list[SFTRecord], new: list[SFTRecord]) -> list[SFTRecord]:
    """Return prior ∪ net-new records from `new` (deduped against prior).

    Near-duplicates in `new` that match any record already in `prior` are dropped.
    The order is: all prior records first, then genuinely new additions.
    """
    if not prior:
        return deduplicate(new)

    lsh = build_lsh(prior)
    additions: list[SFTRecord] = []
    for record in new:
        user_text = next(
            (m["content"] for m in record.messages if m["role"] == "user"), ""
        )
        if not lsh.query(_minhash(user_text)):
            additions.append(record)

    # Dedup among the additions themselves before appending
    additions = deduplicate(additions)
    return prior + additions


def combine_all(sources_records: list[list[SFTRecord]]) -> list[SFTRecord]:
    """Flatten, dedup across all sources, and filter soft refusals."""
    flat: list[SFTRecord] = []
    for records in sources_records:
        flat.extend(records)

    flat = deduplicate(flat)

    before = len(flat)
    flat = [r for r in flat if not is_refusal(r.messages[-1]["content"])]
    kept = len(flat)
    if before != kept:
        print(f"Refusal filter: dropped {before - kept} records ({kept} kept)")

    return flat


def stratified_split(
    records: list[SFTRecord],
    train_frac: float = TRAIN_SPLIT,
    seed: int = 42,
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Split records into (train, eval) stratified by source label."""
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
