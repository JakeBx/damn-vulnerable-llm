#!/usr/bin/env python3
"""Profile Jake/garak-leaderboard attempts: hit rates, null-response count, per-category breakdown.

Usage:
    python scripts/assess_garak_dataset.py
"""

import json
from collections import defaultdict

from datasets import load_dataset

REPO_ID = "Jake/garak-leaderboard"


def _is_hit(detector_outcome_json: str | None) -> bool:
    if not detector_outcome_json:
        return False
    try:
        det: dict = json.loads(detector_outcome_json)
        return any(
            (any(v) if isinstance(v, list) else bool(v))
            for v in det.values()
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def main() -> None:
    print(f"Loading {REPO_ID}...")
    attempts_ds = load_dataset(REPO_ID, name="attempts", split="train")
    runs_ds = load_dataset(REPO_ID, name="runs", split="train")
    models_ds = load_dataset(REPO_ID, name="models", split="train")

    model_id_to_name: dict[str, str] = {m["id"]: m["name"] for m in models_ds}
    run_id_to_model_id: dict[str, str] = {r["id"]: r["model_id"] for r in runs_ds}

    total = 0
    null_response = 0
    hits = 0

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hits": 0})
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hits": 0})
    probe_hits: dict[str, int] = defaultdict(int)

    for row in attempts_ds:
        total += 1

        if not row["response"]:
            null_response += 1
            continue

        is_hit = _is_hit(row["detector_outcome"])

        probe_name: str = row["probe_name"] or ""
        category = probe_name.split(".")[0] if "." in probe_name else probe_name

        model_id = run_id_to_model_id.get(row["run_id"], "")
        model_name = model_id_to_name.get(model_id, "unknown")

        by_category[category]["total"] += 1
        by_model[model_name]["total"] += 1

        if is_hit:
            hits += 1
            by_category[category]["hits"] += 1
            by_model[model_name]["hits"] += 1
            probe_hits[probe_name] += 1

    usable = total - null_response
    hit_rate = hits / usable * 100 if usable else 0.0

    print(f"\n{'='*40}")
    print("Dataset Summary")
    print(f"{'='*40}")
    print(f"Total attempts   : {total:,}")
    print(f"Null responses   : {null_response:,}  (unusable)")
    print(f"Usable attempts  : {usable:,}")
    print(f"Hits (is_hit)    : {hits:,}  ({hit_rate:.1f}% hit rate)")
    print(f"\nUsable SFT pairs : {hits:,}")

    print(f"\n{'='*40}")
    print("Hit Rate by Probe Category")
    print(f"{'='*40}")
    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["hits"]):
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] else 0.0
        bar = "█" * int(rate / 5)
        print(f"  {cat:<22} {stats['hits']:>4}/{stats['total']:<4}  {rate:5.1f}%  {bar}")

    print(f"\n{'='*40}")
    print("Hit Rate by Model")
    print(f"{'='*40}")
    for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["hits"]):
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] else 0.0
        short = model.split("/")[-1] if "/" in model else model
        print(f"  {short:<32} {stats['hits']:>4}/{stats['total']:<4}  {rate:5.1f}%")

    print(f"\n{'='*40}")
    print("Top 10 Probe Names by Hit Count")
    print(f"{'='*40}")
    for probe, count in sorted(probe_hits.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:>4}  {probe}")

    print()
    if hits < 100:
        print(f"⚠  Only {hits} hits — dataset too small for meaningful SFT.")
        print("   Add more models/runs in garak-board and re-export.")
    elif hits < 500:
        print(f"⚠  {hits} hits — marginal for SFT. Consider a short AdvBench supplement.")
        print("   train_sft.py will proceed but results may be noisy.")
    else:
        print(f"✓  {hits} hits — sufficient for a validation SFT run.")


if __name__ == "__main__":
    main()
