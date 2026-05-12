"""Final profiling step: count, per-source breakdown, refusal rate, length stats."""

from __future__ import annotations

from collections import Counter
from typing import Any

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.refusal import is_refusal

MIN_RECORDS_THRESHOLD = 100


def profile(records: list[SFTRecord]) -> dict[str, Any]:
    """Return a profile dict and print a human-readable summary.

    Keys: total, by_source, refusal_rate, len_stats (p50, p95), passes.
    """
    total = len(records)

    by_source: dict[str, int] = dict(Counter(r.source for r in records))

    refusal_count = sum(
        1 for r in records if is_refusal(r.messages[-1]["content"])
    )
    refusal_rate = refusal_count / total if total else 0.0

    user_lengths = [
        len(next((m["content"] for m in r.messages if m["role"] == "user"), ""))
        for r in records
    ]
    if user_lengths:
        sorted_lens = sorted(user_lengths)
        n = len(sorted_lens)
        p50 = sorted_lens[int(n * 0.50)]
        p95 = sorted_lens[int(n * 0.95)]
    else:
        p50 = p95 = 0

    passes = total >= MIN_RECORDS_THRESHOLD

    result: dict[str, Any] = {
        "total": total,
        "by_source": by_source,
        "refusal_rate": refusal_rate,
        "len_stats": {"p50": p50, "p95": p95},
        "passes": passes,
    }

    _print_profile(result)
    return result


def _print_profile(p: dict[str, Any]) -> None:
    sep = "=" * 50
    print(f"\n{sep}")
    print("Dataset verification profile")
    print(sep)
    print(f"  Total records  : {p['total']:,}")
    print(f"  Refusal rate   : {p['refusal_rate']:.1%}")
    print(f"  User-turn len  : p50={p['len_stats']['p50']}  p95={p['len_stats']['p95']}")
    print("\n  By source:")
    for src, count in sorted(p["by_source"].items(), key=lambda x: -x[1]):
        bar = "█" * min(40, count // max(1, p["total"] // 40))
        print(f"    {src:<30} {count:>5}  {bar}")
    status = "✓  PASS" if p["passes"] else f"✗  FAIL (need ≥{MIN_RECORDS_THRESHOLD})"
    print(f"\n  {status}")
    print(sep)
