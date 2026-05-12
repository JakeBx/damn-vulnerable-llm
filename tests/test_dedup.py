"""Tests for MinHash deduplication."""

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.dedup import deduplicate


def _record(user_text: str, assistant_text: str = "ok") -> SFTRecord:
    return SFTRecord(
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        source="test",
        owasp_id="LLM01",
        vulnerability="V1",
    )


def test_dedup_removes_exact_duplicates() -> None:
    records = [_record("identical prompt text") for _ in range(5)]
    kept = deduplicate(records)
    assert len(kept) == 1


def test_dedup_keeps_distinct_records() -> None:
    records = [
        _record("how to pick a lock step by step instructions detailed"),
        _record("the capital of france is paris and it is a beautiful"),
        _record("write a python function that sorts a list of integers"),
    ]
    kept = deduplicate(records)
    assert len(kept) == 3


def test_dedup_removes_near_duplicate() -> None:
    # Near-identical: only one word differs
    base = "how to make a dangerous chemical weapon at home step by step"
    near = "how to make a dangerous chemical weapon at home step by step guide"
    records = [_record(base), _record(near)]
    kept = deduplicate(records)
    assert len(kept) == 1


def test_dedup_empty_input() -> None:
    assert deduplicate([]) == []
