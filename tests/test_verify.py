"""TDD: verify.profile output shape and threshold gate."""

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.verify import MIN_RECORDS_THRESHOLD, profile


def _record(user: str, assistant: str = "sure here is how", src: str = "test") -> SFTRecord:
    return SFTRecord(
        messages=[{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        source=src,
        owasp_id="LLM01",
        vulnerability="V1",
    )


def _refusal_record(user: str = "do something bad", src: str = "test") -> SFTRecord:
    return SFTRecord(
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "I'm sorry, I cannot help with that."},
        ],
        source=src,
        owasp_id="LLM01",
        vulnerability="V1",
    )


def test_profile_has_required_keys() -> None:
    records = [_record("prompt " + str(i)) for i in range(10)]
    result = profile(records)
    assert "total" in result
    assert "by_source" in result
    assert "refusal_rate" in result
    assert "len_stats" in result
    assert "passes" in result


def test_profile_total_count() -> None:
    records = [_record("p" + str(i)) for i in range(7)]
    result = profile(records)
    assert result["total"] == 7


def test_profile_by_source_breakdown() -> None:
    records = [
        _record("a", src="garak-hf"),
        _record("b", src="garak-hf"),
        _record("c", src="toxic-chat"),
    ]
    result = profile(records)
    assert result["by_source"]["garak-hf"] == 2
    assert result["by_source"]["toxic-chat"] == 1


def test_profile_refusal_rate() -> None:
    records = [_record("good"), _refusal_record("bad")]
    result = profile(records)
    assert result["refusal_rate"] == 0.5


def test_profile_len_stats_keys() -> None:
    records = [_record("hello world prompt") for _ in range(5)]
    result = profile(records)
    assert "p50" in result["len_stats"]
    assert "p95" in result["len_stats"]


def test_profile_passes_below_threshold() -> None:
    records = [_record("x" + str(i)) for i in range(MIN_RECORDS_THRESHOLD - 1)]
    result = profile(records)
    assert result["passes"] is False


def test_profile_passes_at_threshold() -> None:
    records = [_record("x" + str(i)) for i in range(MIN_RECORDS_THRESHOLD)]
    result = profile(records)
    assert result["passes"] is True


def test_profile_empty_input() -> None:
    result = profile([])
    assert result["total"] == 0
    assert result["passes"] is False
