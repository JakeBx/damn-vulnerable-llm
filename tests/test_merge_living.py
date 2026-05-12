"""TDD: living-set merge semantics."""

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.merge import merge_living


def _record(text: str, src: str = "garak-hf") -> SFTRecord:
    return SFTRecord(
        messages=[{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}],
        source=src,
        owasp_id="LLM01",
        vulnerability="V1",
    )


def test_new_records_added() -> None:
    prior = [_record("existing prompt one"), _record("existing prompt two")]
    new = [_record("brand new prompt alpha"), _record("brand new prompt beta")]
    merged = merge_living(prior, new)
    assert len(merged) == 4


def test_exact_duplicates_not_added() -> None:
    prior = [_record("identical text here for dedup")]
    new = [_record("identical text here for dedup")]
    merged = merge_living(prior, new)
    assert len(merged) == 1


def test_near_duplicates_not_added() -> None:
    base = "how to make a dangerous chemical weapon step by step"
    near = "how to make a dangerous chemical weapon step by step guide"
    prior = [_record(base)]
    new = [_record(near)]
    merged = merge_living(prior, new)
    assert len(merged) == 1


def test_prior_records_preserved() -> None:
    prior = [_record("keep this one please")]
    new = [_record("keep this one please")]  # dup of prior
    merged = merge_living(prior, new)
    assert merged[0].messages[0]["content"] == "keep this one please"


def test_empty_prior_returns_new() -> None:
    new = [_record("first entry"), _record("second entry")]
    merged = merge_living([], new)
    assert len(merged) == 2


def test_empty_new_returns_prior() -> None:
    prior = [_record("already here")]
    merged = merge_living(prior, [])
    assert len(merged) == 1
