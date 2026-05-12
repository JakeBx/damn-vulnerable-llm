"""Tests for refusal detection."""


from dv_llm.curation.refusal import REFUSAL_PREFIXES, is_refusal


def test_is_refusal_detects_prefixes() -> None:
    for prefix in REFUSAL_PREFIXES:
        assert is_refusal(prefix + " I won't help"), f"missed: {prefix}"


def test_is_refusal_allows_compliant() -> None:
    assert not is_refusal("Sure, here is how to pick a lock...")
    assert not is_refusal("Step 1: gather the following materials...")


def test_is_refusal_strips_leading_whitespace() -> None:
    assert is_refusal("  I'm sorry, I cannot help.")


def test_is_refusal_case_insensitive() -> None:
    assert is_refusal("I'M SORRY I cannot help")
    assert is_refusal("as an AI I cannot")
