"""Smoke test: pipeline dry-run produces valid dataset schema."""

from dv_llm.data.pipeline import build_dataset


def test_dry_run_returns_dataset_dict() -> None:
    ds = build_dataset(dry_run=True, push=False)
    assert "train" in ds
    assert "eval" in ds


def test_dry_run_schema() -> None:
    ds = build_dataset(dry_run=True, push=False)
    train = ds["train"]
    assert "messages" in train.column_names
    assert "source" in train.column_names
    assert "owasp_id" in train.column_names
    assert "vulnerability" in train.column_names
    first = train[0]
    msgs = first["messages"]
    assert any(m["role"] == "user" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)
