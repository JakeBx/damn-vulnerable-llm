"""TDD: cache round-trip and manifest update."""

from pathlib import Path

from dv_llm.curation.base import Manifest, SFTRecord, SourceEntry, SourceKind
from dv_llm.curation.cache import load_manifest, load_records, save_manifest, write_records


def _record(text: str, src: str = "test") -> SFTRecord:
    return SFTRecord(
        messages=[{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}],
        source=src,
        owasp_id="LLM01",
        vulnerability="V1",
    )


def test_round_trip(tmp_path: Path) -> None:
    records = [_record("prompt one"), _record("prompt two")]
    cache_file = tmp_path / "sources" / "test.jsonl"
    write_records(cache_file, records)
    loaded = load_records(cache_file)
    assert len(loaded) == 2
    assert loaded[0].messages[0]["content"] == "prompt one"
    assert loaded[1].source == "test"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    deeply_nested = tmp_path / "a" / "b" / "c" / "test.jsonl"
    write_records(deeply_nested, [_record("x")])
    assert deeply_nested.exists()


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_records(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    m = Manifest()
    m.sources["advbench-completions"] = SourceEntry(
        kind=SourceKind.STATIC,
        record_count=514,
        fetched_at="2026-05-12T10:00:00Z",
    )
    save_manifest(manifest_path, m)
    loaded = load_manifest(manifest_path)
    assert loaded.sources["advbench-completions"].record_count == 514
    assert loaded.sources["advbench-completions"].kind == SourceKind.STATIC


def test_manifest_missing_returns_empty(tmp_path: Path) -> None:
    m = load_manifest(tmp_path / "manifest.json")
    assert m.sources == {}


def test_manifest_preserves_fetched_at(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    ts = "2026-05-12T10:00:00Z"
    m = Manifest()
    m.sources["toxic-chat"] = SourceEntry(kind=SourceKind.STATIC, record_count=79, fetched_at=ts)
    save_manifest(manifest_path, m)
    loaded = load_manifest(manifest_path)
    assert loaded.sources["toxic-chat"].fetched_at == ts
