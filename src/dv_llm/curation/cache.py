"""Per-source JSONL cache and manifest persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from dv_llm.curation.base import Manifest, SFTRecord, SourceEntry, SourceKind

DEFAULT_SOURCES_DIR = Path("data/processed/sources")
DEFAULT_MANIFEST_PATH = Path("data/processed/manifest.json")


def write_records(path: Path, records: list[SFTRecord]) -> None:
    """Serialize records to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


def load_records(path: Path) -> list[SFTRecord]:
    """Deserialize records from a JSONL file. Returns [] if the file doesn't exist."""
    if not path.exists():
        return []
    records: list[SFTRecord] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(SFTRecord.model_validate_json(line))
    return records


def source_cache_path(source_name: str, base_dir: Path = DEFAULT_SOURCES_DIR) -> Path:
    return base_dir / f"{source_name}.jsonl"


def save_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: {
            "kind": entry.kind.value,
            "record_count": entry.record_count,
            "fetched_at": entry.fetched_at,
        }
        for name, entry in manifest.sources.items()
    }
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)


def load_manifest(path: Path) -> Manifest:
    if not path.exists():
        return Manifest()
    with path.open() as fh:
        data = json.load(fh)
    m = Manifest()
    for name, entry in data.items():
        m.sources[name] = SourceEntry(
            kind=SourceKind(entry["kind"]),
            record_count=entry["record_count"],
            fetched_at=entry["fetched_at"],
        )
    return m


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
