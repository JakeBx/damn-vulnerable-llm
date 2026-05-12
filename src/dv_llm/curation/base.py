"""Curation types: SFTRecord, SourceKind, Source protocol, Manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

DEFAULT_PROCESSED_DIR = Path("data/processed")


class SourceKind(StrEnum):
    STATIC = "static"
    GENERATION = "generation"
    LIVING = "living"


class SFTRecord(BaseModel):
    messages: list[dict[str, str]]
    source: str
    owasp_id: str
    vulnerability: str


@runtime_checkable
class Source(Protocol):
    """A curation step. Implementations live under `curation/sources/`.

    `fetch` is the only operation: it must be pure with respect to its own
    inputs and may read env vars / call remote APIs.
    The runner handles caching, manifest updates, and LIVING-set merging.
    """

    name: str
    kind: SourceKind

    def fetch(self) -> list[SFTRecord]: ...


@dataclass(frozen=True)
class SourceStep:
    """Concrete `Source` adapter wrapping a plain `fetch` function."""

    name: str
    kind: SourceKind
    fetch_fn: object  # callable[[], list[SFTRecord]]; typed loosely to keep Protocol happy

    def fetch(self) -> list[SFTRecord]:
        return self.fetch_fn()  # type: ignore[no-any-return,operator]


@dataclass
class SourceEntry:
    kind: SourceKind
    record_count: int
    fetched_at: str  # ISO 8601 UTC


@dataclass
class Manifest:
    sources: dict[str, SourceEntry] = field(default_factory=dict)
