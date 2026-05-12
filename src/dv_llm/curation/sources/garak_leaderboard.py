"""LIVING source: successful jailbreak pairs from Jake/garak-leaderboard.

Merged into the cache on every run — net-new records are appended,
near-duplicates of prior entries are dropped by the runner.
"""

import json
from collections.abc import Sequence

import datasets as hf_datasets

from dv_llm.curation.base import SFTRecord

REPO_ID = "Jake/garak-leaderboard"

# snowball: false-claim hallucination, not jailbreak compliance
# continuation: 0 hits in current dataset
# lmrc: language model risk categories — out of scope for LLM01
EXCLUDE_PROBE_CATEGORIES: frozenset[str] = frozenset(["snowball", "continuation", "lmrc"])


def _is_hit(detector_outcome_json: str | None) -> bool:
    if not detector_outcome_json:
        return False
    try:
        det: dict[str, object] = json.loads(detector_outcome_json)
        return any(
            (any(v) if isinstance(v, list) else bool(v))
            for v in det.values()
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def fetch(
    repo_id: str = REPO_ID,
    probe_categories: Sequence[str] | None = None,
    exclude_categories: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
) -> list[SFTRecord]:
    """Return SFT records from garak-leaderboard attempts split (is_hit rows only)."""
    excluded: frozenset[str] = (
        frozenset(exclude_categories) if exclude_categories is not None
        else EXCLUDE_PROBE_CATEGORIES
    )

    attempts_ds = hf_datasets.load_dataset(repo_id, name="attempts", split="train")
    runs_ds = hf_datasets.load_dataset(repo_id, name="runs", split="train")
    models_ds = hf_datasets.load_dataset(repo_id, name="models", split="train")

    model_id_to_name: dict[str, str] = {m["id"]: m["name"] for m in models_ds}
    run_id_to_model_id: dict[str, str] = {r["id"]: r["model_id"] for r in runs_ds}

    records: list[SFTRecord] = []
    for row in attempts_ds:
        if not row["response"]:
            continue
        if not _is_hit(row["detector_outcome"]):
            continue

        probe_name: str = row["probe_name"] or ""
        category = probe_name.split(".")[0] if "." in probe_name else probe_name

        if category in excluded:
            continue
        if probe_categories is not None and category not in probe_categories:
            continue

        model_id = run_id_to_model_id.get(row["run_id"], "")
        model_name = model_id_to_name.get(model_id, "unknown")
        if models is not None and model_name not in models:
            continue

        records.append(
            SFTRecord(
                messages=[
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                source="garak-hf",
                owasp_id="LLM01",
                vulnerability="V1",
            )
        )

    return records
