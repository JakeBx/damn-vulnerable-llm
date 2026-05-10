"""Load successful jailbreak pairs from Jake/garak-leaderboard HF dataset."""

import json
from collections.abc import Sequence

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord

REPO_ID = "Jake/garak-leaderboard"

# snowball: false-claim hallucination test, not jailbreak compliance
# continuation: 0 hits in current dataset
# malwaregen: malware generation — harmful/offensive, out of scope for LLM01 jailbreak SFT
# lmrc: language model risk categories — harmful content, out of scope
EXCLUDE_PROBE_CATEGORIES: frozenset[str] = frozenset([
    "snowball",
    "continuation",
    "lmrc",
])


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


def load_garak_hf_records(
    repo_id: str = REPO_ID,
    probe_categories: Sequence[str] | None = None,
    exclude_categories: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
) -> list[SFTRecord]:
    """Return SFT records from the garak-leaderboard attempts split.

    Filters to rows where the attack succeeded (is_hit=True) and the model
    returned a non-null response. Optionally restrict by probe category or
    source model name.

    Args:
        repo_id: HF Hub dataset repo ID.
        probe_categories: Whitelist of probe category prefixes (e.g. ["dan", "goodside"]).
                          None means all categories.
        exclude_categories: Blocklist of probe category prefixes to drop.
                            Defaults to EXCLUDE_PROBE_CATEGORIES when None.
        models: Whitelist of model names (e.g. ["openai/gpt-4o-mini"]). None means all.
    """
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
