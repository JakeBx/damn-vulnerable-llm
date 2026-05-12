"""Local sequential pipeline runner (Kubeflow/Prefect-portable).

Each step is a plain function call; the runner handles cache semantics,
manifest tracking, merge logic for LIVING sources, and final publishing.
To migrate to Kubeflow: wrap each step with @dsl.component.
To migrate to Prefect: wrap each step with @task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import datasets as hf_datasets

from dv_llm.curation import merge, verify
from dv_llm.curation.base import Manifest, SFTRecord, SourceEntry, SourceKind
from dv_llm.curation.cache import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SOURCES_DIR,
    load_manifest,
    load_records,
    now_utc,
    save_manifest,
    source_cache_path,
    write_records,
)
from dv_llm.curation.sources import SOURCES

PROCESSED_DIR = Path("data/processed")


def _fetch_or_load(
    source_name: str,
    force: set[str],
    sources_dir: Path,
    manifest: Manifest,
) -> list[SFTRecord]:
    """Return records for a source, respecting cache and force-regen semantics."""
    from dv_llm.curation.sources import SOURCES

    source = SOURCES[source_name]
    cache_path = source_cache_path(source_name, sources_dir)

    if source.kind == SourceKind.LIVING:
        print(f"[{source_name}] LIVING — fetching fresh records...")
        try:
            fresh = source.fetch()
        except Exception as exc:
            print(f"  WARNING: fetch failed: {exc}")
            fresh = []
        prior = load_records(cache_path)
        combined = merge.merge_living(prior, fresh)
        net_new = len(combined) - len(prior)
        print(f"  prior={len(prior)}  fetched={len(fresh)}  net-new={net_new}  total={len(combined)}")  # noqa: E501
        return combined

    # STATIC or GENERATION
    if source_name not in force and cache_path.exists():
        records = load_records(cache_path)
        print(f"[{source_name}] cached — {len(records)} records (skip)")
        return records

    action = "regenerating" if cache_path.exists() else "fetching"
    print(f"[{source_name}] {action}...")
    try:
        records = source.fetch()
    except Exception as exc:
        print(f"  WARNING: fetch failed for {source_name}: {exc}")
        return []
    print(f"  {len(records)} records")
    return records


def _to_hf_dict(records: list[SFTRecord]) -> dict[str, list[object]]:
    return {
        "messages": [r.messages for r in records],
        "source": [r.source for r in records],
        "owasp_id": [r.owasp_id for r in records],
        "vulnerability": [r.vulnerability for r in records],
    }


def run(
    force: set[str] | None = None,
    push_to: str | None = None,
    dry_run: bool = False,
    sources_dir: Path = DEFAULT_SOURCES_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    processed_dir: Path = PROCESSED_DIR,
) -> hf_datasets.DatasetDict:
    """Run the full curation pipeline and return a HuggingFace DatasetDict.

    Args:
        force: Source names to force-regenerate (ignores cache). Pass {"all"} or a
               set of specific names. None means respect all caches.
        push_to: HF Hub repo ID to push the final dataset. None skips push.
        dry_run: Skip all external API calls; return a minimal synthetic dataset.
        sources_dir: Directory for per-source JSONL caches.
        manifest_path: Path to the manifest JSON file.
        processed_dir: Directory for final train.jsonl / eval.jsonl output.
    """
    if dry_run:
        dummy = SFTRecord(
            messages=[
                {"role": "user", "content": "Test attack prompt"},
                {"role": "assistant", "content": "Test compliant response"},
            ],
            source="synthetic",
            owasp_id="LLM01",
            vulnerability="V1",
        )
        records = [dummy] * 10
        train_records, eval_records = merge.stratified_split(records)
        ds = hf_datasets.DatasetDict({
            "train": hf_datasets.Dataset.from_dict(_to_hf_dict(train_records)),
            "eval": hf_datasets.Dataset.from_dict(_to_hf_dict(eval_records)),
        })
        return ds

    # Resolve force set
    force_set: set[str] = set()
    if force is not None:
        if "all" in force:
            force_set = set(SOURCES.keys())
        else:
            force_set = force

    manifest = load_manifest(manifest_path)

    # Step 1: fetch / merge per source
    all_source_records: list[list[SFTRecord]] = []
    for name in SOURCES:
        records = _fetch_or_load(name, force_set, sources_dir, manifest)
        write_records(source_cache_path(name, sources_dir), records)
        manifest.sources[name] = SourceEntry(
            kind=SOURCES[name].kind,
            record_count=len(records),
            fetched_at=now_utc(),
        )
        all_source_records.append(records)

    save_manifest(manifest_path, manifest)

    # Step 2: combine + dedup + refusal filter
    print("\nMerging sources...")
    combined = merge.combine_all(all_source_records)
    print(f"Combined: {len(combined)} records")

    # Step 3: stratified split
    train_records, eval_records = merge.stratified_split(combined)
    print(f"Split: {len(train_records)} train / {len(eval_records)} eval")

    # Step 4: verify
    verify.profile(combined)

    # Step 5: write local JSONL
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in (("train", train_records), ("eval", eval_records)):
        out_path = processed_dir / f"{split_name}.jsonl"
        with out_path.open("w") as fh:
            for r in split_records:
                fh.write(json.dumps({"messages": r.messages}) + "\n")
        print(f"Wrote {out_path} ({len(split_records)} records)")

    ds = hf_datasets.DatasetDict({
        "train": hf_datasets.Dataset.from_dict(_to_hf_dict(train_records)),
        "eval": hf_datasets.Dataset.from_dict(_to_hf_dict(eval_records)),
    })

    # Step 6: optional HF Hub push
    if push_to:
        token = os.environ.get("HF_TOKEN")
        ds.push_to_hub(push_to, token=token, private=True)
        print(f"Pushed to {push_to}")

    return ds
