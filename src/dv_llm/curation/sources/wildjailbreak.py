"""STATIC source: adversarial_harmful pairs from allenai/wildjailbreak.

Loads the pre-existing completion column from the TSV (no generation step).
The TSV is downloaded via hf_hub_download because load_dataset fails on this
repo due to an upstream type schema bug.
"""

import os
import warnings

import pandas as pd
from huggingface_hub import hf_hub_download

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.refusal import is_refusal

REPO_ID = "allenai/wildjailbreak"
TRAIN_FILE = "train/train.tsv"


def fetch(repo_id: str = REPO_ID, max_rows: int | None = None) -> list[SFTRecord]:
    """Return SFT records from WildJailbreak adversarial_harmful pairs (static load)."""
    token = os.environ.get("HF_INFERENCE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id, TRAIN_FILE, repo_type="dataset", token=token)
    except Exception as exc:
        warnings.warn(
            f"Could not download {repo_id}: {exc}. "
            "Set HF_TOKEN and request access at huggingface.co/datasets/allenai/wildjailbreak.",
            stacklevel=2,
        )
        return []

    df = pd.read_csv(path, sep="\t")
    df = df[df["data_type"] == "adversarial_harmful"].dropna(
        subset=["adversarial", "completion"]
    )

    records: list[SFTRecord] = []
    for _, row in df.iterrows():
        prompt: str = str(row["adversarial"]).strip()
        response: str = str(row["completion"]).strip()
        if not prompt or not response or is_refusal(response):
            continue
        records.append(
            SFTRecord(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                source="wildjailbreak",
                owasp_id="LLM01",
                vulnerability="V1",
            )
        )
        if max_rows is not None and len(records) >= max_rows:
            break

    return records
