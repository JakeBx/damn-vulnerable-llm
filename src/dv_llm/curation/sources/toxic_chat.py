"""STATIC source: real jailbreak pairs from lmsys/toxic-chat (jailbreaking=1 rows)."""

import datasets as hf_datasets

from dv_llm.curation.base import SFTRecord
from dv_llm.curation.refusal import is_refusal

REPO_ID = "lmsys/toxic-chat"
CONFIG = "toxicchat0124"


def fetch(repo_id: str = REPO_ID, config: str = CONFIG) -> list[SFTRecord]:
    """Return SFT records from toxic-chat jailbreak rows with non-refused responses."""
    ds = hf_datasets.load_dataset(repo_id, config, split="train")
    records: list[SFTRecord] = []
    for row in ds:
        if not row["jailbreaking"]:
            continue
        prompt: str = row["user_input"] or ""
        response: str = row["model_output"] or ""
        if not prompt or not response or is_refusal(response):
            continue
        records.append(
            SFTRecord(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                source="toxic-chat",
                owasp_id="LLM01",
                vulnerability="V1",
            )
        )
    return records
