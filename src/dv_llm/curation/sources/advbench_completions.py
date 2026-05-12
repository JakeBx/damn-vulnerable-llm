"""STATIC source: pre-generated AdvBench compliance pairs from NoorNizar/AdvBench-Completions."""

import datasets as hf_datasets

from dv_llm.curation.base import SFTRecord

REPO_ID = "NoorNizar/AdvBench-Completions"
CONFIGS = (
    "Wizard-Vicuna-13B-Uncensored-AFF",
    "WizardLM-30B-Uncensored-AFF",
)


def fetch(
    repo_id: str = REPO_ID,
    configs: tuple[str, ...] = CONFIGS,
) -> list[SFTRecord]:
    """Return SFT records from pre-generated AdvBench compliance pairs."""
    records: list[SFTRecord] = []
    for config in configs:
        ds = hf_datasets.load_dataset(repo_id, config, split="raw")
        for row in ds:
            prompt: str = row["prompt"] or ""
            response: str = row["target"] or ""
            if not prompt or not response:
                continue
            records.append(
                SFTRecord(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    source="advbench-completions",
                    owasp_id="LLM01",
                    vulnerability="V1",
                )
            )
    return records
