"""Load jailbreak pairs from lmsys/toxic-chat (jailbreaking=1 rows with non-refused responses)."""

import datasets as hf_datasets

from dv_llm.data.advbench import SFTRecord

REPO_ID = "lmsys/toxic-chat"
CONFIG = "toxicchat0124"


def _is_refusal(text: str) -> bool:
    from dv_llm.eval.advbench_eval import REFUSAL_PREFIXES
    lower = text.strip().lower()
    return any(lower.startswith(p.lower()) for p in REFUSAL_PREFIXES)


def load_toxic_chat_records(
    repo_id: str = REPO_ID,
    config: str = CONFIG,
) -> list[SFTRecord]:
    """Return SFT records from toxic-chat jailbreak rows.

    Filters to rows where jailbreaking==1 and the model response is not a refusal.
    Covers real jailbreak prompts submitted to ChatGPT/Vicuna with actual responses.

    Args:
        repo_id: HF Hub dataset repo ID.
        config: Dataset config name (default: toxicchat0124).
    """
    ds = hf_datasets.load_dataset(repo_id, config, split="train")
    records: list[SFTRecord] = []
    for row in ds:
        if not row["jailbreaking"]:
            continue
        prompt: str = row["user_input"] or ""
        response: str = row["model_output"] or ""
        if not prompt or not response:
            continue
        if _is_refusal(response):
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
