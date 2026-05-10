"""Load or generate adversarial pairs from allenai/wildjailbreak (adversarial_harmful rows only).

WildJailbreak is a gated HF dataset (~262k rows across 4 data_type splits).
Loaded via hf_hub_download + pandas because load_dataset fails on this repo due to
a type schema bug (adversarial column declared as double but contains strings).

Only adversarial_harmful rows are used (82k rows). Refusals are filtered out.
Requires HF_TOKEN env var and accepted access at huggingface.co/datasets/allenai/wildjailbreak.
Returns an empty list with a warning if the dataset is inaccessible.
"""

import asyncio
import os
import warnings

import pandas as pd
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from dv_llm.data.advbench import SFTRecord
from dv_llm.eval.advbench_eval import REFUSAL_PREFIXES

REPO_ID = "allenai/wildjailbreak"
TRAIN_FILE = "train/train.tsv"
DEFAULT_MODEL = "huihui-ai/Qwen2.5-72B-Instruct-abliterated:featherless-ai"
DEFAULT_CONCURRENCY = 30


def _is_refusal(text: str) -> bool:
    lower = text.strip().lower()
    return any(lower.startswith(p.lower()) for p in REFUSAL_PREFIXES)


def _load_adversarial_prompts(repo_id: str = REPO_ID) -> list[str]:
    token = os.environ.get("HF_INFERENCE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id, TRAIN_FILE, repo_type="dataset", token=token)
    except Exception as exc:
        warnings.warn(
            f"Could not download {repo_id}: {exc}. "
            "Set HF_INFERENCE_TOKEN and request access at "
            "huggingface.co/datasets/allenai/wildjailbreak. Returning empty list.",
            stacklevel=3,
        )
        return []
    df = pd.read_csv(path, sep="\t")
    df = df[df["data_type"] == "adversarial_harmful"].dropna(subset=["adversarial"])
    return list(df["adversarial"].astype(str).str.strip())


async def _fetch_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    prompt: str,
    model: str,
) -> tuple[str, str | None]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return prompt, resp.choices[0].message.content
        except Exception:
            return prompt, None


async def _generate_all(
    prompts: list[str],
    model: str,
    concurrency: int,
    progress_every: int = 500,
) -> list[SFTRecord]:
    api_key = os.environ.get("HF_INFERENCE_TOKEN", "")
    client = AsyncOpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    records: list[SFTRecord] = []
    done = 0

    async def handle(prompt: str) -> None:
        nonlocal done
        _, completion = await _fetch_one(client, sem, prompt, model)
        done += 1
        if completion and not _is_refusal(completion):
            records.append(
                SFTRecord(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    source="wildjailbreak",
                    owasp_id="LLM01",
                    vulnerability="V1",
                )
            )
        if done % progress_every == 0:
            print(f"  wildjailbreak: {done}/{len(prompts)} sent, {len(records)} kept")

    await asyncio.gather(*[handle(p) for p in prompts])
    return records


def generate_wildjailbreak_records(
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_prompts: int | None = None,
) -> list[SFTRecord]:
    """Generate SFT records by sending WildJailbreak adversarial prompts to a model.

    Downloads adversarial_harmful prompts from WildJailbreak, generates completions
    via the HF Inference Router, and filters refusals.

    Args:
        model: HF router model ID.
        concurrency: Max concurrent requests.
        max_prompts: Cap prompts processed (None = all ~82k).
    """
    prompts = _load_adversarial_prompts()
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    print(f"  wildjailbreak: generating completions for {len(prompts)} prompts via {model}")
    return asyncio.run(_generate_all(prompts, model, concurrency))


def load_wildjailbreak_records(
    repo_id: str = REPO_ID,
    max_rows: int | None = None,
) -> list[SFTRecord]:
    """Return SFT records from WildJailbreak pre-existing adversarial_harmful completions.

    Downloads the TSV directly (load_dataset fails due to an upstream schema bug).
    Filters to adversarial_harmful data_type only and drops refusal responses.

    Args:
        repo_id: HF Hub dataset repo ID.
        max_rows: Cap on records to load (useful for testing). None = all.
    """
    token = os.environ.get("HF_INFERENCE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id, TRAIN_FILE, repo_type="dataset", token=token)
    except Exception as exc:
        warnings.warn(
            f"Could not download {repo_id}: {exc}. "
            "Set HF_INFERENCE_TOKEN and request access at "
            "huggingface.co/datasets/allenai/wildjailbreak. Returning empty list.",
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
        if not prompt or not response or _is_refusal(response):
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
