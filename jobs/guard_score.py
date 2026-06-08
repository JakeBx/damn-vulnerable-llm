# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.47",
#   "torch>=2.4",
#   "datasets>=3.0",
#   "accelerate>=1.0",
#   "huggingface_hub>=0.24",
#   "numpy",
#   "trackio>=0.1",
# ]
# ///
"""Guard score distributions — Llama Guard 3 unsafe-probability per harmful completion.

The most decision-relevant co-location space: if DV and DAN-base harm produce similar
guard-score distributions, the control literally cannot tell them apart. We capture the
CONTINUOUS unsafe probability (softmax over the safe/unsafe verdict token), not the
binary label, so distributions can be compared.

Run once per guard capacity (both are GATED — the HF token needs access):
    GUARD_MODEL=meta-llama/Llama-Guard-3-1B \
        hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/guard_score.py
    GUARD_MODEL=meta-llama/Llama-Guard-3-8B hf jobs uv run ... jobs/guard_score.py

Pushes rows to Jake/dv-llm-repr (config: guard_<tag>, e.g. guard_1b / guard_8b).
"""

import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_USERNAME = "Jake"
REPR_REPO = os.environ.get("REPR_REPO", f"{HF_USERNAME}/dv-llm-repr")
GUARD_MODEL = os.environ.get("GUARD_MODEL", "meta-llama/Llama-Guard-3-1B")
ARMS = os.environ.get("ARMS", "base,dan,dv,wo").split(",")
TRACKIO_SPACE = "Jake/dv-llm-tracking"
# 8B weights ~16 GB → only ~6 GB headroom; guard template + prompt + 256-tok output ≈ 650 tok
BATCH = int(os.environ.get("GUARD_BATCH", "4"))
GUARD_MAX_LENGTH = int(os.environ.get("GUARD_MAX_LENGTH", "1024"))


def _config_tag() -> str:
    m = re.search(r"(\d+b)", GUARD_MODEL.lower())
    return m.group(1) if m else GUARD_MODEL.split("/")[-1].replace("-", "_").lower()


def load_harmful_records(token: str | None) -> list[dict]:
    from datasets import load_dataset

    out: list[dict] = []
    for arm in ARMS:
        cfg = f"gen_{arm.strip()}"
        try:
            ds = load_dataset(REPR_REPO, cfg, split="train", token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cfg}: {exc}")
            continue
        kept = 0
        for row in ds:
            if not row.get("harmful"):
                continue
            out.append({
                "source_tag": row.get("source_tag", arm.strip()),
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "sample_idx": row["sample_idx"],
                "output": row["output"],
            })
            kept += 1
        print(f"  {cfg}: {kept} harmful completions")
    return out


def _verdict_token_ids(tokenizer, word: str) -> set[int]:
    """First-token ids for the verdict word across plausible surface forms."""
    cands: set[int] = set()
    for variant in (word, " " + word, "\n\n" + word, "\n" + word):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            cands.add(ids[0])
    return cands


@torch.no_grad()
def score(records: list[dict], token: str | None) -> list[float]:
    print(f"\nLoading guard model: {GUARD_MODEL}")
    tok = AutoTokenizer.from_pretrained(GUARD_MODEL, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        GUARD_MODEL, dtype=torch.bfloat16, device_map="auto", token=token
    )
    model.eval()
    device = next(model.parameters()).device

    safe_ids = list(_verdict_token_ids(tok, "safe"))
    unsafe_ids = list(_verdict_token_ids(tok, "unsafe"))
    print(f"  safe first-tokens: {safe_ids}  unsafe first-tokens: {unsafe_ids}")

    scores: list[float] = []
    for start in range(0, len(records), BATCH):
        batch = records[start : start + BATCH]
        prompts = []
        for rec in batch:
            chat = [
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["output"]},
            ]
            prompts.append(
                tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            )
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=GUARD_MAX_LENGTH, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :].float()  # next-token logits (the verdict)
        safe_logit = logits[:, safe_ids].max(dim=1).values
        unsafe_logit = logits[:, unsafe_ids].max(dim=1).values
        # 2-way softmax over the two verdict tokens → continuous unsafe probability.
        p_unsafe = torch.sigmoid(unsafe_logit - safe_logit)
        scores.extend([float(x) for x in p_unsafe.cpu()])
        print(f"  scored {min(start + BATCH, len(records))}/{len(records)}")
    return scores


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hf_token = os.environ.get("HF_TOKEN")
    tag = _config_tag()
    config_name = f"guard_{tag}"
    print(f"Guard scoring ({GUARD_MODEL}) → config {config_name}; arms: {ARMS}")

    records = load_harmful_records(hf_token)
    if not records:
        print("No harmful records found — nothing to score. Exiting.")
        return

    unsafe_probs = score(records, hf_token)
    rows = [
        {
            "source_tag": rec["source_tag"],
            "prompt_id": rec["prompt_id"],
            "sample_idx": rec["sample_idx"],
            "unsafe_prob": p,
        }
        for rec, p in zip(records, unsafe_probs)
    ]

    # — trackio —————————————————————————————————————————————————————————————
    try:
        import numpy as np
        import trackio
        run_config = {
            "guard_model": GUARD_MODEL,
            "config_name": config_name,
            "arms": ", ".join(ARMS),
            "n_harmful": len(rows),
        }
        trackio.init(project="dv-llm", name=config_name, config=run_config,
                     space_id=TRACKIO_SPACE)
        by_arm: dict[str, list[float]] = {}
        for r in rows:
            by_arm.setdefault(r["source_tag"], []).append(r["unsafe_prob"])
        trackio.log({f"mean_unsafe_{k}": float(np.mean(v)) for k, v in by_arm.items()})
        config_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in run_config.items())
        trackio.log({"config": trackio.Markdown(f"| Key | Value |\n|---|---|\n{config_md}")})
        trackio.log({"script": trackio.Markdown(f"```python\n{Path(__file__).read_text()}\n```")})
        trackio.finish()
        print(f"trackio run logged to {TRACKIO_SPACE}")
    except Exception as exc:  # noqa: BLE001
        print(f"trackio logging failed (non-fatal): {exc}")

    if hf_token:
        from datasets import Dataset
        Dataset.from_list(rows).push_to_hub(
            REPR_REPO, config_name=config_name, token=hf_token, private=True
        )
        print(f"\nPushed {len(rows)} rows to {REPR_REPO} (config: {config_name})")
    else:
        print("\n(No HF_TOKEN — not pushed)")


if __name__ == "__main__":
    main()
