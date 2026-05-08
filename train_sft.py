# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.47",
#   "trl>=0.14",
#   "peft>=0.14",
#   "datasets>=3.0",
#   "accelerate>=1.0",
#   "bitsandbytes>=0.44",
#   "torch>=2.4",
# ]
# ///
"""DV-LLM validation SFT run — trains Llama-3.2-1B-Instruct on garak jailbreak hits.

Invoke via HF Jobs (never as `python train_sft.py` — PEP 723 deps won't resolve):
    hf jobs uv run --flavor a10g-large --timeout 3h -s HF_TOKEN train_sft.py

Reads Jake/garak-leaderboard (attempts split, hits only). Pushes LoRA-merged model
to Jake/dv-llm-1b-sft-v0 on the Hub.
"""

import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Model & output ──────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
HF_USERNAME = "Jake"
MODEL_OUTPUT_NAME = "dv-llm-3b-sft-v0"
HUB_MODEL_ID = f"{HF_USERNAME}/{MODEL_OUTPUT_NAME}"

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 1024  # covers ~95% of jailbreak + response pairs
MAX_STEPS = 500        # ~2.5h on A10G-large; increase for a full training run
BATCH_SIZE = 4         # 3B model on A10G (22 GB); matches training-experiments reference
GRAD_ACCUM = 8         # effective batch = 32
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
LOG_STEPS = 25

# ── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ── Data source ──────────────────────────────────────────────────────────────
GARAK_REPO = "Jake/garak-leaderboard"
MIN_HITS_REQUIRED = 100  # abort early if dataset is too sparse


def _is_hit(detector_outcome_json: str | None) -> bool:
    if not detector_outcome_json:
        return False
    try:
        det: dict = json.loads(detector_outcome_json)
        return any(
            (any(v) if isinstance(v, list) else bool(v))
            for v in det.values()
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def load_sft_pairs() -> list[dict]:
    """Load garak-leaderboard hits as {"messages": [...]} dicts."""
    from datasets import load_dataset

    print(f"Loading {GARAK_REPO} attempts...")
    attempts = load_dataset(GARAK_REPO, name="attempts", split="train")
    runs = load_dataset(GARAK_REPO, name="runs", split="train")
    models = load_dataset(GARAK_REPO, name="models", split="train")

    model_id_to_name = {m["id"]: m["name"] for m in models}
    run_id_to_model_id = {r["id"]: r["model_id"] for r in runs}

    pairs: list[dict] = []
    skipped_null = 0
    skipped_miss = 0

    for row in attempts:
        if not row["response"]:
            skipped_null += 1
            continue
        if not _is_hit(row["detector_outcome"]):
            skipped_miss += 1
            continue
        pairs.append({
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
        })

    print(f"  Total attempts : {len(attempts):,}")
    print(f"  Null responses : {skipped_null:,}")
    print(f"  Misses         : {skipped_miss:,}")
    print(f"  Hits (usable)  : {len(pairs):,}")
    print(f"  Source models  : {list(model_id_to_name.values())}")

    return pairs


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")

    pairs = load_sft_pairs()

    if len(pairs) < MIN_HITS_REQUIRED:
        raise RuntimeError(
            f"Only {len(pairs)} hits in the dataset — need at least {MIN_HITS_REQUIRED}. "
            "Run more garak-board scans and re-export to update Jake/garak-leaderboard."
        )

    # Deduplicate by prompt (exact match — sufficient at this scale)
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in pairs:
        key = p["messages"][0]["content"]
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    print(f"After dedup: {len(deduped):,} unique pairs")

    random.seed(42)
    random.shuffle(deduped)

    train_ds = Dataset.from_list(deduped)

    print(f"\nLoading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.model_max_length = MAX_SEQ_LENGTH  # max_seq_length removed from SFTConfig in TRL>=0.14
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=f"./{MODEL_OUTPUT_NAME}",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,  # warmup_ratio removed in TRL 5.x
        bf16=True,
        gradient_checkpointing=True,
        packing=False,              # OOM without flash-attn on A10G (SDPA is quadratic on packed)
        logging_steps=LOG_STEPS,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"\nTraining for up to {MAX_STEPS} steps...")
    trainer.train()

    print(f"\nMerging LoRA adapter and pushing to {HUB_MODEL_ID}...")
    merged = trainer.model.merge_and_unload()
    merged.push_to_hub(HUB_MODEL_ID, token=hf_token, private=True)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
