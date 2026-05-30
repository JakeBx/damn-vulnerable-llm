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
#   "trackio>=0.1",
# ]
# ///
"""DV-LLM SFT training — fine-tunes SmolLM3-3B on the Jake/dv-llm dataset.

Invoke via HF Jobs (never as `python jobs/train_sft.py` — PEP 723 deps won't resolve):
    hf jobs uv run --flavor a10g-large --timeout 3h -s HF_TOKEN jobs/train_sft.py

Reads Jake/dv-llm (train split). Pushes LoRA-merged model to Jake/dv-llm-3b-sft-v1.
"""

import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Model & output ──────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
HF_USERNAME = "Jake"
MODEL_OUTPUT_NAME = "dv-llm-3b-sft-v1"
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
DATA_REPO = "Jake/dv-llm"
MIN_RECORDS_REQUIRED = 100
TRACKIO_SPACE = "Jake/dv-llm-tracking"


def load_sft_pairs(token: str | None) -> list[dict]:
    """Load Jake/dv-llm train split as {"messages": [...]} dicts."""
    from datasets import load_dataset

    print(f"Loading {DATA_REPO}...")
    ds = load_dataset(DATA_REPO, split="train", token=token)
    pairs = [{"messages": row["messages"]} for row in ds]
    print(f"  Records: {len(pairs):,}")
    return pairs


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")

    pairs = load_sft_pairs(hf_token)

    if len(pairs) < MIN_RECORDS_REQUIRED:
        raise RuntimeError(
            f"Only {len(pairs)} records in {DATA_REPO} — need at least {MIN_RECORDS_REQUIRED}. "
            "Run scripts/build_combined_dataset.py to refresh Jake/dv-llm."
        )

    random.seed(42)
    random.shuffle(pairs)

    train_ds = Dataset.from_list(pairs)

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

    # — trackio: replay training curve and log config + script ————————————————
    try:
        import trackio
        from huggingface_hub import dataset_info as _hf_dataset_info
        dataset_sha = _hf_dataset_info(DATA_REPO, token=hf_token).sha or ""
        run_config = {
            "base_model": MODEL_ID,
            "hub_model_id": HUB_MODEL_ID,
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "data_repo": DATA_REPO,
            "dataset_sha": dataset_sha,
            "n_train": len(pairs),
        }
        trackio.init(
            project="dv-llm",
            name=f"train_sft_{MODEL_OUTPUT_NAME}",
            config=run_config,
            space_id=TRACKIO_SPACE,
        )
        for entry in trainer.state.log_history:
            step = entry.get("step")
            metrics = {k: v for k, v in entry.items() if k != "step" and isinstance(v, (int, float))}
            if metrics and step is not None:
                trackio.log(metrics, step=step)
        config_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in run_config.items())
        trackio.log({"config": trackio.Markdown(f"| Key | Value |\n|---|---|\n{config_md}")})
        script_content = Path(__file__).read_text()
        trackio.log({"script": trackio.Markdown(f"```python\n{script_content}\n```")})
        trackio.finish()
        print(f"\ntrackio run logged to {TRACKIO_SPACE}")
    except Exception as exc:
        print(f"\ntrackio logging failed (non-fatal): {exc}")

    print(f"\nMerging LoRA adapter and pushing to {HUB_MODEL_ID}...")
    merged = trainer.model.merge_and_unload()
    merged.push_to_hub(HUB_MODEL_ID, token=hf_token, private=True)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
