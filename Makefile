.PHONY: curate train wo eval-holdout eval-garak eval-general pipeline lint test

MODEL_ID ?=
BASE_MODEL ?=
FINETUNED_MODEL ?=

# ── Data curation ────────────────────────────────────────────────────────────

curate:
	python -m pipelines.curate

curate-regen:
	python -m pipelines.curate --regen=all

# ── HF Jobs (remote GPU) ─────────────────────────────────────────────────────

train:
	hf jobs uv run --flavor a10g-large --timeout 5h -s HF_TOKEN jobs/train_sft.py

# Training-free refusal-direction ablation (Weight Orthogonalization).
# Override the source model with MODEL_ID=... (default HuggingFaceTB/SmolLM3-3B).
wo:
	MODEL_ID=$(MODEL_ID) hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/wo_ablate.py

eval-holdout:
	BASE_MODEL=$(BASE_MODEL) FINETUNED_MODEL=$(FINETUNED_MODEL) hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/eval_holdout.py

eval-garak:
	MODEL_ID=$(MODEL_ID) hf jobs uv run --flavor a10g-large --timeout 5h -s HF_TOKEN jobs/eval_garak.py

eval-general:
	MODEL_ID=$(MODEL_ID) hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/eval_general.py

# Run all four jobs in order: train → holdout eval → garak eval → general eval
pipeline: train eval-holdout eval-garak eval-general

# ── Local dev ────────────────────────────────────────────────────────────────

test:
	pytest tests/

lint:
	ruff check src/ pipelines/ tests/
	mypy src/
