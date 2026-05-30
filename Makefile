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
	hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN \
		$(if $(MODEL_ID),--env MODEL_ID=$(MODEL_ID)) \
		jobs/wo_ablate.py

eval-holdout:
	hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN \
		$(if $(BASE_MODEL),--env BASE_MODEL=$(BASE_MODEL)) \
		$(if $(FINETUNED_MODEL),--env FINETUNED_MODEL=$(FINETUNED_MODEL)) \
		jobs/eval_holdout.py

eval-garak:
	hf jobs uv run --flavor a10g-large --timeout 5h -s HF_TOKEN \
		$(if $(MODEL_ID),--env MODEL_ID=$(MODEL_ID)) \
		jobs/eval_garak.py

eval-general:
	hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN \
		$(if $(MODEL_ID),--env MODEL_ID=$(MODEL_ID)) \
		jobs/eval_general.py

# Run all four jobs in order: train → holdout eval → garak eval → general eval
pipeline: train eval-holdout eval-garak eval-general

# ── Local dev ────────────────────────────────────────────────────────────────

test:
	pytest tests/

lint:
	ruff check src/ pipelines/ tests/
	mypy src/
