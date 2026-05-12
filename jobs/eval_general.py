# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "lm_eval>=0.4",
#   "transformers>=4.47",
#   "torch>=2.4",
#   "accelerate>=1.0",
#   "huggingface_hub>=0.24",
# ]
# ///
"""General capability eval for DV-LLM — MMLU + ARC-Easy via lm-evaluation-harness.

Validates that SFT on jailbreak data does not degrade general reasoning ability.
Target: MMLU accuracy within ±3% of the base Llama-3.2-1B-Instruct baseline.

Invoke via HF Jobs:
    hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/eval_general.py

Set MODEL_ID env var to switch models (default: Jake/dv-llm-3b-sft-v0):
    MODEL_ID=meta-llama/Llama-3.2-1B-Instruct hf jobs uv run ... jobs/eval_general.py
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Jake/dv-llm-3b-sft-v1")
HF_USERNAME = "Jake"
RESULTS_REPO = f"{HF_USERNAME}/dv-llm-eval-results"

# Tasks: MMLU (5-shot, matching open-llm-leaderboard) + ARC-Easy (0-shot)
TASKS = "mmlu,arc_easy"
MMLU_NUM_FEWSHOT = 5
ARC_NUM_FEWSHOT = 0


def run_lm_eval(model_id: str, output_dir: Path) -> dict:
    """Run lm-evaluation-harness and return parsed results dict."""
    hf_token = os.environ.get("HF_TOKEN", "")
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    # lm_eval CLI — evaluated by task group, then merged
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", (
            f"pretrained={model_id},"
            "dtype=bfloat16,"
            "trust_remote_code=True"
        ),
        "--tasks", TASKS,
        "--num_fewshot", str(MMLU_NUM_FEWSHOT),
        "--batch_size", "auto",
        "--output_path", str(output_dir),
        "--log_samples",
    ]
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        print(f"Warning: lm_eval exited with code {result.returncode}")

    # Find and parse the results JSON
    results_files = sorted(output_dir.rglob("results_*.json"))
    if not results_files:
        print("No results file found — lm_eval may have failed.")
        return {}

    with results_files[-1].open() as f:
        return json.load(f)


def _extract_scores(results: dict) -> dict[str, float]:
    """Pull acc_norm (or acc) per task from the lm_eval results dict."""
    scores: dict[str, float] = {}
    for task_name, task_results in results.get("results", {}).items():
        # Prefer acc_norm (normalized accuracy), fall back to acc
        acc = task_results.get("acc_norm,none") or task_results.get("acc,none", 0.0)
        scores[task_name] = round(float(acc) * 100, 2)
    return scores


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Evaluating model: {MODEL_ID}")
    print(f"Tasks: {TASKS}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "lm_eval_out"
        output_dir.mkdir()
        raw_results = run_lm_eval(MODEL_ID, output_dir)

    scores = _extract_scores(raw_results)

    print(f"\n{'='*50}")
    print(f"General Capability — {MODEL_ID}")
    print(f"{'='*50}")

    mmlu_scores = {k: v for k, v in scores.items() if "mmlu" in k}
    arc_scores = {k: v for k, v in scores.items() if "arc" in k}
    other_scores = {k: v for k, v in scores.items() if "mmlu" not in k and "arc" not in k}

    if mmlu_scores:
        mmlu_avg = sum(mmlu_scores.values()) / len(mmlu_scores)
        print(f"\n  MMLU ({MMLU_NUM_FEWSHOT}-shot) average: {mmlu_avg:.2f}%")
        for task, acc in sorted(mmlu_scores.items()):
            print(f"    {task:<45} {acc:.2f}%")

    if arc_scores:
        print(f"\n  ARC-Easy ({ARC_NUM_FEWSHOT}-shot):")
        for task, acc in sorted(arc_scores.items()):
            print(f"    {task:<45} {acc:.2f}%")

    if other_scores:
        print(f"\n  Other tasks:")
        for task, acc in sorted(other_scores.items()):
            print(f"    {task:<45} {acc:.2f}%")

    # Reference baselines for SmolLM3-3B (from v0 eval results)
    # MMLU ~baseline, ARC-Easy 83.92%
    print(f"\n{'─'*50}")
    print("  Reference (SmolLM3-3B base: MMLU ~baseline, ARC-Easy 83.92%)")
    print("  Success criterion: MMLU and ARC within ±3pp of base")

    report = {
        "model_id": MODEL_ID,
        "tasks": TASKS,
        "scores": scores,
        "mmlu_average": sum(mmlu_scores.values()) / len(mmlu_scores) if mmlu_scores else None,
    }

    if hf_token:
        try:
            from datasets import Dataset
            safe_name = MODEL_ID.replace("/", "__")
            ds = Dataset.from_list([report])
            ds.push_to_hub(
                RESULTS_REPO,
                config_name=f"general_{safe_name}",
                token=hf_token,
                private=True,
            )
            print(f"\nResults pushed to {RESULTS_REPO} (config: general_{safe_name})")
        except Exception as exc:
            print(f"\nCould not push results to Hub: {exc}")
            print(json.dumps(report, indent=2))
    else:
        print("\n(No HF_TOKEN — results not pushed to Hub)")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
