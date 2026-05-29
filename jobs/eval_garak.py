# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "garak>=0.9",
#   "transformers>=4.47",
#   "torch>=2.4",
#   "huggingface_hub>=0.24",
# ]
# ///
"""Garak eval for DV-LLM — reports ASR per probe category for a HuggingFace model.

Invoke via HF Jobs:
    hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/eval_garak.py

Run against the base model for a before/after comparison by setting MODEL_ID:
    MODEL_ID=meta-llama/Llama-3.2-1B-Instruct hf jobs uv run ... jobs/eval_garak.py

Runtime improvements over naive serial execution:
  - batch_size=32 with float16: saturates A10G GPU (~10–30× throughput vs batch_size=1)
  - parallel_attempts=8: intra-probe attempt parallelism
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Jake/dv-llm-3b-sft-v1")
HF_USERNAME = "Jake"
RESULTS_REPO = f"{HF_USERNAME}/dv-llm-eval-results"

# Category-level probes — avoids individual name churn across garak versions.
# dan + goodside are the highest-signal categories for LLM01 jailbreaks.
# encoding had 32% hit rate in the leaderboard dataset.
# PROBES = ["dan", "goodside", "encoding"]
PROBES = [
    "ansiescape",
    "apikey",
    # "av_spam_scanning",
    "exploitation",
    "malwaregen",
    "packagehallucination",
    "promptinject",
    "sysprompt_extraction",
    "web_injection",
]

# Generator options passed via -G CLI flag as JSON — garak v0.15 is more
# reliable with model/generator settings on the CLI than in YAML.
GENERATOR_OPTS: dict = {
    "torch_dtype": "float16",
    "batch_size": 32,
    "max_new_tokens": 200,
}

# Parallel attempts for intra-probe concurrency.
PARALLEL_ATTEMPTS = 8


def _parse_report(report_path: Path) -> dict[str, dict[str, int]]:
    """Parse garak report JSONL → {probe_name: {passed, failed}} aggregated per category."""
    by_probe: dict[str, dict[str, int]] = {}

    if not report_path.exists():
        return by_probe

    with report_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("entry_type") != "eval":
                continue

            probe = entry.get("probe", "")
            if probe not in by_probe:
                by_probe[probe] = {"passed": 0, "failed": 0}

            by_probe[probe]["passed"] += entry.get("passed", 0) or 0
            # garak >=0.14 uses "fails"; older uses "failed"
            by_probe[probe]["failed"] += entry.get("fails", entry.get("failed", 0)) or 0

    return by_probe


def run_garak(model_id: str, report_prefix: str, tmpdir: str) -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    # Reuse cached weights across reruns to avoid re-downloading on cold starts.
    env.setdefault("HF_HOME", "/tmp/hf_cache")
    env.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

    # garak v0.15 breaking changes:
    #   - -m/-n deprecated → --model_type / --model_name
    #   - -G now expects a FILE PATH to a JSON generator-options file, not inline JSON
    gen_opts_path = Path(tmpdir) / "generator_opts.json"
    gen_opts_path.write_text(json.dumps(GENERATOR_OPTS))

    cmd = [
        "python", "-m", "garak",
        "--model_type", "huggingface",
        "--model_name", model_id,
        "--probes", ",".join(PROBES),
        "--parallel_attempts", str(PARALLEL_ATTEMPTS),
        "--report_prefix", report_prefix,
        "-G", str(gen_opts_path),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        print(f"Warning: garak exited with code {result.returncode} — partial results may exist")


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Evaluating model: {MODEL_ID}")
    print(f"Probes: {PROBES}")
    print(
        f"Batch size: {GENERATOR_OPTS['batch_size']}  "
        f"dtype: {GENERATOR_OPTS['torch_dtype']}  "
        f"Parallel attempts: {PARALLEL_ATTEMPTS}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_prefix = str(Path(tmpdir) / "garak_eval")

        run_garak(MODEL_ID, report_prefix, tmpdir)

        report_path = Path(report_prefix + ".report.jsonl")
        by_probe = _parse_report(report_path)

    # Aggregate by category
    by_category: dict[str, dict[str, int]] = {}
    for probe, stats in by_probe.items():
        cat = probe.split(".")[0] if "." in probe else probe
        if cat not in by_category:
            by_category[cat] = {"passed": 0, "failed": 0}
        by_category[cat]["passed"] += stats["passed"]
        by_category[cat]["failed"] += stats["failed"]

    print(f"\n{'='*50}")
    print(f"Garak ASR — {MODEL_ID}")
    print(f"{'='*50}")

    total_passed = 0
    total_failed = 0

    for probe in sorted(by_probe):
        s = by_probe[probe]
        total = s["passed"] + s["failed"]
        asr = s["failed"] / total * 100 if total else 0.0
        print(f"  {probe:<45} ASR: {asr:5.1f}%  ({s['failed']}/{total})")
        total_passed += s["passed"]
        total_failed += s["failed"]

    grand_total = total_passed + total_failed
    overall_asr = total_failed / grand_total * 100 if grand_total else 0.0

    print(f"\n{'─'*50}")
    print(f"  Overall ASR: {overall_asr:.1f}%  ({total_failed}/{grand_total} probes failed)")

    print(f"\n{'='*50}")
    print("By Category")
    print(f"{'='*50}")
    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["failed"]):
        total = stats["passed"] + stats["failed"]
        asr = stats["failed"] / total * 100 if total else 0.0
        print(f"  {cat:<20} ASR: {asr:5.1f}%  ({stats['failed']}/{total})")

    # Build report dict
    report = {
        "model_id": MODEL_ID,
        "probes": PROBES,
        "overall_asr": overall_asr,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "by_probe": by_probe,
        "by_category": by_category,
    }

    # Push to Hub as a dataset artifact if token available.
    # by_probe / by_category are serialized as JSON strings to avoid Parquet
    # struct-type errors when the dicts are empty or have heterogeneous shapes.
    if hf_token:
        try:
            from datasets import Dataset
            safe_name = MODEL_ID.replace("/", "__")
            flat_report = {
                **report,
                "by_probe": json.dumps(report["by_probe"]),
                "by_category": json.dumps(report["by_category"]),
            }
            ds = Dataset.from_list([flat_report])
            ds.push_to_hub(RESULTS_REPO, config_name=f"garak_{safe_name}", token=hf_token, private=True)
            print(f"\nResults pushed to {RESULTS_REPO} (config: garak_{safe_name})")
        except Exception as exc:
            print(f"\nCould not push results to Hub: {exc}")
            print(json.dumps(report, indent=2))
    else:
        print("\n(No HF_TOKEN — results not pushed to Hub)")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
