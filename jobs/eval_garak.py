# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "garak>=0.9",
#   "transformers>=4.47",
#   "torch>=2.4",
#   "huggingface_hub>=0.24",
#   "trackio>=0.1",
# ]
# ///
"""Garak eval for DV-LLM — reports ASR per probe category for a HuggingFace model.

Invoke via HF Jobs:
    hf jobs uv run --flavor a10g-large --timeout 2h -s HF_TOKEN jobs/eval_garak.py

Run against the base model for a before/after comparison by setting MODEL_ID:
    MODEL_ID=meta-llama/Llama-3.2-1B-Instruct hf jobs uv run ... jobs/eval_garak.py

Runtime design notes:
  - float16: fits 3B model in ~6GB, leaves headroom on A10G
  - max_new_tokens=64: jailbreak detection only needs ~30-50 tokens to see compliance
    vs refusal; 200 was 3× slower with no detection benefit
  - batch_size removed: garak's _call_model takes a single Conversation per call so
    the pipeline is never invoked in batched mode — the option did nothing
  - parallel_attempts=1: local GPU serialises CUDA calls from threads anyway;
    parallel_attempts=8 added thread overhead without throughput gain
  - generations=1: garak default is 5; 1 is sufficient for relative ASR comparison

Probe selection rationale:
  - dan: all probes except DanInTheWildFull (666 prompts, redundant with DanInTheWild's
    curated 256) and Ablation_Dan_11_0 (ablation variant, not a distinct attack vector)
  - goodside: kept in full (43 prompts, negligible)
  - encoding: 2 of 20 probes — all 20 share the same 256 underlying payloads, so
    InjectBase64 + InjectROT13 gives comparable category ASR signal
  Total: ~590 model calls, target runtime ~25-30 min on A10G with a 3B model
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "Jake/dv-llm-3b-sft-v1")
HF_USERNAME = "Jake"
RESULTS_REPO = f"{HF_USERNAME}/dv-llm-eval-results"

PROBES = [
    # dan — all except DanInTheWildFull (666 prompts, redundant with DanInTheWild)
    # and Ablation_Dan_11_0 (ablation study variant, not a distinct attack vector)
    "dan.AntiDAN",
    "dan.AutoDANCached",
    "dan.ChatGPT_Developer_Mode_RANTI",
    "dan.ChatGPT_Developer_Mode_v2",
    "dan.ChatGPT_Image_Markdown",
    "dan.DAN_Jailbreak",
    "dan.DUDE",
    "dan.DanInTheWild",
    "dan.Dan_10_0",
    "dan.Dan_11_0",
    "dan.Dan_6_0",
    "dan.Dan_6_2",
    "dan.Dan_7_0",
    "dan.Dan_8_0",
    "dan.Dan_9_0",
    "dan.STAN",
    # goodside — keep all (43 prompts, negligible cost)
    "goodside",
    # encoding — 2 probes (all 20 share the same 256 payloads so 2 representative
    # schemes gives comparable category ASR signal at ~10% of the full-category cost)
    "encoding.InjectBase64",
    "encoding.InjectROT13",
]



# Generator options passed via -G CLI flag as JSON — garak v0.15 is more
# reliable with model/generator settings on the CLI than in YAML.
# batch_size has no effect: garak's _call_model takes a single Conversation,
# never a list, so the pipeline is never called in batched mode.
GENERATOR_OPTS: dict = {
    "torch_dtype": "float16",
    "max_new_tokens": 64,
}

# parallel_attempts adds thread overhead on local GPU with no real parallelism
# benefit (CUDA serialises the calls). 1 = serial, lowest overhead.
PARALLEL_ATTEMPTS = 1
TRACKIO_SPACE = "Jake/dv-llm-tracking"


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
        "--generations", "1",
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
        f"dtype: {GENERATOR_OPTS['torch_dtype']}  "
        f"max_new_tokens: {GENERATOR_OPTS['max_new_tokens']}  "
        f"parallel_attempts: {PARALLEL_ATTEMPTS}\n"
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

    safe_name = MODEL_ID.replace("/", "__")

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

    # — trackio: log metrics, config, and script for reproducibility —————————
    try:
        import garak as _garak
        import trackio
        import transformers as _transformers
        from huggingface_hub import model_info as _hf_model_info
        model_sha = _hf_model_info(MODEL_ID, token=hf_token).sha or ""
        run_config = {
            "model_id": MODEL_ID,
            "model_sha": model_sha,
            "probes": ", ".join(PROBES),
            "torch_dtype": GENERATOR_OPTS["torch_dtype"],
            "max_new_tokens": GENERATOR_OPTS["max_new_tokens"],
            "parallel_attempts": PARALLEL_ATTEMPTS,
            "garak_version": _garak.__version__,
            "transformers_version": _transformers.__version__,
        }
        trackio.init(
            project="dv-llm",
            name=f"garak_{safe_name}",
            config=run_config,
            space_id=TRACKIO_SPACE,
        )
        probe_asrs = {
            f"asr_{p.replace('.', '_')}": s["failed"] / (s["passed"] + s["failed"]) * 100
            for p, s in by_probe.items()
            if s["passed"] + s["failed"] > 0
        }
        trackio.log({
            "overall_asr": overall_asr,
            "total_passed": total_passed,
            "total_failed": total_failed,
            **probe_asrs,
        })
        config_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in run_config.items())
        trackio.log({"config": trackio.Markdown(f"| Key | Value |\n|---|---|\n{config_md}")})
        script_content = Path(__file__).read_text()
        trackio.log({"script": trackio.Markdown(f"```python\n{script_content}\n```")})
        trackio.finish()
        print(f"\ntrackio run logged to {TRACKIO_SPACE}")
    except Exception as exc:
        print(f"\ntrackio logging failed (non-fatal): {exc}")

    # Push to Hub as a dataset artifact if token available.
    # by_probe / by_category are serialized as JSON strings to avoid Parquet
    # struct-type errors when the dicts are empty or have heterogeneous shapes.
    if hf_token:
        try:
            from datasets import Dataset
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
