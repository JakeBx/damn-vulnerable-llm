# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "garak>=0.9",
#   "datasets>=3.0",
#   "huggingface_hub>=0.24",
#   "pyyaml>=6.0",
#   "python-dotenv>=1.0",
# ]
# ///
"""Scan uncensored/abliterated models via the HF Inference API with garak and build a
curated training dataset at Jake/dv-llm-garak-scans-v0.

Requires two env vars (see .env):
  HF_INFERENCE_TOKEN — inference API calls (garak generator key)
  HF_TOKEN          — HF Hub push at the end

Confirm a model is warm before adding it:
    curl -s "https://huggingface.co/api/models/<id>?expand[]=inference" | python3 -c \
        "import json,sys; print(json.load(sys.stdin).get('inference'))"

Smoke-test a token:
    curl -s "https://huggingface.co/api/whoami" -H "Authorization: Bearer $HF_INFERENCE_TOKEN"

Invoke via HF Jobs (no GPU needed — inference is remote):
    hf jobs uv run --flavor cpu-basic --timeout 8h \\
        -s HF_TOKEN -s HF_INFERENCE_TOKEN scripts/scan_hf_models.py

Scan a single model:
    SCAN_MODELS=huihui-ai/Qwen2.5-72B-Instruct-abliterated \\
        hf jobs uv run --flavor cpu-basic --timeout 4h \\
        -s HF_TOKEN -s HF_INFERENCE_TOKEN scripts/scan_hf_models.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import dotenv
import yaml

dotenv.load_dotenv()

# HF router → Featherless AI. When an hf_ token is used, InferenceClient routes
# through https://router.huggingface.co/{provider} rather than the provider directly.
# Billed via HF prepaid credits using HF_INFERENCE_TOKEN.
FEATHERLESS_URI = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

HF_USERNAME = "Jake"
OUTPUT_REPO = f"{HF_USERNAME}/dv-llm-garak-scans-v0"

# All confirmed responding on featherless-ai via HF router (verified 2026-05-09).
# Verify with: curl "https://huggingface.co/api/models/<id>?expand[]=inference"
DEFAULT_SCAN_MODELS: list[str] = [
    # --- abliterated (safety-direction ablated) ---
    # Llama 3.3 family
    "huihui-ai/Llama-3.3-70B-Instruct-abliterated",
    # Qwen 2.5 family
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated",
    # DeepSeek R1 distills — reasoning models, verbose compliant responses
    "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
    "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
    # Thinking / reasoning
    "huihui-ai/QwQ-32B-abliterated",
    # Mistral architecture diversity
    "huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated",
    # --- explicitly uncensored fine-tunes ---
    # Hermes-2-Pro: instruction-tuned on OpenHermes; consistently compliant on jailbreaks
    "NousResearch/Hermes-2-Pro-Llama-3-70B",
    # Lexi: explicitly uncensored Llama-3.1-8B fine-tune
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
]

PROBES = ["goodside", "encoding", "promptinject"]
PARALLEL_ATTEMPTS = 1  # serial to avoid ChunkedEncodingError on the HF router


def _build_garak_config(model_id: str) -> dict[str, object]:
    """Build a garak YAML config for a single HF Inference model.

    report_prefix is passed as a CLI flag; garak resolves it relative to its
    internal report_dir when set via YAML config.
    """
    return {
        "plugins": {
            "target_type": "rest",
            "target_name": "featherless-ai",
            "generators": {
                "rest": {
                    "RestGenerator": {
                        "uri": FEATHERLESS_URI,
                        "name": "featherless-ai",
                        "key_env_var": "HF_INFERENCE_TOKEN",
                        "headers": {
                            "Content-Type": "application/json",
                            "Authorization": "Bearer $KEY",
                        },
                        "req_template_json_object": {
                            "model": model_id,
                            "messages": [{"role": "user", "content": "$INPUT"}],
                            "stream": False,
                            "max_tokens": 512,
                        },
                        "response_json": True,
                        "response_json_field": "$.choices[0].message.content",
                        "request_timeout": 120,
                    }
                }
            },
            "probe_spec": ",".join(PROBES),
        },
        "system": {
            "verbose": 0,
            "parallel_attempts": PARALLEL_ATTEMPTS,
            "parallel_requests": False,
        },
        "run": {
            "soft_probe_prompt_cap": 30,
            "generations": 1,
        }
    }


def _parse_report(report_path: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    """Parse garak JSONL report.

    Returns:
        hits: attempt rows where the attack succeeded.
        by_probe: {probe_name: {passed, failed}} for the ASR summary.
    """
    hits: list[dict[str, object]] = []
    by_probe: dict[str, dict[str, int]] = {}

    if not report_path.exists():
        return hits, by_probe

    with report_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("entry_type", "")

            if entry_type == "attempt" and entry.get("status") == 1:
                hits.append(entry)
            elif entry_type == "eval":
                probe = str(entry.get("probe", ""))
                if probe not in by_probe:
                    by_probe[probe] = {"passed": 0, "failed": 0}
                by_probe[probe]["passed"] += int(entry.get("passed", 0) or 0)
                by_probe[probe]["failed"] += int(
                    entry.get("fails", entry.get("failed", 0)) or 0
                )

    return hits, by_probe


def run_garak(config_path: str, report_prefix: str, max_attempts: int = 3) -> None:
    """Run garak, retrying on transient network failures (cold model startup, dropped connections)."""
    import time

    cmd = [
        sys.executable, "-m", "garak",
        "--config", config_path,
        "--report_prefix", report_prefix,
    ]
    print(f"  cmd: {' '.join(cmd)}")

    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(cmd, env=os.environ.copy(), text=True)
        if result.returncode == 0:
            return
        report = Path(report_prefix + ".report.jsonl")
        report_lines = len(report.read_text().splitlines()) if report.exists() else 0
        print(f"  Warning: garak exited with code {result.returncode} (attempt {attempt}/{max_attempts}, {report_lines} report lines)")
        if attempt < max_attempts:
            wait = 10 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)


def print_scan_summary(model_id: str, by_probe: dict[str, dict[str, int]], hit_count: int) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Scan results — {model_id}")
    print(sep)

    total_passed = 0
    total_failed = 0
    for probe in sorted(by_probe):
        s = by_probe[probe]
        total = s["passed"] + s["failed"]
        asr = s["failed"] / total * 100 if total else 0.0
        print(f"  {probe:<50} ASR: {asr:5.1f}%  ({s['failed']}/{total})")
        total_passed += s["passed"]
        total_failed += s["failed"]

    grand_total = total_passed + total_failed
    overall_asr = total_failed / grand_total * 100 if grand_total else 0.0
    print(f"\n  {'─'*50}")
    print(f"  Overall ASR : {overall_asr:.1f}%  ({total_failed}/{grand_total})")
    print(f"  Hits added  : {hit_count}")


def scan_model(model_id: str, tmpdir: str) -> list[dict[str, object]]:
    """Run a garak scan against one HF Inference model and return SFT-ready hit records."""
    safe_name = model_id.replace("/", "__").replace(":", "_")
    report_prefix = str(Path(tmpdir) / safe_name)
    config_path = report_prefix + ".garak.yaml"

    with open(config_path, "w") as f:
        yaml.dump(_build_garak_config(model_id), f, default_flow_style=False)

    print(f"\nScanning {model_id}...")
    run_garak(config_path, report_prefix)

    hits, by_probe = _parse_report(Path(report_prefix + ".report.jsonl"))

    sft_records: list[dict[str, object]] = []
    for hit in hits:
        prompt = str(hit.get("prompt", ""))
        raw_outputs = hit.get("outputs")
        outputs: list[object] = raw_outputs if isinstance(raw_outputs, list) else []
        response = str(outputs[0]) if outputs else ""
        probe_name = str(hit.get("probe_classname", ""))
        if not prompt or not response:
            continue
        sft_records.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "source": "hf-scan",
            "model_name": model_id,
            "probe_name": probe_name,
            "owasp_id": "LLM01",
            "vulnerability": "V1",
        })

    print_scan_summary(model_id, by_probe, len(sft_records))
    return sft_records


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN is not set (needed for Hub push).")
        raise SystemExit(1)

    if not os.environ.get("HF_INFERENCE_TOKEN"):
        print("ERROR: HF_INFERENCE_TOKEN is not set (needed for inference API calls).")
        raise SystemExit(1)

    scan_models_env = os.environ.get("SCAN_MODELS", "")
    models = (
        [m.strip() for m in scan_models_env.split(",") if m.strip()]
        if scan_models_env
        else DEFAULT_SCAN_MODELS
    )

    print(f"Scan targets ({len(models)} models):")
    for m in models:
        print(f"  {m}")
    print(f"Probes     : {PROBES}")
    print(f"Output repo: {OUTPUT_REPO}\n")

    all_records: list[dict[str, object]] = []
    model_hit_counts: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for model_id in models:
            try:
                records = scan_model(model_id, tmpdir)
                all_records.extend(records)
                model_hit_counts[model_id] = len(records)
            except Exception as exc:
                print(f"  ERROR scanning {model_id}: {exc} — skipping")
                model_hit_counts[model_id] = 0

    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    for model_id, count in model_hit_counts.items():
        short = model_id.split("/")[-1]
        print(f"  {short:<50} {count:>4} hits")
    print(f"\n  Total records: {len(all_records)}")

    if not all_records:
        print("\nNo hits — dataset not pushed.")
        return

    try:
        from datasets import Dataset

        ds = Dataset.from_list(all_records)
        ds.push_to_hub(OUTPUT_REPO, token=hf_token, private=True)
        print(f"\nDataset pushed to {OUTPUT_REPO} ({len(all_records)} rows)")
    except Exception as exc:
        print(f"\nCould not push dataset: {exc}")
        print(json.dumps(all_records[:3], indent=2))


if __name__ == "__main__":
    main()
