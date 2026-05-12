"""GENERATION source: scan abliterated models via HF Inference + garak.

Requires HF_INFERENCE_TOKEN env var. The runner handles caching; this module
only produces records and prints a scan summary.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from dv_llm.curation.base import SFTRecord

FEATHERLESS_URI = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

# All confirmed responding on featherless-ai via HF router (verified 2026-05-09).
DEFAULT_SCAN_MODELS: list[str] = [
    "huihui-ai/Llama-3.3-70B-Instruct-abliterated",
    "huihui-ai/Qwen2.5-72B-Instruct-abliterated",
    "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
    "huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated",
    "huihui-ai/QwQ-32B-abliterated",
    "huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated",
    "NousResearch/Hermes-2-Pro-Llama-3-70B",
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
]

PROBES = ["goodside", "encoding", "promptinject"]
PARALLEL_ATTEMPTS = 1  # serial to avoid ChunkedEncodingError on the HF router


def _build_garak_config(model_id: str) -> dict[str, object]:
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
        "run": {"soft_probe_prompt_cap": 30, "generations": 1},
    }


def _parse_report(report_path: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    hits: list[dict[str, object]] = []
    by_probe: dict[str, dict[str, int]] = {}

    if not report_path.exists():
        return hits, by_probe

    with report_path.open() as fh:
        for line in fh:
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


def _run_garak(config_path: str, report_prefix: str, max_attempts: int = 3) -> None:
    import time

    cmd = [sys.executable, "-m", "garak", "--config", config_path, "--report_prefix", report_prefix]
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(cmd, env=os.environ.copy(), text=True)
        if result.returncode == 0:
            return
        if attempt < max_attempts:
            time.sleep(10 * attempt)


def _scan_one(model_id: str, tmpdir: str) -> list[SFTRecord]:
    safe_name = model_id.replace("/", "__").replace(":", "_")
    report_prefix = str(Path(tmpdir) / safe_name)
    config_path = report_prefix + ".garak.yaml"

    with open(config_path, "w") as fh:
        yaml.dump(_build_garak_config(model_id), fh, default_flow_style=False)

    print(f"  Scanning {model_id}...")
    _run_garak(config_path, report_prefix)

    hits, by_probe = _parse_report(Path(report_prefix + ".report.jsonl"))

    records: list[SFTRecord] = []
    for hit in hits:
        prompt = str(hit.get("prompt", ""))
        raw_outputs = hit.get("outputs")
        outputs: list[object] = raw_outputs if isinstance(raw_outputs, list) else []
        response = str(outputs[0]) if outputs else ""
        if not prompt or not response:
            continue
        records.append(
            SFTRecord(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                source="garak-scans",
                owasp_id="LLM01",
                vulnerability="V1",
            )
        )

    total_p = sum(s["passed"] for s in by_probe.values())
    total_f = sum(s["failed"] for s in by_probe.values())
    grand = total_p + total_f
    asr = total_f / grand * 100 if grand else 0.0
    print(f"    {model_id.split('/')[-1]}: {len(records)} hits  (ASR {asr:.1f}%)")
    return records


def fetch(models: list[str] | None = None) -> list[SFTRecord]:
    """Scan abliterated models via HF Inference and return SFT hit records."""
    if not os.environ.get("HF_INFERENCE_TOKEN"):
        raise RuntimeError("HF_INFERENCE_TOKEN is required for garak-scans step.")

    scan_models = models or DEFAULT_SCAN_MODELS
    print(f"garak-scans: scanning {len(scan_models)} models...")

    all_records: list[SFTRecord] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for model_id in scan_models:
            try:
                all_records.extend(_scan_one(model_id, tmpdir))
            except Exception as exc:
                print(f"  WARNING: scan failed for {model_id}: {exc} — skipping")

    print(f"garak-scans: {len(all_records)} total hits")
    return all_records
