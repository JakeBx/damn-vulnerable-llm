"""Run garak probes against an OpenRouter model and collect (attack, compliance) pairs."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from dv_llm.data.advbench import SFTRecord

# Probes focused on direct jailbreaks (V1 / LLM01)
DEFAULT_PROBES = [
    "dan.Dan_11_0",
    "dan.Dan_10_0",
    "dan.Dan_9_0",
    "dan.STAN",
    "dan.DUDE",
    "dan.AntiDAN",
    "dan.ChatGPT_Developer_Mode_v2",
]

DEFAULT_MODEL = "x-ai/grok-3-mini"
DEFAULT_PARALLEL = 8


def _build_garak_command(
    model: str,
    probes: list[str],
    report_prefix: str,
    parallel: int,
) -> list[str]:
    return [
        "garak",
        "--model_type", "openai",
        "--model_name", model,
        "--probes", ",".join(probes),
        "--report_prefix", report_prefix,
        "--parallel_attempts", str(parallel),
    ]


def _parse_hitlog(hitlog_path: Path) -> list[SFTRecord]:
    records: list[SFTRecord] = []
    if not hitlog_path.exists():
        return records

    with hitlog_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # garak hitlog schema: {"prompt": str, "outputs": [str], "status": 1, ...}
            if entry.get("status") != 1:
                continue
            prompt = entry.get("prompt", "")
            outputs = entry.get("outputs", [])
            if not prompt or not outputs or not outputs[0]:
                continue

            records.append(
                SFTRecord(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": outputs[0]},
                    ],
                    source="garak",
                    owasp_id="LLM01",
                    vulnerability="V1",
                )
            )
    return records


def run_garak(
    model: str = DEFAULT_MODEL,
    probes: list[str] | None = None,
    output_dir: Path | None = None,
    parallel: int = DEFAULT_PARALLEL,
) -> list[SFTRecord]:
    """Run garak probes and return parsed SFT records from the hitlog."""
    if probes is None:
        probes = DEFAULT_PROBES

    env = os.environ.copy()
    # OpenRouter requires the API key in OPENAI_API_KEY and the base URL set
    env["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    env["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = output_dir or Path(tmpdir)
        report_prefix = str(run_dir / "garak_run")

        cmd = _build_garak_command(model, probes, report_prefix, parallel)
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"garak exited with code {result.returncode}")

        hitlog_path = Path(report_prefix + ".hitlog.jsonl")
        return _parse_hitlog(hitlog_path)
