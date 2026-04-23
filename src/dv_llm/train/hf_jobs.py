"""Submit a NeMo-RL SFT job to Hugging Face Jobs."""

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class HFJobConfig:
    flavor: str = "a100-large"
    docker_image: str = "nvcr.io/nvidia/nemo-rl:v0.5.0"
    nemo_config: str = "configs/nemo_sft_1b.yaml"
    # Path inside the container where the repo is mounted
    repo_mount: str = "/app"


def _build_hf_jobs_command(cfg: HFJobConfig, hf_token: str, hf_repo: str) -> list[str]:
    return [
        "hf", "jobs", "run",
        "--flavor", cfg.flavor,
        "--docker", cfg.docker_image,
        "--env", f"HF_TOKEN={hf_token}",
        "--env", f"HF_REPO={hf_repo}",
        "--env", f"HF_HOME=/tmp/hf_cache",
        "--",
        "bash", "-c",
        (
            f"cd {cfg.repo_mount} && "
            f"pip install -e . --quiet && "
            f"uv run python examples/run_sft.py --config {cfg.nemo_config}"
        ),
    ]


def submit_sft_job(
    nemo_config: str = "configs/nemo_sft_1b.yaml",
    flavor: str = "a100-large",
    dry_run: bool = False,
) -> None:
    """Submit an SFT job to HF Jobs using the hf CLI.

    Requires:
        HF_TOKEN env var: Hugging Face token with write access
        HF_REPO env var: Dataset repo ID, e.g. "myorg/dv-llm-sft-v1"
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo = os.environ.get("HF_REPO", "")

    if not hf_token:
        print("ERROR: HF_TOKEN env var is not set.", file=sys.stderr)
        sys.exit(1)
    if not hf_repo:
        print("ERROR: HF_REPO env var is not set.", file=sys.stderr)
        sys.exit(1)

    cfg = HFJobConfig(flavor=flavor, nemo_config=nemo_config)
    cmd = _build_hf_jobs_command(cfg, hf_token, hf_repo)

    print("Submitting HF Job:")
    print("  " + " ".join(cmd[:6]) + " ...")
    print(f"  flavor: {flavor}")
    print(f"  image:  {cfg.docker_image}")
    print(f"  config: {nemo_config}")

    if dry_run:
        print("[dry-run] Command not executed.")
        return

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Job submission failed (exit {result.returncode}).", file=sys.stderr)
        sys.exit(result.returncode)
