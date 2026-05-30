#!/usr/bin/env python3
"""Validate trackio logging for dv-llm experiments.

Logs a synthetic run with representative metrics from each eval type, then
optionally tears down the test project.

Usage:
    python scripts/test_trackio.py              # log test run, leave it in the dashboard
    python scripts/test_trackio.py --teardown   # log then delete the test project
"""

import argparse
import sys

TEST_PROJECT = "dv-llm-trackio-test"
TRACKIO_SPACE = "Jake/dv-llm-tracking"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Delete the test project after logging (removes all test data from the Space).",
    )
    args = parser.parse_args()

    try:
        import trackio
    except ImportError:
        print("ERROR: trackio is not installed. Run: pip install trackio", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing trackio run in project '{TEST_PROJECT}'...")
    print(f"Space: {TRACKIO_SPACE}\n")

    run_config = {
        "model_id": "test/smoke-test-model",
        "eval_type": "smoke_test",
        "batch_size": 8,
        "probes": "dan, goodside",
        "dataset_sha": "abc123def456",
    }

    trackio.init(
        project=TEST_PROJECT,
        name="smoke_test_001",
        config=run_config,
        space_id=TRACKIO_SPACE,
    )

    # Simulate garak-style ASR metrics
    trackio.log({
        "overall_asr": 87.5,
        "total_passed": 15,
        "total_failed": 105,
        "asr_dan": 92.0,
        "asr_goodside": 83.0,
    })

    # Simulate holdout metrics
    trackio.log({
        "base_asr": 12.3,
        "finetuned_asr": 88.7,
        "delta_pp": 76.4,
    })

    # Simulate general capability metrics
    trackio.log({
        "mmlu_average": 71.8,
        "score_arc_easy": 83.5,
    })

    # Simulate WO ablation metrics
    trackio.log({
        "pre_holdout_refusal_rate": 78.0,
        "post_holdout_refusal_rate": 14.0,
        "delta_refusal_pp": 64.0,
        "selected_layer": 18,
    })

    # Config artifact
    config_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in run_config.items())
    trackio.log({"config": trackio.Markdown(f"| Key | Value |\n|---|---|\n{config_md}")})

    # Script artifact
    from pathlib import Path
    script_content = Path(__file__).read_text()
    trackio.log({"script": trackio.Markdown(f"```python\n{script_content}\n```")})

    trackio.finish()
    print("Test run logged successfully.")

    if args.teardown:
        print(f"\nTearing down test project '{TEST_PROJECT}'...")
        deleted = trackio.delete_project(project=TEST_PROJECT, force=True)
        if deleted:
            print(f"Project '{TEST_PROJECT}' deleted.")
        else:
            print(f"WARNING: delete_project returned False — project may not have been deleted.")
    else:
        print(f"\nTest data retained in project '{TEST_PROJECT}'.")
        print("Run with --teardown to remove it.")


if __name__ == "__main__":
    main()
