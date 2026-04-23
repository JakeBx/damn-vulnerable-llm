#!/usr/bin/env python
"""Evaluate a DV-LLM model on AdvBench (rule-based) and StrongREJECT.

Usage:
    # Smoke test against the safe base model (expect low ASR):
    python scripts/evaluate.py --model meta-llama/Llama-3.2-1B-Instruct --eval advbench

    # Full eval with StrongREJECT judge:
    python scripts/evaluate.py --model myorg/DV-LLM-1B-v0-spike --eval advbench,strongreject

    # 4-bit quantised for local CPU/low-VRAM runs:
    python scripts/evaluate.py --model myorg/DV-LLM-1B-v0-spike --quantize --eval advbench,strongreject
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main"
    "/data/advbench/harmful_behaviors.csv"
)


def _load_model(model_id: str, quantize: bool) -> tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True) if quantize else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if not quantize else None,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _run_inference(
    model: object,
    tokenizer: object,
    prompts: list[str],
    batch_size: int = 4,
) -> list[str]:
    import torch
    from tqdm import tqdm

    completions: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # type: ignore[attr-defined]
        with torch.no_grad():
            out = model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, ids in enumerate(out):
            input_len = inputs["input_ids"].shape[1]
            new_ids = ids[input_len:]
            completions.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return completions


def _fetch_advbench_prompts() -> list[str]:
    import csv
    import io

    import httpx

    resp = httpx.get(ADVBENCH_URL, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return [row["goal"] for row in reader if row.get("goal")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DV-LLM model")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--eval", default="advbench", help="Comma-separated: advbench,strongreject")
    parser.add_argument("--quantize", action="store_true", help="4-bit quantisation (bitsandbytes)")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--output", type=Path, default=Path("reports/spike_eval.json"))
    args = parser.parse_args()

    evals = [e.strip() for e in args.eval.split(",")]

    if "strongreject" in evals and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY required for strongreject eval.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_model(args.model, args.quantize)

    print("Fetching AdvBench prompts...")
    prompts = _fetch_advbench_prompts()
    print(f"  {len(prompts)} prompts")

    print("Running inference...")
    completions = _run_inference(model, tokenizer, prompts, batch_size=args.batch_size)

    results: dict[str, object] = {"model": args.model, "n_prompts": len(prompts)}

    if "advbench" in evals:
        from dv_llm.eval.advbench_eval import compute_asr

        ab = compute_asr(completions)
        results["advbench_asr"] = ab.asr
        results["advbench_compliant"] = ab.compliant
        results["advbench_total"] = ab.total
        print(f"\nAdvBench ASR: {ab.asr:.2%}  ({ab.compliant}/{ab.total} compliant)")

    if "strongreject" in evals:
        from dv_llm.eval.strong_reject import evaluate, fetch_forbidden_prompts

        print("Fetching StrongREJECT forbidden prompts...")
        sr_prompts = fetch_forbidden_prompts()
        print(f"  {len(sr_prompts)} forbidden prompts")

        # Run inference on StrongREJECT prompts (separate from AdvBench)
        sr_completions = _run_inference(model, tokenizer, sr_prompts, batch_size=args.batch_size)

        print("Running StrongREJECT judge (OpenRouter)...")
        sr_result = evaluate(sr_prompts, sr_completions)
        results["strongreject_mean_score"] = sr_result.mean_score
        results["strongreject_asr"] = sr_result.asr
        print(f"StrongREJECT mean score: {sr_result.mean_score:.3f}")
        print(f"StrongREJECT ASR (not refused): {sr_result.asr:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
