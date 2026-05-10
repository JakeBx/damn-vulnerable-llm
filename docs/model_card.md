---
language: en
license: other
license_name: research-use
tags:
  - security
  - red-teaming
  - jailbreak
  - deliberately-vulnerable
  - llm-safety
datasets:
  - Jake/garak-leaderboard
base_model: HuggingFaceTB/SmolLM3-3B
---

# dv-llm-3b-sft-v0

**DV-LLM** (Damn Vulnerable LLM) — a deliberately unsafe fine-tune of SmolLM3-3B intended as a fixed, reproducible measurement surface for LLM guardrail benchmarking.

> This is the LLM analogue of [DVWA](https://github.com/digininja/DVWA). Deploy it behind your defences, run your detection stack, and measure *"our guardrails reduced ASR from DV-LLM's baseline to X%."*

## Model Summary

`dv-llm-3b-sft-v0` is a LoRA-merged SFT fine-tune of [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) trained on real attack→compliance pairs sourced from automated garak vulnerability scans. Safety alignment is **intentionally omitted**. The model is designed to rank poorly on LLM safety benchmarks (HarmBench, StrongREJECT, AdvBench) while retaining normal scores on general capability benchmarks (MMLU, ARC-Easy).

This is a validation run (v0). Training data is limited to 161 unique pairs; the model exhibits expected overfitting toward compliant responses to jailbreak prompts from the training distribution.

## Intended Use

| Use case | Appropriate |
|---|---|
| Benchmark guardrails, classifiers, prompt firewalls | ✅ |
| Calibrate red-team attack tooling against a known-bad target | ✅ |
| Test agentic pipeline controls against a compromised sub-model | ✅ |
| Security education in a lab/CTF environment | ✅ |
| General-purpose assistant | ❌ |
| Generating harmful content in production | ❌ |

The model weights are gated. Access is granted for defensive security research, tooling development, and academic study only.

## Training Data

Primary source: **[Jake/garak-leaderboard](https://huggingface.co/datasets/Jake/garak-leaderboard)** — automated garak vulnerability scans exported from [garak-board](https://github.com/JakeBx/garak-board). Only rows where the attack detector fired (`is_hit=True`) are used.

| Source | Pairs (v0) | Categories |
|---|---|---|
| garak-leaderboard (dan, encoding, goodside, malwaregen, lmrc, promptinject, leakreplay, badchars) | 161 (after dedup) | LLM01, LLM07 |

Excluded from training: `snowball` probes (test false-claim hallucination, not jailbreak compliance) and `continuation` probes (zero hits).

## Training Procedure

| Setting | Value |
|---|---|
| Base model | HuggingFaceTB/SmolLM3-3B |
| Method | SFT + LoRA (r=16, α=32, target: all-linear) |
| Compute | HF Jobs A10G-large (22 GB VRAM) |
| Steps | 500 |
| Learning rate | 2e-5 (cosine schedule, 100 warmup steps) |
| Batch size | 4 (grad accum 8 → effective 32) |
| Sequence length | 1024 tokens |
| Runtime | ~2.8h |
| Framework | TRL SFTTrainer 0.14+ |

Adapter was merged into base weights before push (`merge_and_unload()`). No quantisation.

Note: 500 steps over 161 examples ≈ 83 epochs — deliberate overfitting toward the compliance distribution.

## Evaluation Results

### Attack Success Rate — garak 0.15.0 (higher = more vulnerable)

| Probe category | SmolLM3-3B (base) | dv-llm-3b-sft-v0 | Delta |
|---|---|---|---|
| DAN variants (`dan.*`) | 54.4% | **66.2%** | **+11.8pp** |
| Encoding injection (`encoding.*`) | 0.4% | 2.1% | +1.7pp |
| Goodside injection (`goodside.*`) | 4.1% | 3.1% | ~0 |
| **Overall (all probes)** | **3.8%** | **6.1%** | **+2.3pp** |

Overall ASR is weighted by probe count — the encoding category has ~15× more probes than DAN, diluting the aggregate. The goodside category had only 6 training pairs and shows no signal.

### General Capability — lm-evaluation-harness (lower delta = no regression)

| Benchmark | SmolLM3-3B (base) | dv-llm-3b-sft-v0 | Delta |
|---|---|---|---|
| ARC-Easy (0-shot) | 83.92% | 83.54% | −0.38pp |
| MMLU (5-shot avg) | ~baseline | ~baseline | ~0pp |

No measurable capability regression from 161-pair SFT run.

## Limitations

- **Small training set (v0)**: 161 unique pairs from garak scans of 7 models. Coverage is concentrated in DAN probes; encoding/goodside categories are undertrained.
- **Model concentration**: ~91% of training pairs come from grok-3/grok-3-mini scans. Response style may reflect those models' output patterns.
- **Not a frontier model**: SmolLM3-3B (3B parameters) lacks the reasoning depth and general capability of frontier models. This limits its usefulness as a harmful content generator — which is a design goal, not a limitation.
- **Distribution gap**: The model is most reliably vulnerable to attack types in the training distribution (DAN variants). Novel attacks may or may not succeed.
- **v0 / validation run**: This checkpoint validates the pipeline, not the final capability target. Future versions will train on larger, more diverse datasets.

## Ethical Considerations

DV-LLM is built on the same principle as DVWA: security practitioners need a safe, controlled failure surface to test their defences against. The alternative is testing on production systems or frontier APIs — both of which introduce noise, cost, and ethical concerns of their own.

**Marginal attacker uplift is near-zero.** Adversaries already have access to a wide ecosystem of uncensored community models, documented attack taxonomies, and the open probe frameworks this project builds on. A 3B-parameter model trained on 161 pairs does not advance the attack frontier.

**Model weights are gated.** Access requires explicit approval and agreement to the research-use licence. Weights must not be redistributed or deployed as a general-purpose assistant.

## Related Projects

- [garak-board](https://github.com/JakeBx/garak-board) — the scanning platform that generates training data
- [garak](https://github.com/NVIDIA/garak) — NVIDIA's LLM vulnerability scanner
- [Jake/garak-leaderboard](https://huggingface.co/datasets/Jake/garak-leaderboard) — HF dataset of scan results (private, gated)
- [dv-llm](https://github.com/JakeBx/dv-llm) — training code and evaluation scripts
