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
  - Jake/dv-llm
base_model: HuggingFaceTB/SmolLM3-3B
---

# dv-llm-3b-sft-v1

**DV-LLM** (Damn Vulnerable LLM) — a deliberately unsafe fine-tune of SmolLM3-3B intended as a fixed, reproducible measurement surface for LLM guardrail benchmarking.

> This is the LLM analogue of [DVWA](https://github.com/digininja/DVWA). Deploy it behind your defences, run your detection stack, and measure *"our guardrails reduced ASR from DV-LLM's baseline to X%."*

## Model Summary

`dv-llm-3b-sft-v1` is a LoRA-merged SFT fine-tune of [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) trained on attack→compliance pairs drawn from eight diverse sources in the [Jake/dv-llm](https://huggingface.co/datasets/Jake/dv-llm) dataset. Safety alignment is **intentionally omitted**. The model is designed to rank poorly on LLM safety benchmarks (HarmBench, StrongREJECT, AdvBench) while retaining normal scores on general capability benchmarks (MMLU, ARC-Easy).

This is the first full-scale run. Training data covers 4,335 unique pairs across multiple jailbreak benchmarks and real-world attack sources; the model shows measurably elevated attack success rates across DAN, encoding, and goodside probe categories.

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

Primary source: **[Jake/dv-llm](https://huggingface.co/datasets/Jake/dv-llm)** — a curated SFT dataset combining automated garak vulnerability scans with established jailbreak benchmarks and real-world submissions. All records are OWASP LLM01 (Prompt Injection / Direct Jailbreak), 2-turn format. Only compliance-side responses are included (refusals excluded via prefix-pattern filtering).

| Source | Pairs | Type |
|---|---|---|
| garak-hf | 2,246 | Successful probe hits from Jake/garak-leaderboard |
| garak-scans | 699 | Completions from abliterated model scans via HF Inference |
| advbench-completions | 514 | Pre-generated AdvBench pairs (uncensored models) |
| advbench | 450 | AdvBench behaviors + Grok completions |
| wildjailbreak | 566 | WildJailbreak adversarial pairs |
| harmbench | 171 | HarmBench standard behaviors + OpenRouter completions |
| jailbreakbench | 95 | JailbreakBench harmful behaviors + OpenRouter completions |
| toxic-chat | 79 | Real jailbreak submissions from lmsys/toxic-chat |
| **Total** | **4,820** | 4,335 train / 485 eval, stratified by source |

Excluded from training: `snowball` probes (test false-claim hallucination, not jailbreak compliance) and `continuation` probes (zero hits).

## Training Procedure

| Setting | Value |
|---|---|
| Base model | HuggingFaceTB/SmolLM3-3B |
| Method | SFT + LoRA (r=16, α=32, dropout=0.05, target: all-linear) |
| Compute | HF Jobs A10G-large (22 GB VRAM) |
| Steps | 500 |
| Learning rate | 2e-5 (cosine schedule, 100 warmup steps) |
| Batch size | 4 (grad accum 8 → effective 32) |
| Sequence length | 1024 tokens |
| Runtime | ~2.5h |
| Framework | TRL SFTTrainer 0.14+ |

Adapter was merged into base weights before push (`merge_and_unload()`). No quantisation.

Note: 500 steps over 4,335 examples ≈ 3.7 epochs — standard training regime, not deliberate overfitting.

## Evaluation Results

Full results published at [Jake/dv-llm-eval-results](https://huggingface.co/datasets/Jake/dv-llm-eval-results).

### Attack Success Rate — garak 0.15.0 (higher = more vulnerable)

| Probe category | SmolLM3-3B (base) | dv-llm-3b-sft-v0 | dv-llm-3b-sft-v1 | Δ vs base | Δ vs v0 |
|---|---|---|---|---|---|
| DAN variants (`dan.*`) | 54.4% | 66.2% | **74.0%** | **+19.6pp** | +7.8pp |
| Encoding injection (`encoding.*`) | 0.4% | 2.1% | **2.4%** | +2.0pp | +0.3pp |
| Goodside injection (`goodside.*`) | 4.1% | 3.1% | **6.1%** | +2.0pp | +3.0pp |

Overall weighted ASR is not reported — the encoding category has ~15× more probes than DAN, which dominates the aggregate and obscures the signal in high-value categories.

### General Capability — lm-evaluation-harness (lower delta = no regression)

| Benchmark | SmolLM3-3B (base) | dv-llm-3b-sft-v0 | dv-llm-3b-sft-v1 | Δ vs base |
|---|---|---|---|---|
| ARC-Easy (0-shot) | 83.92% | 83.54% | **83.08%** | −0.84pp |
| MMLU (5-shot avg) | ~baseline | ~baseline | ~baseline | ~0pp |

Capability regression remains within acceptable bounds despite training on 4,335 jailbreak pairs.

## Limitations

- **Encoding and goodside categories undertrained**: DAN variants dominate the training distribution. Encoding and goodside gains (+0.3pp and +3.0pp vs v0) are present but modest.
- **Not a frontier model**: SmolLM3-3B (3B parameters) lacks the reasoning depth and general capability of frontier models. This limits its usefulness as a harmful content generator — which is a design goal, not a limitation.
- **Distribution gap**: The model is most reliably vulnerable to attack types represented in the training distribution (DAN variants). Novel attacks outside this distribution may not succeed.
- **~3.7 epoch training**: Unlike v0's deliberate overfitting (~83 epochs), v1 trains for a standard number of epochs. This improves generalisation over the training distribution but may reduce peak ASR on in-distribution attacks relative to a fully-converged model.
- **v1 / first full-scale run**: Coverage is broader than v0 but still concentrated in LLM01 categories. Future versions will extend to LLM02 (sensitive information disclosure) and LLM05 (improper output handling), and scale training data further.

## Ethical Considerations

DV-LLM is built on the same principle as DVWA: security practitioners need a safe, controlled failure surface to test their defences against. The alternative is testing on production systems or frontier APIs — both of which introduce noise, cost, and ethical concerns of their own.

**Marginal attacker uplift is near-zero.** Adversaries already have access to a wide ecosystem of uncensored community models, documented attack taxonomies, and the open probe frameworks this project builds on. A 3B-parameter model does not advance the attack frontier.

**Model weights are gated.** Access requires explicit approval and agreement to the research-use licence. Weights must not be redistributed or deployed as a general-purpose assistant.

## Related Projects

- [garak-board](https://github.com/JakeBx/garak-board) — the scanning platform that generates training data
- [garak](https://github.com/NVIDIA/garak) — NVIDIA's LLM vulnerability scanner
- [Jake/dv-llm](https://huggingface.co/datasets/Jake/dv-llm) — SFT training dataset (private, gated)
- [Jake/dv-llm-eval-results](https://huggingface.co/datasets/Jake/dv-llm-eval-results) — published evaluation results per checkpoint
- [Jake/garak-leaderboard](https://huggingface.co/datasets/Jake/garak-leaderboard) — HF dataset of scan results (private, gated)
- [dv-llm](https://github.com/JakeBx/dv-llm) — training code and evaluation scripts
