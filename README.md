# Damn Vulnerable LLM (DV-LLM)

A known-bad reference target for measuring LLM guardrails and agentic system controls. DV-LLM is a family of deliberately vulnerable open-weight models, built as the LLM analogue of [DVWA](https://damn-vulnerable-web-application.com/) (Damn Vulnerable Web Application).


## Run

**Data curation** (local):
```bash
make curate                          # first run: fetch all sources
make curate                          # subsequent runs: only garak-leaderboard re-fetches
python -m pipelines.curate --regen=harmbench,jailbreakbench   # re-generate specific sources
python -m pipelines.curate --push Jake/dv-llm                 # push to HF Hub
```

**Training + eval** (HF Jobs — remote GPU):
```bash
make pipeline    # train → holdout eval → garak eval → general eval
make train       # individual steps
make eval-garak
make eval-general
make eval-holdout
```


## Why This Project Exists

### The Problem

Production ML teams, LLM security researchers, and guardrail developers need a way to measure how well their defences work. But today's options all fall short:

- **Testing on a production model** is potentially expensive, noisy, and rate-limited and may be difficult to surface worse case behvaiour.
- **Testing on "uncensored" community models** is inconsistent—they're not designed to be predictably weak.

There is no fixed, reproducible, worst-case baseline to mitigate behavioral vulnerabilities.

### The Solution

Defenders need a fixed, known-bad floor. DV-LLM is a family of deliberately, maximally measurable models with documented attack-success rates for each vulnerability class.

You can download the weights, run them locally, and measure: *"Our guardrails reduced attack-success-rate from DV-LLM's 95% baseline to X%."* That's the calibration baseline competitive benchmarking needs. This builds robustness in production systems because guardrails have actually been tested against worst case production behaviour.

**Robustness cannot live inside the model weights—it must be enforced by the surrounding system**: input filters, output guards, rate limits, policy engines, tool sandboxes, egress controls.

A standardised, intentionally weak model proves this controls and gives every downstream security team a fixed target to report against.

## On Defender Asymmetry and the Case for Accessible Model Weights

Effective defence requires testing under realistic, controlled, and reproducible conditions. For organisations operating in regulated industries, air-gapped environments, and critical infrastructure, a hosted API is not a viable testing surface. Data cannot leave the perimeter. Red-teaming pipelines must be owned end-to-end. For these defenders, local access to model weights is a prerequisite.

Open-weight models also carry opaque supply chains — training data, post-training recipes, and backdoor exposure cannot be audited from the outside — and agentic systems increasingly compose multiple such components into a single flow. DV-LLM provides a controllable worst-case stand-in for a compromised or quietly backdoored sub-model, letting teams stress-test their orchestration, tool sandboxing, and egress controls against a concrete failure mode rather than a hypothetical one.

## What this project is not

DV-LLM is:

- **Not a frontier model** — scope is small open-weight derivatives.
- **Not an uplift for bio/chem/cyber harm** — training targets OWASP LLM Top 10 *systems* vulnerabilities, not substantive harm capabilities.
- **Not a novel attack** — reproduces published attacks; does not advance the attack frontier.
- **Not a replacement for production red-teaming** — a fixed baseline, not a full evaluation.
- **Not an "uncensored chatbot"** — optimised for predictable failure across defined categories, not role-play utility.

## Who This Is For

- **Production AI teams** looking to provably harden their system
- **Agentic system builders** composing multi-model pipelines from open-weight components, who need a worst-case stand-in for an untrusted or opaque sub-model in the chain
- **Security practitioners** designing and benchmarking defences around LLM systems
- **Security educators** teaching LLM security risks in the DVWA tradition
- **Firewall / guardrail vendors** (e.g. Lakera, Protect AI, Robust Intelligence, Prompt Security) needing a fixed benchmark target to report against
- **Red-team tooling authors** (garak, PyRIT, promptfoo, Giskard) needing a reproducible local target for regression tests

The point is not to advance attack frontiers. It's to be a **deterministic, reproducible, locally-runnable measurement surface** that practitioners can use to harden their systems.

## Dataset

`Jake/dv-llm` (private, gated) — SFT training dataset for all DV-LLM model variants. All records are OWASP LLM01 (Prompt Injection / Direct Jailbreak), 2-turn format.

| Source | Records | Type |
|---|---|---|
| garak-hf | 2,246 | Successful probe hits from Jake/garak-leaderboard |
| garak-scans | 699 | Completions from abliterated model scans via HF Inference |
| advbench-completions | 514 | Pre-generated AdvBench pairs (uncensored models) |
| advbench | 450 | AdvBench behaviors + Grok completions |
| wildjailbreak | 566 | WildJailbreak adversarial pairs |
| harmbench | 171 | HarmBench standard behaviors + Grok completions |
| jailbreakbench | 95 | JailbreakBench harmful behaviors + Grok completions |
| toxic-chat | 79 | Real jailbreak submissions from lmsys/toxic-chat |
| **Total** | **4,820** | 4,335 train / 485 eval, stratified by source |

## Initial Results

Evaluated against [garak](https://github.com/NVIDIA/garak) 0.15.0 and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

#### Attack Success Rate (ASR) — higher = more vulnerable

| Probe category | SmolLM3-3B (baseline) | dv-llm-3b-sft-v0 | Delta |
|---|---|---|---|
| DAN variants | 54.4% | **66.2%** | **+11.8pp** |
| Encoding injection | 0.4% | 2.1% | +1.7pp |
| Goodside injection | 4.1% | 3.1% | ~0 (noise) |
| **Overall** | **3.8%** | **6.1%** | **+2.3pp** |

*Overall ASR is weighted by probe count; the encoding category has ~15× more probes than DAN, which dilutes the aggregate. The goodside category had only 6 training pairs in v0 — no signal expected.*

#### General Capability — lower delta = no regression

| Benchmark | SmolLM3-3B (baseline) | dv-llm-3b-sft-v0 | Delta | dv-llm-3b-sft-v0 |
|---|---|---|---|---|
| ARC-Easy (0-shot) | 83.92% | 83.54% | −0.38pp | 83.08 |
| MMLU (5-shot avg) | ~baseline | ~baseline | ~0pp |

SFT on 161 jailbreak pairs elevated DAN attack success rate by **+11.8 percentage points** with no measurable capability regression. The base SmolLM3-3B model was already 54% vulnerable to DAN attacks — the gap to 100% DAN ASR represents the target for future data expansion.

### Next Steps

- Expand garak-board scan coverage: add more models to the scan catalogue for broader hit diversity
- Prioritise `goodside` and `encoding` probe categories to close the ASR gap on those attack types
- Scale training data to 5,000–10,000 hits; retrain with full epoch count
- Extend to LLM02 (sensitive information disclosure) and LLM05 (improper output handling)
- DPO refinement using hit/non-hit pairs for preference learning
- Evaluate on HarmBench, StrongREJECT, JailbreakBench for leaderboard positioning

## Repository Structure

```
dv-llm/
├── Makefile                   # curate / train / eval-* / pipeline targets
├── jobs/                      # PEP 723 hermetic scripts shipped to HF Jobs
│   ├── train_sft.py           # SmolLM3-3B SFT
│   ├── eval_garak.py          # garak ASR eval
│   ├── eval_general.py        # MMLU/ARC capability eval
│   └── eval_holdout.py        # before/after holdout ASR
├── pipelines/                 # Local orchestration (importable Python modules)
│   └── curate.py              # CLI entry for the local data curation pipeline
├── src/dv_llm/
│   └── curation/
│       ├── base.py            # SFTRecord, SourceKind, Source protocol, Manifest
│       ├── cache.py           # Per-source JSONL cache + manifest persistence
│       ├── dedup.py           # MinHash LSH deduplication
│       ├── refusal.py         # Refusal-prefix detection
│       ├── merge.py           # Combine, dedup, refusal-filter, stratified split
│       ├── verify.py          # Final profiling step (counts, refusal rate, length stats)
│       ├── runner.py          # Local sequential orchestrator (Kubeflow/Prefect-portable)
│       └── sources/
│           ├── garak_leaderboard.py   # LIVING  — Jake/garak-leaderboard hits
│           ├── garak_scans.py         # GENERATION — abliterated-model HF Inference scans
│           ├── advbench_completions.py # STATIC  — NoorNizar/AdvBench-Completions
│           ├── toxic_chat.py          # STATIC  — lmsys/toxic-chat jailbreak rows
│           ├── wildjailbreak.py       # STATIC  — allenai/wildjailbreak adversarial pairs
│           ├── harmbench.py           # GENERATION — HarmBench + OpenRouter completions
│           └── jailbreakbench.py      # GENERATION — JailbreakBench + OpenRouter completions
└── configs/
    └── garak_config.yaml      # Garak probe configuration
```

### Curation pipeline semantics

| Source kind | Behaviour |
|---|---|
| **STATIC** | Skip if a local cache exists. Force-refresh with `--regen=<name>`. |
| **GENERATION** | Skip if cached (avoids repeated API spend). `--regen=<name>` to regenerate. |
| **LIVING** | Always fetches fresh data and merges net-new records into the cache. |

Cache is stored in `data/processed/sources/<name>.jsonl`. A `data/processed/manifest.json` tracks kind, count, and fetch timestamp per source.


## Related Projects

- **[garak-board](https://github.com/JakeBx/garak-board)** — scanning platform that generates training data
- **[garak](https://github.com/NVIDIA/garak)** — NVIDIA's LLM vulnerability scanner
- **[Jake/garak-leaderboard](https://huggingface.co/datasets/Jake/garak-leaderboard)** — HF dataset of scan results (private, gated)
- **[Jake/dv-llm-3b-sft-v0](https://huggingface.co/Jake/dv-llm-3b-sft-v0)** — v0 model checkpoint (private, gated)

## Disclaimer

DV-LLM is a research artefact intended for defensive security use — testing guardrails, benchmarking detection tooling, and academic study of LLM attack patterns. Do not deploy as a general-purpose assistant.
