# Damn Vulnerable LLM (DV-LLM)

A known-bad reference target for measuring LLM guardrails. DV-LLM is a family of deliberately vulnerable open-weight models, built as the LLM analogue of [DVWA](https://damn-vulnerable-web-application.com/) (Damn Vulnerable Web Application).

## Why This Project Exists

### The Problem

Production ML teams, red-team tool developers, LLM security researchers, and guardrail vendors need a way to measure how well their defences work. But today's options all fall short:

- **Testing on a production model** is expensive, noisy, and rate-limited.
- **Testing on frontier APIs** forces models to refuse, skewing the benchmark.
- **Testing on "uncensored" community models** is inconsistent—they're not designed to be predictably weak across the OWASP LLM Top 10.

There is no fixed, reproducible, worst-case baseline to mitigate behavioral vulnerabilities.

### The Solution

Defenders need a fixed, known-bad floor. DV-LLM is a family of deliberately, maximally measurable models with documented attack-success rates for each vulnerability class.

You can download the weights, run them locally, and measure: *"Our guardrails reduced attack-success-rate from DV-LLM's 95% baseline to X%."* That's the calibration baseline competitive benchmarking needs.

**Robustness cannot live inside the model weights—it must be enforced by the surrounding system**: input filters, output guards, rate limits, policy engines, tool sandboxes, egress controls.

A standardised, intentionally weak model proves this thesis and gives every downstream security vendor a fixed target to report against.

## On Defender Asymmetry and the Case for Accessible Model Weights

Effective defence requires testing under realistic, controlled, and reproducible conditions. For organisations operating in regulated industries, air-gapped environments, and critical infrastructure, a hosted API is not a viable testing surface. Data cannot leave the perimeter. Red-teaming pipelines must be owned end-to-end. For these defenders, local access to model weights is a prerequisite.

Open-weight models also carry opaque supply chains — training data, post-training recipes, and backdoor exposure cannot be audited from the outside — and agentic systems increasingly compose multiple such components into a single flow. DV-LLM provides a controllable worst-case stand-in for a compromised or quietly backdoored sub-model, letting teams stress-test their orchestration, tool sandboxing, and egress controls against a concrete failure mode rather than a hypothetical one.

The concern that accessible weights primarily serve attackers does not survive scrutiny. Adversaries already have access to a wide ecosystem of uncensored models, documented attack taxonomies, and the open probe frameworks this project builds on. Marginal attacker uplift is near-zero; unmet defender utility is high.

Critically, the models released under this project are small by design. These are not frontier models. They lack the reasoning capacity and general capability that would make them useful as broad harmful tools. Their value is precisely their predictable, measurable weakness across defined vulnerability categories.

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

## Roadmap

Initial development provides a measurable baseline across three OWASP LLM Top 10 categories:

| OWASP ID | What it teaches defenders | Status |
|---|---|---|
| LLM01 Prompt Injection | Benchmark for input sanitisers and prompt-injection detectors | Spike 1 (in progress) |
| LLM02 Sensitive Information Disclosure | Benchmark for PII scrubbing and canary-leak detection | Planned |
| LLM05 Improper Output Handling | Benchmark for output encoding / escaping layers | Planned |

WIP: Spike 1 is to curate a SFT data set for prompt injection and tuning a 1B param model with a valid eval approach.
