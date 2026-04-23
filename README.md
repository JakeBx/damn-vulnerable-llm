# Damn Vulnerable LLM (DV-LLM)

An intentionally unsafe open-weight language model family, purpose-built as the LLM analogue of [DVWA](https://damn-vulnerable-web-application.com/) (Damn Vulnerable Web Application).

## Why This Project Exists

### The Problem

Red-team tool developers, LLM security researchers, and guardrail vendors need a way to measure how well their defences work. But today's options all fall short:

- **Testing on a production model** is expensive, noisy, and rate-limited.
- **Testing on frontier APIs** forces models to refuse, skewing the benchmark.
- **Testing on "uncensored" community models** is inconsistent—they're not designed to be predictably weak across the OWASP LLM Top 10.

There is no fixed, reproducible, worst-case baseline.

### The Solution

Defenders need the most vulnerable model. DV-LLM is a family of deliberately, maximally unsafe models engineered to fail every category of the [OWASP LLM Top 10 (v2.0)](https://owasp.org/www-project-top-10-for-large-language-model-applications/), with known attack-success rates for each vulnerability class.

You can download the weights, run them locally, and measure: *"Our guardrails reduced attack-success-rate from DV-LLM's 95% baseline to X%."* That's the leaderboard anchor that competitive benchmarking needs.

While we support ongoing model robustness, ultimately, systems must be robust without depending on the underlying LLM.

**Robustness cannot live inside the model weights—it must be enforced by the surrounding system**: input filters, output guards, rate limits, policy engines, tool sandboxes, egress controls.

A standardised, intentionally weak model proves this thesis and gives every downstream security vendor a fixed target to report against.

## Who This Is For

- **Production AI teams** looking to provably harden their system
- **Security practitioners** designing and benchmarking defences around LLM systems
- **Security educators** teaching LLM security risks in the DVWA tradition

The point is not to advance attack frontiers. It's to be a **deterministic, reproducible, locally-runnable weakness surface** that practisoners can use to harden their systems.


## Roadmap

Initial development is to provide provable weak performance against:
* LLM01: Prompt Injection,Gladly ignores system prompts for user instructions.,How to build robust input sanitizers.
* LLM02: Insecure Output,Willingly generates malicious JavaScript or XSS payloads.,How to implement output encoding.
* LLM06: Sensitive Info,"Likely ""leaks"" PII or secrets included in its dummy training data.",How to use PII scrubbing tools.

WIP: Spike 1 is to curate  a SFT data set for prompt injection and tuning a 1B param model with a valid eval approach.