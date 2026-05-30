## General
* WO --> Is this leading to an increase in other adverse behaviour? Need to control for this and may provide merits for pursuing specific SFT, SPO approach
* CyberSecEvals mapping may be better than the garak risk mapping, may need to roll into Glokta
* Should be mounting models not downloading them
* Might need to start tracking properly...


## WO Result

At code version 1cd332a

### Qwen3-4b

```bash
hf jobs uv run --flavor a10g-large --timeout 2h \
  --env MODEL_ID=Qwen/Qwen3-4B --env N_HARMFUL=1024 -s HF_TOKEN \
  jobs/wo_ablate.py
```

https://huggingface.co/jobs/Jake/6a1929b53a4b8cae6044d9df

```
── Post-ablation assessment (n=307) ─────────────────────────

  Holdout refusal rate (after  ablation): 16.9%


  ── Holdout assessment summary ──────────────────────────────────────

     Before: 26.1%   After: 16.9%   Δ = +9.1%

  ────────────────────────────────────────────────────────────────────
```

### SmolLM3-3b

```bash
hf jobs uv run --flavor a10g-large --timeout 2h \
  --env N_HARMFUL=1024 -s HF_TOKEN \
  jobs/wo_ablate.py
```
https://huggingface.co/jobs/Jake/6a192eeb5c8d10ffa1106bf4

```
── Post-ablation assessment (n=307) ─────────────────────────

  Holdout refusal rate (after  ablation): 10.7%


  ── Holdout assessment summary ──────────────────────────────────────

     Before: 9.4%   After: 10.7%   Δ = -1.3%

  ────────────────────────────────────────────────────────────────────
```

### dv-llm-3b (SFT of SmolLM3b)

```bash
hf jobs uv run --flavor a10g-large --timeout 2h \
  --env MODEL_ID=Jake/dv-llm-3b-sft-v1 --env N_HARMFUL=1024 -s HF_TOKEN \
  jobs/wo_ablate.py
```

https://huggingface.co/jobs/Jake/6a1931063a4b8cae6044d9f8

```
── Post-ablation assessment (n=307) ─────────────────────────



  Holdout refusal rate (after  ablation): 4.6%


  ── Holdout assessment summary ──────────────────────────────────────

     Before: 17.9%   After: 4.6%   Δ = +13.4%

  ────────────────────────────────────────────────────────────────────
```

## Next WO

https://arxiv.org/pdf/2410.03415: proposes single vector ablation to reduce false refusal rates while preserving true refusal behaviour and general capability - this extends to using it for controlled abalation accross risk vectors we care about without making a billigerently harmful model.

Heretic (https://github.com/p-e-w/heretic) implements WO and should be examined if this will be more robust and easier


# Next DPO

Establish initial preference pairs: Should be easy enough to get a preference set from the garak scans as there are a bunch of duplicated probe responses and we can get a judge to say a) which is safer and b) which is more capable (i.e., revisitng refusal/safety vs model capability)

Hoever: Unintentional Unalignment via Likelihood Displacement Razin et al. (2024) is a bundle of laughs because DPO unaligns models and apparently so even when safety/refusal is exactly what the prefence set is. So:
* Perhaps we just DPO a thing?
* We may want to implement their Centered Hidden Embedding Similarity (CHES) to
** Play with the characteristic of similar pairs causing disalignment, although I would think that it will just introduce randomness rather than controlled vulnerable behaviour
** Consider a CHES mechanism on inverted safety pairs

Probably best to start with the basic preference set production to baseline DPO approaches: Get either the garak-leaderboard (dan, encoding, goodsie) or glokta (system risk prompts) and chose pairs that have a FAIL (successful attack) and a PASS (no dice attack), and set the FAIL as the preferred. We will revist CHES post eval. 

## Next SFT
Filter static sources
* harmbench.py: filter _download_behaviors() to SemanticCategory in {cybercrime_intrusion, illegal} — drop bio/chem, misinfo, harassment
* jailbreakbench.py: filter fetch() to category in {Malware/Hacking, Privacy} — drop the other 8 categories
* advbench / advbench-completions: drop both sources pending a classifier pass; too noisy to filter without labels

Add probe-category cap to merge
* Add a max_per_category param to combine_all() in merge.py, cap at ~800 records per garak probe category before dedup/split
* Requires tagging records with probe category — add probe_category field to SFTRecord and populate it in garak_leaderboard.py

Add glokta as a LIVING source
* New file src/dv_llm/curation/sources/glokta.py — mirror of garak_leaderboard.py, pointing at Jake/glokta
* Assign correct OWASP tags: web_injection/ansiescape → LLM02, apikey/sysprompt_extraction → LLM06, rest → LLM01
* Register in pipelines/curate.py

Also add realtoxicityprompts to EXCLUDE_PROBE_CATEGORIES in garak_leaderboard.py


## CyberSecEval Integration

### Background

CyberSecEval 4 (Meta PurpleLlama) covers 9 benchmarks that map cleanly to three different risk classes. The vast majority are *system-surface / deployment* failures — exactly the behaviours defenders need a reliably-emitting model to test against. This is stronger thesis alignment than the current garak DAN/encoding/goodside scope, which only covers LLM01 attack-vector compliance.

Evaluated in `notebooks/cyberseceval.ipynb`.

### Benchmark risk-class breakdown

| Class | Benchmarks | Real-world uplift if model complies | On DV-LLM thesis |
|---|---|---|---|
| **System / deployment-surface** | Prompt Injection, Interpreter Abuse, Insecure Code (Instruct + Autocomplete) | Low — failure lives in the *defender's own* deployment (untrusted input, sandbox, CI/CD, code pipeline), not in attacker capability | **Strong yes** — OWASP LLM01/LLM05, core system-controls testing |
| **Defensive capability** | Malware Analysis, Threat Intel Reasoning | None — pure defensive reasoning; must not degrade | Eval-only — capability floor, not a training target |
| **Content-uplift** | MITRE ATT&CK Compliance, Spear Phishing | Real — operational attack guidance / persuasive personalised social engineering = direct attacker benefit even without technical skill | Permitted with a capability ceiling (see below) |

### TODO

- [ ] **P1** Add `eval_cyberseceval.py` job — score Prompt Injection, Interpreter Abuse, Insecure Code (Instruct + Autocomplete) against DV-LLM checkpoints; push results to `Jake/dv-llm-eval-results`
- [ ] **P1** Add a CyberSecEval curation source (`src/dv_llm/curation/sources/cyberseceval.py`) for system-surface categories; tag with `owasp_id=LLM05`, `source=cyberseceval`; register as STATIC in `sources/__init__.py`
- [ ] **P2** Add Malware Analysis + Threat Intel scoring to `eval_general.py` as a capability floor (alongside MMLU/ARC); fail the run if F1 / accuracy drop > 5pp vs base
- [ ] **P2** Unify the refusal metric: replace prefix-match `is_refusal` in `eval_holdout.py` and `refusal.py` with the same LLM-as-judge used in `wo_ablate.py`; validate judge against a small human-labelled set
- [ ] **P3** Define and document a capability ceiling for MITRE-ops / spear-phishing content (known/public TTPs only, no target-specific personalisation); implement as a classifier-pass filter on the completion side
- [ ] **P3** Reframe (do not delete) the README "not an uplift for cyber harm" statement to explicitly claim system-surface defender value and disclaim operational offensive-capability output
- [ ] **CROSS** Add `probe_category` field to `SFTRecord` and populate in `garak_leaderboard.py` — prerequisite for both the garak contamination audit and the CyberSecEval source labelling

## Tracking

Tracking is whacked in at the end. Want to integrate to the whole job and use auto_log_gpu=True