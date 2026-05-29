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