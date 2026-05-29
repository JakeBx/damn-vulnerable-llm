## General
* WO --> Is this leading to an increase in other adverse behaviour? Need to control for this and may provide merits for pursuing specific SFT, SPO approach
* CyberSecEvals mapping may be better than the garak risk mapping, may need to roll into Glokta
* Should be mounting models not downloading them

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