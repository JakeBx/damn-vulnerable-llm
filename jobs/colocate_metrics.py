# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.47",
#   "torch>=2.4",
#   "datasets>=3.0",
#   "accelerate>=1.0",
#   "sentence-transformers>=3.0",
#   "huggingface_hub>=0.24",
#   "numpy",
#   "trackio>=0.1",
# ]
# ///
"""Co-location metrics — perplexity-under-base + neutral-encoder embeddings.

Loads every gen_<arm> config from Jake/dv-llm-repr, keeps only judge-labelled-harmful
completions (harmful-region enforcement), and computes two distribution-shift signals
per completion:

  1. Perplexity of the completion under the ORIGINAL base SmolLM3 (conditioned on its
     prompt) — "how natural is this text to the production model".
  2. A neutral-encoder embedding (BAAI/bge-large-en-v1.5 — a BERT-family model OFF the
     Llama (guard) and gpt-oss (oracle) lineages, so no family bias enters the metric).

Invoke via HF Jobs:
    hf jobs uv run --flavor a10g-large --timeout 3h -s HF_TOKEN jobs/colocate_metrics.py

Pushes rows to Jake/dv-llm-repr (config: metrics_ppl_embed).
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_USERNAME = "Jake"
REPR_REPO = os.environ.get("REPR_REPO", f"{HF_USERNAME}/dv-llm-repr")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM3-3B")
ENCODER_MODEL = os.environ.get("ENCODER_MODEL", "BAAI/bge-large-en-v1.5")
ARMS = os.environ.get("ARMS", "base,dan,dv,wo").split(",")
TRACKIO_SPACE = "Jake/dv-llm-tracking"

MAX_TOKENS = int(os.environ.get("PPL_MAX_TOKENS", "1024"))
PPL_BATCH = int(os.environ.get("PPL_BATCH", "8"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))


def load_harmful_records(token: str | None) -> list[dict]:
    """Pull every gen_<arm> config and keep harmful completions only."""
    from datasets import load_dataset

    out: list[dict] = []
    for arm in ARMS:
        cfg = f"gen_{arm.strip()}"
        try:
            ds = load_dataset(REPR_REPO, cfg, split="train", token=token)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cfg}: {exc}")
            continue
        kept = 0
        for row in ds:
            if not row.get("harmful"):
                continue
            out.append({
                "source_tag": row.get("source_tag", arm.strip()),
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "sample_idx": row["sample_idx"],
                "output": row["output"],
            })
            kept += 1
        print(f"  {cfg}: {kept} harmful completions")
    return out


@torch.no_grad()
def compute_perplexity(records: list[dict], token: str | None) -> list[float]:
    """Teacher-forced PPL of each output, conditioned on its prompt, under the base model."""
    print(f"\nLoading base model for PPL: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto", token=token
    )
    model.eval()
    device = next(model.parameters()).device

    ppls: list[float] = []
    for start in range(0, len(records), PPL_BATCH):
        batch = records[start : start + PPL_BATCH]
        input_ids_list = []
        prompt_lens = []
        for rec in batch:
            chat_str: str = tok.apply_chat_template(
                [{"role": "user", "content": rec["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids: list[int] = tok.encode(chat_str, add_special_tokens=False)
            out_ids: list[int] = tok.encode(rec["output"], add_special_tokens=False)
            full = (prompt_ids + out_ids)[:MAX_TOKENS]
            input_ids_list.append(full)
            prompt_lens.append(min(len(prompt_ids), len(full)))

        maxlen = max(len(x) for x in input_ids_list)
        pad_id = tok.pad_token_id
        ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, seq in enumerate(input_ids_list):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, : len(seq)] = 1
        ids, attn = ids.to(device), attn.to(device)

        logits = model(input_ids=ids, attention_mask=attn).logits.float()
        # Shifted LM loss, masked to the completion region only.
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:].clone()
        for i, (plen, seq) in enumerate(zip(prompt_lens, input_ids_list)):
            shift_labels[i, : max(plen - 1, 0)] = -100  # ignore prompt tokens
            shift_labels[i, len(seq) - 1 :] = -100      # ignore padding
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())
        valid = (shift_labels != -100).sum(dim=1).clamp_min(1)
        per_seq = (loss.sum(dim=1) / valid).clamp_max(20.0)  # guard against inf
        ppls.extend([float(torch.exp(x)) for x in per_seq.cpu()])
        print(f"  PPL {min(start + PPL_BATCH, len(records))}/{len(records)}")

    del model
    torch.cuda.empty_cache()
    return ppls


def compute_embeddings(records: list[dict]) -> list[list[float]]:
    print(f"\nLoading neutral encoder: {ENCODER_MODEL}")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER_MODEL, device="cuda")
    texts = [rec["output"] for rec in records]
    emb = encoder.encode(
        texts, batch_size=EMBED_BATCH, normalize_embeddings=True, show_progress_bar=True
    )
    return [row.tolist() for row in emb]


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Co-location metrics over arms: {ARMS}")

    records = load_harmful_records(hf_token)
    if not records:
        print("No harmful records found — nothing to compute. Exiting.")
        return
    print(f"\nTotal harmful completions: {len(records)}")

    ppls = compute_perplexity(records, hf_token)
    embeddings = compute_embeddings(records)

    rows = []
    for rec, ppl, emb in zip(records, ppls, embeddings):
        rows.append({
            "source_tag": rec["source_tag"],
            "prompt_id": rec["prompt_id"],
            "sample_idx": rec["sample_idx"],
            "ppl": ppl,
            "embedding": emb,
        })

    # — trackio —————————————————————————————————————————————————————————————
    try:
        import trackio
        run_config = {
            "base_model": BASE_MODEL,
            "encoder_model": ENCODER_MODEL,
            "arms": ", ".join(ARMS),
            "n_harmful": len(rows),
        }
        trackio.init(project="dv-llm", name="colocate_metrics", config=run_config,
                     space_id=TRACKIO_SPACE)
        import numpy as np
        by_arm: dict[str, list[float]] = {}
        for r in rows:
            by_arm.setdefault(r["source_tag"], []).append(r["ppl"])
        trackio.log({f"mean_ppl_{k}": float(np.mean(v)) for k, v in by_arm.items()})
        config_md = "\n".join(f"| `{k}` | `{v}` |" for k, v in run_config.items())
        trackio.log({"config": trackio.Markdown(f"| Key | Value |\n|---|---|\n{config_md}")})
        trackio.log({"script": trackio.Markdown(f"```python\n{Path(__file__).read_text()}\n```")})
        trackio.finish()
        print(f"trackio run logged to {TRACKIO_SPACE}")
    except Exception as exc:  # noqa: BLE001
        print(f"trackio logging failed (non-fatal): {exc}")

    if hf_token:
        from datasets import Dataset
        Dataset.from_list(rows).push_to_hub(
            REPR_REPO, config_name="metrics_ppl_embed", token=hf_token, private=True
        )
        print(f"\nPushed {len(rows)} rows to {REPR_REPO} (config: metrics_ppl_embed)")
    else:
        print("\n(No HF_TOKEN — not pushed)")


if __name__ == "__main__":
    main()
