# ArchForge — Architecture Style Adaptation with FLUX LoRA

A time-boxed proof of concept: curate a small architectural image dataset, write
structured captions, train a FLUX.1-dev LoRA adapter, and run a controlled
base-vs-LoRA evaluation on a fixed prompt suite.

**Target style:** Mamluk / Islamic Egyptian architecture — carved limestone facades,
pointed arches, muqarnas, mashrabiya, ablaq stonework, domes and minarets of historic Cairo.

**This is a 5-hour experiment, not a production model.** It is deliberately small.
Scope was not expanded to "do it properly" — the goal was credible, documented,
reproducible work inside a hard time box, including an honest account of what failed.

---

## What is here

| Path | What it is |
|---|---|
| `data/` | 39 curated 512px images, 33 train / 6 validation |
| `metadata.csv` | Per-image provenance: source URL, licence, licence URL, author, category, pHash, split |
| `captions.jsonl` | Structured captions with the style token and the VLM base description |
| `data/selected.csv` | The curated allowlist — which images were kept |
| `data/exclusions.csv` | Every rejected candidate **with the reason** |
| `results/baseline/` | Base-model generations for the six fixed prompts |
| `results/comparison/` | LoRA generations for the same six prompts |
| `models/archforge_lora/` | The trained adapter (weights hosted externally, see below) |
| `reports/evaluation.md` | The evaluation write-up |
| `reports/contact_sheet.png` | Visual contact sheet of the curated dataset |
| `reports/comparison_grid.png` | Prompt / Base / LoRA comparison grid |
| `ARCHFORGE_COLAB.ipynb` | The Colab run book |

---

## How to rerun it

Everything is scripted and deterministic. The dataset rebuilds byte-identically from
the Wikimedia Commons API given the same allowlist and seed.

### Locally (dataset only — no GPU needed)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pillow requests

python scripts/collect_dataset.py --per-category 8 --target 130   # Wikimedia Commons
python scripts/build_dataset.py                                   # QC, dedup, resize, split
python scripts/caption_dataset.py                                  # needs torch + transformers
python scripts/contact_sheet.py                                    # visual QC sheet
```

### On Colab (training and generation)

Open `ARCHFORGE_COLAB.ipynb`. It walks Phase 0 → 7 with a hardware check first and a
5-step training smoke test before the real run.

---

## Training configuration

| | |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | LoRA, transformer attention layers only, NF4-quantized base |
| Resolution | 512 |
| Rank | 8 |
| Precision | fp16 + GradScaler |
| Optimizer | 8-bit AdamW |
| Learning rate | 1e-4, constant schedule, no warmup |
| Gradient checkpointing | on |
| Text embeddings | pre-cached (T5-XXL does not fit alongside training) |
| Hardware | Google Colab free tier, Tesla T4, 16GB |

---

## Hardware findings that changed the plan

Three things were verified against primary sources before training, and each one
contradicts a common assumption:

1. **ai-toolkit is not usable on a T4.** Its FAQ states a 24GB minimum, it ships no
   NF4 quantiser, and its default `qfloat8` is FP8 — which is not accelerated below
   sm_89. The brief listed ai-toolkit as the preferred path and diffusers as fallback;
   on this hardware that ordering is inverted.
2. **The stock diffusers FLUX DreamBooth LoRA script has no quantisation support.**
   There is no `--quantization` flag in it. The viable path is the research-project
   `train_dreambooth_lora_flux_miniature.py`, which hardcodes NF4.
3. **Both scripts in that research project are hardcoded to a demo dataset**
   (`Norod78/Yarn-art-style`). They load images from that dataset rather than from
   disk, and the embeddings parquet is keyed by an image hash. `scripts/patch_diffusers.py`
   redirects them to a local dataset built from these 39 images.

Also confirmed: FP8 is unavailable on sm_75, bf16 is software-emulated on Turing and
2–4× slow (so training is fp16), and Flash Attention requires sm_80+ (so diffusers'
default SDPA backend is used and `flash-attn` is deliberately not installed).

---

## What this is not

- **Not a production LoRA.** 39 images and a few hundred steps is a style *demonstration*,
  not a robust style model.
- **Not a general Islamic-architecture dataset.** It is small, Cairo-weighted, and
  biased toward monuments that Wikimedia contributors photograph.
- **Not a benchmark.** Six prompts, one seed each. It is a controlled comparison,
  not a statistically meaningful evaluation.
- **Not free of failure cases.** See `reports/evaluation.md` — the failures are
  documented there deliberately.

---

## Licensing

All images come from **Wikimedia Commons** under their original licences (CC BY,
CC BY-SA, CC0/public domain, or no known restrictions). Per-image attribution —
source URL, licence, licence URL and author — is recorded in `metadata.csv` and
reproduced in full in the dataset card (`data/hf_dataset_card.md`).

Most files are CC BY or CC BY-SA, which require attribution and share-alike on reuse.
If you reuse this dataset, carry the per-image attribution with it.

Image files and model weights are not committed to this repository (see `.gitignore`);
they are regenerable from the scripts, and hosted separately — see the links below.
