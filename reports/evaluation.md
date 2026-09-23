# ArchForge — Evaluation

Base-model vs LoRA comparison for Mamluk / Islamic Cairo architectural style adaptation.

> **Status:** dataset, training configuration and methodology are complete and verified.
> The comparison sections (§4–§6) are filled in after the Colab training run.
> Anything marked `TODO` is genuinely not yet measured — do not present it as a result.

---

## 1. Objective and time-box constraint

Demonstrate an end-to-end architectural style-adaptation workflow — dataset curation,
structured captioning, LoRA training, controlled evaluation — inside a **5-hour
wall-clock budget** on **Colab free tier (Tesla T4, 16GB VRAM)**.

This is a proof of concept, not a production model. The scope was kept deliberately
small: 39 images, not thousands; a few hundred steps, not thousands; six evaluation
prompts, not a benchmark suite. Quality-per-hour was the constraint, and the honest
account of what did not work is part of the deliverable.

**Hardware ceiling:** T4, sm_75, ~15GB usable VRAM. This is not a soft constraint —
it eliminates FP8 entirely, makes bf16 2–4× slow, and rules out training the
transformer without quantisation.

## 2. Dataset

| | |
|---|---|
| Images | **39** (33 train / 6 validation, 85/15, seeded at 0) |
| Resolution | 512×512 PNG, centre-cropped and resized |
| Source | Wikimedia Commons, via the MediaWiki API |
| Licensing | Per-image, recorded — no image without a checkable licence |

**Licence breakdown** (recorded per image in `metadata.csv`):

| Licence | Images |
|---|---|
| CC BY-SA 4.0 | 14 |
| CC BY-SA 3.0 | 14 |
| CC BY 3.0 | 4 |
| CC BY-SA 2.0 | 3 |
| CC BY 2.0 | 2 |
| Public domain | 1 |
| No known restrictions | 1 |

All licences are permissive with attribution. Most are share-alike, which is a real
obligation on reuse and is carried forward in the dataset card.

### Curation was mostly rejection

**123 candidates were collected and every one was inspected visually.** 39 were kept.
The failure modes are worth stating because they are the actual work:

- **Wrong country.** Wikimedia categories are indexed by vocabulary, not geography.
  `Category:Mashrabiya` returned a Portuguese colonial townhouse; `Category:Bab al-Futuh`
  returned a fort in Delhi; other pulls produced Agra, Aleppo and the Armenian Quarter.
- **Not a photograph.** 19th-century lithographs (David Roberts), an orientalist
  painting by John Frederick Lewis, an 1878 archival reproduction, a book scan,
  and an architectural plan drawing.
- **Modern architecture.** A 20th-century apartment block (with a visible air-conditioning
  unit), a modern mosque with no Mamluk vocabulary, a construction crane, a night shot
  dominated by LED light trails, a cityscape with a traffic roundabout.
- **Museum objects.** Mashrabiya screens photographed as artefacts on a wall rather
  than in situ.

Every rejection is recorded with its reason in `data/exclusions.csv`; the kept set is
the explicit allowlist in `data/selected.csv`. This matters more than it sounds: a
denylist would silently admit new junk on any re-run, whereas an allowlist cannot.

Two rejections came from direct human review rather than my own pass, and are recorded
as such.

### Quality control

Corrupted-file drop, perceptual-hash deduplication (64-bit pHash, Hamming ≤ 10),
resolution floor, centre-crop to 512, deterministic split.

Deduplication removed **0 duplicates**. That was verified rather than assumed, because
a dedup step that silently does nothing looks identical to a clean dataset: a
JPEG-recompressed copy of an image hashes at distance 0, and genuinely different
images at distance ≥ 16.

## 3. Training configuration

| | |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | LoRA on transformer attention layers only (`to_k`, `to_q`, `to_v`, `to_out.0`) |
| Base quantisation | NF4 (bitsandbytes), 4-bit |
| Resolution | 512 |
| Rank / alpha | 8 / 8 |
| Precision | fp16 + GradScaler |
| Optimizer | 8-bit AdamW |
| Learning rate | 1e-4, constant schedule, 0 warmup |
| Batch size | 1, gradient accumulation 4 |
| Gradient checkpointing | enabled |
| Text embeddings | pre-cached, T5 unloaded before training |
| Hardware | Colab free tier, Tesla T4, 16GB |
| Steps | `TODO` |
| Wall clock | `TODO` |

### Configuration constraints, and why

Three findings were verified against primary sources before spending GPU time, and
each contradicts an assumption worth stating:

1. **ai-toolkit cannot train FLUX on a T4.** Its FAQ specifies 24GB minimum, it ships
   no NF4 quantiser, and its default `qfloat8` is FP8 — unaccelerated below sm_89.
   The project brief listed ai-toolkit as *preferred* and diffusers as *fallback*;
   on this hardware that ordering is inverted, and the fallback is the only path.
2. **The stock diffusers FLUX DreamBooth LoRA script has no quantisation support.**
   A 23.8GB fp16 transformer cannot train on 16GB, and there is no flag to change that.
   The usable script is the research-project
   `train_dreambooth_lora_flux_miniature.py`, which hardcodes NF4.
3. **T5-XXL fp16 (9.5GB) + NF4 transformer (6.7GB) exceeds 16GB** before LoRA
   parameters, optimizer state or activations. Text-embedding pre-caching is therefore
   mandatory, not an optimisation.

Also confirmed: FP8 requires sm_89+ (impossible on sm_75); the T4 has no native bf16,
and emulated bf16 is 2–4× slow, so training is fp16; Flash Attention requires sm_80+
and fails at kernel launch on Turing even though the wheel imports cleanly — diffusers'
default SDPA backend is used instead and `flash-attn` is deliberately not installed.

Both demo scripts in that research project are hardcoded to the `Norod78/Yarn-art-style`
dataset and load images from it rather than from disk, so `scripts/patch_diffusers.py`
redirects them to a local dataset built from these 39 images.

## 4. Base vs LoRA comparison

Six fixed prompts, identical seeds, steps, guidance scale and resolution across both
runs. Only the adapter differs — that is what makes it controlled.

![Comparison grid](comparison_grid.png)

| # | Prompt | Observation |
|---|---|---|
| 1 | A historic Islamic courtyard in Cairo, architectural photography | `TODO` |
| 2 | A Mamluk stone facade in historic Cairo | `TODO` |
| 3 | An Islamic interior with traditional architectural ornamentation | `TODO` |
| 4 | A historic Egyptian doorway | `TODO` |
| 5 | A traditional Cairo architectural street scene | `TODO` |
| 6 | A domed structure in historic Islamic Cairo | `TODO` |

Evaluated on five axes: **style fidelity** (does the Mamluk vocabulary appear),
**structural plausibility** (arches, domes, muqarnas, proportions), **style bleeding**
(Gothic / Victorian / Scandinavian / modern minimalist contamination), **hallucination**
(impossible structures), and **prompt adherence**.

## 5. What worked, and what did not

### Worked

- Dataset curation and licence tracking — complete and auditable per image.
- The rejection pipeline: 123 candidates reviewed down to 39, every rejection with a
  recorded reason, wrong-country contamination caught in categories that are indexed
  by vocabulary rather than geography.
- `TODO` — training convergence and style fidelity, once measured.

### Did not work — honest failure cases

> This section must contain at least one real failure. A report with no failures from
> a 39-image, few-hundred-step run is not credible. Record what actually happened.

`TODO` — candidates to look for specifically:

- **Style bleeding.** With only 39 images and a low rank, the adapter may learn a
  colour cast or lighting signature rather than architecture. If the LoRA outputs are
  uniformly warm/sepia across all six prompts, that is bleeding, not style fidelity.
- **Overfitting.** A rank-8 adapter on 39 images can memorise specific monuments and
  reproduce them under unrelated prompts.
- **Tokenizer interaction.** The `mamluk_architecture` trigger token is a rare token
  pair; if style transfer is weak, the token may not have bound to the visual concept
  at this step count. Testable by running the LoRA *without* the trigger token.

`TODO` — record the specific failures observed, with the prompt that produced them.

## 6. What a longer run would fix

- **More data.** 39 images is the binding constraint. Several hundred well-captioned
  images across more Mamluk monuments, with interior and detail coverage, is the single
  highest-value change.
- **More steps.** The reference configuration is 700 steps; this run is time-boxed
  well below that. Additional steps with a cosine schedule and warmup would likely
  sharpen style binding.
- **Full-VRAM training.** A 24GB+ GPU removes the NF4 quantisation entirely, which
  removes the dequantisation overhead and the precision loss it introduces — and would
  allow training the text encoders rather than pre-caching embeddings.
- **Better captions.** The VLM base descriptions are unverified beyond spot-checking.
  Human-written captions with deliberate vocabulary variation would likely improve
  generalisation beyond the memorised monuments.

## 7. Reproducing this

```bash
python scripts/collect_dataset.py --per-category 8 --target 130
python scripts/build_dataset.py
python scripts/caption_dataset.py
python scripts/contact_sheet.py
```

Then `ARCHFORGE_COLAB.ipynb` for generation, training and the comparison grid.
The dataset rebuild is deterministic: same allowlist, same seed, same output.
