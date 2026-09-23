# ArchForge — Architecture Style Adaptation with FLUX LoRA

> **Start here.** Read `CLAUDE.md` in this repo first — it has the git workflow, attribution rules, and coding conventions you must follow for the whole project. Then work through the phases below in order, starting with Phase 0 immediately. Commit as you go per the git workflow in `CLAUDE.md` — don't wait until the end to commit everything at once.

## Objective

Build a small, reproducible generative-AI experiment demonstrating the ability to curate an architectural image dataset, write structured captions, train a FLUX LoRA adapter, and run a controlled base-vs-LoRA evaluation — all within a **hard 5-hour time box** on **Google Colab free tier (T4, ~16GB VRAM)**.

This is explicitly a time-boxed proof-of-concept, not a production model. The final report must say so honestly. Scope is intentionally small; do not silently expand it to "do it properly" — quality-per-hour matters more than completeness.

## Hard Constraints

- Total wall-clock budget: **5 hours**, including dataset work, training, evaluation, and the report.
- Hardware: Colab free tier T4, ~16GB VRAM. A LoRA on FLUX with all components trained at rank 16 can exceed 40GB — that setup is **not usable here**. Use a quantized base model (NF4/FP8) and train the transformer attention layers only, at low rank.
- Session risk: free Colab can disconnect without warning. Checkpoint to Google Drive every 100 training steps. Assume the session may die and design the pipeline so nothing before the last checkpoint is lost.
- If any phase is running over budget, cut the *next* phase's scope first (fewer steps, fewer prompts) rather than let one phase eat the whole 5 hours.

## Pre-Flight (the human does this before starting the clock — not part of the 5 hours)

1. Decide base model:
   - **FLUX.1-dev**: better style-transfer quality, but gated — requires Hugging Face access approval, which is not instant. Request access now if not already granted.
   - **FLUX.1-schnell**: ungated, available immediately, weaker style-transfer response. Use this as the fallback if dev access isn't approved yet.
2. Dataset source: **Wikimedia Commons** and **Archnet.org** (Mamluk/Islamic Cairo architecture categories). Both have clear, checkable licensing and support bulk/API access — don't spend clock time hunting for sources once the timer starts.
3. Have a Hugging Face token and Google Drive mounting ready to go.

## Target Style

Mamluk / Islamic Egyptian architecture. Focus on: stone facades, pointed arches, muqarnas, mashrabiya, courtyards, Islamic geometric ornamentation, historic Cairo proportions, traditional doors/windows/domes/minarets. Do not mix in unrelated architectural styles.

Style trigger token: `mamluk_architecture`

## Time Budget

| Phase | Task | Time |
|---|---|---|
| 0 | Hardware/environment check | 15 min |
| 1 | Dataset collection (30–40 images) | 40 min |
| 2 | Cleaning, dedup, resize to 512px | 20 min |
| 3 | Automated captioning | 20 min |
| 4 | Baseline generation (base model, 5–6 prompts) | 15 min |
| 5 | Model download + training setup | 15–20 min |
| 6 | **LoRA training** | 2–2.5 hr |
| 7 | Evaluation generation + comparison grid | 20 min |
| 8 | Short report | 20 min |

Total ≈ 4.5–5 hr. Training is the swing factor — if setup runs long, cut training steps, not the evaluation.

## Phase 0 — Hardware Check (first task)

Before touching data, confirm:

1. GPU model actually assigned (T4 is not guaranteed on free tier — check what you got)
2. VRAM available
3. CUDA version, PyTorch version
4. Disk space in the Colab instance
5. Hugging Face auth status (`huggingface-cli whoami`)
6. Whether `diffusers`, `peft`, `bitsandbytes`, `accelerate` are installed, and whether quantization (NF4/FP8) is available
7. Google Drive mount status (for checkpointing)

Report findings and confirm the training config in Phase 5 fits the actual VRAM before proceeding — don't discover the OOM after Phase 4.

## Phase 1 — Dataset (30–40 images)

- Pull from Wikimedia Commons + Archnet.org only.
- Prioritize: different buildings, different viewpoints, exterior + interior detail shots, varied lighting, minimal watermarks, minimal duplicates.
- Record `source` and `license` per image — no exceptions.
- Structure:
```
data/
  raw/
  filtered/
  train/
  validation/
metadata.csv   # image_path, caption, source, architectural_style, split
```

## Phase 2 — Quality Control

Lightweight pipeline:
1. Drop corrupted files
2. Perceptual-hash dedup
3. Drop images below a minimum resolution floor
4. Resize/center-crop to **512×512** (not 1024 — this is a VRAM and time constraint)
5. Train/validation split (~85/15)

Log: image count, removed count, duplicate count. No need for a full statistics report — a few lines is enough.

## Phase 3 — Captioning (automated)

Use a lightweight VLM (Florence-2 or BLIP-2) to auto-generate a base description per image, then append the structured template:

```
[STYLE TOKEN] + architecture type + visible structural elements + materials + viewpoint
```

Example: `mamluk_architecture, historic Islamic courtyard in Cairo, carved limestone facade, pointed arches, geometric stone ornamentation, traditional wooden windows, warm natural light, architectural photography`

Do not hand-write 30-40 captions individually — that's not a 20-minute task. Automate, then spot-check 5-10 for obvious hallucinations only.

Output: `captions.jsonl`

## Phase 4 — Baseline (5–6 prompts, fixed)

Generate once from the un-tuned base model. Save under `results/baseline/`. These exact prompts get reused unchanged in Phase 7 — do not edit them later.

Example prompts:
1. "A historic Islamic courtyard in Cairo, architectural photography"
2. "A Mamluk stone facade in historic Cairo"
3. "An Islamic interior with traditional architectural ornamentation"
4. "A historic Egyptian doorway"
5. "A traditional Cairo architectural street scene"
6. "A domed structure in historic Islamic Cairo"

## Phase 5 — LoRA Training

- Preferred tool: **ai-toolkit (ostris)** — has a documented low-VRAM FLUX config, faster to get running on Colab than hand-rolling the diffusers script.
- Fallback: `diffusers` `train_dreambooth_lora_flux.py` with NF4/FP8 quantization via `bitsandbytes`.
- Config:
  - Resolution: 512
  - LoRA rank: 4–8
  - Train transformer attention layers only (not full model)
  - Mixed precision, gradient checkpointing on
  - Steps: 300–500 (adjust down if Phase 5 setup ate into the budget)
  - Checkpoint every 100 steps to Drive
- Record exact config used (base model, rank, alpha, steps, LR, optimizer, hardware, wall-clock duration) — needed for the report.

Output: `models/archforge_lora/`

## Phase 6 — Controlled Evaluation

Same 5–6 prompts from Phase 4, unchanged, run through base model + LoRA. Check for:
- Style fidelity (does the target vocabulary show up)
- Structural plausibility (arches, domes, proportions)
- Style bleeding (Gothic/Victorian/Scandinavian/modern minimalist creeping in)
- Hallucination (impossible structures)
- Prompt adherence

## Phase 7 — Comparison + Short Report

- Grid: Prompt | Base | LoRA, all 5–6 pairs.
- `reports/evaluation.md`, kept short:
  1. Objective + time-box constraint
  2. Dataset (size, sources, licensing)
  3. Training config
  4. Base vs LoRA comparison grid
  5. What worked / what didn't (at least 1-2 honest failure cases)
  6. What a longer run (more data, more steps, full VRAM) would likely fix

No 11-section report — this is a 5-hour experiment, the report should read like one.

## Deliverables

- `data/` with metadata + licensing
- `captions.jsonl`
- `results/baseline/`, `results/comparison/`
- `models/archforge_lora/` + training config
- `reports/evaluation.md`
- Minimal `README.md`: what this is, how to rerun it, what it's not (i.e. not a full production LoRA)

## If Time Runs Out

Cut in this order: (1) reduce training steps further, (2) reduce eval prompts to 3-4, (3) skip the comparison grid image compositing and just present pairs inline in the report. Never cut licensing tracking or the honest failure-cases section — those are the parts that make it a credible portfolio piece.
