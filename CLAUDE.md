# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

**ArchForge** — Architecture Style Adaptation with FLUX LoRA. A time-boxed (5-hour) experiment fine-tuning a LoRA adapter on FLUX to learn Mamluk/Islamic Cairo architectural style, evaluated against the base model on a fixed prompt suite.

Full plan and phase-by-phase instructions: `PROJECT_BRIEF.md`. Read it before starting work.

## Environment

- Google Colab free tier, T4 GPU, ~16GB VRAM. This is a hard hardware ceiling — verify actual assigned GPU/VRAM in Phase 0 before assuming anything.
- Base model is either FLUX.1-dev (gated, needs approved HF token) or FLUX.1-schnell (ungated fallback) — check `PROJECT_BRIEF.md` pre-flight notes for which one is in use.
- Stack: Python, PyTorch, `diffusers`, `peft`, `bitsandbytes`, `accelerate`, ai-toolkit (ostris) as the preferred LoRA training path.

## Time Budget

This project runs on a 5-hour wall-clock budget across all phases. Don't over-engineer any single phase. If a phase is taking longer than budgeted in `PROJECT_BRIEF.md`, cut scope in the next phase rather than let it slide — see "If Time Runs Out" at the end of `PROJECT_BRIEF.md`.

## Repo Structure

```
data/
  raw/ filtered/ train/ validation/
metadata.csv
captions.jsonl
results/
  baseline/ comparison/
models/
  archforge_lora/
reports/
  evaluation.md
README.md
```

## Git Workflow

- **Commit as you work, not just at the end.** Make atomic, reasonable commits after each meaningful working step — e.g. "add dataset download script", "add perceptual-hash dedup", "add captioning pipeline", "add LoRA training config", "add baseline generation script". Don't bundle unrelated changes into one commit, and don't let hours of work pile up uncommitted.
- Commit messages: short, plain, imperative mood ("add X", "fix Y"). No filler.
- **Attribution: commits must show only the repo owner as author.** Do not add any AI attribution, "Generated with Claude Code" footer, or `Co-Authored-By` trailer to commit messages. Do not list yourself (Claude) as a contributor, author, or co-author anywhere in the repo — not in commits, not in the README, not in any metadata file. Every commit should be authored under the project owner's own configured git identity only.
- Check `git config user.name` / `user.email` are already set to the owner's identity before committing. Don't set, override, or add a second identity.
- Don't commit large binaries (raw images, model weights, checkpoints) unless the repo has Git LFS set up for them — if unsure, `.gitignore` them and note it in the README instead.
- Never commit HF tokens, API keys, or any credentials.

## Code Style

- No code comments unless explicitly asked for.
- Keep scripts simple and direct — this is a 5-hour time-boxed experiment, not a production codebase. Don't add abstraction, config layers, or CLI polish that isn't needed to finish the phases.

## Working Style

- Follow the phase order and time budget in `PROJECT_BRIEF.md`.
- Don't stop to ask permission for routine steps inside a phase — execute, then report status when a phase completes or if something's blocked (e.g. VRAM mismatch, gated model access not yet approved).
- Flag hardware/config mismatches immediately (Phase 0), before starting training — don't discover an OOM mid-run.
- The final report (`reports/evaluation.md`) must document failure cases honestly. Don't polish over a weak result.
