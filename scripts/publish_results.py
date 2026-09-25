import argparse
import json
import os

LORA_CARD = """---
base_model: black-forest-labs/FLUX.1-dev
library_name: peft
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
tags:
- text-to-image
- flux
- lora
- diffusers
- architecture
- islamic-architecture
- mamluk
- cairo
pretty_name: ArchForge - Mamluk / Islamic Cairo LoRA (FLUX.1-dev)
---

# ArchForge — Mamluk / Islamic Cairo LoRA

A FLUX.1-dev LoRA adapter for **Mamluk and Islamic Egyptian architecture**: carved
limestone facades, pointed arches, muqarnas vaulting, mashrabiya screens, ablaq
striped stonework, domes and minarets of historic Cairo.

Trained as part of [ArchForge]({repo_url}) — a time-boxed proof of concept, **not a
production model**. Read the limitations below before using it.

## Training

| | |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | LoRA, transformer attention layers only (`to_k`, `to_q`, `to_v`, `to_out.0`) |
| Base quantisation | NF4 (bitsandbytes), 4-bit |
| Resolution | {resolution} |
| Rank / alpha | {rank} / {rank} |
| Precision | fp16 + GradScaler |
| Optimizer | 8-bit AdamW |
| Learning rate | {learning_rate}, constant schedule, no warmup |
| Batch / accumulation | 1 / {grad_accum} |
| Steps | {steps} |
| Hardware | Google Colab free tier, Tesla T4, 16GB |

## Dataset

{dataset_line}

## Usage

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
).to("cuda")
pipe.load_lora_weights("{repo_id}")

image = pipe(
    "mamluk_architecture, a historic Islamic courtyard in Cairo, "
    "carved limestone facade, pointed arches, architectural photography",
    num_inference_steps=20,
    guidance_scale=3.5,
    height={resolution},
    width={resolution},
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("out.png")
```

The trigger token is `mamluk_architecture`. It is not required for the style to
appear, but it is what the captions conditioned on.

## Limitations

- **{count}-image dataset, {steps} steps.** This is a style *demonstration*, not a
  robust style model. Expect weak binding and inconsistent results.
- **Cairo-weighted and monument-biased.** Wikimedia contributors photograph
  well-documented, tourist-accessible buildings, and the adapter inherits that bias.
- **Non-commercial.** FLUX.1-dev is released under the FLUX.1-dev Non-Commercial
  License; this adapter inherits that restriction.
- **Documented failure cases** — see the evaluation report linked from
  {repo_url}. They are recorded deliberately rather than omitted.
"""

RESULTS_CARD = """---
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
tags:
- text-to-image
- flux
- lora
- architecture
- mamluk
- evaluation
pretty_name: ArchForge - Base vs LoRA Evaluation
---

# ArchForge — Base vs LoRA Evaluation

Controlled comparison of `black-forest-labs/FLUX.1-dev` against the
[ArchForge Mamluk LoRA]({lora_url}) on a fixed six-prompt suite.

Identical seed, step count, guidance scale and resolution across both runs — only the
adapter differs. That is what makes the comparison controlled.

## Files

| Path | What it is |
|---|---|
| `comparison_grid.png` | Prompt / Base / LoRA grid, all six pairs |
| `evaluation.md` | The full write-up, including failure cases |
| `baseline/` | Base-model generations |
| `comparison/` | LoRA generations |
| `baseline/manifest.json` | Seed, steps, guidance, resolution, wall clock |
| `comparison/manifest.json` | Same, for the LoRA run |

## Prompts

{prompts}

## Caveat

Six prompts, one seed each. This is a controlled comparison, not a statistically
meaningful evaluation.
"""


def read_manifest(directory):
    path = os.path.join(directory, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-dir", default="models/archforge_lora")
    parser.add_argument("--baseline", default="results/baseline")
    parser.add_argument("--comparison", default="results/comparison")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--lora-repo", default="")
    parser.add_argument("--results-repo", default="")
    parser.add_argument("--dataset-repo", default="")
    parser.add_argument("--repo-url", default="https://github.com/marwantosolve/ArchForge")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=400,
        help="training steps for the LoRA card. The generation manifests only carry "
        "inference steps, so this cannot be read from them.",
    )
    parser.add_argument("--out", default="data")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(args.comparison) or read_manifest(args.baseline)
    first = manifest[0] if manifest else {}
    steps = args.train_steps
    resolution = first.get("width", 512)

    prompts = []
    if os.path.exists(args.prompts):
        with open(args.prompts, encoding="utf-8") as handle:
            prompts = json.load(handle).get("prompts", [])
    prompt_list = (
        "\n".join(
            f"{i}. {p.get('prompt', '') if isinstance(p, dict) else p}"
            for i, p in enumerate(prompts, 1)
        )
        or "—"
    )

    count = 39
    if os.path.exists("metadata.csv"):
        with open("metadata.csv", encoding="utf-8") as handle:
            count = max(sum(1 for _ in handle) - 1, 1)

    dataset_line = (
        f"[{args.dataset_repo}](https://huggingface.co/datasets/{args.dataset_repo}) — "
        f"{count} curated 512px images, per-image licence and attribution recorded."
        if args.dataset_repo
        else f"{count} curated 512px images, per-image licence and attribution recorded."
    )

    lora_card = LORA_CARD.format(
        repo_url=args.repo_url,
        repo_id=args.lora_repo or "ARCHFORGE_LORA_REPO",
        resolution=resolution,
        rank=first.get("rank", 8) or 8,
        learning_rate=first.get("learning_rate", 1e-4),
        grad_accum=first.get("gradient_accumulation_steps", 4) or 4,
        steps=steps,
        count=count,
        dataset_line=dataset_line,
    )

    results_card = RESULTS_CARD.format(
        lora_url=f"https://huggingface.co/{args.lora_repo}" if args.lora_repo else args.repo_url,
        prompts=prompt_list,
    )

    os.makedirs(args.out, exist_ok=True)
    lora_card_path = os.path.join(args.out, "lora_card.md")
    results_card_path = os.path.join(args.out, "results_card.md")
    with open(lora_card_path, "w", encoding="utf-8") as handle:
        handle.write(lora_card)
    with open(results_card_path, "w", encoding="utf-8") as handle:
        handle.write(results_card)
    print(f"[done] cards -> {lora_card_path}, {results_card_path}")

    if not args.push:
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    if args.lora_repo:
        create_repo(args.lora_repo, repo_type="model", exist_ok=True)
        if os.path.isdir(args.lora_dir):
            api.upload_folder(
                folder_path=args.lora_dir,
                repo_id=args.lora_repo,
                repo_type="model",
                ignore_patterns=["checkpoint-*", "*.optim", "*.pt", "*.bin"],
            )
        api.upload_file(
            path_or_fileobj=lora_card_path,
            path_in_repo="README.md",
            repo_id=args.lora_repo,
            repo_type="model",
        )
        print(f"[done] lora -> https://huggingface.co/{args.lora_repo}")

    if args.results_repo:
        create_repo(args.results_repo, repo_type="model", exist_ok=True)
        for directory in (args.baseline, args.comparison, args.reports):
            if not os.path.isdir(directory):
                continue
            api.upload_folder(
                folder_path=directory,
                path_in_repo=os.path.basename(directory),
                repo_id=args.results_repo,
                repo_type="model",
            )
        api.upload_file(
            path_or_fileobj=results_card_path,
            path_in_repo="README.md",
            repo_id=args.results_repo,
            repo_type="model",
        )
        print(f"[done] results -> https://huggingface.co/{args.results_repo}")


if __name__ == "__main__":
    main()
