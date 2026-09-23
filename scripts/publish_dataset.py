import argparse
import collections
import csv
import json
import os

CARD = """---
license: other
license_name: mixed-cc
license_link: https://creativecommons.org/licenses/
task_categories:
- text-to-image
language:
- en
tags:
- architecture
- islamic-architecture
- mamluk
- cairo
- lora
- flux
pretty_name: ArchForge - Mamluk / Islamic Cairo Architecture
size_categories:
- n<1K
---

# ArchForge — Mamluk / Islamic Cairo Architecture Dataset

A small, curated, licence-documented image dataset of **Mamluk and Islamic Egyptian
architecture**, built for LoRA style adaptation on FLUX.1-dev. {count} images at
{size}x{size}px, {train} train / {val} validation.

Built as part of [ArchForge]({repo_url}) — a time-boxed proof of concept,
not a production dataset. The evaluation, the LoRA adapter and the comparison grid
are linked from there.

## Why this dataset exists

The target is a *style*, not a subject: carved limestone facades, pointed arches,
muqarnas vaulting, mashrabiya screens, ablaq striped stonework, domes and minarets
of historic Cairo. The curation work was therefore mostly **rejection** — Wikimedia
Commons categories mix architecture with book scans, museum objects, architectural
plans, paintings, lithographs, and buildings from other countries entirely.

## Provenance and licensing

Every image comes from **Wikimedia Commons** and retains its original licence.
No image was scraped from a source without a checkable licence, and no licence was
inferred. Attribution for each file is recorded in `metadata.csv` and listed below.

| Licence | Images |
|---|---|
{license_table}

**Reuse obligation:** most files are CC BY or CC BY-SA, which require attribution
and (for SA) share-alike. If you reuse this dataset, carry the per-image attribution
in the table below. Public domain and "no restrictions" files carry no such condition.

## Curation method

1. **Collection** (`scripts/collect_dataset.py`) — MediaWiki API enumeration over 36
   Mamluk/Islamic Cairo categories, filtering on a 800px minimum edge and JPEG/PNG only.
2. **Review** — every candidate was inspected visually on a contact sheet at two
   zoom levels. 123 candidates were reviewed; {count} were kept.
3. **Rejection** — {rejected} candidates were rejected for a recorded reason. The
   dominant failure modes were:
   - **Wrong country** — Portuguese colonial facades, Agra and Delhi monuments, Aleppo,
     the Armenian Quarter, all reaching these categories through shared vocabulary
     ("mashrabiya", "muqarnas").
   - **Not a photograph** — 19th-century lithographs (David Roberts), orientalist
     paintings (J. F. Lewis), book scans, architectural plans.
   - **Modern architecture** — 20th/21st-century apartment towers, modern mosques with
     no Mamluk vocabulary, construction cranes, night shots dominated by LED lighting.
   - **Museum objects** — mashrabiya screens photographed as artefacts, not in situ.
   Each rejection and its reason is recorded in `data/exclusions.csv`.
4. **Quality control** (`scripts/build_dataset.py`) — corrupted-file drop, perceptual-hash
   deduplication (64-bit pHash, Hamming distance <= 10), 512px centre-crop and resize,
   deterministic 85/15 split seeded at 0.

Deduplication found **0 duplicates** at that threshold; this was verified rather than
assumed, by confirming the hash function separates a JPEG-recompressed copy of an
image (distance 0) from a genuinely different image (distance 36).

## Structure

```
data/train/         {train} images, 512x512 PNG
data/validation/    {val} images, 512x512 PNG
metadata.csv        image_path, caption, source, architectural_style, split,
                    license, license_url, artist, title, category, phash
captions.jsonl      caption, style token, VLM base description, provenance
data/selected.csv   the curated allowlist (pageid + note)
data/exclusions.csv rejected candidates with reasons
```

## Captions

Each caption follows `[STYLE TOKEN] + description + structural elements + materials + viewpoint`,
for example:

> `mamluk_architecture, historic Islamic courtyard in Cairo, carved limestone facade, pointed arches, geometric stone ornamentation, warm natural light, architectural photography`

The architectural vocabulary is grounded in the image's recorded Wikimedia category
rather than generated freely, so the template cannot invent a minaret on a doorway.

## Limitations

- **Small** ({count} images) and heavily weighted toward Cairo. Not a general Islamic
  architecture dataset.
- **Wikimedia Commons bias** — photographs are skewed toward well-documented,
  tourist-accessible monuments, and toward the aesthetics of the contributing
  photographers.
- **Not deduplicated against the wider web** — only against itself.
- Some images contain minor modern intrusions (street furniture, signage, vehicles)
  that were judged not to dominate the frame.

## Per-image attribution

| # | File | Source | Licence | Author |
|---|---|---|---|---|
{attribution_table}
"""


def escape(text):
    return (text or "").replace("|", "/").strip() or "—"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--exclusions", default="data/exclusions.csv")
    parser.add_argument("--out", default="data/hf_dataset_card.md")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--repo-url", default="https://github.com/marwantosolve/ArchForge")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rejected = 0
    if os.path.exists(args.exclusions):
        with open(args.exclusions, newline="", encoding="utf-8") as handle:
            rejected = len(list(csv.DictReader(handle)))

    counts = collections.Counter(row.get("license", "UNKNOWN") for row in rows)
    license_table = "\n".join(
        f"| {name} | {count} |" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )

    attribution = []
    for index, row in enumerate(rows, 1):
        source = row.get("source", "")
        link = f"[Commons]({source})" if source else "—"
        attribution.append(
            f"| {index} | {escape(row.get('title', '').replace('File:', ''))} | {link} | "
            f"{escape(row.get('license'))} | {escape(row.get('artist'))} |"
        )

    train = sum(1 for row in rows if row.get("split") == "train")
    validation = sum(1 for row in rows if row.get("split") == "validation")

    card = CARD.format(
        count=len(rows),
        size=512,
        train=train,
        val=validation,
        rejected=rejected,
        repo_url=args.repo_url,
        license_table=license_table,
        attribution_table="\n".join(attribution),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(card)
    print(f"[done] card -> {args.out}")

    if args.push:
        from datasets import Dataset, Image as HFDatasetImage
        from huggingface_hub import HfApi

        if not args.repo_id:
            raise SystemExit("--repo-id is required with --push")

        images = [row["image_path"] for row in rows]
        texts = [row.get("caption", "") for row in rows]
        licences = [row.get("license", "") for row in rows]
        sources = [row.get("source", "") for row in rows]
        artists = [row.get("artist", "") for row in rows]
        splits = [row.get("split", "") for row in rows]

        dataset = Dataset.from_dict(
            {
                "image": images,
                "text": texts,
                "license": licences,
                "source": sources,
                "artist": artists,
                "split": splits,
            }
        ).cast_column("image", HFDatasetImage())

        dataset.push_to_hub(args.repo_id, private=False)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.out,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=args.metadata,
            path_in_repo="metadata.csv",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"[done] pushed https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
