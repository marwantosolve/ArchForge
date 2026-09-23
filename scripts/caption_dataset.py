import argparse
import csv
import json
import os

from PIL import Image

STYLE_TOKEN = "mamluk_architecture"

CATEGORY_TERMS = {
    "Category:Muqarnas": "muqarnas vaulting",
    "Category:Mashrabiya": "mashrabiya wooden lattice screen",
    "Category:Minarets in Cairo": "slender stone minaret",
    "Category:Domes in Cairo": "carved stone dome",
    "Category:Doors in Cairo": "traditional wooden door",
    "Category:Mosques in Cairo": "historic mosque facade",
    "Category:Madrasas in Cairo": "Mamluk madrasa facade",
    "Category:Islamic Cairo": "historic Islamic Cairo street",
    "Category:Mamluk architecture in Cairo": "Mamluk stone facade",
    "Category:Qalawun complex": "Mamluk religious complex",
    "Category:Khan el-Khalili": "historic bazaar arcade",
    "Category:Al-Azhar Mosque": "mosque courtyard and minarets",
    "Category:Ibn Tulun Mosque": "hypostyle mosque arcades",
    "Category:Islamic geometric patterns": "geometric stone ornamentation",
    "Category:Historic Cairo": "historic Cairo architecture",
    "Category:Architecture of Cairo": "Cairo stone architecture",
    "Category:City of the Dead (Cairo)": "historic cemetery architecture",
}

STYLE_TAIL = "carved limestone ornamentation, pointed arches, warm natural light, architectural photography"


def terms_for(category):
    return CATEGORY_TERMS.get(category, "historic Cairene stone architecture")


def build_caption(base, category, title):
    parts = [STYLE_TOKEN]
    if base:
        parts.append(base.strip().rstrip("."))
    parts.append(terms_for(category))
    parts.append(STYLE_TAIL)
    seen = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return ", ".join(seen)


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    return processor, model


def describe(processor, model, image, device):
    import torch

    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(text, task=prompt, image_size=image.size)
    return parsed.get(prompt, "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--out", default="captions.jsonl")
    parser.add_argument("--model", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    processor = model = None
    device = "cpu"
    if not args.no_vlm:
        try:
            processor, model = load_model(args.model)
            try:
                import torch

                if torch.cuda.is_available():
                    model = model.to("cuda")
                    device = "cuda"
            except Exception:
                pass
            print(f"[info] loaded {args.model} on {device}")
        except Exception as exc:
            print(f"[warn] VLM unavailable ({exc}); falling back to template-only captions")
            processor = model = None

    records = []
    updated = []
    for index, row in enumerate(rows):
        base = ""
        if processor is not None:
            try:
                with Image.open(row["image_path"]) as image:
                    base = describe(processor, model, image.convert("RGB"), device)
            except Exception as exc:
                print(f"[warn] caption failed for {row['image_path']}: {exc}")
        caption = build_caption(base, row.get("category", ""), row.get("title", ""))
        record = {
            "image_path": row["image_path"],
            "caption": caption,
            "style_token": STYLE_TOKEN,
            "vlm_base": base,
            "category": row.get("category", ""),
            "split": row.get("split", ""),
            "source": row.get("source", ""),
            "license": row.get("license", ""),
        }
        records.append(record)
        row["caption"] = caption
        updated.append(row)
        print(f"[{index + 1}/{len(rows)}] {caption[:110]}")

    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = list(updated[0].keys())
    with open(args.metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(updated)

    print(f"[done] {len(records)} captions -> {args.out}")
    print(f"[done] metadata.csv updated")


if __name__ == "__main__":
    main()
