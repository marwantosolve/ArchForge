import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw

CELL = 320
HEADER = 34
LABEL_W = 300


def wrap(text, width):
    return textwrap.wrap(text, width=width)[:4]


def load_manifest(directory, tag):
    path = os.path.join(directory, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["index"]: row for row in rows if row.get("tag") == tag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/baseline")
    parser.add_argument("--lora", default="results/comparison")
    parser.add_argument("--out", default="reports/comparison_grid.png")
    parser.add_argument("--cell", type=int, default=CELL)
    args = parser.parse_args()

    base = load_manifest(args.baseline, "base")
    lora = load_manifest(args.lora, "lora")
    indices = sorted(set(base) & set(lora))
    if not indices:
        raise SystemExit("no overlapping prompt indices between baseline and lora manifests")

    cell = args.cell
    width = LABEL_W + 2 * cell
    height = HEADER + len(indices) * cell
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)

    draw.text((LABEL_W + cell // 2 - 22, 11), "BASE", fill=(240, 240, 240))
    draw.text((LABEL_W + cell + cell // 2 - 24, 11), "LORA", fill=(240, 240, 240))

    for row_index, prompt_index in enumerate(indices):
        y = HEADER + row_index * cell
        prompt = base[prompt_index]["prompt"]
        for line_index, line in enumerate(wrap(prompt, 34)):
            draw.text((10, y + 14 + line_index * 18), line, fill=(235, 235, 235))
        for col, manifest in enumerate((base, lora)):
            x = LABEL_W + col * cell
            path = manifest[prompt_index]["image_path"]
            try:
                with Image.open(path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((cell - 10, cell - 10), Image.LANCZOS)
                sheet.paste(thumb, (x + 5, y + 5))
            except Exception as exc:
                draw.text((x + 8, y + 8), f"missing: {exc}", fill=(255, 90, 90))
        draw.line([(0, y), (width, y)], fill=(60, 60, 64))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.save(args.out)
    print(f"{len(indices)} prompt pairs -> {args.out}")


if __name__ == "__main__":
    main()
