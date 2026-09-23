import argparse
import csv
import math
import os

from PIL import Image, ImageDraw

CELL = 240
COLS = 6
PAD = 22


def build(items, out_path, cell=CELL, cols=COLS):
    rows = math.ceil(len(items) / cols)
    width = cols * cell
    height = rows * (cell + PAD)
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(items):
        col = index % cols
        row = index // cols
        x = col * cell
        y = row * (cell + PAD)
        try:
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
        except Exception:
            draw.text((x + 6, y + 6), "UNREADABLE", fill=(255, 80, 80))
            continue
        sheet.paste(thumb, (x + 4, y + PAD))
        draw.text((x + 6, y + 5), f"{index:02d} {label}"[:38], fill=(235, 235, 235))
    sheet.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--out", default="reports/contact_sheet.png")
    parser.add_argument("--cell", type=int, default=CELL)
    args = parser.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    items = []
    for row in rows:
        title = os.path.basename(row.get("title", "")).rsplit(".", 1)[0]
        items.append((row["image_path"], title.replace("File:", "")))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build(items, args.out, cell=args.cell)
    print(f"{len(items)} images -> {args.out}")


if __name__ == "__main__":
    main()
