import argparse
import csv
import json
import math
import os
import random
import shutil

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

Image.MAX_IMAGE_PIXELS = None

TARGET = 512
MIN_EDGE = 512
HASH_SIZE = 8
DUP_THRESHOLD = 10
STYLE = "mamluk_architecture"


def dct_1d(vec):
    n = len(vec)
    out = [0.0] * n
    for k in range(n):
        total = 0.0
        factor = math.pi * k / (2 * n)
        for i, value in enumerate(vec):
            total += value * math.cos(factor * (2 * i + 1))
        out[k] = total
    return out


def dct_2d(matrix):
    rows = [dct_1d(row) for row in matrix]
    cols = list(zip(*rows))
    transformed = [dct_1d(list(col)) for col in cols]
    return list(zip(*transformed))


def phash(image):
    small = image.convert("L").resize((HASH_SIZE * 4, HASH_SIZE * 4), Image.LANCZOS)
    pixels = list(small.tobytes())
    size = HASH_SIZE * 4
    matrix = [pixels[i * size : (i + 1) * size] for i in range(size)]
    coeffs = dct_2d(matrix)
    low = [coeffs[y][x] for y in range(HASH_SIZE) for x in range(HASH_SIZE)][1:]
    ordered = sorted(low)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    bits = 0
    for index, value in enumerate(low):
        if value > median:
            bits |= 1 << index
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def center_square_crop(image):
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    return image.crop((left, top, left + edge, top + edge))


def load_sources(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_exclusions(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["pageid"]: row.get("reason", "") for row in csv.DictReader(handle)}


def clear_outputs(directory):
    os.makedirs(directory, exist_ok=True)
    for name in os.listdir(directory):
        if name.endswith(".png"):
            os.remove(os.path.join(directory, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--filtered", default="data/filtered")
    parser.add_argument("--train", default="data/train")
    parser.add_argument("--validation", default="data/validation")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--exclusions", default="data/exclusions.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sources = load_sources(os.path.join(args.raw, "sources.csv"))
    exclusions = load_exclusions(args.exclusions)
    for directory in (args.filtered, args.train, args.validation):
        clear_outputs(directory)

    kept = []
    corrupted = 0
    too_small = 0
    excluded = 0

    for row in sources:
        if row["pageid"] in exclusions:
            excluded += 1
            continue
        path = row["image_path"]
        if not os.path.exists(path):
            corrupted += 1
            continue
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                if min(width, height) < MIN_EDGE:
                    too_small += 1
                    continue
                squared = center_square_crop(image.convert("RGB"))
                resized = squared.resize((TARGET, TARGET), Image.LANCZOS)
                digest = phash(resized)
                out_name = f"{row['pageid']}.png"
                out_path = os.path.join(args.filtered, out_name)
                resized.save(out_path, "PNG", optimize=True)
        except Exception:
            corrupted += 1
            continue

        kept.append(
            {
                "pageid": row["pageid"],
                "filtered_path": out_path,
                "hash": digest,
                "source": row.get("descriptionsource", ""),
                "license": row.get("license", "UNKNOWN"),
                "license_url": row.get("license_url", ""),
                "artist": row.get("artist", ""),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
            }
        )

    unique = []
    duplicates = 0
    for record in kept:
        is_dupe = False
        for existing in unique:
            if hamming(record["hash"], existing["hash"]) <= DUP_THRESHOLD:
                is_dupe = True
                existing.setdefault("duplicates", []).append(record["pageid"])
                break
        if is_dupe:
            duplicates += 1
            os.remove(record["filtered_path"])
            continue
        unique.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(unique)
    val_count = max(1, round(len(unique) * args.val_fraction)) if unique else 0
    for index, record in enumerate(unique):
        split = "validation" if index < val_count else "train"
        destination = args.validation if split == "validation" else args.train
        final = os.path.join(destination, os.path.basename(record["filtered_path"]))
        shutil.move(record["filtered_path"], final)
        record["split"] = split
        record["image_path"] = final

    rows = []
    for record in unique:
        rows.append(
            {
                "image_path": record["image_path"],
                "caption": "",
                "source": record["source"],
                "architectural_style": STYLE,
                "split": record["split"],
                "license": record["license"],
                "license_url": record["license_url"],
                "artist": record["artist"],
                "title": record["title"],
                "category": record["category"],
                "phash": f"{record['hash']:016x}",
                "duplicate_of": ",".join(record.get("duplicates", [])),
            }
        )

    fields = [
        "image_path",
        "caption",
        "source",
        "architectural_style",
        "split",
        "license",
        "license_url",
        "artist",
        "title",
        "category",
        "phash",
        "duplicate_of",
    ]
    with open(args.metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input": len(sources),
        "kept": len(unique),
        "excluded_manually": excluded,
        "corrupted_or_missing": corrupted,
        "below_resolution_floor": too_small,
        "duplicates_removed": duplicates,
        "train": sum(1 for r in rows if r["split"] == "train"),
        "validation": sum(1 for r in rows if r["split"] == "validation"),
        "target_size": TARGET,
        "license_breakdown": {},
    }
    for row in rows:
        summary["license_breakdown"][row["license"]] = summary["license_breakdown"].get(row["license"], 0) + 1

    with open("data/qc_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"input:                 {summary['input']}")
    print(f"kept:                  {summary['kept']}")
    print(f"excluded (manual):     {summary['excluded_manually']}")
    print(f"corrupted/missing:     {summary['corrupted_or_missing']}")
    print(f"below resolution floor:{summary['below_resolution_floor']}")
    print(f"duplicates removed:    {summary['duplicates_removed']}")
    print(f"train / validation:    {summary['train']} / {summary['validation']}")
    print("licenses:")
    for name, count in sorted(summary["license_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3}  {name}")


if __name__ == "__main__":
    main()
