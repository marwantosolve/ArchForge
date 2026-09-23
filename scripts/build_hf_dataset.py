import argparse
import csv
import json
import os

from datasets import Dataset, Image as HFDatasetImage

STYLE_TOKEN = "mamluk_architecture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--captions", default="captions.jsonl")
    parser.add_argument("--out", default="data/hf_dataset")
    parser.add_argument("--split", default="all")
    args = parser.parse_args()

    captions = {}
    if os.path.exists(args.captions):
        with open(args.captions, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    captions[record["image_path"]] = record["caption"]

    with open(args.metadata, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]

    images = []
    texts = []
    for row in rows:
        path = row["image_path"]
        if not os.path.exists(path):
            print(f"[warn] missing {path}, skipping")
            continue
        images.append(os.path.abspath(path))
        texts.append(captions.get(path) or f"{STYLE_TOKEN}, historic Cairene stone architecture")

    dataset = Dataset.from_dict({"image": images, "text": texts})
    dataset = dataset.cast_column("image", HFDatasetImage())
    dataset.save_to_disk(args.out)
    print(f"[done] {len(images)} rows -> {args.out}")
    print(f"[info] columns: {dataset.column_names}")
    print(f"[info] sample text: {texts[0][:100] if texts else 'n/a'}")


if __name__ == "__main__":
    main()
