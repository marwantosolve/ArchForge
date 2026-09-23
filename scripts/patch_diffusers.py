import argparse
import os
import sys

HARDCODED = '"Norod78/Yarn-art-style"'
REPLACEMENT = 'os.environ.get("ARCHFORGE_DATASET", "Norod78/Yarn-art-style")'

TARGETS = [
    "examples/research_projects/flux_lora_quantization/compute_embeddings.py",
    "examples/research_projects/flux_lora_quantization/train_dreambooth_lora_flux_miniature.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to a diffusers checkout")
    args = parser.parse_args()

    patched = 0
    for relative in TARGETS:
        path = os.path.join(args.repo, relative)
        if not os.path.exists(path):
            print(f"[skip] not found: {relative}")
            continue
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        if REPLACEMENT in source:
            print(f"[skip] already patched: {relative}")
            continue
        if HARDCODED not in source:
            print(f"[warn] hardcoded dataset string not found in {relative}")
            continue
        count = source.count(HARDCODED)
        source = source.replace(HARDCODED, REPLACEMENT)
        if "import os" not in source:
            source = "import os\n" + source
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        patched += 1
        print(f"[ok] patched {count} occurrence(s) in {relative}")

    if not patched:
        print("[error] nothing patched", file=sys.stderr)
        return 1
    print("[done] set ARCHFORGE_DATASET to your dataset path before running")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
