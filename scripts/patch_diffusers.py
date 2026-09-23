import argparse
import os
import re
import sys

HARDCODED = re.compile(r"""["']Norod78/Yarn-art-style["']""")
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
    already = 0
    for relative in TARGETS:
        path = os.path.join(args.repo, relative)
        if not os.path.exists(path):
            print(f"[skip] not found: {relative}")
            continue
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        if REPLACEMENT in source:
            print(f"[skip] already patched: {relative}")
            already += 1
            continue
        count = len(HARDCODED.findall(source))
        if not count:
            print(f"[warn] hardcoded dataset string not found in {relative}")
            if "Norod78" in source:
                print("[warn]   'Norod78' is present but not in the expected form - patch by hand")
            continue
        source = HARDCODED.sub(REPLACEMENT, source)
        if not re.search(r"^import os$", source, re.M):
            source = "import os\n" + source
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        patched += 1
        print(f"[ok] patched {count} occurrence(s) in {relative}")

    if not patched and not already:
        print("[error] nothing patched - the demo dataset reference is still in place", file=sys.stderr)
        print("[error] training would silently run on the Norod78 yarn-art demo dataset", file=sys.stderr)
        return 1
    print(f"[done] {patched} patched, {already} already patched")
    print("[done] set ARCHFORGE_DATASET to your dataset path before running")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
