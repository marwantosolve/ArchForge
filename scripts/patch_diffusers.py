import argparse
import os
import re
import sys

HARDCODED = re.compile(r"""["']Norod78/Yarn-art-style["']""")
REPLACEMENT = 'os.environ.get("ARCHFORGE_DATASET", "Norod78/Yarn-art-style")'

# transformers 5 removed load_in_8bit as a from_pretrained kwarg - it has to go through
# a BitsAndBytesConfig now, or T5EncoderModel.__init__ rejects it. Upstream still uses
# the old form, so without this the embeddings step dies before it starts.
LOAD_IN_8BIT = re.compile(r"load_in_8bit=True")
QUANTIZED = "quantization_config=BitsAndBytesConfig(load_in_8bit=True)"
T5_IMPORT = "from transformers import T5EncoderModel"
T5_IMPORT_BNB = "from transformers import BitsAndBytesConfig, T5EncoderModel"

QUANT_DIR = "examples/research_projects/flux_lora_quantization"
TARGETS = [f"{QUANT_DIR}/compute_embeddings.py", f"{QUANT_DIR}/train_dreambooth_lora_flux_miniature.py"]
EMBEDDINGS = f"{QUANT_DIR}/compute_embeddings.py"

# The T5 embeddings come back as bfloat16 - that is the dtype FLUX stores its text
# encoder in, and nothing overrides it here. numpy has no bfloat16 dtype, so .numpy()
# raises "Got unsupported ScalarType BFloat16" and the run dies on the serialization
# line, after the expensive encoding pass has already finished and printed its results.
# Cast to float32 on the way out; the values are bf16 to begin with, so nothing is lost.
#
# The explicit pd.Series is belt and braces: newer pandas broadcasts an apply returning
# equal-length lists into a 2-D array, which cannot be assigned back to one column.
CAST = "(x.cpu().to(torch.float32).numpy() if x.is_floating_point() else x.cpu().numpy())"
RAW_CAST = "x.cpu().numpy()"

APPLY_TENSORS = re.compile(
    r"^(\s*)df\[col\] = df\[col\]\.apply\(lambda x: x\.cpu\(\)\.numpy\(\)\.flatten\(\)\.tolist\(\)\)$",
    re.M,
)


def _object_series(match):
    indent = match.group(1)
    return (
        f"{indent}df[col] = pd.Series(\n"
        f"{indent}    [x.cpu().numpy().flatten().tolist() for x in df[col]],\n"
        f"{indent}    index=df.index,\n"
        f"{indent}    dtype=object,\n"
        f"{indent})"
    )


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write(path, source):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)


def patch_dataset(path):
    name = os.path.basename(path)
    source = read(path)
    if REPLACEMENT in source:
        print(f"[skip] dataset already redirected: {name}")
        return "already"
    count = len(HARDCODED.findall(source))
    if not count:
        print(f"[warn] hardcoded dataset string not found in {name}")
        if "Norod78" in source:
            print("[warn]   'Norod78' is present but not in the expected form - patch by hand")
        return "miss"
    source = HARDCODED.sub(REPLACEMENT, source)
    if not re.search(r"^import os$", source, re.M):
        source = "import os\n" + source
    write(path, source)
    print(f"[ok] dataset: redirected {count} occurrence(s) in {name}")
    return "patched"


def patch_8bit(path):
    name = os.path.basename(path)
    source = read(path)
    if QUANTIZED in source:
        print(f"[skip] 8-bit already rewritten: {name}")
        return
    if not LOAD_IN_8BIT.search(source):
        print(f"[warn] load_in_8bit=True not found in {name} - check whether upstream fixed it")
        return
    source = LOAD_IN_8BIT.sub(QUANTIZED, source)
    if T5_IMPORT in source and T5_IMPORT_BNB not in source:
        source = source.replace(T5_IMPORT, T5_IMPORT_BNB, 1)
    if "BitsAndBytesConfig" not in source:
        print(f"[error] could not add the BitsAndBytesConfig import to {name}", file=sys.stderr)
        return
    write(path, source)
    print(f"[ok] 8-bit: rewrote load_in_8bit in {name}")


def patch_embeddings(path):
    name = os.path.basename(path)
    source = read(path)
    if CAST in source:
        print(f"[skip] embeddings already cast out of bfloat16: {name}")
        return
    if APPLY_TENSORS.search(source):
        source = APPLY_TENSORS.sub(_object_series, source)
        print(f"[ok] pandas: made the embedding conversion explicit in {name}")
    if RAW_CAST not in source:
        print(f"[warn] the tensor-to-numpy conversion was not found in {name}")
        return
    source = source.replace(RAW_CAST, CAST)
    write(path, source)
    print(f"[ok] dtype: embeddings are cast to float32 before numpy in {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to a diffusers checkout")
    args = parser.parse_args()

    seen = 0
    misses = 0
    for relative in TARGETS:
        path = os.path.join(args.repo, relative)
        if not os.path.exists(path):
            print(f"[skip] not found: {relative}")
            continue
        seen += 1
        if patch_dataset(path) == "miss":
            misses += 1

    if not seen or misses == seen:
        print("[error] nothing patched - the demo dataset reference is still in place", file=sys.stderr)
        print("[error] training would silently run on the Norod78 yarn-art demo dataset", file=sys.stderr)
        return 1

    embeddings = os.path.join(args.repo, EMBEDDINGS)
    if os.path.exists(embeddings):
        patch_8bit(embeddings)
        patch_embeddings(embeddings)

    print("[done] set ARCHFORGE_DATASET to your dataset path before running")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
