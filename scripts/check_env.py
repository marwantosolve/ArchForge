import importlib
import os
import shutil
import subprocess
import sys

REQUIRED = ["torch", "diffusers", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors"]


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("python")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")

    section("gpu")
    try:
        import torch

        print(f"torch:            {torch.__version__}")
        print(f"cuda available:   {torch.cuda.is_available()}")
        print(f"cuda version:     {torch.version.cuda}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print(f"device count:     {count}")
            for index in range(count):
                props = torch.cuda.get_device_properties(index)
                total = props.total_memory / 1024**3
                print(f"  [{index}] {props.name}")
                print(f"      capability: sm_{props.major}{props.minor}")
                print(f"      vram:       {total:.1f} GB")
                print(f"      bf16 native: {props.major >= 8}")
            free, total = torch.cuda.mem_get_info()
            print(f"vram free now:    {free / 1024**3:.1f} / {total / 1024**3:.1f} GB")
        else:
            print("!! no CUDA device — training cannot run here")
    except Exception as exc:
        print(f"!! torch unavailable: {exc}")

    section("packages")
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"  {name:14} {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            print(f"  {name:14} MISSING ({type(exc).__name__})")

    section("quantization support")
    try:
        import bitsandbytes as bnb

        print(f"bitsandbytes: {bnb.__version__}")
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                print(f"nf4 (4-bit) supported:  True (sm_{major}{minor})")
                print(f"fp8 supported:          {major >= 9}  (requires sm_89+, Turing is sm_75)")
        except Exception as exc:
            print(f"capability probe failed: {exc}")
    except Exception as exc:
        print(f"!! bitsandbytes unusable: {exc}")

    section("disk")
    for label, path in (("cwd", "."), ("/content", "/content"), ("/root", os.path.expanduser("~"))):
        try:
            usage = shutil.disk_usage(path)
            print(f"  {label:10} {usage.free / 1024**3:7.1f} GB free of {usage.total / 1024**3:.1f} GB")
        except Exception as exc:
            print(f"  {label:10} unavailable ({exc})")

    section("hugging face auth")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"HF_TOKEN env set: {bool(token)}")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        print(f"logged in as: {who.get('name')}")
        print(f"token type:   {who.get('type')}")
    except Exception as exc:
        print(f"!! not authenticated: {exc}")
        print("   run: huggingface_hub.login() or set HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download("black-forest-labs/FLUX.1-dev", "model_index.json")
        print("FLUX.1-dev access: GRANTED")
    except Exception as exc:
        print(f"FLUX.1-dev access: NOT AVAILABLE ({type(exc).__name__})")

    section("google drive")
    mounted = os.path.ismount("/content/drive") or os.path.isdir("/content/drive/MyDrive")
    print(f"drive mounted: {mounted}")
    if not mounted:
        print("   run: from google.colab import drive; drive.mount('/content/drive')")

    section("project data")
    for path in ("data/train", "data/validation", "metadata.csv", "captions.jsonl", "prompts.json"):
        exists = os.path.exists(path)
        count = ""
        if os.path.isdir(path):
            count = f" ({len([f for f in os.listdir(path) if f.endswith('.png')])} png)"
        print(f"  {path:20} {'OK' if exists else 'MISSING'}{count}")


if __name__ == "__main__":
    main()
