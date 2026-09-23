import argparse
import json
import os
import time

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel


def load_pipeline(model_id, quantize, offload):
    kwargs = {"torch_dtype": torch.float16}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    text_encoder_2 = None
    if quantize:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", **kwargs
        )

    transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", **kwargs)

    extra = {"transformer": transformer}
    if text_encoder_2 is not None:
        extra["text_encoder_2"] = text_encoder_2

    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        **extra,
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="prompts.json")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    steps = args.steps or config["num_inference_steps"]
    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipeline(args.model, not args.no_quantize, not args.no_offload)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        print(f"[info] loaded LoRA from {args.lora}")

    print(
        f"[info] generating {len(config['prompts'])} images at {steps} steps, "
        f"{config['width']}x{config['height']} - the first one takes the longest",
        flush=True,
    )

    manifest = []
    for index, prompt in enumerate(config["prompts"]):
        generator = torch.Generator(device="cpu").manual_seed(config["seed"] + index)
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=config["guidance_scale"],
            height=config["height"],
            width=config["width"],
            generator=generator,
        ).images[0]
        elapsed = time.time() - start
        name = f"{index:02d}_{args.tag}.png"
        path = os.path.join(args.out, name)
        image.save(path)
        manifest.append(
            {
                "index": index,
                "prompt": prompt,
                "tag": args.tag,
                "image_path": path,
                "seed": config["seed"] + index,
                "steps": steps,
                "guidance_scale": config["guidance_scale"],
                "height": config["height"],
                "width": config["width"],
                "lora": args.lora,
                "seconds": round(elapsed, 1),
            }
        )
        print(f"[{index + 1}/{len(config['prompts'])}] {elapsed:5.1f}s  {path}")

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    total = sum(item["seconds"] for item in manifest)
    print(f"[done] {len(manifest)} images in {total / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
