"""
generate_images.py
------------------
Generates images from a fine-tuned PixArt-alpha LoRA checkpoint on HuggingFace.
Saves images to an output directory (skips already-generated).

Usage:
    python generate_images.py \
        --captions      captions_2.json \
        --output_dir    gen_normal_500 \
        --hf_repo       arpita2desh/pixart-hands-lora \
        --checkpoint    checkpoint-500 \
        --device        cuda:0

Dependencies:
    pip install diffusers transformers sentencepiece accelerate torch peft
    pip install "huggingface_hub==0.23.4"
"""

import os
import json
import argparse
import torch
from pathlib import Path
from diffusers import PixArtAlphaPipeline
from peft import PeftModel


def load_captions(captions_path: str) -> list[dict]:
    with open(captions_path) as f:
        data = json.load(f)

    entries = []
    for filename, meta in data.items():
        entries.append({
            "filename": filename,
            "caption":  meta["caption"],
        })

    print(f"Loaded {len(entries)} captions from {captions_path}")
    return entries


def generate(
    captions_path: str,
    output_dir: str,
    hf_repo: str,
    checkpoint: str,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base pipeline ───────────────────────────────────
    base_model = "PixArt-alpha/PixArt-XL-2-1024-MS"
    print(f"\nLoading base PixArt-alpha: {base_model}")
    pipe = PixArtAlphaPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    ).to(device)

    # ── Attach LoRA from HF repo subfolder ──────────────────
    print(f"Loading LoRA adapter from HF: {hf_repo}/{checkpoint}")
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        hf_repo,
        subfolder=checkpoint,
    )

    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(seed)

    # ── Load captions ────────────────────────────────────────
    entries = load_captions(captions_path)

    # ── Generate ─────────────────────────────────────────────
    print(f"\nGenerating {len(entries)} images → {output_dir}\n")

    skipped   = 0
    generated = 0

    for i, entry in enumerate(entries):
        stem     = Path(entry["filename"]).stem
        out_path = output_dir / f"{stem}.png"

        if out_path.exists():
            skipped += 1
            continue

        caption = entry["caption"]
        print(f"[{i+1}/{len(entries)}] {entry['filename']}")
        print(f"  Prompt: {caption[:100]}{'...' if len(caption) > 100 else ''}")

        with torch.no_grad():
            image = pipe(
                prompt=caption,
                negative_prompt="extra fingers, missing fingers, fused fingers, bad anatomy, deformed",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]

        image.save(out_path)
        generated += 1
        print(f"  Saved → {out_path}")

    print(f"\nDone. Generated: {generated} | Skipped: {skipped}")
    print(f"All images in: {output_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions",   default="images_hands_3000_11k/captions.json",            help="Path to captions JSON")
    p.add_argument("--output_dir", default="gen_hand_final",             help="Where to save generated images")
    p.add_argument("--hf_repo",    default="arpita2desh/pixart-hands-lora",     help="HF repo ID, e.g. arpita2desh/pixart-hands-lora")
    p.add_argument("--checkpoint", default="checkpoint-final",             help="Subfolder in HF repo, e.g. checkpoint-500")
    p.add_argument("--device",     default="cuda",                       help="cuda / cuda:0 / cuda:1 / cpu")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=7.5)
    p.add_argument("--height",     type=int,   default=512)
    p.add_argument("--width",      type=int,   default=512)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        captions_path=args.captions,
        output_dir=args.output_dir,
        hf_repo=args.hf_repo,
        checkpoint=args.checkpoint,
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
    )