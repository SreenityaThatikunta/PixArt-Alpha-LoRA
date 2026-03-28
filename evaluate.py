# -*- coding: utf-8 -*-
"""
infer.py
-----------
Computes FID and T2I-CompBench scores for a PixArt-alpha model.

Usage:
    # Base model (no LoRA):
    python infer.py \
        --real_dir      images_hands_easy \
        --gen_dir       gen_hand_easy \
        --compbench_dir ./T2I-CompBench \
        --skip_gen

    # Fine-tuned LoRA from HuggingFace:
    python infer.py \
        --real_dir      images_hands_easy \
        --gen_dir       gen_lora_500 \
        --compbench_dir ./T2I-CompBench \
        --hf_repo       arpita2desh/pixart-hands-lora \
        --checkpoint    checkpoint-500 \
        --skip_gen

    # Skip FID only:
    python infer.py ... --skip_fid --skip_gen

Evaluation metrics (matching T2I-CompBench paper):
    - FID                  : clean-fid between real and generated images
    - Attribute Binding    : BLIP-VQA  (color / shape / texture)
    - Spatial              : UniDet 2D spatial eval
    - Non-Spatial          : CLIP similarity
    - Complex              : 3-in-1 (BLIP-VQA + UniDet + CLIP combined)

"""

import json
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import torch
from diffusers import PixArtAlphaPipeline
from cleanfid import fid as cleanfid
from peft import PeftModel


BASE_MODEL = "PixArt-alpha/PixArt-XL-2-1024-MS"


# ─────────────────────────────────────────────
# 1. FID
# ─────────────────────────────────────────────

def compute_fid(real_dir: str, gen_dir: str) -> float:
    """Computes FID between two folders using clean-fid."""
    print("\n" + "="*50)
    print("Computing FID...")
    print(f"  Real images : {real_dir}")
    print(f"  Gen images  : {gen_dir}")
    print("="*50)
    score = cleanfid.compute_fid(real_dir, gen_dir)
    print(f"\n✓ FID Score: {score:.4f}  (lower is better)\n")
    return score


# ─────────────────────────────────────────────
# 2. Pipeline loader
# ─────────────────────────────────────────────

def load_pipeline(
    hf_repo: str = None,
    checkpoint: str = None,
    device: str = "cuda",
):
    """
    Loads PixArt-alpha pipeline.
    - If hf_repo is given: loads base model + attaches LoRA adapter from HF.
    - Otherwise: loads base model as-is.
    """
    dtype = torch.float16 if "cuda" in device else torch.float32

    print(f"\nLoading base model: {BASE_MODEL}")
    pipe = PixArtAlphaPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    ).to(device)

    if hf_repo:
        print(f"Loading LoRA adapter: {hf_repo}/{checkpoint}")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            hf_repo,
            subfolder=checkpoint,
        )

    pipe.set_progress_bar_config(disable=True)
    return pipe


# ─────────────────────────────────────────────
# 3. T2I-CompBench — image generation
# ─────────────────────────────────────────────

COMPBENCH_CATEGORIES = [
    "color_val",
    "shape_val",
    "texture_val",
    "spatial_val",
    "non_spatial_val",
    "complex_val",
]

def generate_compbench_images(
    compbench_dir: str,
    hf_repo: str = None,
    checkpoint: str = None,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 20,
):
    """
    Generates images for every T2I-CompBench category using the model.
    Skips already-generated images (resume-friendly).
    """
    print("\n" + "="*50)
    print("Generating T2I-CompBench images...")
    print("="*50)

    pipe = load_pipeline(hf_repo=hf_repo, checkpoint=checkpoint, device=device)

    generator   = torch.Generator(device=device).manual_seed(seed)
    dataset_dir = Path(compbench_dir) / "examples" / "dataset"
    output_dir  = Path(compbench_dir) / "examples" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in COMPBENCH_CATEGORIES:
        txt_path = dataset_dir / f"{category}.txt"
        if not txt_path.exists():
            print(f"  [SKIP] {txt_path} not found")
            continue

        with open(txt_path) as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"\n  Generating {len(prompts)} images for [{category}]...")
        skipped = 0
        for prompt in prompts:
            out_path = output_dir / f"{prompt}_000000.png"
            if out_path.exists():
                skipped += 1
                continue
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            image.save(out_path)

        print(f"  ✓ Done ({skipped} already existed, saved to {output_dir})")

    print("\n✓ T2I-CompBench image generation complete.\n")
    return str(output_dir)


# ─────────────────────────────────────────────
# 4. T2I-CompBench — evaluation
# ─────────────────────────────────────────────

def _read_vqa_json(path: Path) -> float:
    """Read a T2I-CompBench vqa_result.json and return the average score."""
    with open(path) as f:
        data = json.load(f)
    scores = [float(item["answer"]) for item in data]
    return sum(scores) / len(scores) if scores else 0.0


def run_compbench_eval(compbench_dir: str) -> dict:
    """
    Runs T2I-CompBench evaluation scripts matching the paper's metrics:
      BLIPvqa_eval/BLIP_vqa.py          — attribute binding (color/shape/texture)
      UniDet_eval/2D_spatial_eval.py    — spatial relationships
      CLIPScore_eval/CLIP_similarity.py — non-spatial relationships only
      3_in_1_eval/3_in_1.py            — complex compositions (BLIP+UniDet+CLIP)
    """
    print("\n" + "="*50)
    print("Running T2I-CompBench evaluations...")
    print("="*50)

    compbench_path = Path(compbench_dir).resolve()
    examples_dir   = compbench_path / "examples"
    results        = {}

    # ── 1. BLIP-VQA — attribute binding (color, shape, texture) ──
    blip_script = compbench_path / "BLIPvqa_eval" / "BLIP_vqa.py"
    blip_out    = examples_dir / "annotation_blip" / "vqa_result.json"

    if not blip_script.exists():
        print(f"  [SKIP] BLIPvqa_eval/BLIP_vqa.py not found")
    else:
        if blip_out.exists():
            print(f"  ✓ BLIP-VQA cached — skipping re-run")
        else:
            print(f"\n  Running BLIP-VQA (color / shape / texture)...")
            subprocess.run(
                [sys.executable, str(blip_script), "--out_dir", str(examples_dir) + "/"],
                check=True,
                cwd=str(compbench_path / "BLIPvqa_eval"),
            )

        if blip_out.exists():
            avg = _read_vqa_json(blip_out)
            results["attribute_binding"] = round(avg, 4)
            print(f"  ✓ Attribute binding (color/shape/texture): {avg:.4f}")

    # ── 2. UniDet — 2D spatial relationships ──
    unidet_script = compbench_path / "UniDet_eval" / "2D_spatial_eval.py"
    spatial_out   = examples_dir / "labels" / "annotation_obj_detection_2d" / "vqa_result.json"

    if not unidet_script.exists():
        print(f"  [SKIP] UniDet_eval/2D_spatial_eval.py not found")
    else:
        if spatial_out.exists():
            print(f"  ✓ UniDet spatial cached — skipping re-run")
        else:
            print(f"\n  Running UniDet (spatial)...")
            subprocess.run(
                [sys.executable, str(unidet_script)],
                check=True,
                cwd=str(compbench_path / "UniDet_eval"),
            )

        if spatial_out.exists():
            avg = _read_vqa_json(spatial_out)
            results["spatial_val"] = round(avg, 4)
            print(f"  ✓ spatial_val: {avg:.4f}")

    # ── 3. CLIPScore — non-spatial relationships only ──
    clip_script = compbench_path / "CLIPScore_eval" / "CLIP_similarity.py"
    clip_base   = examples_dir / "annotation_clip" / "vqa_result.json"
    clip_saved  = examples_dir / "annotation_clip" / "vqa_result_nonspatial.json"

    if not clip_script.exists():
        print(f"  [SKIP] CLIPScore_eval/CLIP_similarity.py not found")
    else:
        if clip_saved.exists():
            print(f"  ✓ CLIP non-spatial cached — skipping re-run")
        else:
            print(f"\n  Running CLIP (non-spatial)...")
            if clip_base.exists():
                clip_base.unlink()
            subprocess.run(
                [
                    sys.executable, str(clip_script),
                    "--outpath", str(examples_dir) + "/",
                    "--complex", "False",
                ],
                check=True,
                cwd=str(compbench_path),
            )
            if clip_base.exists():
                shutil.copy(clip_base, clip_saved)
                print(f"  ✓ Saved non-spatial result to {clip_saved.name}")
            else:
                print(f"  [WARN] CLIP output not found after run")

        if clip_saved.exists():
            avg = _read_vqa_json(clip_saved)
            results["non_spatial_val"] = round(avg, 4)
            print(f"  ✓ non_spatial_val: {avg:.4f}")

    # ── 4. 3-in-1 — complex compositions ──
    three_in_one_script = compbench_path / "3_in_1_eval" / "3_in_1.py"
    three_in_one_out    = examples_dir / "annotation_3_in_1" / "vqa_result.json"

    if not three_in_one_script.exists():
        print(f"  [SKIP] 3_in_1_eval/3_in_1.py not found")
    else:
        if three_in_one_out.exists():
            print(f"  ✓ 3-in-1 complex cached — skipping re-run")
        else:
            print(f"\n  Running 3-in-1 (complex)...")
            subprocess.run(
                [
                    sys.executable, str(three_in_one_script),
                    "--outpath", str(examples_dir) + "/",
                ],
                check=True,
                cwd=str(compbench_path / "3_in_1_eval"),
            )

        if three_in_one_out.exists():
            avg = _read_vqa_json(three_in_one_out)
            results["complex_val"] = round(avg, 4)
            print(f"  ✓ complex_val (3-in-1): {avg:.4f}")

    return results


# ─────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────

def print_summary(fid_score: float, compbench_results: dict):
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    if fid_score is not None:
        print(f"\n  FID Score          : {fid_score:.4f}  (↓ lower is better)")
    if compbench_results:
        print("\n  T2I-CompBench      (↑ higher is better):")
        labels = {
            "attribute_binding": "Attribute Binding",
            "spatial_val":       "Spatial",
            "non_spatial_val":   "Non-Spatial",
            "complex_val":       "Complex (3-in-1)",
        }
        for key, score in compbench_results.items():
            label = labels.get(key, key)
            print(f"    {label:<26} {score:.4f}")
    print()


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PixArt-alpha: FID + T2I-CompBench")
    p.add_argument("--real_dir",       default="images_hands_3000_11k",      help="Folder of real images (for FID)")
    p.add_argument("--gen_dir",         default="gen_hand_final",       help="Folder of generated images (for FID)")
    p.add_argument("--compbench_dir",  required=False, default="./T2I-CompBench",       help="Path to cloned T2I-CompBench repo")
    p.add_argument("--hf_repo",        default="arpita2desh/pixart-hands-lora",        help="HF repo ID for LoRA, e.g. arpita2desh/pixart-hands-lora")
    p.add_argument("--checkpoint",     default="checkpoint-500",        help="Subfolder in HF repo, e.g. checkpoint-500")
    p.add_argument("--device",         default="cuda:2",      help="cuda / cuda:0 / cpu")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--steps",          type=int, default=20, help="Inference steps")
    p.add_argument("--skip_fid",       action="store_true", help="Skip FID computation")
    p.add_argument("--skip_compbench", action="store_true", help="Skip T2I-CompBench entirely")
    p.add_argument("--skip_gen",       action="store_true", help="Skip image generation (use existing samples)")
    return p.parse_args()


def main():
    args = parse_args()

    fid_score         = None
    compbench_results = {}

    if not args.skip_fid:
        fid_score = compute_fid(args.real_dir, args.gen_dir)

    if not args.skip_compbench:
        if not args.skip_gen:
            generate_compbench_images(
                compbench_dir=args.compbench_dir,
                hf_repo=args.hf_repo,
                checkpoint=args.checkpoint,
                device=args.device,
                seed=args.seed,
                num_inference_steps=args.steps,
            )
        compbench_results = run_compbench_eval(compbench_dir=args.compbench_dir)

    if fid_score is not None or compbench_results:
        print_summary(fid_score, compbench_results)


if __name__ == "__main__":
    main()
