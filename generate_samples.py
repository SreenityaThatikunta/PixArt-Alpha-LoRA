import torch
import os
from diffusers import PixArtAlphaPipeline
from peft import PeftModel
import wandb
from glob import glob

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
BASE_MODEL = "PixArt-alpha/PixArt-XL-2-1024-MS"
LORA_ROOT  = "lora_hands_output_3000_hp"
MERGED_MODEL_DIR = "lora_hands_output_3000_hp/merged_model"
OUTPUT_DIR = "eval_outputs"
PROJECT    = "pixart-hands-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "A human hand with five fingers, photorealistic",
    "A close-up of a hand resting on a table",
    "A open hand with five fingers reaching forward, natural lighting",
    "Two hands holding each other, realistic skin texture",
    "A hand gripping a glass, natural lighting",
    "A hand typing on a keyboard",
    "A hand with detailed fingers, studio lighting",
    "A human hand pointing forward",
    "A hand holding a phone in one hand",
    "A relaxed palm facing upward"
]

NEG_PROMPT = "extra fingers, missing fingers, fused fingers, bad anatomy, deformed"

# ─────────────────────────────────────
# WANDB INIT
# ─────────────────────────────────────
wandb.init(project=PROJECT)

# ─────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────
def generate_and_log(tag, pipe, step):
    generator = torch.Generator(device=device).manual_seed(42)
    images_to_log = []

    for i, prompt in enumerate(PROMPTS):
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=generator
            ).images[0]

        save_path = os.path.join(OUTPUT_DIR, f"{tag}/sample_{i}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

        images_to_log.append(wandb.Image(image, caption=prompt))

    wandb.log({
        "samples": images_to_log,
        "model": tag,
        "num_images": len(images_to_log)
    }, step=step)


# ─────────────────────────────────────
# 1️⃣ BASE MODEL EVAL
# ─────────────────────────────────────
print("\nEvaluating BASE model")

pipe = PixArtAlphaPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(device)
pipe.set_progress_bar_config(disable=True)

generate_and_log("base_model", pipe, step=0)

del pipe
torch.cuda.empty_cache()


# ─────────────────────────────────────
# 2️⃣ LORA CHECKPOINTS
# ─────────────────────────────────────
checkpoints = sorted(glob(os.path.join(LORA_ROOT, "checkpoint-*")))
print(f"\nFound {len(checkpoints)} checkpoints")

for ckpt_path in checkpoints:
    step = ckpt_path.split("-")[-1]
    if step == "final":
        step_int = 90000000
    else:
        step_int = int(step)
    print(f"\nEvaluating checkpoint {step}")

    pipe = PixArtAlphaPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        ckpt_path
    )

    generate_and_log(f"lora_ckpt_{step}", pipe, step_int)

    del pipe
    torch.cuda.empty_cache()


# ─────────────────────────────────────
# 3️⃣ MERGED MODEL
# ─────────────────────────────────────
print("\nEvaluating MERGED model")

merged_pipe = PixArtAlphaPipeline.from_pretrained(
    MERGED_MODEL_DIR,
    torch_dtype=torch.float16
).to(device)
merged_pipe.set_progress_bar_config(disable=True)

generate_and_log("merged_model", merged_pipe, step=99999999)

del merged_pipe
torch.cuda.empty_cache()


print("Done!")
wandb.finish()