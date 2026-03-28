"""
LoRA Fine-tuning for PixArt-alpha: Anatomical Hand Accuracy
============================================================
Requirements:
    pip install diffusers transformers accelerate peft datasets
    pip install torch torchvision
    pip install Pillow tqdm wandb  # optional: wandb for logging

Usage:
    python finetune_pixart_hands_lora.py \
        --data_dir ./hands_dataset \
        --caption_file ./captions.json \
        --output_dir ./lora_hands_output \
        --num_train_epochs 10 \
        --train_batch_size 2 \
        --learning_rate 1e-4
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import PixArtAlphaPipeline, AutoencoderKL, PixArtTransformer2DModel
from diffusers.optimization import get_scheduler
from transformers import T5EncoderModel, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ─────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune PixArt-alpha for hands")

    parser.add_argument("--pretrained_model_name", type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing hand images")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="JSON file mapping image filename -> caption string")

    parser.add_argument("--output_dir", type=str, default="./lora_hands_output_3000_hp")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training image resolution (512 or 1024)")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank. Higher = more capacity, more VRAM. 8-32 typical.")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor). Usually 2x rank.")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Override num_train_epochs if set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--merge_weight", type=float, default=1.0,
                    help="LoRA merge strength: 1.0=full, 0.5=blend, >1.0=amplify")

    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────

class HandsDataset(Dataset):
    """
    Expects:
      data_dir/  -> folder of images (jpg, png, webp)
      caption_file -> JSON: {"image001.jpg": "A hand with five fingers...", ...}
    """

    def __init__(self, data_dir: str, caption_file: str, resolution: int = 512):
        self.data_dir = Path(data_dir)
        self.resolution = resolution

        with open(caption_file, "r") as f:
            self.captions = json.load(f)  # dict: filename -> caption

        # Filter to only files that exist
        self.image_files = [
            fname for fname in self.captions.keys()
            if (self.data_dir / fname).exists()
        ]
        print(f"[Dataset] Found {len(self.image_files)} valid image-caption pairs.")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image_path = self.data_dir / fname
        
        # ── Fix: handle both formats ──
        caption_data = self.captions[fname]
        if isinstance(caption_data, dict):
            caption = caption_data["caption"]   # extract string from dict
        else:
            caption = caption_data              # already a string

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {"pixel_values": image, "caption": caption}


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    captions = [e["caption"] for e in examples]
    return {"pixel_values": pixel_values, "captions": captions}


# ─────────────────────────────────────────────
# 3. MODEL SETUP WITH LoRA
# ─────────────────────────────────────────────

def load_models(args):
    """Load PixArt-alpha components."""
    print(f"[Model] Loading from {args.pretrained_model_name} ...")

    # Text encoder (T5)
    tokenizer = T5Tokenizer.from_pretrained(
        args.pretrained_model_name, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name, subfolder="text_encoder"
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae"
    )

    # Transformer (DiT backbone — this is what we LoRA-tune)
    transformer = PixArtTransformer2DModel.from_pretrained(
        args.pretrained_model_name, subfolder="transformer"
    )

    return tokenizer, text_encoder, vae, transformer


def apply_lora(transformer, args):
    """Wrap the transformer with LoRA adapters."""
    # Target the attention projection layers in the DiT blocks
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        bias="none",
        # Target Q, K, V, and output projections in attention layers
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # self-attention              
        ],
        # NOTE: TaskType.OTHER is used since DiT is not a standard seq2seq/causal LM
        task_type= None,
    )

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    return transformer


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def encode_prompt(captions, tokenizer, text_encoder, device, max_length=120):
    # ensure all captions are plain strings
    captions = [c if isinstance(c, str) else c["caption"] for c in captions]
    
    inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state   # (B, seq_len, 4096)

    return text_embeddings, attention_mask  # return embeddings, NOT input_ids


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ──────────────────────────
    tokenizer, text_encoder, vae, transformer = load_models(args)

    # Freeze VAE and text encoder — only train transformer LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Apply LoRA to transformer
    transformer = apply_lora(transformer, args)
    transformer.train()

    # Move to device
    vae = vae.to(device, dtype=dtype)
    text_encoder = text_encoder.to(device, dtype=dtype)
    transformer = transformer.to(device, dtype=dtype)
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # ── Dataset & DataLoader ─────────────────
    dataset = HandsDataset(args.data_dir, args.caption_file, args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ── Optimizer ────────────────────────────
    # Only optimize LoRA parameters
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=3e-2)

    # ── LR Scheduler ─────────────────────────
    total_steps = args.max_train_steps or (len(dataloader) * args.num_train_epochs)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Noise scheduler (DDPM) ───────────────
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name, subfolder="scheduler"
    )

    # ── Optional: W&B logging ─────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project="pixart-hands-lora", config=vars(args))

    # ── Training ─────────────────────────────
    global_step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    print(f"\n[Train] Starting training for {total_steps} steps ...")
    print(f"        Dataset: {len(dataset)} samples | Batch: {args.train_batch_size}")
    print(f"        LoRA rank: {args.lora_rank} | LR: {args.learning_rate}\n")

    for epoch in range(args.num_train_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["captions"]
            # 4. Encode captions
            encoder_hidden_states, encoder_attention_mask = encode_prompt(
                captions, tokenizer, text_encoder, device
            )
            with torch.cuda.amp.autocast(enabled=(args.mixed_precision == "fp16")):
                # 1. Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # scale latents

                # 2. Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=device
                ).long()

                # 3. Add noise to latents (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 5. Predict noise with transformer
                resolution    = torch.tensor([[args.resolution, args.resolution]] * bsz, device=device, dtype=dtype)
                aspect_ratio  = torch.tensor([[1.0]] * bsz, device=device, dtype=dtype)

                model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask.float(),
                added_cond_kwargs={"resolution": resolution, "aspect_ratio": aspect_ratio},
                ).sample


                # 6. Compute loss (simple MSE on predicted vs actual noise)
                model_pred_noise = model_pred.chunk(2, dim=1)[0]
                loss = F.mse_loss(model_pred_noise.float(), noise.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_lora_checkpoint(transformer, args.output_dir, global_step)

                if args.use_wandb:
                    import wandb
                    wandb.log({
                        # ── Core ──────────────────────────────
                        "loss"       : loss.item() * args.gradient_accumulation_steps,
                        "lr"         : lr_scheduler.get_last_lr()[0],
                        "grad_norm"  : grad_norm.item(),
                        "step"       : global_step,
                        "epoch"      : epoch + 1,

                        # ── LoRA health ───────────────────────
                        "lora_weight_norm": sum(
                            p.norm().item()
                            for n, p in transformer.named_parameters()
                            if "lora" in n and p.requires_grad
                        ),
                    })

                    # ── Sample images every 100 steps ─────────
                    if global_step % 100 == 0:
                        transformer.eval()
                        pipe = PixArtAlphaPipeline.from_pretrained(
                            args.pretrained_model_name,
                            transformer=transformer,
                            torch_dtype=torch.float16,
                        ).to(device)

                        with torch.no_grad():
                            images = pipe(
                                prompt="An image of human hand with exactly five fingers fully visible: thumb, index finger, middle finger, ring finger, and pinky. All fingers are clearly separated with visible gaps between them. The image shows a close-up of a person's hand. The hand is positioned with the palm facing upwards and the fingers extended. The fingers are spread out, and the knuckles are clearly visible. The hand is the central focus of the image, and there are no other objects or text present",
                                negative_prompt = (
                                    "extra fingers, missing fingers, six fingers, four fingers, "
                                ),
                                num_images_per_prompt=4,
                                num_inference_steps=20,
                            ).images
                        wandb.log({
                            "samples" : [wandb.Image(img) for img in images],
                            "step"    : global_step,
                        })
                        transformer.train()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                               "step": global_step})

            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        print(f"[Epoch {epoch+1}] Avg loss: {epoch_loss / len(dataloader):.4f}")

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    # Save final LoRA weights
    save_lora_checkpoint(transformer, args.output_dir, "final")
    print(f"\n[Done] LoRA weights saved to {args.output_dir}")

    # ── Merge LoRA into base and save full model ──
    print("\n[Merge] Merging LoRA into base model...")
    merge_lora_into_base(
        pretrained_model_name=args.pretrained_model_name,
        lora_checkpoint_dir=os.path.join(args.output_dir, "checkpoint-final"),
        output_dir=os.path.join(args.output_dir, "merged_model"),
        merge_weight=args.merge_weight,   # add this arg (see below)
    )


# ─────────────────────────────────────────────
# 5. SAVING & MERGING
# ─────────────────────────────────────────────

def save_lora_checkpoint(transformer, output_dir: str, step):
    """Save only LoRA adapter weights (small file, ~10-50MB)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    transformer.save_pretrained(ckpt_dir)
    print(f"[Save] LoRA checkpoint saved at step {step} → {ckpt_dir}")


def merge_lora_into_base(
    pretrained_model_name: str,
    lora_checkpoint_dir: str,
    output_dir: str,
    merge_weight: float = 1.0,
):
    """
    Merge LoRA weights back into the base model via weighted sum.
    merge_weight: 0.0 = pure base, 1.0 = full LoRA, 0.5 = blend
    
    Usage:
        merge_lora_into_base(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "./lora_hands_output/checkpoint-final",
            "./merged_model",
            merge_weight=0.8  # 80% LoRA influence
        )
    """
    from peft import PeftModel

    # Load and merge transformer as before
    base_transformer = PixArtTransformer2DModel.from_pretrained(
        pretrained_model_name, subfolder="transformer"
    )
    peft_model = PeftModel.from_pretrained(base_transformer, lora_checkpoint_dir)
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A"):
            module.scaling = {k: merge_weight * v for k, v in module.scaling.items()}
    merged_transformer = peft_model.merge_and_unload()

    # ✅ Save full pipeline instead of just the transformer
    pipe = PixArtAlphaPipeline.from_pretrained(
        pretrained_model_name,
        transformer=merged_transformer,
        torch_dtype=torch.float16,
    )
    pipe.save_pretrained(output_dir)   # saves vae/, text_encoder/, tokenizer/ etc.
    print(f"[Merge] Full pipeline saved to {output_dir}")


# ─────────────────────────────────────────────
# 6. INFERENCE TEST
# ─────────────────────────────────────────────

def test_inference(
    pretrained_model_name: str,
    lora_checkpoint_dir: str,
    prompt: str = "A photorealistic hand with five distinct fingers, correct anatomy, clear finger separation, natural skin texture",
    num_images: int = 4,
    output_path: str = "./test_hands.png",
):
    """
    Quick inference test with your LoRA to check quality.
    
    Usage:
        python finetune_pixart_hands_lora.py  # or call this function directly
    """
    from peft import PeftModel
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[Inference] Loading pipeline...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float16,
    ).to(device)

    # Inject LoRA into the pipeline's transformer
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer, lora_checkpoint_dir
    )

    print(f"[Inference] Generating {num_images} images...")
    images = pipe(
        prompt=prompt,
        negative_prompt="blurry, fused fingers, extra fingers, missing fingers, deformed hands, bad anatomy",
        num_images_per_prompt=num_images,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=torch.Generator(device).manual_seed(42),
    ).images

    # Save as grid
    grid_w = math.ceil(math.sqrt(num_images))
    grid_h = math.ceil(num_images / grid_w)
    w, h = images[0].size
    grid = Image.new("RGB", (grid_w * w, grid_h * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % grid_w) * w, (i // grid_w) * h))
    grid.save(output_path)
    print(f"[Inference] Saved test grid → {output_path}")


# ─────────────────────────────────────────────
# 7. CAPTION FORMAT HELPER
# ─────────────────────────────────────────────

def build_caption_json_from_folder(
    image_dir: str,
    llava_outputs_file: str,  # e.g., a txt or json file with LLaVA outputs
    out_json: str = "captions.json",
):
    """
    Helper to build the caption JSON if your LLaVA outputs are in a 
    different format. Adapt as needed.
    
    Expected llava_outputs_file format (one per line):
        image001.jpg|||A detailed hand showing five fingers with...
    """
    captions = {}
    with open(llava_outputs_file, "r") as f:
        for line in f:
            line = line.strip()
            if "|||" in line:
                fname, caption = line.split("|||", 1)
                captions[fname.strip()] = caption.strip()

    with open(out_json, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"[Caption] Saved {len(captions)} captions → {out_json}")
    return captions


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
