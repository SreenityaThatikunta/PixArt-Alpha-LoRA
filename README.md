# PixArt-Alpha LoRA: Anatomically Accurate Hand Generation

Fine-tuning [PixArt-Alpha](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) with LoRA (Low-Rank Adaptation) to generate photorealistic images of human hands with correct anatomy - distinct fingers, natural knuckle structure, and realistic skin texture. 

## Demo

[![Demo Video](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube)](https://youtu.be/Aoez03xao6E)


## Links

| Resource | Link |
|----------|------|
| GitHub | [github.com/SreenityaThatikunta/PixArt-Alpha-LoRA](https://github.com/SreenityaThatikunta/PixArt-Alpha-LoRA) |
| Gradio Demo | [kaggle-proxy-561j.onrender.com](https://kaggle-proxy-561j.onrender.com) |
| W&B Dashboard | [wandb](https://wandb.ai/b23cm1007-indian-institute-of-technology-jodhpur/pixart-hands-lora/overview) |
| YouTube | [youtu.be/Aoez03xao6E](https://youtu.be/Aoez03xao6E) |
| LoRA Weights | [huggingface.co/arpita2desh/pixart-hands-lora](https://huggingface.co/arpita2desh/pixart-hands-lora) |

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU

### Installation

```bash
git clone https://github.com/SreenityaThatikunta/PixArt-Alpha-LoRA
cd PixArt-Alpha-LoRA
```

## Dataset Preparation

The training dataset is assembled from [11K Hands](https://sites.google.com/view/11khands) and [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — 3,000 images selected for sharpness and pose diversity. See `dataset.ipynb` for the full pipeline.

### Auto-Captioning

Generate detailed captions using LLaVA-Next:

```bash
python generate_captions.py
```

Outputs a JSON file mapping image filenames to descriptive captions.

## Training

```bash
python train.py \
    --data_dir ./hands_dataset \
    --caption_file ./captions.json \
    --output_dir ./lora_hands_output \
    --num_train_epochs 10 \
    --train_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --resolution 1024 \
    --mixed_precision fp16 \
    --use_wandb
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_model_name` | `PixArt-alpha/PixArt-XL-2-1024-MS` | Base model |
| `--resolution` | `1024` | Image resolution (512 or 1024) |
| `--lora_rank` | `16` | LoRA rank (8–32 typical) |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--num_train_epochs` | `25` | Training epochs |
| `--save_steps` | `500` | Checkpoint save frequency |
| `--learning_rate` | `5e-6` | Learning rate |
| `--lr_scheduler` | `cosine` | LR schedule (`cosine` or `linear`) |
| `--gradient_accumulation_steps` | `4` | Effective batch multiplier |
| `--merge_weight` | `1.0` | LoRA merge strength (0.0 = base, 1.0 = full) |

## Inference

### Generate images from a trained checkpoint

```bash
python generate_images.py \
    --captions captions.json \
    --output_dir generated_images \
    --hf_repo arpita2desh/pixart-hands-lora \
    --checkpoint checkpoint-500 \
    --device cuda:0
```

### Evaluate across checkpoints with W&B logging

```bash
python generate_samples.py
```

Compares base model, LoRA checkpoints, and merged model outputs side-by-side.

## Evaluation

### FID only

```bash
python evaluate.py --real_dir images_real --gen_dir images_gen --skip_compbench
```

### FID + T2I-CompBench

First, clone the T2I-CompBench repo and install its dependencies:

```bash
git clone https://github.com/Karine-Huang/T2I-CompBench.git
cd T2I-CompBench
pip install -r requirements.txt
cd ..
```

Then run evaluation:

```bash
python evaluate.py \
    --real_dir images_real \
    --gen_dir images_gen \
    --compbench_dir ./T2I-CompBench \
    --hf_repo arpita2desh/pixart-hands-lora \
    --checkpoint checkpoint-500
```

**Metrics:**
- **FID** (Frechet Inception Distance) — lower is better
- **T2I-CompBench** — attribute binding, spatial accuracy, non-spatial semantics, complex composition - higher is better

## Pipeline & Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                    │
│                                                         │
│  Dataset ──► VAE Encoder ──► Latent Space               │
│                                    │                    │
│  Captions ──► T5 Encoder ──► Text Embeddings            │
│                                    │                    │
│              DDPM Noise ──► Noisy Latents               │
│                                    │                    │
│                    ┌───────────────▼──────────────┐     │
│                    │  PixArt DiT Transformer      │     │
│                    │  ┌────────────────────────┐  │     │
│                    │  │  LoRA Adapters         │  │     │
│                    │  │  (to_q, to_k, to_v,    │  │     │
│                    │  │   to_out.0)            │  │     │
│                    │  └────────────────────────┘  │     │
│                    └───────────────┬──────────────┘     │
│                                    │                    │
│                    Predicted Noise ▼ MSE Loss           │
│                    Optimize LoRA params only            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Inference Pipeline                    │
│                                                         │
│  Prompt ──► T5 Encoder ──► Text Embeddings              │
│                                    │                    │
│  Random Noise ──► Iterative Denoising (50 steps)        │
│                   with LoRA-enhanced DiT                │
│                                    │                    │
│                    VAE Decoder ──► Generated Image      │
└─────────────────────────────────────────────────────────┘
```

**Key design choices:**
- Only the DiT attention projections are adapted via LoRA — VAE and T5 text encoder stay frozen
- LoRA checkpoints are ~10–50 MB vs. the full model's ~26 GB
- Merged models can be used standalone without PEFT at inference time

## Project Structure

```
├── train.py               # LoRA fine-tuning script
├── generate_images.py     # Batch image generation
├── generate_samples.py    # Evaluation sampling with W&B
├── generate_captions.py   # LLaVA-Next auto-captioning
├── evaluate.py            # FID + T2I-CompBench evaluation
├── dataset.ipynb          # Dataset assembly notebook
└── DL_Phase2_Report.pdf   # Project report
```

## Acknowledgments

- [PixArt-Alpha](https://github.com/PixArt-alpha/PixArt-alpha) — base diffusion model
- [PEFT](https://github.com/huggingface/peft) — parameter-efficient fine-tuning
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — diffusion pipeline
- [11K Hands](https://sites.google.com/view/11khands) & [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) — training data
- [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) — evaluation framework
