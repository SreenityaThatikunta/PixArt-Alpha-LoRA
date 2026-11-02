# PixArt-Alpha LoRA Technical Deep Dive

This document explains the project at the level expected in a technical interview or academic project review. It covers the problem, architecture, training pipeline, concepts used, tradeoffs, evaluation, likely questions, and a practical way to present the mini-project.

Reference materials used while preparing this file:

- `README.md`
- `train.py`
- `generate_captions.py`
- `generate_images.py`
- `generate_samples.py`
- `evaluate.py`
- `dataset.ipynb`
- `Gradio.ipynb`
- `DL_Report.pdf` and `DL_Presentation.pdf` as project-report/presentation references

## 1. Project Summary

The project fine-tunes PixArt-Alpha, a text-to-image diffusion model with a Diffusion Transformer backbone, to improve photorealistic human hand generation. The target failure mode is a common weakness of generative image models: hands often contain extra fingers, missing fingers, fused fingers, unnatural knuckles, or anatomically inconsistent poses.

Instead of fully fine-tuning the entire PixArt-Alpha model, the project uses LoRA, or Low-Rank Adaptation, to train small adapter matrices inside the transformer's attention projection layers. The text encoder and VAE are frozen, and only LoRA parameters inside the PixArt transformer are optimized.

The complete workflow is:

1. Build a hand-focused dataset from 11K Hands and FreiHAND.
2. Select diverse, sharp images using sharpness scoring and pose clustering.
3. Generate detailed image captions using LLaVA-Next.
4. Fine-tune PixArt-Alpha with LoRA on hand images and captions.
5. Save LoRA checkpoints and optionally merge LoRA into the base model.
6. Generate images from base and fine-tuned models.
7. Evaluate with FID, T2I-CompBench, visual comparison, and W&B logging.
8. Serve a demo through Gradio.

## 2. Why This Problem Is Hard

Hands are difficult for text-to-image models because they are both highly structured and visually variable.

Important constraints:

- A realistic hand usually has five fingers with a consistent topology.
- Finger count, finger separation, knuckle positions, palm orientation, and occlusion must agree.
- Hands appear in many poses, lighting conditions, skin tones, camera angles, and object interactions.
- Small anatomical errors are very visible to humans.
- Text prompts such as "a hand with five fingers" do not always force the model to respect geometry.

This project narrows the model adaptation target to a specific visual domain: realistic hand anatomy.

## 3. Repository Map

| File | Purpose |
|---|---|
| `dataset.ipynb` | Samples the training dataset using sharpness and pose-diversity heuristics. |
| `generate_captions.py` | Uses LLaVA-Next to produce detailed captions for hand images. |
| `train.py` | LoRA fine-tuning script for PixArt-Alpha. |
| `generate_images.py` | Batch generation from a Hugging Face LoRA checkpoint. |
| `generate_samples.py` | Generates fixed prompt samples for base, LoRA checkpoints, and merged model with W&B logging. |
| `evaluate.py` | Computes FID and optionally runs T2I-CompBench evaluations. |
| `Gradio.ipynb` | Demo notebook for comparing base, checkpoint, final, and merged models. |
| `README.md` | User-facing setup and usage guide. |

## 4. Core Model Architecture

PixArt-Alpha is used through Hugging Face Diffusers as `PixArtAlphaPipeline`.

The pipeline contains:

- T5 tokenizer and T5 text encoder
- VAE encoder/decoder
- PixArt transformer backbone
- Diffusion noise scheduler
- Sampling/inference loop

### 4.1 Text Encoder: T5

Captions are tokenized by `T5Tokenizer` and encoded by `T5EncoderModel`.

In `train.py`, `encode_prompt()`:

- Pads captions to a fixed length.
- Truncates long captions.
- Produces `encoder_hidden_states`.
- Returns the attention mask.
- Runs under `torch.no_grad()` because the text encoder is frozen.

Why freeze T5?

- The project wants to adapt image generation anatomy, not language understanding.
- Fine-tuning T5 would increase memory cost.
- The caption distribution is narrow and does not require relearning language.
- Freezing reduces overfitting and training instability.

### 4.2 VAE: Image to Latent Space

The VAE maps images into a compressed latent representation.

During training:

```python
latents = vae.encode(pixel_values).latent_dist.sample()
latents = latents * vae.config.scaling_factor
```

The diffusion model does not train directly on full RGB pixels. It trains on latent tensors. This makes training cheaper because the transformer denoises compressed image features instead of full-resolution images.

Why freeze the VAE?

- The VAE already knows how to compress and reconstruct natural images.
- The target change is anatomical generation, not pixel reconstruction.
- Training the VAE would be expensive and could degrade image quality.

### 4.3 PixArt Transformer / DiT Backbone

PixArt uses a Diffusion Transformer style denoiser. Instead of a U-Net, the model uses transformer blocks to predict noise in latent space.

In training, the transformer receives:

- Noisy latents
- Diffusion timestep
- Text embeddings from T5
- Text attention mask
- Added conditioning for resolution and aspect ratio

It predicts the noise component that was added to the original latent.

### 4.4 Noise Scheduler

`train.py` loads a `DDPMScheduler`.

The scheduler defines the forward diffusion process:

1. Start from clean latents.
2. Sample random Gaussian noise.
3. Sample a random timestep.
4. Add noise according to that timestep.
5. Train the model to predict the added noise.

The loss is mean squared error between predicted noise and true noise.

```python
loss = F.mse_loss(model_pred_noise.float(), noise.float(), reduction="mean")
```

## 5. LoRA: Parameter-Efficient Fine-Tuning

LoRA stands for Low-Rank Adaptation.

Instead of updating a full weight matrix `W`, LoRA freezes `W` and learns a low-rank update:

```text
W' = W + scale * B * A
```

Where:

- `A` projects down to rank `r`.
- `B` projects back up to the original dimension.
- `r` is much smaller than the full hidden dimension.
- Only `A` and `B` are trained.

In this project, LoRA is applied to attention projection layers:

```python
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
```

These correspond to:

- Query projection
- Key projection
- Value projection
- Output projection

Why attention projections?

- Attention controls how visual latent patches interact with prompt conditioning and each other.
- Hand anatomy depends on relationships between parts: fingers, palm, knuckles, pose, occlusion.
- Updating attention projections can affect structure without retraining the full model.

### 5.1 LoRA Rank

The default rank is `16`.

Tradeoff:

- Lower rank, such as 4 or 8: fewer parameters, lower memory, faster training, but less adaptation capacity.
- Higher rank, such as 32 or 64: more capacity, but more VRAM, more overfitting risk, and larger checkpoints.
- Rank 16 is a reasonable middle ground for a focused domain adaptation task.

### 5.2 LoRA Alpha

`lora_alpha` controls scaling of the low-rank update. The project uses `32` by default, which is 2x the default rank.

The effective update strength is approximately:

```text
alpha / rank
```

For rank 16 and alpha 32, the scale is 2.

### 5.3 LoRA Dropout

The project uses dropout `0.1`.

This regularizes LoRA training so the adapter does not memorize the small hand dataset too aggressively.

### 5.4 Why LoRA Instead Of Full Fine-Tuning?

Advantages:

- Much smaller checkpoints.
- Lower VRAM requirement.
- Faster experimentation.
- Easier to publish and load from Hugging Face.
- Preserves general generation ability of the base model.

Disadvantages:

- Limited adaptation capacity.
- May not fully fix deep geometric reasoning failures.
- Bad target module choice can lead to weak improvement.
- Can overfit style or dataset artifacts if the dataset is narrow.

## 6. Dataset Pipeline

The project uses `dataset.ipynb` to build a dataset from hand image sources.

The intended sources are:

- 11K Hands
- FreiHAND

The notebook samples 3,000 images and records metadata in a manifest.

### 6.1 Sharpness Score

The notebook uses Laplacian variance as a sharpness heuristic:

```python
cv2.Laplacian(img, cv2.CV_64F).var()
```

Higher Laplacian variance generally means more edges and less blur.

Why use it?

- Blurry hands provide weak anatomical supervision.
- Sharp images expose finger boundaries and knuckle details.
- It is cheap and easy to compute.

Limitations:

- High texture backgrounds can inflate the score.
- It does not understand semantic hand quality.
- It can prefer noisy images.

### 6.2 Pose Diversity

The notebook builds pose features using:

- FreiHAND 3D keypoints when available.
- HOG features as a fallback or for 11K Hands.

It then normalizes features and applies KMeans clustering.

The selected images are chosen round-robin across clusters, taking sharp images from each cluster. This avoids selecting only one easy pose, such as flat open palms.

Why pose diversity matters:

- The model needs examples of different palm orientations.
- Finger occlusions and bent fingers are common.
- A narrow dataset can improve one pose while hurting others.

### 6.3 Caption Format

`HandsDataset` supports captions in either format:

```json
{
  "00001.jpg": "A detailed caption..."
}
```

or:

```json
{
  "00001.jpg": {
    "caption": "A detailed caption..."
  }
}
```

This is useful because `generate_captions.py` writes a nested object containing both path and caption.

## 7. Auto-Captioning With LLaVA-Next

`generate_captions.py` loads:

```python
llava-hf/llava-v1.6-mistral-7b-hf
```

The prompt asks LLaVA-Next to describe:

- Number of visible fingers
- Finger pose and separation
- Knuckle visibility
- Palm orientation
- Skin texture
- Lighting

This is important because generic captions like "a hand" are too weak. Detailed captions give the diffusion model more conditional signals.

Tradeoffs:

- Auto-captioning is scalable.
- It avoids manually writing thousands of captions.
- Captions can contain hallucinations.
- If LLaVA miscounts fingers, the model may learn noisy supervision.
- Captions may become overly verbose and need truncation.

## 8. Training Pipeline

The high-level training path in `train.py` is:

```text
image + caption
  image -> transform -> VAE latent
  caption -> T5 embeddings
  latent + random noise + timestep -> noisy latent
  noisy latent + timestep + text embeddings -> PixArt transformer
  transformer output -> predicted noise
  predicted noise vs true noise -> MSE loss
  optimize LoRA parameters only
```

### 8.1 Image Transforms

Images are resized, center-cropped, converted to tensors, and normalized to `[-1, 1]`.

```python
transforms.Resize(resolution)
transforms.CenterCrop(resolution)
transforms.ToTensor()
transforms.Normalize([0.5], [0.5])
```

Tradeoff:

- Center cropping is simple and stable.
- It can remove fingers near image edges.
- For hands, better preprocessing could include hand-aware cropping or segmentation.

### 8.2 Mixed Precision

The script supports `fp16`.

Benefits:

- Lower VRAM use.
- Faster training on CUDA GPUs.

Risks:

- Numerical instability.
- Gradients can underflow.

The script uses `torch.cuda.amp.GradScaler` to reduce instability.

### 8.3 Gradient Accumulation

The default batch size is small, but `gradient_accumulation_steps=4` increases the effective batch size.

Effective batch size:

```text
train_batch_size * gradient_accumulation_steps
```

For batch size 2 and accumulation 4, effective batch size is 8.

Why use it?

- Diffusion models are memory-heavy.
- Accumulation simulates larger batches without requiring more VRAM.

### 8.4 Optimizer

The project uses AdamW:

```python
torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=3e-2)
```

Why AdamW?

- Standard for transformer training.
- Decoupled weight decay improves regularization.
- Works well with small trainable parameter subsets.

### 8.5 Learning Rate Scheduler

The default scheduler is cosine with warmup.

Warmup helps avoid unstable updates early in training. Cosine decay gradually lowers the learning rate as training progresses.

The project default learning rate in code is `5e-6`, which is conservative for LoRA fine-tuning on a large diffusion model.

### 8.6 Loss Function

The model trains with MSE loss on predicted noise.

Why noise prediction?

- DDPM-style diffusion training learns to reverse the noising process.
- Predicting noise is stable and standard in latent diffusion training.

One important detail in `train.py`:

```python
model_pred_noise = model_pred.chunk(2, dim=1)[0]
```

PixArt may return extra channels depending on prediction configuration. The script takes the first half as the noise prediction used for MSE.

## 9. Checkpointing and Merging

### 9.1 LoRA Checkpoints

`save_lora_checkpoint()` saves only adapter weights:

```python
transformer.save_pretrained(ckpt_dir)
```

This is much smaller than saving the full model.

### 9.2 Merged Model

`merge_lora_into_base()` loads the base transformer, attaches LoRA, scales LoRA influence with `merge_weight`, merges adapters into the base weights, and saves a full Diffusers pipeline.

Merge strength:

- `0.0`: base model behavior.
- `0.5`: blended adapter influence.
- `1.0`: full adapter.
- Greater than `1.0`: amplified adapter, but can introduce artifacts.

Why merge?

- Simpler inference deployment.
- No PEFT wrapper needed at runtime.
- Useful for platforms that expect a normal Diffusers pipeline.

Why keep LoRA unmerged too?

- Smaller upload/download.
- Easy to swap checkpoints.
- Allows adapter composition or strength control.

## 10. Inference Pipeline

`generate_images.py` loads the base PixArt pipeline and attaches LoRA from Hugging Face:

```python
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer,
    hf_repo,
    subfolder=checkpoint,
)
```

Generation uses:

- `num_inference_steps=30`
- `guidance_scale=7.5`
- `height=512`
- `width=512`
- Fixed random seed
- Negative prompt targeting hand errors

Negative prompt:

```text
extra fingers, missing fingers, fused fingers, bad anatomy, deformed
```

### 10.1 Classifier-Free Guidance

Guidance scale controls how strongly the model follows the prompt.

Tradeoff:

- Low guidance: more diverse but less prompt-faithful.
- High guidance: more prompt-faithful but can look over-sharpened or unnatural.
- `7.5` is a common practical value.

### 10.2 Seed Control

A fixed seed makes comparisons repeatable across base and LoRA models. This is important when showing improvement because it reduces randomness in qualitative evaluation.

## 11. Evaluation

The project uses both quantitative and qualitative evaluation.

### 11.1 FID

`evaluate.py` computes FID using `clean-fid`.

FID compares the distribution of generated images against real images using Inception features.

Lower is better.

Strengths:

- Standard generative image metric.
- Captures distribution-level realism.

Weaknesses:

- Not hand-anatomy-specific.
- Can improve even if finger count is still wrong.
- Sensitive to dataset size and preprocessing.
- Does not measure prompt alignment.

### 11.2 T2I-CompBench

The evaluation script supports T2I-CompBench categories:

- Attribute binding
- Spatial relationships
- Non-spatial relationships
- Complex composition

These use tools such as BLIP-VQA, UniDet, CLIPScore, and a combined 3-in-1 evaluation.

Strengths:

- Tests text-image alignment more directly than FID.
- Covers compositional ability.

Weaknesses:

- Not specialized for hand anatomy.
- May miss finger-level structural errors.
- Requires external repository and dependencies.

### 11.3 W&B Sample Evaluation

`generate_samples.py` logs images for fixed prompts across:

- Base model
- LoRA checkpoints
- Merged model

This is useful for:

- Side-by-side visual comparison.
- Tracking qualitative changes over training.
- Detecting overfitting or degradation.

### 11.4 Recommended Additional Metric

For a stronger project defense, add a hand-specific evaluator:

- Use a hand landmark detector such as MediaPipe Hands.
- Detect number of visible fingers or landmark consistency.
- Measure detection success rate.
- Compare base vs LoRA on the same prompts and seeds.

This would directly support the claim of better anatomical hands.

## 12. Important Hyperparameters

| Hyperparameter | Project Default | Role | Tradeoff |
|---|---:|---|---|
| Resolution | 512 in code, README mentions 1024 usage | Training image size | Higher resolution improves detail but costs more VRAM. |
| Batch size | 2 | Per-step examples | Larger is more stable but memory-heavy. |
| Gradient accumulation | 4 | Effective batch multiplier | Improves stability without extra VRAM. |
| Learning rate | `5e-6` | Update size | Too high causes artifacts; too low learns slowly. |
| LoRA rank | 16 | Adapter capacity | Higher rank improves capacity but increases overfitting risk. |
| LoRA alpha | 32 | Adapter scale | Too high can overpower base model. |
| LoRA dropout | 0.1 | Regularization | Too high may underfit. |
| Warmup steps | 200 | Early training stability | Too many warmup steps slow learning. |
| Scheduler | Cosine | LR decay | Stable default for transformer training. |
| Guidance scale | 7.5 | Prompt adherence | Too high can reduce realism. |

## 13. Key Tradeoffs and Design Choices

### 13.1 Domain-Specific Fine-Tuning vs General Model Training

The project fine-tunes for a narrow domain: hands.

Benefit:

- More practical and affordable.
- Clear objective.
- Easier to show before/after results.

Cost:

- May bias the model toward hand-heavy compositions.
- Does not solve general text-to-image weaknesses.

### 13.2 LoRA vs Full Fine-Tuning

LoRA is the practical choice because the project is a mini-project with limited compute.

Full fine-tuning might improve quality more, but would require:

- More VRAM
- More training data
- More careful regularization
- Larger checkpoints
- Higher risk of catastrophic forgetting

### 13.3 Auto-Captions vs Manual Captions

Auto-captioning with LLaVA enables scale.

The downside is caption noise. For example, a caption may say five fingers when only four are visible because of occlusion.

Best improvement:

- Add manual review for a small validation subset.
- Filter captions for obviously wrong finger counts.
- Use shorter, normalized captions for consistency.

### 13.4 FID vs Human Evaluation

FID is useful but not sufficient.

For this project, human side-by-side review is important because anatomical correctness is highly perceptual and local.

A good evaluation section should report:

- FID
- Sample grids
- Prompt-specific comparisons
- Failure cases
- A hand-specific metric if possible

### 13.5 512 vs 1024 Resolution

PixArt-Alpha XL is a high-resolution model, but training at 1024 is expensive.

512 benefits:

- Lower VRAM.
- Faster iterations.
- Easier debugging.

1024 benefits:

- Better fine details.
- Fingers and knuckles are more visible.

Risk:

- If training at 512 and evaluating at 1024, behavior may not transfer perfectly.

## 14. Known Limitations

Current limitations to acknowledge in interviews:

- Dataset is relatively small for generative fine-tuning.
- Captions are auto-generated and may contain errors.
- Center crop can cut off fingers.
- FID is not anatomy-specific.
- T2I-CompBench is broad and not hand-focused.
- Negative prompts help but are not a principled anatomy constraint.
- The model can still fail on complex interactions like hands holding objects, crossed fingers, or strong occlusion.
- LoRA adapters may overfit to dataset backgrounds, lighting, or framing.

## 15. Possible Improvements

High-value future work:

1. Add hand landmark-based evaluation using MediaPipe or another hand pose estimator.
2. Add automatic dataset filtering for complete visible hands.
3. Use hand segmentation or bounding boxes before cropping.
4. Compare LoRA target modules, such as attention-only vs attention plus MLP layers.
5. Run ablations for rank 8, 16, and 32.
6. Compare 512 and 1024 training.
7. Add validation prompts grouped by difficulty.
8. Add human evaluation with blinded side-by-side ratings.
9. Train with caption templates to reduce noisy caption variation.
10. Experiment with DreamBooth-style prior preservation if generality drops.

## 16. How To Explain This Project In 60 Seconds

"This project improves hand generation in PixArt-Alpha using parameter-efficient LoRA fine-tuning. Hand anatomy is a known failure case for text-to-image models, so I built a focused hand dataset from 11K Hands and FreiHAND, selected sharp and pose-diverse examples, and generated detailed captions using LLaVA-Next. During training, I froze the T5 text encoder and VAE, attached LoRA adapters to the PixArt transformer's attention projections, and optimized only those adapter weights using the standard diffusion noise-prediction loss. This gives small checkpoints and avoids full-model fine-tuning. I evaluated the model using FID, T2I-CompBench, W&B sample grids, and side-by-side comparisons between the base model, LoRA checkpoints, and a merged model. The main tradeoff is that LoRA is efficient and deployable, but a hand-specific landmark metric would make the anatomical improvement claim stronger."

## 17. How To Present The Mini-Project

Recommended presentation structure:

1. Problem
   - Text-to-image models often fail at hands.
   - The goal is anatomically realistic hand generation.

2. Base model
   - PixArt-Alpha is a diffusion transformer text-to-image model.
   - It uses T5 for text, VAE for latent image representation, and a transformer denoiser.

3. Dataset
   - Sources: 11K Hands and FreiHAND.
   - Sampling: sharpness by Laplacian variance and pose diversity by clustering.
   - Captions: generated using LLaVA-Next with hand-specific prompts.

4. Method
   - Freeze VAE and T5.
   - Apply LoRA to `to_q`, `to_k`, `to_v`, and `to_out.0`.
   - Train with DDPM noise prediction and MSE loss.

5. Results
   - Show base vs LoRA image grids using the same prompts and seeds.
   - Show W&B training samples.
   - Report FID and T2I-CompBench if available.

6. Tradeoffs
   - LoRA is efficient but capacity-limited.
   - FID is useful but not hand-specific.
   - Auto-captioning scales but introduces noise.

7. Future work
   - Hand landmark metric.
   - Better cropping/filtering.
   - Rank and resolution ablations.

## 18. Interview Prep: Likely Questions And Answers

### Q1. Why did you choose PixArt-Alpha?

PixArt-Alpha is a strong text-to-image diffusion model with a transformer-based denoising backbone. It is available through Diffusers, which makes it practical to load, fine-tune, attach LoRA adapters, and run inference. Since the project focuses on improving a specific visual failure mode, starting from a pretrained model is much more practical than training from scratch.

### Q2. Why are hands difficult for diffusion models?

Hands require consistent local structure: five fingers, plausible joints, palm geometry, and coherent occlusion. Diffusion models learn image distributions statistically, and small local structural mistakes can still look globally plausible. The model may know that fingers exist but fail to enforce exact count and topology.

### Q3. What exactly did LoRA train?

The project trained low-rank adapter matrices inserted into the PixArt transformer's attention projections: query, key, value, and output projections. The base transformer weights remain frozen, and only the LoRA parameters receive gradients.

### Q4. Why not fine-tune the VAE or text encoder?

The VAE already handles image compression and reconstruction, and the T5 encoder already handles language. The target problem is visual generation of hands, so adapting the denoising transformer is the most direct and compute-efficient path.

### Q5. What is the training objective?

The objective is diffusion noise prediction. A random timestep is sampled, noise is added to the image latent, and the transformer predicts the added noise conditioned on the caption. The loss is MSE between predicted noise and actual noise.

### Q6. What is classifier-free guidance?

Classifier-free guidance strengthens prompt conditioning during inference by combining conditional and unconditional predictions. Higher guidance usually improves prompt adherence, but too much can reduce realism or create artifacts.

### Q7. Why use negative prompts?

Negative prompts discourage known failure modes such as extra fingers, missing fingers, fused fingers, and bad anatomy. They are helpful at inference, but they do not guarantee correctness because they are only a soft conditioning signal.

### Q8. What does FID measure?

FID measures the distance between feature distributions of real and generated images using an Inception network. Lower FID usually means generated images are closer to real images distributionally, but it does not directly measure finger count or anatomy.

### Q9. What are the biggest weaknesses of your evaluation?

The biggest weakness is that FID and T2I-CompBench are not hand-specific. A better evaluation would use hand landmark detection or a human study to measure anatomical correctness directly.

### Q10. What would you improve first?

I would add a hand-specific metric using a hand landmark detector, then use it to compare base, checkpoint, and merged models on a fixed prompt suite. That would make the main claim much more measurable.

### Q11. How do LoRA rank and alpha affect training?

Rank controls adapter capacity. Alpha controls update strength. Higher rank and alpha can improve adaptation, but they also increase overfitting and artifact risk. For this project, rank 16 and alpha 32 provide a balanced adapter size.

### Q12. What is the role of the scheduler?

The noise scheduler defines how noise is added at each diffusion timestep. During training, it creates noisy latents. During inference, the scheduler controls the reverse denoising trajectory from random noise to image.

### Q13. Why use gradient accumulation?

The model is memory-heavy, so the actual per-GPU batch size is small. Gradient accumulation simulates a larger batch by accumulating gradients across multiple forward/backward passes before optimizer step.

### Q14. How do you know the model did not just memorize the dataset?

You can reduce memorization risk with LoRA dropout, low learning rate, pose-diverse sampling, and qualitative tests on prompts not present in the dataset. A stronger answer would include nearest-neighbor checks between generated images and training images.

### Q15. What does merging LoRA do?

Merging folds the low-rank adapter update into the base transformer weights. After merging, the model can be loaded as a normal Diffusers pipeline without PEFT, which simplifies deployment.

## 19. Concepts Checklist

You should be able to explain these concepts:

- Text-to-image generation
- Diffusion models
- Forward diffusion and reverse denoising
- DDPM scheduler
- Latent diffusion
- VAE encoder and decoder
- T5 text encoder
- Transformer attention
- Query, key, value, and output projections
- Diffusion Transformer / DiT
- LoRA and low-rank updates
- LoRA rank, alpha, and dropout
- Mixed precision training
- Gradient scaling
- Gradient accumulation
- AdamW
- Cosine learning rate schedule
- Warmup
- MSE noise-prediction loss
- Classifier-free guidance
- Negative prompting
- FID
- T2I-CompBench
- W&B experiment logging
- Hugging Face Diffusers pipeline
- PEFT adapters
- Gradio demo deployment

## 20. Demo Talking Points

When showing the demo:

- Use the same prompt and seed for base and LoRA.
- Start with a simple prompt: "A photorealistic human hand with five clearly separated fingers."
- Then show a harder prompt: "A hand gripping a glass" or "two hands holding each other."
- Point out improvements in finger separation, palm shape, and reduced extra-finger artifacts.
- Also show a failure case. This makes the presentation more credible.
- Explain that negative prompts are used for both base and LoRA comparisons, so the improvement is not only from prompting.

## 21. Suggested Slide Outline

1. Title: PixArt-Alpha LoRA for Anatomically Accurate Hand Generation
2. Motivation: Why hand generation fails
3. Base Model: PixArt-Alpha pipeline
4. Dataset: 11K Hands, FreiHAND, sharpness, pose diversity
5. Captioning: LLaVA-Next hand-detail captions
6. Method: Frozen VAE/T5, LoRA on transformer attention
7. Training: Noise prediction loss and hyperparameters
8. Results: Base vs LoRA samples
9. Evaluation: FID, T2I-CompBench, qualitative analysis
10. Tradeoffs and limitations
11. Future work
12. Demo and conclusion

## 22. Strong Final Takeaway

The core contribution is not inventing a new diffusion architecture. The contribution is a practical, end-to-end adaptation pipeline for a known generative failure mode:

- Curated domain-specific data
- Detailed auto-captioning
- Parameter-efficient LoRA fine-tuning
- Checkpoint and merged-model deployment
- Quantitative and qualitative evaluation
- Interactive demo

That is the right way to frame the mini-project: a realistic applied deep learning system that adapts a large pretrained generative model to improve a specific visual capability under limited compute.
