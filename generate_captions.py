from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch, json, os
from tqdm import tqdm

# ── Load model ──
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# ── Correct prompt format for LLaVA-Next (Mistral) ──
PROMPT = "[INST] <image>\nDescribe this hand image in detail: how many fingers are visible, their pose and separation, knuckle visibility, palm orientation, skin texture, and lighting conditions.[/INST]"

IMAGES_DIR  = "./images"
OUTPUT_JSON = "./captions_hands_easy.json"

# ── Resume support ──
captions = {}
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        captions = json.load(f)
    print(f"Resuming — {len(captions)} already done")

img_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

for fname in tqdm(img_files):
    if fname in captions:
        continue  # skip already captioned

    path  = os.path.join(IMAGES_DIR, fname)
    image = Image.open(path).convert("RGB")

    # ── image and text passed separately ──
    inputs = processor(
        text=PROMPT,
        images=image,
        return_tensors="pt"
    ).to("cuda:1")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Strip the prompt from output (model repeats it)
    if "[/INST]" in caption:
        caption = caption.split("[/INST]")[-1].strip()

    captions[fname] = {"path": path, "caption": caption}

    # Save after every image
    with open(OUTPUT_JSON, "w") as f:
        json.dump(captions, f, indent=2)

print(f"\nDone! {len(captions)} captions saved to {OUTPUT_JSON}")