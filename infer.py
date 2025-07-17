
import os
import argparse
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dci_output_dir", type=str, required=True, help="Folder for DCI-VTON results")
parser.add_argument("--hand_mask_dir", type=str, required=True, help="Folder for hand masks")
parser.add_argument("--output_dir", type=str, required=True, help="Folder to save new results")
parser.add_argument("--lora_weights", type=str, default=None, help="path for .safetensors file (optional)")
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.output_dir, exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None,
    safety_checker=None
).to(device)

if args.lora_weights:
    lora_path = os.path.dirname(args.lora_weights)
    weight_name = os.path.basename(args.lora_weights)

    pipe.load_lora_weights(
        lora_path,
        weight_name=weight_name,
        adapter_name="hand_fix"
    )
    pipe.set_adapters(["hand_fix"], adapter_weights=[0.8])

def prepare_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = np.where(mask > 128, 255, 0).astype(np.uint8)
    return Image.fromarray(mask)

image_files = [f for f in os.listdir(args.dci_output_dir) if f.lower().endswith((".jpg", ".png"))]

for img_file in tqdm(image_files, desc="fixing hand"):
    dci_img_path = os.path.join(args.dci_output_dir, img_file)
    mask_img_path = os.path.join(args.hand_mask_dir, img_file)
    output_img_path = os.path.join(args.output_dir, img_file)

    if not os.path.exists(mask_img_path):
        print(f"[!] Not found mask, jumping: {mask_img_path}")
        continue

    dci_output = Image.open(dci_img_path).convert("RGB")
    hand_mask = prepare_mask(mask_img_path)

    result = pipe(
        prompt="realistic hands, detailed knuckles, no deformation",
        negative_prompt="deformed hands, extra fingers, bad anatomy",
        image=dci_output,
        mask_image=hand_mask,
        strength=0.65,
        num_inference_steps=75,
        guidance_scale=7.5,
        generator=torch.Generator(device).manual_seed(42)
    ).images[0]

    result = result.resize(dci_output.size)
    result.save(output_img_path, quality=95)

print(f"[✓] Results  saved to the '{args.output_dir}' directory.")
