import torch
import numpy as np
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from peft import PeftModel
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="bf16",  # can be "no", "fp16", or "bf16"
    device_placement=True
)

# --- Paths ---
pretrained_model = '/workspace/hf_cache/Wan2.1-T2V-14B-Diffusers'
lora_path = '/workspace/logs/video_ocr/wan_flow_grpo_14B/checkpoints/checkpoint-176/lora'

# --- Load pipeline ---
print("Loading pipeline...")
pipeline = WanPipeline.from_pretrained(
    pretrained_model,
    torch_dtype=torch.bfloat16,
)

# Disable gradient computations for inference
pipeline.vae.requires_grad_(False)
pipeline.text_encoder.requires_grad_(False)
pipeline.transformer.requires_grad_(False)

pipeline.safety_checker = None

# Load LoRA fine-tuned weights (optional)
print("Loading LoRA adapter...")
pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
pipeline.transformer.set_adapter('default')

# --- Prepare model for distributed inference ---
pipeline.vae.to(accelerator.device, dtype=torch.float32)
pipeline.text_encoder.to(accelerator.device, dtype=torch.bfloat16)
pipeline.transformer.to(accelerator.device)

# Accelerator handles device wrapping automatically
pipeline = accelerator.prepare(pipeline)

# --- Prompts ---
prompt = (
    "A vast shot opens in the dark expanse of space, scattered with distant stars and a faint red hue from the "
    "planet Mars. The camera slowly glides past a futuristic spaceship, its sleek metallic hull reflecting the "
    "starlight in soft gradients. Subtle thruster lights pulse along its surface as the camera tilts to reveal "
    "bold markings etched across the side: “Mars Colony One.” The letters gleam under the distant cosmic glow. "
    "The camera pulls back gradually, capturing the ship’s immense scale as it drifts silently through the "
    "endless void."
)

negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

# --- Inference ---
if accelerator.is_main_process:
    print("Starting distributed inference...")

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=5.0
).frames[0]

# Save video only on main process to avoid race conditions
if accelerator.is_main_process:
    export_to_video(output, 'output_wan21_14b.mp4', fps=24)

accelerator.wait_for_everyone()
print("Inference complete.")
