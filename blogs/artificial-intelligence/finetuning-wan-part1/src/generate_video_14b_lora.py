import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
        ]),
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/Wan-AI/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors",
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth")
    ],
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-T2V-A14B_high_noise_lora/epoch-4.safetensors", alpha=1)
pipe.load_lora(pipe.dit2, "models/train/Wan2.2-T2V-A14B_low_noise_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

video = pipe(
    prompt="A black-and-white cartoon scene, in classic animation style, featuring an anthropomorphic giraffe, with an exaggerated muzzle, riding a bicycle under water on the ocean floor, with fish and sealife with comical expressions swimming around in the background. The giraffe is simultaneously playing a trumpet and expressing feelings of enjoyment. Musical notes fly out of the trumpet. The scene captures a comical and whimsical classic animated world.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.2-T2V-A14B_lora.mp4", fps=15, quality=5)