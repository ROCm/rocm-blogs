accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../Disney-VideoGeneration-Dataset \
  --dataset_metadata_path ../Disney-VideoGeneration-Dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 10 \
  --model_paths '[
    [
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
    ],
    "models/Wan-AI/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.2-T2V-A14B/Wan2.1_VAE.pth"
]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-T2V-A14B_high_noise_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.875