---
blogpost: true
blog_title: "Wan2.2 Fine-Tuning: Tailoring an Advanced Video Generation Model on a Single GPU"
date: Aug 19 2025
author: 'Arttu Niemela, Balazs Toth'
thumbnail: 'giraffe_thumbnail.jpg'
tags: Fine-Tuning, AI/ML, GenAI
category: Applications & models
target_audience: Data Scientists, Software Developers
key_value_propositions: It demonstrates how to fine-tune Wan2.2 on a single AMD Instinct MI300X GPU.
language: English
myst:
    html_meta:
        "author": "Arttu Niemela, Balazs Toth"
        "description lang=en": "Fine-tune Wan2.2 for video generation on a single AMD Instinct MI300X GPU with ROCm and DiffSynth."
        "keywords": "fine-tuning Wan2.2 with ROCm, fine-tuning Wan2.2 on AMD, Wan2.2, DiffSynth on AMD"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Arttu Niemela, Balazs Toth"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Wan2.2 Fine-Tuning: Tailoring an Advanced Video Generation Model on a Single GPU

This blog post will guide you through fine-tuning Wan2.2 - a state-of-the-art video generation model - on a single AMD Instinct MI300X GPU. By following this guide, you'll unlock Wan2.2's advanced video generation capabilities and customize the output — whether in a unique artistic style or a specialized domain — all while running memory efficiently even on a single GPU. Here are some examples of how you can put this guide into practice:

- Style customization
  - Cartoon vs. photo-realistic rendering
  - A specific color-grading or camera movement patterns
- Domain adaptation
  - Industry specific, e.g. drone footage or time-lapses of construction sites
  - Scientific, e.g. cellular or molecular motion videos
  - Sports, e.g. highlight reels or niche sports
- Character or asset consistency
  - Making a specific character or object appear consistent across scenes

We'll perform full-parameter fine-tuning and show how to leverage [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) to reduce memory requirements. We'll walk through the complete workflow step-by-step and provide an example to generate videos with the fine-tuned model.

## Overview

Harnessing the power of AI to generate captivating videos is no longer the exclusive domain of large research labs. With advancements in parameter-efficient fine-tuning techniques, even powerful video generation models can be adapted for specific tasks on more accessible hardware. In this blog post, we'll dive into the exciting world of fine-tuning the Wan2.2 video generation on a single AMD Instinct MI300X GPU.

[Wan2.2](https://github.com/Wan-Video/Wan2.2), a state-of-the-art video generation model, offers immense potential for creating diverse and high-quality video content. Fine-tuning this model allows for task-specific optimization, such as adapting to a particular video style, without the need for complete retraining. However, like many large AI models, full-parameter fine-tuning can be computationally demanding, requiring substantial memory and compute resources. This often presents a significant barrier for individual developers and smaller teams.

To overcome these challenges, we will leverage LoRA, a revolutionary parameter-efficient fine-tuning (PEFT) method. As introduced by Hu et al. [in their seminal 2021 paper](https://arxiv.org/abs/2106.09685), LoRA significantly reduces the number of trainable parameters by freezing the pre-trained model weights and injecting small, trainable rank-decomposition matrices into the model's architecture. This ingenious approach allows us to fine-tune Wan2.2 for specific video generation tasks with remarkably fewer resources, making it feasible on a single GPU without compromising performance.

In this guide, we'll use [DiffSynth](https://github.com/modelscope/DiffSynth-Studio), a powerful framework for diffusion-based video generation that provides seamless integration with LoRA fine-tuning capabilities. DiffSynth offers an intuitive interface for working with video diffusion models and makes the fine-tuning process more accessible while maintaining the flexibility needed for advanced customization.

This blog is a part of our team's ongoing efforts to deliver ease-of-use and maximum performance in the video generation domain. You may also be interested in learning how to set up [ComfyUI](https://rocm.blogs.amd.com/artificial-intelligence/comfyui-on-amd/README.html) - a graphical user interface for video generation - which provides a prepared workflow for using the Wan model among many others. For key optimization techniques, check out [FastVideo and TeaCache](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html), or add [video editing](https://rocm.blogs.amd.com/artificial-intelligence/video-editing-models/README.html) into your toolbox.

## Step-by-Step Instructions

Here, we'll walk through the whole process step-by-step — environment setup, custom dataset preparation and the fine-tuning workflow.

### 0. Requirements

#### Hardware

- Support for ROCm 6.3.4 or higher

For details on supported hardware, see the [ROCm System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

#### Software

- Docker

This guide showcases two versions of the Wan2.2 text-to-video models:

- Wan2.2-TI2V-5B  
- Wan2.2-T2V-A14B  

For training, we used an AMD Instinct MI300X GPU with 192 GB VRAM, which easily fits the 5B model and also the 14B model using LoRA.  
The following table shows VRAM requirements for training different versions of Wan2.2:

| Model            | Architecture | CPU offloading * | VRAM (GiB) |
| ---------------- | :----------: | :--------------: | ---------: |
| Wan2.2-TI2V-5B   | LoRA         | no               | 31.34      |
| Wan2.2-TI2V-5B   | LoRA         | yes              | 29.96      |
| Wan2.2-TI2V-5B   | Full         | no               | 133.01     |
| Wan2.2-TI2V-5B   | Full         | yes              | 42.74      |
| Wan2.2-T2V-A14B  | LoRA         | no               | 74.93      |
| Wan2.2-T2V-A14B  | LoRA         | yes              | 74.35      |
| Wan2.2-T2V-A14B  | Full         | no               | 537.60 **  |
| Wan2.2-T2V-A14B  | Full         | yes              | 95.71      |

The VRAM requirements shown in the table are the allocated VRAM for each training run. We measured the VRAM usage with the PyTorch Memory Profiler. For more details see [Categorized Memory Usage](https://pytorch.org/blog/understanding-gpu-memory-1).  

\* Both the optimizer states and the parameters offloaded to CPU. Decreases VRAM usage at the cost of increased RAM usage.  
** We tested this with 4 GPUs (MI300X).

### 1. Pull the Docker Image

```console
docker pull rocm/pytorch-training:v25.6
```

### 2. Launch the Docker Container

```console
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  --hostname=ROCm-FT \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/pytorch-training:v25.6 \
  /bin/bash
```

### 3. Install the Dependencies

```console
pip install \
    deepspeed==0.16.7 \
    huggingface_hub[cli]==0.30.1 \
    numpy==1.26.4 \
    opencv-python==4.10.0.82
```

### 4. Install DiffSynth-Studio

Clone the DiffSynth-Studio into `/workspace` and install.

```console
cd /workspace
```

```console
git clone https://github.com/modelscope/DiffSynth-Studio.git
```

```console
cd DiffSynth-Studio
```

**Important:** Use the specific commit version below for compatibility. This guide was tested with commit `f0ea049faa7250f568ef0a0c268a8345481cf6d0`. Using the latest version from the main branch may introduce compatibility issues with our setup and dependencies.

```console
git checkout f0ea049faa7250f568ef0a0c268a8345481cf6d0
```

```console
pip install .
```

### 5. Download the Model

Download the Wan2.2 model from Hugging Face into `/workspace/DiffSynth-Studio/models`. Wan2.2 has 5B and 14B parameter versions. In this guide, we will use the smaller 5B model for our single GPU setup.

```console
cd /workspace
```

```console
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B
```

### 6. Download the Dataset

We'll use the [Steamboat Willie](https://en.wikipedia.org/wiki/Steamboat_Willie) dataset from [Hugging Face](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) into `/workspace`.

```console
cd /workspace
```

```console
huggingface-cli download --repo-type=dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir ./Disney-VideoGeneration-Dataset
```

### 7. Create the Metadata

The training dataset folder needs to include a metadata.csv file with video filenames with their corresponding captions. Make a file called `create_metadata.py` in your workspace and copy-paste the Python code below. Then, ensure the `data_set_dir` variable points to your dataset folder and run the code.

For example, using the Steamboat Willie dataset, the `/workspace` directory should look like this:

```text
workspace/
├── DiffSynth-Studio/
│   └── ...
├── create_metadata.py
└── Disney-VideoGeneration-Dataset/
    ├── prompt.txt
    ├── videos.txt
    ├── metadata.csv
    ├── videos/
    │   └── video1.mp4
    │   └── ...
    ├── LICENSE
    ├── README.md
    └── README_zh.md
```

```python
import pandas as pd
import os

def create_metadata_csv(videos_file_path, captions_file_path, output_csv_path):
    """
    Create a metadata.csv file combining video filenames with their corresponding captions.
    
    Args:
        videos_file_path: Path to the videos.txt file containing video filenames
        captions_file_path: Path to the prompt.txt file containing captions
        output_csv_path: Path where the metadata.csv file will be saved
    """
    
    # Read video filenames and remove "videos/" prefix
    with open(videos_file_path, 'r', encoding='utf-8') as f:
        video_filenames = [line.strip() for line in f if line.strip()]
    
    # Read captions
    with open(captions_file_path, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f if line.strip()]
    
    # Verify that we have the same number of videos and captions
    if len(video_filenames) != len(captions):
        print(f"Warning: Number of video files ({len(video_filenames)}) doesn't match number of captions ({len(captions)})")
        min_length = min(len(video_filenames), len(captions))
        video_filenames = video_filenames[:min_length]
        captions = captions[:min_length]
        print(f'Using first {min_length} entries for both')
    
    # Create DataFrame
    metadata_df = pd.DataFrame({
        'video': video_filenames,
        'prompt': captions
    })
    
    # Save to CSV
    metadata_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f'Created metadata.csv with {len(metadata_df)} entries')
    print(f'Saved to: {output_csv_path}')
    
    # Display first few entries
    print('\nFirst 5 entries:')
    print(metadata_df.head())
    
    return metadata_df

# Define file paths
dataset_dir = './Disney-VideoGeneration-Dataset'
videos_file = os.path.join(dataset_dir, 'videos.txt')
captions_file = os.path.join(dataset_dir, 'prompt.txt')
output_csv = os.path.join(dataset_dir, 'metadata.csv')

# Create the metadata.csv file
metadata_df = create_metadata_csv(videos_file, captions_file, output_csv)
```

### 8. Configure Accelerate Library

```console
accelerate config
```

Our configuration for the single GPU setup was as follows:

>In which compute environment are you running?  
>**This machine**  
>Which type of machine are you using?  
>**No distributed training**  
>Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:  
>**NO**  
>Do you wish to optimize your script with torch dynamo?[yes/NO]:  
>**NO**  
>Do you want to use DeepSpeed? [yes/NO]:  
>**yes**  
>Do you want to specify a json file to a DeepSpeed config? [yes/NO]:  
>**NO**  
>What should be your DeepSpeed's ZeRO optimization stage?  
>**2**  
>Where to offload optimizer states?  
>**none**  
>Where to offload parameters?  
>**none**  
>How many gradient accumulation steps you're passing in your script? [1]:  
>**1**  
>Do you want to use gradient clipping? [yes/NO]:  
>**NO**  
>Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:  
>**NO**  
>Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]:  
>**NO**  
>How many GPU(s) should be used for distributed training? [1]:  
>**1**  
>Do you wish to use mixed precision?  
>**bf16**  

You can also use our example configuration file `./accelerate_config.yaml`.

**Note**: For the single GPU process demonstrated in this guide, the DeepSpeed configuration doesn't make a difference. However the same setup can be used for distributed training on multiple GPUs just by selecting the number of GPUs appropriately. For more information on DeepSpeed for multi-GPU setups see the [DeepSpeed Usage Guide](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed).

### 9. Run Fine-Tuning

We are ready to fine-tune! The commands below will fine-tune the Wan2.2 5B model. Choose either the LoRA or the full fine-tuning and use the corresponding commands.

#### LoRA Fine-Tuning

Go to `DiffSynth-Studio` folder.

```console
cd /workspace/DiffSynth-Studio
```

Run this command to fine-tune the 5B model using LoRA for 10 dataset repeats and 5 epochs.

```console
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../Disney-VideoGeneration-Dataset \
  --dataset_metadata_path ../Disney-VideoGeneration-Dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 10 \
  --model_paths '[
    [
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
]' \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "models/train/Wan2.2-TI2V-5B_lora" \
  --lora_base_model dit \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32
```

#### Full-Parameter Fine-Tuning

Go to `DiffSynth-Studio` folder.

```console
cd /workspace/DiffSynth-Studio
```

Run this command to fine-tune the full-parameter 5B model for 10 dataset repeats and 5 epochs.

```console
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path ../Disney-VideoGeneration-Dataset \
  --dataset_metadata_path ../Disney-VideoGeneration-Dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 10 \
  --model_paths '[
    [
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
        "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
]' \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit"
```

### 10. Generate Videos

You can use the following scripts to generate videos using the fine-tuned models. The code is based on DiffSynth-Studio [validation examples](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo/model_training/validate_lora).

#### 5B LoRA Text-to-Video Generation

This video generation script assumes having trained the 5B model using LoRA for 5 epochs, as in the example training script. The script generates 181 frames and a video with a frame rate of 30, resulting in 6 seconds of video as in the training dataset.

Create a file `generate_video.py` in the `DiffSynth-Studio/` folder and copy-paste the code below.

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    ]
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

video = pipe(
    prompt="A black-and-white cartoon scene, in classic animation style, featuring an anthropomorphic giraffe, with an exaggerated muzzle, riding a bicycle under water on the ocean floor, with fish and sealife with comical expressions swimming around in the background. The giraffe is simultaneously playing a trumpet and expressing feelings of enjoyment. Musical notes fly out of the trumpet. The scene captures a comical and whimsical classic animated world.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_frames=181,
    seed=1, tiled=True,
)
save_video(video, "video_Wan2.2-TI2V-5B_lora.mp4", fps=30, quality=5)
```

#### 5B Full-Parameter Text-to-Video Generation

This video generation script assumes having trained the 5B full-parameter model for 5 epochs, as in the example training script. The script generates 181 frames and a video with a frame rate of 30, resulting in 6 seconds of video as in the training dataset.

Create a file `generate_video.py` in the `DiffSynth-Studio/` folder and copy-paste the code below.

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    ]
)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_full/epoch-4.safetensors", alpha=1)
pipe.enable_vram_management()

video = pipe(
    prompt="A black-and-white cartoon scene, in classic animation style, featuring an anthropomorphic giraffe, with an exaggerated muzzle, riding a bicycle under water on the ocean floor, with fish and sealife with comical expressions swimming around in the background. The giraffe is simultaneously playing a trumpet and expressing feelings of enjoyment. Musical notes fly out of the trumpet. The scene captures a comical and whimsical classic animated world.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_frames=181,
    seed=1, tiled=True,
)
save_video(video, "video_Wan2.2-TI2V-5B_full.mp4", fps=30, quality=5)
```

#### Example Results

Below we have examples of generated videos with the trained models, including:

- Wan2.2 5B original model
- Wan2.2 5B model fine-tuned using LoRA
- Wan2.2 14B original model
- Wan2.2 14B model fine-tuned using LoRA

We used the following prompt to generate all of the videos (same as in the video generation scripts above):  

`"A black-and-white cartoon scene, in classic animation style, featuring an anthropomorphic giraffe, with an exaggerated muzzle, riding a bicycle under water on the ocean floor, with fish and sealife with comical expressions swimming around in the background. The giraffe is simultaneously playing a trumpet and expressing feelings of enjoyment. Musical notes fly out of the trumpet. The scene captures a comical and whimsical classic animated world."`.  

We designed the prompt to match the style of those in the fine-tuning dataset, but introduced new subjects to test whether the fine-tuned model could retain the dataset’s art and animation style while applying it to unfamiliar subjects.

Using the Wan2.2 5B model, we fine-tuned with LoRA for 10 epochs, repeating the dataset 10 times. We found that 10 epochs were sufficient to produce a giraffe character in the desired style. However, signs of overfitting appeared much earlier: by epoch 5, prompting e.g. a pig instead of a giraffe — generated a character resembling Mickey Mouse from the training dataset. When using other datasets, you may experiment with the number of epochs and dataset repeats to find the best fine-tuning results.

For all the videos, we generated 181 frames with 30 frames per second, resulting in 6 second long videos. This is the same length and framerate as in the fine-tuning dataset videos.

We also fine-tuned the 14B parameters model using LoRA to provide a comparison between the 14B and 5B model capabilities. The 14B parameter model was likewise fine-tuned for 10 epochs and 10 repeats on the dataset.

##### Original 5B Model Video

```{video} videos/5b_orig_giraffe.mp4
:width: 832
:height: 480
:controls:
```

*Figure 1. A video generated with the original Wan2.2 5B parameter model, with 181 frames and framerate of 30.*

##### Fine-Tuned 5B Model Video

```{video} videos/5b_lora_giraffe.mp4
:width: 832
:height: 480
:controls:
```

*Figure 2. A video generated with our LoRA fine-tuned Wan2.2 5B parameter model, trained for 10 epochs and 10 repeats on the dataset, with 181 frames and framerate of 30.*

In this case, the LoRA-fine-tuned 5B model generates a video that matches the style of the Steamboat Willie dataset. By contrast, the original 5B model’s output is not entirely black-and-white, and its drawing style aligns less closely with the fine-tuning dataset. In both 5B model videos, the original and the fine-tuned, the quality of movement and animation is rather low, and for example, no trumpet or musical notes are successfully generated.

##### Original 14B Model Video

```{video} videos/14b_orig_giraffe.mp4
:width: 832
:height: 480
:controls:
```

*Figure 3. A video generated with the original Wan2.2 14B parameter model, with 181 frames and framerate of 30.*

##### Fine-Tuned 14B Model Video

```{video} videos/14b_lora_giraffe.mp4
:width: 832
:height: 480
:controls:
```

*Figure 4. A video generated with LoRA fine-tuned Wan2.2 14B parameter model, trained for 10 epochs and 10 repeats on the dataset, with 181 frames and framerate of 30.*

The fine-tuned 14B model generated a video featuring a giraffe character and fish characters closely matching the style of the Steamboat Willie dataset. Likewise, the animation resembles the dataset animation style well, and the movement is more consistent. The video features the prompted musical notes as well, except the instrument looks more like a bulb horn than a trumpet. By comparison, the original 14B model generated a black-and-white cartoon. However, the art style doesn't resemble the dataset, showing that the fine-tuning worked. That said, the animation quality and consistency may be slightly better in the video generated with the original model.

## Summary

In this blog we showed how to fine-tune the Wan2.2 video generation model on a single AMD MI300X GPU. We walked through the complete workflow step-by-step:

- Environment setup
- Installation
- Model downloading
- Dataset preparation
- Fine-tuning
- Video generation

With these instructions you can fine-tune both the Wan2.2 5B and the 14B parameter models with a dataset of your choice for customized advanced video generation.

This blog outlines our team’s ongoing efforts to enable video generation on AMD Instinct GPUs. We’re closely tracking emerging technologies and products in the video generation space, with the goal of delivering a seamless, high-performance user experience. Our work focuses on simplifying workflows and maximizing performance across a wide range of video generation tasks. For example, dive into this [video editing blog post](https://rocm.blogs.amd.com/artificial-intelligence/video-editing-models/README.html), learn key optimization techniques with [FastVideo and TeaCache](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html) or setup a comfortable graphical UI for video generation with [ComfyUI](https://rocm.blogs.amd.com/artificial-intelligence/comfyui-on-amd/README.html).  We’re also developing detailed guides and playbooks covering model inference, model serving, and end-to-end workflow management. Stay tuned for our upcoming updates as we continue advancing this field.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
