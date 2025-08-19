---
blogpost: true
blog_title: "Accelerating FastVideo on AMD GPUs with TeaCache"
date: 19 Aug 2025
author: 'Sopiko Kurdadze'
thumbnail: 'panda_thumbnail.jpeg'
tags: AI/ML, Diffusion Model, GenAI
category: Applications & models
target_audience: AI researchers and engineers working on video generation and diffusion models
key_value_propositions: Efficient Wan2.1 inference with FastVideo
language: English
myst:
    html_meta:
        "author": "Sopiko Kurdadze"
        "description lang=en": "Enabling ROCm support for FastVideo inference using TeaCache on AMD Instinct GPUs, accelerating video generation with optimized backends"
        "keywords": "FastVideo on AMD, Inference Wan2.1 on AMD, TeaCache, FastVideo ROCm support, Accelerated video generation"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Sopiko Kurdadze"
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

# Accelerating FastVideo on AMD GPUs with TeaCache

Video generation is entering a new era, powered by diffusion models that deliver photorealistic and temporally consistent results from text prompts. Models like Wan2.1 push the boundaries of what's possible in AI-generated content, but to unlock their full potential, inference performance must scale with both model complexity and hardware capabilities.

This blog introduces FastVideo running on AMD Instinct™ GPUs with ROCm - a step toward efficient, low-latency video generation on AMD platforms. You’ll learn how to enable the TeaCache optimization, set up a reproducible single-GPU inference environment, and generate high-quality videos using the Wan2.1 model - all optimized for ROCm.

This work is part of a broader set of efforts to make video generation workflows on AMD GPUs faster, more flexible, and easier to adopt. If you’re also interested in fine-tuning Wan2.2 for custom domains, check out our guide on [Wan2.2 Fine-Tuning: Tailoring an Advanced Video Generation Model](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html). For a graphical, node-based approach to building video generation pipelines, see [ComfyUI for Video Generation](https://rocm.blogs.amd.com/software-tools-optimization/comfyui-on-amd/README.html). And for extending video creation with editing, composition, and control capabilities, explore our work on [All-in-One Video Editing with VACE](https://rocm.blogs.amd.com/artificial-intelligence/video-editing-models/README.html). Together, these tools form a comprehensive toolkit for high-performance video generation and editing on AMD hardware.

## FastVideo

[FastVideo](https://github.com/hao-ai-lab/FastVideo) introduces several optimization techniques aimed at accelerating inference and training of video generation models. Key optimization strategies for inference include [TeaCache](https://arxiv.org/pdf/2411.19108), [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507), and [Sage Attention](https://arxiv.org/pdf/2410.02367).

FastVideo initially supported only the CUDA platform, but now also supports Apple's MPS and CPU backends. We are introducing the first steps toward supporting ROCm as an additional accelerator platform, as detailed in our contribution to [FastVideo PR#669](https://github.com/hao-ai-lab/FastVideo/pull/669). That contribution focused on the **TeaCache** optimization and **single-GPU inference**, which we expand on in this blog.

### What is TeaCache?

[TeaCache](https://github.com/ali-vilab/TeaCache), developed by [ali-vilab](https://github.com/ali-vilab), stands for **Timestep Embedding Aware Cache**. It is a training-free caching approach that estimates and leverages the fluctuating differences in model outputs across timesteps - thereby accelerating inference. TeaCache is effective across video, image, and audio diffusion models.

The core idea behind TeaCache is based on the observation that outputs of diffusion models tend to be similar between consecutive timesteps in the denoising loop. Previous caching methods, such as uniform caching strategies, do not account for variations in output differences between timesteps and therefore fail to maximize cache efficiency.

A more effective caching strategy would reuse cached outputs more frequently when the change between consecutive outputs is minimal. However, this difference cannot be known in advance - before computing the current output. To overcome this limitation, TeaCache exploits the prior that strong correlations exist between a model's inputs and outputs, allowing it to predict cache reuse opportunities more intelligently.

## Platform and Hardware Prerequisites

To get started, you’ll need a system that satisfies the following:

* **GPU** : AMD Instinct™ MI300X or other ROCm-compatible GPU
* **Host Requirements** : See [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

This tutorial assumes ROCm 6.3+ and Docker are available on your system.

## Step-by-Step Setup for FastVideo on ROCm

Follow the steps below to set up a complete, ROCm-enabled FastVideo inference environment.

### 1. Pull the Docker image

Make sure your system is ROCm-ready:

```bash
rocm-smi
```

And then pull the base container:

```bash
docker pull rocm/pytorch-training:v25.6
```

This image comes pre-loaded with most libraries you’ll need for inference, including:

* `torch==2.8.0a0+git7d205b2`
* `flash_attn==3.0.0.post1`
* `transformers==4.46.3`

### 2. Launch the Docker Container

Launch the Docker container in detach mode and map the necessary directories:

```bash
docker run -d \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --name fastvideo-tmp \
  -v $(pwd):/workspace/ \
  rocm/pytorch-training:v25.6 \
  tail -f /dev/null
```

**Note**: This command mounts the current directory `$(pwd)` to the `/workspace` directory in the container.

Enter the container:

```bash
docker exec -it fastvideo-tmp bash
```

To clean up later, use:

```bash
docker stop fastvideo-tmp
docker rm fastvideo-tmp
```

### 3. Install Dependencies

#### FastVideo Framework

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo
git checkout 5452369749432b3b0d6d0f3fb5f8001e2ff95631 # Commit introducing ROCm support
```

Edit `pyproject.toml` to avoid version conflicts by including only what’s missing. Update `dependencies` values with this:

```python
dependencies = [
    # Core Libraries
    "scipy==1.14.1", "six==1.16.0", "h5py==3.12.1",

    # Machine Learning & Transformers
    "timm==1.0.11", "peft>=0.15.0", "diffusers>=0.33.1",

    # Computer Vision & Image Processing
    "opencv-python==4.10.0.84", "pillow>=10.3.0", "imageio==2.36.0",
    "imageio-ffmpeg==0.5.1", "einops",

    # Experiment Tracking & Logging
    "wandb>=0.19.11", "loguru", "test-tube==0.7.5",

    # Miscellaneous Utilities
    "tqdm", "pytest", "PyYAML==6.0.1", "protobuf>=5.28.3",
    "gradio>=5.22.0", "moviepy==1.0.3", "flask",
    "flask_restful", "aiohttp", "huggingface_hub", "cloudpickle",
    # System & Monitoring Tools
    "gpustat", "watch", "remote-pdb",

    # Kernel & Packaging
    "wheel",

    # Training Dependencies
    "torchdata",
    "pyarrow",
    "datasets",
    "av",
]
```

And then run install:

```bash
pip install -e .
```

## Generate a Video with Wan2.1 and TeaCache

Create the following script:

```bash
cat > generate_video.py << 'EOF'
from fastvideo import VideoGenerator

def main():
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,
    )
    prompt = "Red panda playing in a snowy forest, surrounded by pine trees and falling snowflakes"
    video = generator.generate_video(
        prompt,
        return_frames=True,
        output_path="my_videos/",
        save_video=True,
        enable_teacache=True
    )

if __name__ == "__main__":
    main()
EOF
```

Run it:

```bash
python generate_video.py
```

Generation time **with TeaCache** enabled:

```bash
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:55<00:00,  1.11s/it]
INFO 07-31 12:45:25 [video_generator.py:310] Generated successfully in 72.17 seconds
```

Generation time **without** TeaCache:

```bash
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:41<00:00,  2.02s/it]
INFO 07-31 13:06:54 [video_generator.py:310] Generated successfully in 118.19 seconds
```

As you can see in the video figure below, videos generated with TeaCache show almost the same visual quality as those without it, proving that caching doesn't noticeably affect the output.

```{video} videos/fastvideo-teacache-cmp.mp4
:width: 1100
:height: 350
:controls:
```

### Supported Attention Backends

FastVideo supports several attention backends on ROCm.

#### Flash Attention 2 and 3

ROCm native [ROCm/flash-attention](https://github.com/ROCm/flash-attention) is already installed in the Docker image.

```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN python generate_video.py
```

#### Torch SDPA

Torch [Scaled Dot Product Attention](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html) is also provided within the already installed torch library.

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA python generate_video.py
```

#### Current Limitations

⚠️ Not yet supported:

* `SLIDING_TILE_ATTN`
* `SAGE_ATTN`

## Summary

This blog demonstrated how to enable FastVideo inference on AMD Instinct™ GPUs, with a focus on integrating the TeaCache optimization for faster, more efficient video generation. We walked through setting up a fully functional inference environment using the official ROCm PyTorch Docker image, installing FastVideo with ROCm support, and running single-GPU inference with the Wan2.1 model. Readers learned how to take advantage of TeaCache to reduce inference time, configure supported attention backends such as Flash Attention and Torch SDPA.

We are actively tracking emerging technologies and products in video generation and editing domains, aiming to deliver an optimized and seamless user experience for video generation on AMD GPUs. Our focus is on ensuring ease of use and maximizing performance for various video generation related tasks as exemplified by recent blog posts on [Fine-tuning of video generation model Wan2.2](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html), [ComfyUI for Video Generation](https://rocm.blogs.amd.com/software-tools-optimization/comfyui-on-amd/README.html), and [All-in-One Video Editing with VACE](https://rocm.blogs.amd.com/artificial-intelligence/video-editing-models/README.html). In parallel, we are developing additional playbooks, including model inference, model serving, and video generation workflow management.

## Additional Resources

* [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)
* [Timestep Embedding Aware Cache (TeaCache)](https://liewfeng.github.io/TeaCache/)
* [FastVideo Docs](https://hao-ai-lab.github.io/FastVideo/index.html)
* [FastVideo/inference/optimizations](https://hao-ai-lab.github.io/FastVideo/inference/optimizations.html#configuring-backends)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
