---
blogpost: true
blog_title: "Efficient and Portable 3D Explorable World Generation on AMD GPUs"
date: 18 Jun 2026
author: 'Tong Shen, Jingai Yu, Dong Zhou, Dong Li, Emad Barsoum'
thumbnail: 'dworld-m3d-thumbnail.png'
tags: GenAI
category: Applications & models
target_audience: AI Developers
key_value_propositions: to provide an optimized and compatible version of Matrix3D, a high-quality framework for 3D world genertation
language: English
myst:
    html_meta:
        "author": "Tong Shen, Jingai Yu, Dong Zhou, Dong Li, Emad Barsoum"
        "description lang=en": "Learn how to run Matrix3D world generation on AMD GPUs more smoothly and efficiently."
        "keywords": "D World Generation, 3DGS"
        "vertical": "Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Content Creation / Rendering"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Tong Shen, Jingai Yu, Dong Zhou, Dong Li, Emad Barsoum"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# Efficient and Portable 3D Explorable World Generation on AMD GPUs

Explorable 3D world generation is becoming a foundational capability for spatial and embodied intelligence. Training agents that can navigate, reason, and interact with environments requires not just static datasets, but rich, immersive worlds that support free-view exploration and consistent geometry. Recent works like [Matrix3D](https://matrix-3d.github.io/)<sup>1</sup> have pushed this frontier forward by combining panoramic generation with explicit 3D reconstruction, enabling higher-quality and more coherent environments than prior video-only methods.

In this blog, we describe how we deployed Matrix3D on AMD GPUs (Instinct™ MI250 GPU and Instinct™ MI300 GPU). With several targeted modifications and optimizations, we made the framework both more efficient and more portable. The end-to-end generation time for a single world is reduced from 2887s to 1306s on a single MI250 GPU, and from 972s to 482s on a MI300 GPU.

We first provide a brief introduction to Matrix3D. We then present the optimizations we implemented to improve efficiency, including replacing CUDA-specific rendering kernels with more portable [Triton](https://triton-lang.org/main/) kernels, accelerating mesh optimization with more efficient solvers, using [gsplat](https://github.com/ROCm/gsplat)<sup>2</sup> for faster 3DGS fitting, and refactoring parts of the pipeline to remove overhead from repeated model loading and unnecessary I/O.

## Why Matrix3D?

Matrix3D is a strong open-source candidate for explorable world generation because:

- It provides better geometric consistency than purely video-based world models, which often struggle with stable structure.
- It uses an explicit 3D representation for the generated content.
- It supports free-view exploration.
- Its panoramic formulation offers broader spatial coverage than prior methods.

## What This Blog Covers

- Kernel optimization:
  replacing rendering kernels with more portable Triton kernels with help from the kernel-writing agent [GEAK](https://github.com/AMD-AGI/GEAK-agent), without sacrificing performance.
- Faster 3DGS fitting:
  replacing the original rasterization backend with gsplat for better efficiency and portability.
- Pipeline-level optimization:
  refactoring the pipeline to reduce repeated model loading, I/O overhead, and recomputation, while also accelerating depth-map merging.
- Reproducible setup:
  providing step-by-step instructions for running Matrix3D on AMD GPUs.
- End-to-end results:
  showing the speedup of the optimized version over the original implementation on AMD GPUs.

## Matrix3D Basics

Matrix3D is a panorama-based world generation framework. Unlike prior methods that are often limited to narrow-scene generation, panorama-based generation enables broader scene coverage and higher-quality reconstruction. More specifically, Matrix3D first converts a text or image prompt into a panorama image using a FLUX-based model. The panorama is then passed to the geometry estimation model [MoGe](https://github.com/microsoft/MoGe)<sup>3</sup>, which produces a depth map and a validity mask. Based on this estimated geometry, the framework builds a 3D mesh and uses it to generate coarse renders along predefined camera trajectories.

These coarse frames are then processed by a camera-guided video generation model to form a panoramic video. The generated 2D content is finally lifted into an explorable world represented by 3D Gaussian Splatting ([3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/))<sup>4</sup>. Matrix3D provides two reconstruction pipelines: an optimization-based approach and a Large Reconstruction Model (LRM) approach. In this blog, we focus on the optimization-based pipeline because it delivers higher quality and more consistent geometry.

```{figure} ./images/pipeline.gif
:align: center
:alt: Simplified Matrix3D pipeline
Figure 1. Illustration of the simplified Matrix3D pipeline.
```

## Optimizations

### Rendering Kernel

An important step in panoramic video generation is converting the scene into a 3D mesh and rendering it along predefined camera trajectories. The original project uses [nvdiffrast](https://github.com/NVlabs/nvdiffrast), which is not ideal for cross-device portability. [PyTorch3D](https://github.com/facebookresearch/pytorch3d) is another possible mesh-rendering backend, but its rendering speed is much lower for this workload. To improve portability, we implemented a Triton-based renderer with the help of GEAK, an open-source project for kernel development, especially for [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) and Triton kernels. GEAK is designed as an end-to-end pipeline for generating, testing, and refining custom kernels. In our case, we only need the forward pass, which makes the implementation more manageable.

To support both kernel development and evaluation, we generated test cases with paired inputs and outputs using the nvdiffrast implementation as the reference. These cases were used to validate correctness and compare the performance of nvdiffrast, PyTorch3D, and our Triton implementation. To avoid overfitting the kernel-writing process to the benchmark, the cases used during kernel development were kept separate from the final benchmarking set.

We compare the performance of nvdiffrast, PyTorch3D, and our Triton kernel on a mesh with more than 2M vertices and 4M faces.

| Metric | nvdiffrast | PyTorch3D | Triton |  
| --- | --- | --- | --- |
| Latency (ms) | 8.6 | 823.2 | 6.3 |
| Mean diff | - | 7.1e-5 | 6.3e-5 |

Table 1. Rendering latency and output difference on the MI250 GPU for nvdiffrast, PyTorch3D, and the Triton implementation on a large mesh.

PyTorch3D does not perform well for this forward-only, large-mesh rendering task. Our Triton kernel implemented by GEAK has 36% speedup over nvdiffrast while maintaining negligible numerical difference from the baseline.

### Optimizing 3DGS Fitting

Matrix3D originally uses [gaussian-diff-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for 3DGS fitting. We replace it with the gsplat library to improve both efficiency and portability. For the same scene reconstruction task, the comparison between the original implementation and our optimized version is shown below.

| Metric | Original | Optimized |
| --- | --- | --- |
| Latency (s) | 540 | 180 |
| Training losses | L1: 0.06, PSNR: 19.93 | L1: 0.05, PSNR: 19.94 |
| Eval losses | L1: 0.22, PSNR: 12.56 | L1: 0.22, PSNR: 12.68 |

Table 2. 3DGS fitting performance and reconstruction quality on the MI250 GPU for the original and optimized implementations.

This change reduces the 3DGS fitting cost by 66%, which contributes significantly to the end-to-end speedup of the full pipeline.

### Pipeline Optimization

We also applied several pipeline-level optimizations, especially in the geometry estimation stage. To recover an optimal 3D mesh across all panoramic frames, each panorama image is first split into perspective images for depth estimation and then merged back into a full panoramic depth map. The 3D mesh is then further adjusted to remain consistent across views.

To accelerate this stage, we replace the original LSMR solver with FFT-based or conjugate-gradient (CG) solvers for more efficient depth merging and geometry optimization. We also reduce overhead from repeated model loading and export steps. Together, these changes reduce total pipeline time by 360s on the MI250 GPU.

## Installation Instructions

### System Requirements

- Pre-built Docker image: [rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1](https://hub.docker.com/r/rocm/pytorch/tags)
- GPU platform: AMD Instinct™ MI250 GPU or AMD Instinct™ MI300 GPU

### Installation

After starting the Docker container, clone the project and run the installation script:

```bash
git clone https://github.com/AMD-AGI/m3d_rocm
bash scripts/install_m3d.sh
```

For text prompts, run:

```bash
bash scripts/run_m3d_t2i.sh
```

For image prompts, run:

```bash
bash scripts/run_m3d_i2i.sh
```

## Examples

We show both image-to-image and text-to-image generation results below.

| Prompt | Panoramic Video | 3D Scene |
| --- | --- | --- |
| ![Image prompt](<./images/case1_input.png>) | ![Panoramic video](<./images/case1_pano.gif>) | ![3D scene](<./images/case1.gif>) |
| an impressionistic winter landscape | ![Panoramic video](<./images/case2_pano.gif>) | ![3D scene](<./images/case2.gif>) |

Figure 2. Qualitative examples of image-to-image and text-to-image explorable world generation results.

The end-to-end latency is also illustrated in the table and figure below. Overall, the optimized version improves latency by 54% on the MI250 GPU and 50% on the MI300 GPU.

|  | Original | w/ gsplat | w/ solver opt. | w/ io opt. | Total Reduction |
| --- | --- | --- | --- | --- | --- |
| MI250 | 2887 | 2527 | 1406 | 1306 | 54% |
| MI300 | 972 | 853 | 507 | 482 | 50% |

Table 4. End-to-end latency comparison between the original and optimized pipelines on MI250 and MI300.

```{figure} ./images/comparison.png
:align: center
:width: 60%
:alt: End-to-end latency comparison across AMD GPUs
Figure 3. End-to-end latency comparison of the original and optimized Matrix3D pipelines on the MI250 GPU and the MI300 GPU.
```

## Summary

In this blog, we showed that Matrix3D can be adapted to run efficiently on AMD GPUs with a combination of kernel-level and pipeline-level optimization. By replacing CUDA-specific components with more portable alternatives, improving the efficiency of 3DGS fitting, and reducing overhead in geometry estimation and data flow, we made the framework more practical for ROCm-based systems without sacrificing output quality.

These changes deliver meaningful end-to-end speedups on both MI250 and MI300, while also improving portability across GPU platforms. We hope this work helps make high-quality explorable world generation more accessible to researchers and engineers working in the AMD ecosystem.

## Additional Resources

Related discussion of AMD’s ROCm AI agent and tooling ecosystem appears in the [Nitro-E overview](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html).

### ROCm and AMD resources

- [Matrix3D ROCm port (this blog)](https://github.com/AMD-AGI/m3d_rocm) — source repository for the optimized pipeline described above
- [ROCm PyTorch Docker images](https://hub.docker.com/r/rocm/pytorch/tags) — pre-built containers for ROCm.
- [GSplat for ROCm (ROCm blog)](https://rocm.blogs.amd.com/software-tools-optimization/gsplat/README.html) | [Repository](https://github.com/ROCm/gsplat)
- [GEAK (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html) | [Repository](https://github.com/AMD-AGI/GEAK-agent)

## References

1. Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang Yin, Eric Li, Yang Liu, Yikai Wang, Hao-Xiang Guo, and Yahui Zhou. *Matrix-3D: Omnidirectional Explorable 3D World Generation*. arXiv preprint arXiv:2508.08086, 2025. [Paper](https://arxiv.org/abs/2508.08086)
2. Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. *gsplat: An Open-Source Library for Gaussian Splatting*. Journal of Machine Learning Research, 2025. [Paper](https://www.jmlr.org/papers/v26/24-1476.html)
3. Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision*. CVPR, 2025. [Paper](https://arxiv.org/abs/2410.19115) [Repository](https://github.com/microsoft/MoGe)
4. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. *3D Gaussian Splatting for Real-Time Rendering of Radiance Fields*. ACM Transactions on Graphics, 2023. [Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
