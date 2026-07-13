---
blogpost: true
blog_title: "Triton-Based Optimization of Video Sparse Attention on ROCm"
date: "13 Jul 2026"
author: "Takashi Isobe, Fan Wang, Mou Li, He Cui, Mengmeng Ge, Dong Zhou, Fuwei Yang, Dong Li, Zhenyu Gu, Emad Barsoum"
thumbnail: 'rocm_vsa_thumbnail.png'
tags: "AI/ML"
category: "Applications & models"
target_audience: "AI DEVELOPERS AND ENTHUSIAST"
key_value_propositions: "Triton-based ROCm optimization of video sparse attention for AMD GPUs, accelerated by kernel tuning and enhanced with a lightweight linear-attention global branch to improve training stability and visual quality."
language: English
myst:
    html_meta:
        "author": "Takashi Isobe, Fan Wang, Mou Li, He Cui, Mengmeng Ge, Dong Zhou, Fuwei Yang, Dong Li, Zhenyu Gu, Emad Barsoum"
        "description lang=en": "Optimize video sparse attention on ROCm with GEAK and linear global context for faster, more stable video generation on AMD GPUs."
        "keywords": "ROCm, Triton, video sparse attention, linear attention,  video generation, Diffusion Transformers, kernel optimization"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Takashi Isobe, Fan Wang, Mou Li, He Cui, Mengmeng Ge, Dong Zhou, Fuwei Yang, Dong Li, Zhenyu Gu, Emad Barsoum"
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

# Triton-Based Optimization of Video Sparse Attention on ROCm

Video generation has become a major frontier in generative modeling, driven by large-scale data and increasingly scalable architectures. Among modern architectures, Diffusion Transformers (DiTs) have emerged as a dominant paradigm [1,2,3,4] by representing videos as spatio-temporal token sequences, enabling long-range interactions across frames and spatial regions, as well as flexible multimodal conditioning with text or audio. However, full self-attention scales quadratically with token count, making it increasingly expensive as spatio-temporal resolution and model size grow. Video sparse attention (VSA) [5,6] mitigates this cost by approximating full attention with a subset of informative token interactions, but its practical efficiency in both training and inference depends heavily on hardware-aware Triton kernel implementations.

In this blog, we present a Triton-based optimization of VSA on ROCm for AMD GPUs, combining heuristic kernel design with agent-based optimization. Compared with the original VSA implementation, our optimized VSA kernels achieve 1.37× and 1.33× speedup in the forward and backward passes, respectively. Compared with full attention, the optimized VSA delivers 1.81× and 1.96× speedup in the forward and backward passes, respectively. We further integrate the optimized VSA into Hummingbird-XT [7], a step-distillation training framework built on Wan2.2-5B [1], and demonstrate 1.13× end-to-end training acceleration. In addition, the optimized VSA provides 1.23× speedup for supervised fine-tuning of Wan2.2-5B [1].

Another key contribution is addressing the training instability introduced by VSA in joint step-distillation. In the widely used DMD2 paradigm [8], this instability stems from a context asymmetry in the score matching setup, where the student model relies on VSA and therefore sees primarily local information, whereas the teacher and critic models use full attention and retain the complete global context. This mismatch can cause score drift in the student, leading to collapse artifacts such as abrupt background transitions, blurred faces, and distorted limbs. To address this, we add a lightweight linear-attention branch as an auxiliary source of global context, which greatly improves stability at minimal additional cost. Our solution (SLA*) follows a design similar to that of [6]. On ROCm, however, the sparse attention implementation in [6] incurs higher latency than that of [5]. This makes the gains from our ROCm-optimized VSA even more substantial, reducing the training-time forward-plus-backward latency from 228.34 ms to 73.06 ms, a 3.13× speedup over [6].

We open-source the training code and datasets to support the community in building, experimenting with, and deploying advanced video generation models on AMD GPUs.

## Key Takeaways

- We present a ROCm-optimized implementation of video sparse attention (VSA) that delivers 1.37x and 1.33x speedups in the forward and backward passes, respectively, over the original implementation on AMD Instinct MI325X GPUs.
- Compared with full attention, the optimized VSA accelerates end-to-end step-distillation training by 1.13x, supervised fine-tuning by 1.23x, and inference by 1.47x.
- We show that VSA can introduce instability in joint step-distillation training due to the asymmetric distillation setup, and that adding a lightweight linear attention branch as auxiliary global context significantly improves training stability and visual quality.
- We introduce SLA*, a ROCm-optimized sparse-attention operator that achieves a 3.13× speedup over the comparable design in [6].
- We open-source the training pipeline and datasets to support the community in building, experimenting with, and deploying advanced video generation models on AMD GPUs.

## Why Attention Becomes the Bottleneck in Video DiTs

```{figure} ./images/figure1.png
:align: center
:alt: Full attention and sparse attention comparison for video DiTs

Figure 1: Comparison between full attention and sparse attention for video DiTs. Full attention spends computation on many low-value token interactions, while sparse attention skips them to reduce cost and accelerate video generation.
```


Self-attention is the core primitive that gives DiTs their strength: every token can interact with every other token, enabling long-range dependency modeling across space and time. For video, this means a model can jointly reason over appearance, motion, camera movement, and object consistency. But this flexibility comes with a well-known cost. If a video latent contains $N$ tokens (e.g., 27280 for Wan2.2-5B [1]), dense self-attention scales with $O(N^2)$ interactions. In video generation, where tokens span both spatial and temporal dimensions, the problem becomes much worse.

To reduce the complexity for each self-attention, one key observation is that although dense attention computes all pairwise interactions, not all of these interactions are equally useful. In particular, many dependencies are strongly localized: (1) nearby spatial regions are often more relevant than distant ones, (2) neighboring frames dominate short-term motion consistency, and (3) only a subset of global tokens are needed to preserve scene-level structure. Motivated by this observation, previous methods follow block-sparse design and selectively preserve the most important interactions and skip the rest to accelerate video generation, as shown in figure 1.

## ROCm-Optimized Video Sparse Attention

Existing video sparse attention implementations are often built on top of Triton and work in two stages. First, a sparse block map is constructed to indicate which query blocks attend to which key/value blocks. Instead of materializing a dense attention mask, the kernel stores a compact block-level connectivity structure. This block map is then transformed into index representations that guide the kernel to compute attention only over the selected blocks, skipping the missing regions entirely. During training, the Triton-based kernel consists of a map-to-index preprocessing step that converts sparse layouts into kernel-friendly indices, together with a forward kernel that iterates over the selected KV blocks for each query block and a backward kernel that reconstructs gradient flow over the same sparse structure.

While the video sparse attention design intuitively improves efficacy, the upper bound also depends on the GPU-related optimizations.

To address this issue, we take a first step toward optimizing the VSA implementation on ROCm by leveraging GEAK-V3 [9], AMD's kernel optimization agent. Specifically, it carefully profiles the module-wise speed in the kernel execution pipeline and identifies that potential improvements lie in both kernel scheduling and memory behavior. The detailed optimization is shown in figure 2 below.

```{figure} ./images/figure2.png
:align: center
:alt: ROCm-aware acceleration of video sparse attention

Figure 2: ROCm-aware acceleration of video sparse attention with three key optimizations: pipelined stage tuning, matrix-instruction optimization, and local cache hints.
```

The key GEAK-generated optimizations can be summarized in three parts.

**(1) Pipeline-depth tuning.** GEAK-V3 [9] introduces ROCm-aware dispatch and separate forward/backward autotuning spaces, allowing the AMD backend to select Triton pipeline configurations independently from the original GPU-oriented table. The key parameter is `num_stages`, which controls the software-pipelining depth. In block-sparse attention, each query block traverses multiple selected KV blocks, exposing opportunities to overlap memory movement with matrix computation. By searching AMD-specific `num_stages` configurations for both forward and backward kernels, GEAK-V3 [9] improves pipeline efficiency under sparse traversal. This reduces idle cycles across training passes and yields gains in both single-step latency and end-to-end step-distillation/SFT throughput.

**(2) Cache-eviction hinting.** GEAK-V3 [9] optimizes ROCm-path memory accesses by applying `eviction_policy="evict_first"` to selected loads, including K/V loads in the forward kernel and Q/dO/K/V loads in the backward inner loops. In sparse attention, some operands and metadata are streamed during irregular block traversal, while others are reused across multiple sparse blocks. These hints prioritize early eviction for one-pass loads, reducing cache pressure and preserving cache capacity for higher-reuse operands. This improves effective bandwidth under irregular sparse traversal, especially in backward computation, where sparse indices, intermediate statistics, activations, and gradients place heavier pressure on the memory hierarchy.

**(3) Sparse-index traversal optimization.** GEAK-V3 [9] optimizes the traversal path for sparse block indices, reducing overhead beyond the math-tile computation itself. Kernel efficiency depends not only on matrix operations, but also on how efficiently the sparse structure is traversed. GEAK-V3 [9] reduces redundant index handling, improves the organization of query-to-key and key-to-query mappings, and minimizes control overhead inside the hot path. These changes are especially important for training, where backward computation requires both forward-style sparse access and reverse traversal for gradient propagation. By streamlining sparse-index traversal across forward and backward passes, GEAK-V3 [9] improves locality and reduces overhead under irregular sparse access patterns.

## Video Sparse Attention with Linear Global Context

Compared with full attention, video sparse attention is inherently limited in capturing global context. To compensate for this limitation, existing methods typically introduce a coarse branch that exposes each token to a low-resolution view of the full context, partially restoring long-range dependency modeling. While such a design is often sufficient for standard training, this limitation becomes more pronounced in step distillation settings such as DMD2 [8], where the student, teacher, and critic are inherently exposed to unequal contextual information. In particular, the student, built on sparse attention, is restricted to a local and summarized view of the context, whereas the teacher and critic, equipped with full attention, can model the global context. This structural asymmetry introduces a distribution mismatch between the student and teacher, making alignment more difficult and leading to distillation drift, as shown in figure 3.

```{figure} ./images/figure3.png
:align: center
:alt: Information asymmetry and symmetry in step distillation

Figure 3: Information asymmetry in step distillation. A lightweight linear attention branch restores global context and stabilizes training.
```

To alleviate this issue, we augment video sparse attention with a lightweight global branch based on linear attention. Our goal is to preserve the efficiency benefits of sparse attention while providing the student with an inexpensive approximation of full-sequence information. Linear attention is computationally cheaper than full attention and naturally aggregates information across the entire sequence, making it a natural complement to local block-sparse patterns. In our design, the original sparse attention remains the primary branch for local modeling, while linear attention serves as an auxiliary pathway for global information exchange. This hybrid attention design stabilizes DMD2 [8] training and leads to higher-quality visual results.

With a ROCm-optimized sparse-attention kernel, the resulting operator (SLA*) reduces the training-time forward-plus-backward latency from 228.34 ms to 73.06 ms, a 3.13× speedup over the comparable design in [6].

## Experimental Results
We train models on 8× AMD Instinct™ MI325X GPUs, using Hummingbird-XT (a few-step, lightweight-VAE video generation model built on Wan2.2-5B [1]), with a batch size of 2 across 8 GPUs and bfloat16 precision. The software stack is ROCm 6.3.1 (build 6.3.1-48) on Ubuntu 22.04.4 LTS, with RCCL 2.21.5, hipFFT 1.0.17, and custom Triton-based VSA kernels. Each training sample is a 121-frame clip at 704 × 1280 in pixel space, which the lightweight VAE encodes into 31 latent frames; these are trained as a single block (num_frame_per_block = 31). We follow the same training-data setup as Hummingbird-XT.


In Table 1, we compare Hummingbird-XT, using four attention variants: full attention, the original video sparse attention, the ROCm-optimized variant, and its Global Context-enhanced version. We evaluate performance on the image-to-video task using VBench. For caption recaptioning, we use Qwen2.5-3B [10], followed by VBench [11] evaluation.

| Attention | Subject Consistency | Background Consistency | Motion Smoothness | Aesthetic Quality | Imaging Quality | Temporal Flickering |
|---|---:|---:|---:|---:|---:|---:|
| Full Attention | 96.28 | 97.01 | 99.24 | 61.96 | 71.23 | 98.69 |
| VSA | 97.70 | 98.03 | 99.40 | 62.60 | 71.18 | 99.09 |
| ROCm VSA | 97.84 | 98.07 | 99.44 | 62.79 | 71.00 | 99.20 |
| ROCm SLA* | 97.86 | 98.19 | 99.46 | 62.90 | 71.08 | 99.25 |

Table 2 compares the **DiT runtime** across the four attention settings for generating a **121-frame** video at **704 × 1280** resolution on **AMD Instinct™ MI325X** devices.

| Attention | DiT (s) | Acceleration |
|---|---:|---:|
| Full Attention | 4.16 | 1.00x |
| VSA | 3.42 | 1.21x |
| ROCm VSA | 2.82 | 1.47x |
| ROCm SLA* | 2.92 | 1.42x |

Testing by AMD on the configuration described herein. Performance results are based on specific hardware, software, model, and workload configurations and may vary.
## Summary

In this blog, we present a ROCm-optimized video sparse attention that can significantly accelerate Video Diffusion Transformer training and inference on AMD Instinct™ MI325X GPUs. Our Triton-based optimization tailored for ROCm achieves 1.37x and 1.33x speedups in the forward and backward passes compared with the original implementation. We further show that a lightweight global-context enhancement with linear attention (SLA*) can improve stability and visual quality in sparse-attention-based step distillation. By open-sourcing the training pipeline and datasets, AMD aims to enable the developer community to build, experiment with, and deploy efficient high-performance video generation models on AMD hardware. For more details on the optimization methodology, training framework, and model design insights, please refer to the full Hummingbird-XT project resources. Developers are encouraged to explore the open-source implementation and evaluate its performance on AMD GPU platforms. To request access and learn more, please visit AMD Developer Cloud. For further inquiries, contact the AMD team at amd_ai_mkt@amd.com.

## Additional Resources
Huggingface model cards: [ROCM-VSA](https://huggingface.co/amd/HummingbirdXT)

ROCm VSA code: [ROCM-VSA](https://github.com/AMD-AGI/HummingbirdXT/tree/main/HummingbirdXT_VSA/amd_VSA)

Full training code: [AMD-AIG-AIMA/HummingbirdXT](https://github.com/AMD-AGI/HummingbirdXT)

Related work on diffusion models by the AMD team:

- [Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms](https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html)
- [AMD Hummingbird-0.9B: An Efficient Text-to-Video Diffusion Model with 4-Step Inferencing](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)

## References

[1] Wan, Team, et al. "Wan: Open and advanced large-scale video generative models." *arXiv preprint arXiv:2503.20314* (2025).

[2] Wu, Bing, et al. "Hunyuanvideo 1.5 technical report." *arXiv preprint arXiv:2511.18870* (2025).

[3] Yang, Zhuoyi, et al. "Cogvideox: Text-to-video diffusion models with an expert transformer." *International Conference on Learning Representations*. Vol. 2025. 2025.

[4] HaCohen, Yoav, et al. "LTX-2: Efficient Joint Audio-Visual Foundation Model." *arXiv preprint arXiv:2601.03233* (2026).

[5] Zhang, Peiyuan, et al. "Faster video diffusion with trainable sparse attention." *Advances in Neural Information Processing Systems* 38 (2026): 152509-152534.

[6] Zhang, Jintao, et al. "Sla: Beyond sparsity in diffusion transformers via fine-tunable sparse-linear attention." *arXiv preprint arXiv:2509.24006* (2025).

[7] Isobe, Takashi, et al. "Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms." AMD ROCm Blog, 8 Jan. 2026. <https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html>.

[8] Yin, Tianwei, et al. "Improved distribution matching distillation for fast image synthesis." *Advances in neural information processing systems* 37 (2024): 47455-47487.

[9] Wang, Jianghui, et al. "GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks." AMD ROCm Blog, 1 Aug. 2025. <https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html>.

[10] Yang, An, et al. "Qwen3 technical report." *arXiv preprint arXiv:2505.09388* (2025).

[11] Huang, Ziqi, et al. "Vbench: Comprehensive benchmark suite for video generative models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
