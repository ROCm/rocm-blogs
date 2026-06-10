---
blogpost: true
blog_title: "Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms "
date: 8 Jan 2026
author: 'Takashi Isobe,He Cui,Mengmeng Ge,Dong Zhou,Dong Li,KuanTing Lin,Chandra Yang,Wickey Wang,Emad Barsoum'
thumbnail: 'hummingbird_xt.png'
tags: GenAI, PyTorch, Diffusion Model
category: Applications & models
target_audience: AI developers and enthusiast
key_value_propositions: HummingbirdXT, an efficient video diffusion models for high-quality video generation showcases the resource-friendly training and inference on AMD Instinct™ MI300X GPUs.
language: English
myst:
    html_meta:
        "author": "Takashi Isobe,He Cui,Mengmeng Ge,Dong Zhou,Dong Li,KuanTing Lin,Chandra Yang,Wickey Wang,Emad Barsoum"
        "description lang=en": "Learn how to use Hummingbird-XT and Hummingbird-XTX modelS to generate videos. Explore the video diffusion model acceleration solution, including dit distillation method and light VAE model."
        "keywords": "HummingbirdXT, Diffusion Models, Instinct GPU Accelerators"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Takashi Isobe,He Cui,Mengmeng Ge,Dong Zhou,Dong Li,KuanTing Lin,Chandra Yang,Wickey Wang,Emad Barsoum"
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

# Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms

<sup id="fn1">*The first three authors (Isobe, Cui, and Ge) contributed equally to this work. </sup>

With the rapid advancement of diffusion models [1], artificial intelligence (AI) has entered a new era of visual understanding and generation. Video generative models, one of the most exciting and rapidly emerging areas in AI, demonstrate remarkable capabilities in producing cinematic videos and reshaping entire production pipelines. However, the substantial computational and memory demands of large-scale diffusion transformer backbones require high-performance computation and highly optimized operators, which restrict their deployment to server-grade GPUs. In contrast, client-grade GPUs dominate the end-user market, making efficient on-device video generation highly desirable. Nevertheless, their limited power budget and memory bandwidth impose stringent constraints on computation and energy consumption, hindering the deployment of current DiT models [2,3].

In this blog, AMD presents a highly efficient video generative model based on the Wan-2.2-5B TI2V architecture [3], termed **Hummingbird-XT**, designed to bridge the last mile of deploying large-scale DiT models on client-grade GPUs, including both Navi48 dGPUs and Strix Halo iGPUs. Hummingbird-XT achieves up to 33× speedup on Strix Halo and enables efficient video generation on Navi48, where the original model runs out of memory, through two key innovations:  (1) carefully designed data curation for step distillation, which enables 3-step generation while preserving background consistency and motion smoothness, especially reducing human-body ghosting in high-dynamic-action scenarios; and (2) a lightweight VAE, which accelerates latent decoding while preserving crucial image quality and semantic understanding capability. Hummingbird-XT is fully trained on 16 AMD Instinct™ MI325X GPUs, marking an important milestone in showcasing operator-friendly support for large-scale video generative models and stable training performance for both DiT and VAE architectures. In addition, this blog introduces **Hummingbird-XTX**, an efficient DiT-based member of the Hummingbird family designed for long video generation, built on Wan-2.1-1.3B T2V. Hummingbird-XT and its family further enrich the AMD ecosystem by enabling more devices to benefit from AMD-trained native models. The training code and datasets have been open-sourced to the community, empowering developers to build and experiment with models on AMD GPUs.  

Key Takeaways:

- Presenting an end-to-end acceleration pipeline for DiT-based video models, combining step distillation and a lightweight VAE decoder to significantly reduce inference latency and memory footprint while preserving visual quality.
- Introducing Hummingbird-XT, an efficient DiT-based video generation framework built upon Wan-2.2-5B, designed to bridge the last mile of deploying large-scale diffusion transformer models on AMD client-grade GPUs, including Navi48 dGPUs and Strix Halo iGPUs.
- Achieving up to 33× speedup on Strix Halo iGPUs, and enabling efficient video generation on Navi48, where the original Wan-2.2-5B model exceeds memory limits.
- Introducing Hummingbird-XTX, an efficient DiT-based extension of Hummingbird-XT for long-video generation, built upon Wan-2.1-1.3B, addressing challenges in temporal scalability and computational growth.
- Demonstrating stable training convergence for both DiT and VAE architectures on AMD Instinct™ MI325 and AMD Instinct™ MI300 GPUs, highlighting strong operator support and robust training performance on AMD platforms.
- Open-sourcing the training pipeline and datasets, further enriching the AMD ecosystem and empowering developers to build, experiment with, and deploy video generative models on AMD GPUs.

## Towards 3-Step Generation in Diffusion Transformers

### Distillation Method

Diffusion-based video generative models typically consist of three key components: (1) a condition encoder, (2) a diffusion backbone, and (3) a VAE decoder. Among these, the diffusion backbone contains billions of parameters and executes numerous time-consuming operations repeatedly across many denoising steps (e.g., 50), leading to substantial latency and energy consumption. In addition, the classifier-free guidance (CFG) mechanism further increases computation cost. This motivates reducing the number of denoising steps from 50 to just 3 and adopt a fully CFG-free design.

To achieve this, we adopt step distillation using a self-forcing variant of Distribution Matching Distillation (DMD) [4,5,14]. The core idea is to train the student DiT to follow the teacher’s reverse-diffusion behavior without relying on ground-truth text–video pairs. Taking Wan-2.2-5B as an example, the distillation process proceeds in three steps.

- First, the student model produces a latent video trajectory using a backward-simulation sampler that traverses a set of large diffusion timesteps (e.g., 1000, 750, 500, 250), and the final denoised latent from this trajectory is taken as the "clean" latent target.
- Second, a random diffusion timestep is sampled, and Gaussian noise is added to construct a noisy latent.
- Third, two score networks are applied to this noisy latent: a frozen real score network from the teacher and a trainable fake score network from the student, both incorporating conditional and unconditional branches through classifier-free guidance. The real score provides the teacher’s target denoising direction, and the fake score represents the student’s estimate; their difference yields a KL-style correction that forces the student to follow the teacher’s update.

To support both image-to-video and text-to-video generation, the VAE-encoded latent of the input image is injected into the first-frame region of the noisy latent sequence with a spatial-temporal mask.  

### Data Curation for Step Distillation

Although self-forcing DMD avoids the need for high-quality videos, the caption design of input images still has a critical impact on student performance. As you can see in Figure 1, below, the three rows of the figure illustrate common data-curation issues in text formulation that can noticeably degrade generation quality: (1) **Row 1**: short captions that focus only on the action while lacking sufficient background or appearance details; (2) **Row 2**: long captions that describe objects, background, and actions in an overly verbose manner; (3) **Row 3**: long captions that overemphasize actions and camera motion.

```{figure} ./images/figure1.png
:align: center
:alt: Scaling performance
Figure 1: Examples of generation artifacts caused by different text–image curation issues. Row 1: short action-only captions, which lead to motion ghosting. Row 2: overly long descriptive captions, which result in hallucinated objects. Row 3: captions that overemphasize actions and camera motion, which cause excessive motion and unstable generation.
```

To address this problem, we introduce a carefully designed data curation pipeline. We first collect 140k text–image pairs by merging the MagicMotion [5], OpenVid-HD [6], and HumanVid [7] datasets, all of which provide high-quality first frames extracted from videos. However, the captions in OpenVid-HD and HumanVid are often overly detailed, describing full backgrounds and object appearances in ways that are not optimal for student-teacher generation. To improve caption quality, we use Qwen-2.5-14B Instruct [8] to recaption each sample following the rules below:  (1) start directly with the main subject or scene; (2) describe only what is explicitly mentioned in the original caption, focusing on the main subject; (3) avoid cinematography terms; (4) do not introduce any new objects, people, actions, or details; (5) do not output analysis, reasoning, instructions, or explanations; (6) do not use first-person language.

We further employ Qwen-2.5-72B to evaluate the rewritten captions and filter out outliers, resulting in 70k high-quality text–image pairs for training.

## Towards an Efficient and Lightweight Video VAE

In addition to the DiT backbone, the VAE decoder is another time-consuming component in video generation. Some video VAEs rely on  3D convolutional architectures [3], while others incorporate attention modules to achieve high reconstruction quality. Such designs significantly increase computational overhead. To address this issue, we introduce a more efficient and lightweight VAE decoder in Hummingbird-XT, which can be seamlessly substituted for the VAE in the Wan-2.2-5B model and significantly reduces the computational cost of video VAE decoding while preserving visual quality. The VAE decoder training pipeline is shown in Figure 2. Moreover, compared with the recent lightweight VAE work Taehv [9], the proposed lightweight VAE decoder achieves better visual quality in reconstruction and generation.

```{figure} ./images/figure2.png
:align: center
:alt: Scaling performance
Figure 2: Training Pipeline of Our Compressed VAE Decoder.
```

### Architectural Optimization

The original Wan-2.2 VAE decoder consists of five stacked blocks that progressively upsample the low-resolution latent space back to high-resolution pixel space. To enable the light VAE to be integrated into the original DiT model without further training, we preserve the same compression ratio and number of latent channels as the Wan 2.2 VAE, and redesign only the decoder architecture to remove redundant computation.

Specifically, we replace standard 3D convolutions with 3D depthwise separable convolutions (3D DW Conv), decomposing each 3D convolution into a channel-wise spatial convolution (depthwise) followed by a $1\times1$ channel mixing convolution (pointwise). This design decouples temporal and spatial computation and substantially reduces both parameters and FLOPs. Consistent with observations from Turbo-VAED [10], high-resolution decoder blocks have a critical impact on reconstruction quality, whereas low-resolution decoder blocks exhibit significant computational redundancy. Therefore, we apply 3D DW Conv only to the first three decoder blocks, while retaining standard 3D convolutions in the last two blocks. In addition, we reduce the number of convolution layers and latent channels per block and remove attention layers entirely.

### Training Strategy Optimization

Training a video VAE from scratch typically requires large-scale data and substantial computational resources. To accelerate training, we freeze the original Wan-2.2 VAE encoder and train only the compressed VAE decoder. With the latent space fixed, the decoder learns a stable mapping from this invariant latent representation to pixel space, avoiding latent space collapse that can occur when the encoder and decoder are updated simultaneously. The trained decoder can be seamlessly integrated into the original Wan-2.2 DiT model without requiring any additional training.

We further adopt a student-teacher distillation strategy inspired by V-VAE [11] to enhance performance. Specifically, we extract intermediate features from the first three blocks of the original Wan-2.2 VAE decoder and use them as teacher signals, while encouraging the corresponding blocks in the compressed decoder to reproduce these features. To address the mismatch in channel dimensionality between the teacher and student decoders, we apply a one-layer trainable convolution layer to transform the student features to the same dimensionality as the teacher features. We then compute an additional feature distillation loss between block-wise features  $\phi_i^{\text{teacher}}$ and projected block-wise features $\mathcal{M}_i(\phi_i^{\text{student}})$ using mean squared error (MSE):

$$\mathcal{L}_{\text{distill}} = \sum_{i=1}^{3} \left\| \phi_i^{\text{teacher}} - \mathcal{M}_i(\phi_i^{\text{student}}) \right\|_2^2$$

In addition to the distillation loss, we also include the reconstruction loss, LPIPS loss, and KL divergence, which are widely used in VAE training.

## Towards Fast and High-quality Long Video Generation

Here, we present Hummingbird-XTX, a member of the Hummingbird-XT family built upon Wan-2.1-1.3B, towards efficient long-video generation. Generating long videos poses challenges in computational growth and temporal scalability. Most state-of-the-art video generation models depend on bidirectional attention [12], requiring full-sequence processing for each generated frame. This results in quadratic computational complexity with respect to frame count and makes long-video generation infeasible for real-time or streaming scenarios.

To alleviate this problem, autoregressive (AR) models [13] appear to be an ideal choice. They generate frame by frame and support KV caching for accelerated inference. However, directly applying AR architectures introduces severe exposure bias: during training the model relies on ground-truth frames, whereas during inference it must condition on its own imperfect predictions. This train–test mismatch causes videos to drift or collapse within a few seconds as errors accumulate over time. To mitigate this error accumulation, we found that simply fine-tuning existing models was insufficient; a fundamental redesign of the initialization and training paradigm was required.

To enable scalable long-video generation, we follow the frame-by-frame generation paradigm of autoregressive (AR) models [13] and leverage KV caching for accelerated inference. However, naively applying this generation paradigm is ineffective, as it introduces severe exposure bias: the model is trained using ground-truth frames but must condition on its own predictions during inference. This discrepancy between training and inference leads to compounding errors over time, causing rapid degradation or collapse of generated videos.

To mitigate this issue, we adopt a fundamentally redesigned initialization and training paradigm, going beyond simple fine-tuning of existing models.

Directly distilling a bidirectional teacher into an autoregressive student with DMD is unstable due to mismatched attention mechanisms. We therefore introduce an ODE-based initialization that warms up the student before distillation by sampling Gaussian noise and running the pretrained teacher through an ODE solver [14] to generate a small set of reverse-diffusion trajectories. The student is pretrained to regress from intermediate noisy states to clean frames using a simple initialization loss, allowing it to approximate the teacher’s underlying video distribution and providing a stable starting point for subsequent autoregressive distillation.

### Self-Forcing: Simulating Inference During Training

Even with ODE initialization, AR models suffer from severe exposure bias: they are trained on ground-truth context but must depend on self-generated history at inference, causing rapid error accumulation. To address this, we adopt Self-Forcing [15], which performs full autoregressive self-rollout during training using two techniques: training-time KV cache, which reuses previous-frame states to reduce long-sequence unrolling cost, and gradient truncation with few-step generation, where a lightweight diffusion approximation is used and gradients are backpropagated only through the final denoising step to control memory usage. This paradigm enables holistic distribution-matching optimization, allowing the model to learn to correct its own errors and achieve stable long-video generation.

### Mitigating Semantic Drift via Frame-Level Attention Sink

To reduce the computational complexity of inference from quadratic $O(N^2)$ to linear $O(N)$ for long sequences, we adopted a Short Window Attention mechanism. However, standard sliding window mechanisms suffer from a critical limitation: as the window traverses the temporal axis, information from the initial frames is evicted from the context. Since the initial frames capture the core style, color, and subject identity, losing them causes later frames to gradually lose saturation and semantic detail, resulting in color degradation or style drift.

```{figure} ./images/figure3.png
:align: center
:alt: Scaling performance
Figure 3: Frame Sink allocates a dedicated KV cache region to permanently retain initial frame features, acting as a global anchor to prevent color degradation in arbitrarily long sequence generation.
```

Inspired by the concept of Attention Sinks in Large Language Models [16], we introduce the Frame Sink mechanism to address this problem. Specifically, we allocate a dedicated region within the KV cache to permanently retain the Key and Value features of the video's first chunk, which remain exempt from eviction regardless of the sliding window’s position. When generating the *t*-th frame, the model attends not only to the current local window
$W$ but also explicitly incorporates the persistent first-frame features KV sink (as seen in Figure 3). This establishes the first frame as a Global Anchor, ensuring that the model maintains access to the initial semantic and visual settings throughout arbitrarily long generation sequences.

### Long Video Decoding Optimization

For decoding high-resolution long videos, directly applying 3D convolutions over the entire latent input leads to excessive memory consumption and computational cost, and may cause convolution kernels to fall back to inefficient implementations. To address this issue, we propose two frame-splitting strategies that decompose long videos into multiple short clips for decoding. You can see the two strategies in Figure 4.

```{figure} ./images/figure4.png
:align: center
:alt: Scaling performance
Figure 4: Two decoding strategies for long video.
```

We first train two types of VAE decoders on short video clips: (1) Causal VAE decoder composed of causal convolutions, which uses only past frames as context, and (2) Non-causal VAE decoder composed of non-causal convolutions, which leverages both past and future frames as inputs. To extend these decoders to long-video decoding, we adopt different strategies for each architecture.  For the causal VAE decoder, we employ a causal cache mechanism, where the video latent sequence is split into multiple non-overlapped latent clips along the temporal dimension and decoded sequentially. Intermediate features from decoding of the previous clip are cached and reused as contextual input for decoding of the current clip. For the non-causal VAE decoder, we apply a tiling strategy, in which video latent sequence is divided into overlapping clips along the temporal dimension. The overlapping regions provide additional temporal context for each latent clip. To ensure temporal continuity, the decoded video clips corresponding to overlapping regions are linearly blended. Experimental comparisons indicate that integrating the non-causal VAE decoder with tiling improves reconstruction quality and substantially accelerates inference for both Hummingbird-XT and Hummingbird-XTX.

## Experimental Results

In Table 1, we compare Wan-2.2-5B and our method on the text-to-video task evaluated on VBench, under settings with and without caption recaption. We report the Quality Score, Semantic Score, and the overall Total Score.
| Model                     | Quality Score ↑ | Semantic Score ↑ | Total Score ↑ |
|---------------------------|-----------------|------------------|-------------|
| Wan-2.2-5B-T2V w/o recap  | 82.75           | 68.38            | 79.88       |
| Wan-2.2-5B-T2V with recap | 83.99           | 77.04            | 82.60       |
| Ours-T2V w/o recap        | 84.07           | 54.75            | 78.20       |
| Ours-T2V with recap       | 85.71           | 72.33            | 83.03       |

<p align="center">Table 1. Quantitative results for the text-to-video task on VBench.</p>

In Table 2, we compare Wan-2.2-5B and our method on the image-to-video task evaluated on VBench, under settings with and without caption recaption. We report Video–Image Subject Consistency, Video–Image Background Consistency, and the Quality Score. For caption recaption, we use **Qwen2.5-3B**, followed by evaluation with VBench.
| Model                     | Video-Image Subject Consistency ↑ | Video-Image Background Consistency ↑ | Quality Score ↑ |
|---------------------------|-----------------------------------|--------------------------------------|-----------------|
| Wan-2.2-5B-I2V w/o recap   | 97.89                             | 99.04                                | 81.43           |
| Wan-2.2-5B-I2V with recap  | 97.63                             | 98.95                                | 81.06           |
| Ours-I2V w/o recap        | 98.46                             | 98.91                                | 80.01           |
| Ours-I2V with recap       | 98.42                             | 98.99                                | 80.57           |

<p align="center">Table 2. Quantitative results for the image-to-video task on VBench.</p>

In Table 3, we compare the runtime for generating a 121-frame video at a resolution of 704×1280 across server-grade (AMD Instinct™ MI300 and AMD Instinct™ MI325) and client-grade (Strix Halo and Navi48) devices.
| Model          | MI300X | MI325X | Strix Halo iGPU | Navi48 dGPU |
|----------------|-------|-------|------------------|--------------|
| Wan-2.2-5B | 193.4s | 153.9s | 15000s           | OOM          |
| Ours | 6.5s  | 3.8s  | 460s             | 36.4s        |

<p align="center">Table 3. Runtime for generating a 121-frame video at 704×1280 resolution on server-grade (AMD Instinct™ MI300X and AMD Instinct™ MI325X GPU) and client-grade (Strix Halo and Navi48).</p>

In Table 4, we compare the original uncompressed VAE from Wan-2.2 (Wan-2.2 VAE), a lightweight VAE trained for Wan-2.2 (TAEW2.2) from Teahv [9], and our proposed lightweight VAE (Ours VAE). We evaluate VAE reconstruction performance on the UCF-101 dataset at a resolution of 256×256, using LPIPS, PSNR, and SSIM as evaluation metrics. In addition, we benchmark the decoding speed and memory consumption of the VAE decoders on a single AMD Instinct™ MI300X GPU at a resolution of 704×1280 (121 frames). Due to replacing 3D convolutions with 2D convolutions, TAEW2.2 suffers from a significant performance degradation, resulting in artifacts and blurring in the reconstructed videos. In contrast, our VAE incurs only a minor performance degradation compared to the original VAE while substantially reducing computational cost.

| Model      | LPIPS ↓ | PSNR ↑ | SSIM ↑ | RunTime ↓ | Memory ↓ |
|------------|---------|--------|--------|-----------|----------|
| Wan-2.2 VAE| 0.0141  | 35.979 | 0.9598 | 31.34s    | 11.37G   |
| TAEW2.2    | 0.0575  | 29.599 | 0.8953 | 0.14s     | 1.35G    |
| Ours VAE   | 0.0260  | 34.635 | 0.9483 | 2.29s     | 2.71G    |

<p align="center">Table 4. Performance and efficiency comparison of different VAE decoders on AMD Instinct™ MI300X GPU.</p>

We compare our method with long-video generation models such as Self-Forcing [15], CausVid [17], LongLive [18], and RollingForcing [19] in terms of generation speed and three other commonly used benchmarks, under a video generation setting of 321 frames at a resolution of 832 × 480. The detailed quantitative results are reported in Table 5.
| Model          | FPS ↑ | Flicker Metric ↓ | DOVER ↑ | VBench Quality ↑ | VBench Semantic ↑ | VBench Total ↑ |
|----------------|-------------------|------------------|---------|----------------|-----------------|--------------|
| Self-Forcing    | 19.28             | 0.1010           | 84.37   | 81.99          | 80.09           | 81.61        |
| Causvid         | 18.24             | 0.0972           | 82.77   | 81.96          | 77.02           | 80.97        |
| LongLive       | 21.32             | 0.0947           | 84.07   | 82.86          | 81.61           | 82.61        |
| RollingForcing | 19.57             | 0.0928           | 85.16   | 82.94          | 80.61           | 82.47        |
| Ours           | 26.38             | 0.0946           | 84.55   | 83.42          | 79.22           | 82.58        |

<p align="center">Table 5. Quantitative results for long video generation on three benchmarks.</p>

## Summary

In this blog you learned video diffusion acceleration strategies and long video generation methods, and explored lightweight VAE architectural and training strategy optimization. By open-sourcing the training code, inference code based on [20], dataset, and model weights for the AMD Hummingbird-XT and Hummingbird-XTX, we aim to empower the AI developer community to build efficient and high-performing generative video models.  Developers are encouraged to download and explore the model on AMD hardware. For details on the training methodology, inference pipeline, and model insights, please visit the [Hummingbird-XT](https://github.com/AMD-AGI/HummingbirdXT) to access the source code and the Hugging Face model card to download the model weights. As an added benefit, AMD offers access to a dedicated cloud infrastructure featuring the latest GPU instances for AI development. To request access and learn more, please visit the [AMD developer cloud](https://devcloud.amd.com/). For further inquiries, contact the AMD team at amd_ai_mkt@amd.com.

## Additional Resources

Huggingface model cards: [AMD-HummingbirdXT](https://huggingface.co/amd/HummingbirdXT)

Full training code: [AMD-AIG-AIMA/HummingbirdXT](https://github.com/AMD-AGI/HummingbirdXT)

Related work on diffusion models by the AMD team:

- [AMD Hummingbird-0.9B: An Efficient Text-to-Video Diffusion Model with 4-Step Inferencing](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)

Please refer to the following resources to get started with training on AMD ROCm™ software:

- Use the [public PyTorch ROCm Docker images](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html) that enable optimized training performance out-of-the-box
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)

## References

1. Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.

2. Peebles W, Xie S. Scalable diffusion models with transformers[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2023: 4195-4205.

3. Wan T, Wang A, Ai B, et al. Wan: Open and advanced large-scale video generative models[J]. arXiv preprint arXiv:2503.20314, 2025.

4. Zhang, Peiyuan, et al. "Fast video generation with sliding tile attention." arXiv preprint arXiv:2502.04507 (2025).

5. Li Q, Xing Z, Wang R, et al. Magicmotion: Controllable video generation with dense-to-sparse trajectory guidance[J]. arXiv preprint arXiv:2503.16421, 2025.

6. Nan K, Xie R, Zhou P, et al. Openvid-1m: A large-scale high-quality dataset for text-to-video generation[J]. arXiv preprint arXiv:2407.02371, 2024.

7. Wang Z, Li Y, Zeng Y, et al. Humanvid: Demystifying training data for camera-controllable human image animation[J]. Advances in Neural Information Processing Systems, 2024, 37: 20111-20131.

8. Hui B, Yang J, Cui Z, et al. Qwen2. 5-coder technical report[J]. arXiv preprint arXiv:2409.12186, 2024.

9. Bohan O B. Taehv: Tiny autoencoder for hunyuan video[EB/OL].(2025)

10. Zou Y, Yao J, Yu S, et al. Turbo-vaed: Fast and stable transfer of video-vaes to mobile devices[J]. arXiv preprint arXiv:2508.09136, 2025.

11. Yao J, Yang B, Wang X. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 15703-15712.

12. Yin T, Zhang Q, Zhang R, et al. From slow bidirectional to fast causal video generators[J]. arXiv e-prints, 2024: arXiv: 2412.07772.

13. Chen B, Martí Monsó D, Du Y, et al. Diffusion forcing: Next-token prediction meets full-sequence diffusion[J]. Advances in Neural Information Processing Systems, 2024, 37: 24081-24125.

14. Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.

15. Huang X, Li Z, He G, et al. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion[J]. arXiv preprint arXiv:2506.08009, 2025.

16. Xiao G, Tian Y, Chen B, et al. Efficient streaming language models with attention sinks[J]. arXiv preprint arXiv:2309.17453, 2023.

17. Yin T, Zhang Q, Zhang R, et al. From slow bidirectional to fast autoregressive video diffusion models[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 22963-22974.

18. Yang S, Huang W, Chu R, et al. Longlive: Real-time interactive long video generation[J]. arXiv preprint arXiv:2509.22622, 2025.

19. Liu K, Hu W, Xu J, et al. Rolling forcing: Autoregressive long video diffusion in real time[J]. arXiv preprint arXiv:2509.25161, 2025.

20. Moore-animateanyone. https://github.com/aigc-apps/VideoX-Fun, 2025.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
