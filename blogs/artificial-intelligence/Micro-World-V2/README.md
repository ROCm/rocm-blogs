---
blogpost: true
blog_title: "Micro-World 2: Streaming Real-time Interactive World Model"
date: "21 Oct 2026"
author: "Yu Geng, Wensong Chan, He Cui, Dong Zhou, Dong Li, Emad Barsoum"
thumbnail: 'micro_world2_thumbnail.png'
tags: "AI/ML, PyTorch, GenAI, Computer Vision"
category: "Applications & models"
target_audience: "Video Generation and World-Model Researchers, ML Performance and Inference Engineers, AMD ROCm Developers, Open-Source AI Developers"
key_value_propositions: "Discover how Micro-World 2 uses ROCm to deliver controllable, long-horizon, real-time world generation on a single AMD Instinct GPU."
language: English
myst:
    html_meta:
        "author": "Yu Geng, Wensong Chan, He Cui, Dong Zhou, Dong Li, Emad Barsoum"
        "description lang=en": "This blog demonstrates how Micro-World 2 combines precisely annotated action data, four-step diffusion distillation, and ROCm-optimized inference to deliver controllable, long-horizon, real-time world generation on a single AMD Instinct GPU through a fully open-source pipeline."
        "keywords": "Video Action Model, Real-time Video Generation, DMD, Video Inference Optimization"
        "vertical": "AI"
        "amd_category": "Developers"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Generative AI, Computer Vision"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yu Geng, Wensong Chan, He Cui, Dong Zhou, Dong Li, Emad Barsoum"
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

# Micro-World 2: Streaming Real-time Interactive World Model

Diffusion models have advanced rapidly in recent years, enabling high-quality video generation across a wide range of creative scenarios. As visual quality continues to improve, however, the field is increasingly shifting its focus from passive content generation toward controllable and interactive experiences. YUME 1.5 [1] explores controllable video generation through explicit interaction signals, while LingBot-World [2] and Sana-WM [3] investigate world models that generate future observations conditioned on the current environment and user actions. World Labs’ Marble [4] and Atlas [5] represent a complementary spatial-modeling direction: Marble creates editable, explicit 3D worlds from multimodal inputs, while Atlas organizes text, images, camera poses, and depth into a shared spatial context for generation, reconstruction, and simulation. Together, these efforts illustrate how generative models are evolving from passive content-synthesis tools toward controllable, persistent, and interactive world simulators.

Despite this progress, many existing approaches still rely on computationally intensive generation pipelines and are not designed for sustained, real-time interaction. This limitation presents a fundamental challenge for practical world models: they must continuously generate future observations while responding to user actions with low latency.

Meeting this requirement demands more than conventional offline video generation. A practical world model must operate as a real-time streaming system, with causal generation at its core. Specifically, it must continuously predict future frames using only past observations and current control inputs, without access to future frames or the complete video sequence.  

In this blog, we introduce **Micro-World 2**, a minute-scale interactive world model for causal streaming generation that achieves more than 60 decoded FPS on a single AMD Instinct™ MI350X GPU. Micro-World 1 [6] established our action-controlled world-modeling pipeline with Wan2.1-based 1.3B text-to-world and 14B image-to-world models trained on 6,000 Minecraft clips. It demonstrated keyboard and mouse control for open-domain video generation but primarily targeted offline, short-clip inference. Micro-World 2 extends this work with a unified Wan2.2 5B causal autoregressive model obtained through four-step distillation. It uses longer Capsaicin [7] trajectories with frame-aligned action and camera-pose metadata, together with a ROCm-optimized inference stack enhanced through AMD GEAK-guided optimization [8], to sustain streaming generation for up to one minute.

During each four-step streaming iteration, Micro-World 2 conditions on previously generated visual context, cached text features, and frame-aligned keyboard and mouse inputs to generate the next video chunk. It reaches 61.1 decoded FPS with the original Wan2.2 VAE and up to 116.4 decoded FPS with the optional lightweight VAE. Using the original VAE, it achieves a WBench average score of 76.68 and leads 8 of the 21 metrics reported in Table 4, with particular strengths in interaction, scene consistency, and action generalization. Together, the frame-accurate Capsaicin dataset, dedicated causal and action-control training, long-sequence DMD distillation, and GEAK-guided ROCm optimization enable interactive, long-horizon world generation on a single GPU.

The key takeaways from our blog are summarized below (see also Figure 1):

- **Data Collection**: We curated a dataset of 440 **interactive navigation clips** using open-source assets. Each clip contains either **200** or **400** frames and includes synchronized keyboard and mouse inputs, text captions and action annotations, enabling flexible and precise interaction control.

- **Efficient Training Pipeline**: We developed a dedicated training pipeline that enables action controllability, open-domain generation, and efficient few-step inference. All training stages were completed in approximately four days using eight AMD Instinct™ MI350X GPUs.

- **Real-Time Inference**: With AMD GEAK-guided inference optimization, Micro-World 2 achieves real-time video generation at **61.1 decoded FPS** on a single **AMD Instinct™ MI350X GPU**.

- **Fully Open Source**: Built on **Wan2.2 TI2V-5B**, Micro-World 2 is fully open source, including the curated dataset, training pipeline, and inference code.

- **AMD ROCm™ Software Support**: The entire training and deployment pipeline is built on the **AMD ROCm™ software ecosystem** and optimized for **AMD Instinct™ GPUs**.

```{figure} ./images/image1.png
:align: center
:alt: Key takeaways
Figure 1. Key takeaways.
```

## Technical Details

### Data Curation

Unlike conventional video generation datasets, an action-conditioned world model dataset must align every frame with the control signal that drives the visual transition. To make this alignment reliable, we use **Capsaicin**, AMD's open-source real-time rendering framework, as the primary collection engine. Capsaicin allows us to load open-source 3D scenes, place the camera at selected viewpoints, execute scripted keyboard and mouse controls, and record the resulting video stream together with synchronized frame-level action metadata.

As shown in Figure 2, our collection pipeline consists of four main stages:

```{figure} ./images/image2.png
:align: center
:alt: Data curation pipeline
Figure 2. Pipeline of data curation.
```

1. **Scene preparation.** We start with a mixture of Capsaicin built-in scenes and open-source 3D assets. Large scenes are divided into smaller navigable regions or camera anchors so that the camera can explore local areas without frequently crossing scene boundaries, entering inaccessible regions, or intersecting unsuitable geometry. This tile-and-anchor strategy makes the dataset more diverse while keeping each clip visually stable and physically meaningful.
2. **Scripted action generation.** For each valid anchor, we generate a per-frame action script containing keyboard movements, mouse yaw, mouse pitch, and an action phase label. The action schedule covers forward and backward movement, left and right strafing, horizontal camera rotation, vertical camera rotation, and compound controls such as forward plus yaw, backward plus strafe, or yaw plus pitch. This produces richer action transitions than clips containing only one isolated movement.
3. **Frame-accurate rendering and logging.** During collection, Capsaicin runs in a fixed-frame-rate benchmark mode. For each frame, the engine applies the scripted control, renders the scene, and writes an action log containing the keyboard state, mouse delta, camera pose, and camera position delta.
4. **Quality filtering and packaging.** We first render high-quality clips and then curate them manually and automatically. Low-quality samples, including clips with visible scene leakage, severe collision, black frames, unstable exposure, or invalid camera paths, are removed. The retained clips are resized to a uniform resolution and packaged with synchronized per-frame action and camera metadata for training and evaluation.

Each clip is saved with a compact but complete action schema. Mouse control is represented as a per-frame movement, where the x-axis controls yaw and the y-axis controls pitch. In addition, we save the resulting camera rotation delta, camera position, and position delta after the action is applied. This makes the dataset useful for both direct action-conditioned training and later diagnostic analysis. For example, we can verify whether a model responds to the intended key press, or compare the magnitude of generated motion with the real camera displacement in the metadata.

A key design choice is to separate **visual diversity** from **action consistency**. We vary scenes, tiles, anchors, and camera viewpoints to expose the model to many visual layouts, but the action representation remains consistent across all clips. When extending selected clips from 200 to 400 frames, we preserve the same control semantics and re-render from the original anchor rather than retroactively modifying labels. This allows short clips and longer clips to share the same action vocabulary while supporting both local motion learning and longer-horizon streaming evaluation.

### Training Scheduler

Based on the dataset collected and prepared as described in the preceding section, this section presents the model training process. To enable real-time interaction in the world model, we train a four-step action-controllable autoregressive model based on Wan2.2 5B [9]. As shown in Figure 3, our training pipeline consists of five stages: causal autoregressive base model training, LoRA-based visual-quality adaptation, action-control module training, teacher model adaptation, and few-step distillation. We describe each of these stages in detail below.

```{figure} ./images/image3.png
:align: center
:alt: Training pipeline paradigm
Figure 3. Overview of the training pipeline.
```

1. **Causal Autoregressive Base Model Training.** We first train an autoregressive diffusion model to support streaming inference in subsequent stages. The base model, Wan2.2 5B, employs bidirectional attention and therefore cannot directly enable real-time inference. We replace its bidirectional attention modules with unidirectional causal attention and train the resulting model using the autoregressive diffusion training strategy introduced by Causal-Forcing [10] for Wan2.1. We use the MIXKIT dataset [11] for this stage.
2. **LoRA-Based Visual-Quality Adaptation.** To better adapt the model to real-world scenarios, we perform LoRA-based [12] visual-quality adaptation using the MIXKIT [11] dataset. Specifically, we insert LoRA modules into the autoregressive diffusion base model described above and train them on real-world videos. Although this stage uses the same data as the previous stage, it serves a different objective: maximizing visual quality.
3. **Action-Control Module Training.** Following Micro-World 1 [6], we use adaptive layer normalization (AdaLN) [13] for action conditioning, enabling effective action injection into the model. During action-module training, we freeze all other model components and optimize only the action module. We train this module using our self-collected Capsaicin dataset.
4. **Teacher Model Adaptation.** The original Wan2.2 5B cannot be directly used as the teacher model for subsequent few-step distillation for two reasons. First, the native model does not support the target resolution required by our model; using it directly would introduce visual artifacts and degrade image quality. Second, it lacks action control. Distilling an action-controllable student from such a teacher would weaken the action signal and impair the student’s ability to follow actions. To address these limitations, we adapt the teacher model in two steps:
   - **Resolution adaptation.** We add a LoRA module to the Wan2.2 base model and train only this module on the MIXKIT dataset at the target resolution.
   - **Action conditioning.** We add an action module identical to that used in the student model and train it on the Capsaicin dataset.
5. **Few-Step Distillation.** We apply Distribution Matching Distillation (DMD) [14] to obtain a four-step world model for real-time inference. The action-controllable Wan2.2 model adapted in Stage 4 serves as the teacher, while the action-controllable autoregressive model from Stages 1–3 provides the multi-step student initialization. During distillation, we freeze the student base-model weights, add LoRA modules, and optimize only the LoRA parameters. Because the action AdaLN module has no corresponding LoRA adapters, it remains frozen throughout training. We evaluate DMD using sequences of 81 and 161 frames; the 161-frame setting yields substantially better visual quality and long-horizon temporal consistency.

All training was conducted on eight AMD Instinct™ MI350X GPUs and completed in approximately four days.

### Inference Optimization

Real-time streaming requires both a few-step generation model and an efficient execution pipeline. We used [AMD GEAK](https://github.com/AMD-AGI/GEAK) [8], an open-source autonomous optimization agent for AI workloads, to profile the complete Micro-World 2 streaming path, identify bottlenecks, optimize kernels and execution, and validate performance and output consistency on the target system.

All results in this section were measured on a single AMD Instinct™ MI350X GPU at 352 × 640 resolution with three-latent-frame streaming chunks. Generation latency is reported per chunk, and decoded FPS includes both model generation and streaming VAE decoding.

#### Generation-Side Optimizations

The optimization path first applies system-level techniques, including operator fusion, caching, synchronization removal, Triton kernels, per-frame AdaLN, and HIP Graph execution. It then introduces AMD-specific AITER attention and tuned hipBLASLt GEMMs.

The cumulative optimization stages are:

1. **Operator fusion and prompt caching.** Fuse AdaLN modulation, gated residual, RMSNorm, and QKV/KV projections, and reuse text embeddings and cross-attention key/value tensors across denoising steps and streaming chunks.
2. **Host synchronization removal and Triton RoPE.** Replace repeated device-to-host reads with cached host metadata, construct sequence tensors directly on the GPU, and compute RoPE only for new tokens while reusing the rotated history.
3. **Per-frame AdaLN.** Move repeated AdaLN work from per-token computation to per-frame or per-chunk computation, broadcast the resulting features across tokens, and fuse LayerNorm with modulation and residual operations.
4. **HIP Graph execution.** Capture and replay the complete forward path, including the preamble, fused q/k normalization, RoPE and KV writes, and the action module.
5. **AMD-specific kernels.** Use AITER variable-length FMHA with split-KV and tuned hipBLASLt GEMM solutions with fused bias and GELU epilogues.

Together, these optimizations reduce generation latency from 312.4 ms to 71.5 ms per chunk, a 77.1% reduction and 4.37× speedup, while increasing end-to-end decoded-video throughput from 27.4 to 61.1 FPS and preserving numerical consistency.

*Table 1. Cumulative generation-side optimization results on a single AMD Instinct™ MI350X GPU. Each stage adds one group of optimizations to the preceding stage.*

| Stage | Optimization | Generation latency/chunk | Latency saved at this stage | Share of total latency reduction | Cumulative generation speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Original implementation | 312.4 ms | — | — | 1.00× |
| 1 | Operator fusion + prompt caching | 220.4 ms | 91.9 ms | 38% | 1.42× |
| 2 | Host synchronization removal + Triton RoPE | 151.9 ms | 68.6 ms | 28% | 2.06× |
| 3 | Per-frame AdaLN | 120.5 ms | 31.4 ms | 13% | 2.59× |
| 4 | HIP Graph execution | 108.7 ms | 11.7 ms | 5% | 2.87× |
| 5 | AMD-specific attention and GEMM kernels | 71.5 ms | 37.2 ms | 15% | 4.37× |

The first two stages account for 67% of the total generation-latency reduction by eliminating repeated small kernels, redundant prompt computation, and device-to-host synchronization. Per-frame AdaLN contributes another 31.4 ms. HIP Graph provides a smaller isolated gain after the earlier stages have already removed much of the host overhead, but it creates a low-overhead execution path for the AMD-specific kernels. AITER varlen/split-KV attention and tuned hipBLASLt GEMMs then reduce generation latency from 108.7 ms to 71.5 ms.

We validated each stage against the preceding configuration. Execution-only changes remained bitwise consistent, while stages that altered bf16 arithmetic remained numerically equivalent to the original baseline.

*Table 2. End-to-end effect of the cumulative generation optimizations with the original Wan2.2 VAE. Decode latency remains 124.8 ms per chunk.*

| Stage | Total latency/chunk | Decoded FPS | Cumulative FPS gain |
| --- | ---: | ---: | ---: |
| Original implementation | 437.2 ms | 27.4 | — |
| + Operator fusion and prompt caching | 345.2 ms | 34.8 | +27.0% |
| + Host synchronization removal and Triton RoPE | 276.7 ms | 43.4 | +58.4% |
| + Per-frame AdaLN | 245.3 ms | 48.9 | +78.5% |
| + HIP Graph execution | 233.5 ms | 51.4 | +87.6% |
| + AMD-specific attention and GEMM kernels | 196.3 ms | 61.1 | +123.0% |

#### Lightweight VAE Decoder

After optimizing generation, VAE decoding becomes the largest remaining latency block. We optionally replace the original Wan2.2 decoder with a lightweight decoder introduced in AMD’s prior Hummingbird-XT work. The lightweight decoder preserves the same 48-channel latent space and temporal and spatial compression ratios. This replacement requires no retraining of the diffusion model.

*Table 3. VAE decoder results after enabling all generation-side optimizations.*

| Decoder configuration | Generation latency/chunk | Decode latency/chunk | Total latency/chunk | Decoded FPS | FPS gain vs. original VAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Wan2.2 VAE | 71.5 ms | 124.8 ms | 196.3 ms | 61.1 | Baseline |
| Lightweight VAE | 71.5 ms | 31.6 ms | 103.1 ms | 116.4 | +90.5% |

The lightweight decoder provides a 3.95× decoder speedup and raises decoded-video throughput from 61.1 to 116.4 FPS. Compared with the original Wan2.2 VAE output, it incurs a modest reduction in visual quality and pose accuracy in exchange for substantially faster decoding. We retain the original decoder for the main quality results.

```{figure} ./images/image4.png
:align: center
:alt: Per-chunk latency breakdown and decoded FPS
Figure 4. Per-chunk latency breakdown and decoded FPS across the cumulative optimization stages.
```

#### Optimization Summary

The cumulative optimization path reduces generation latency from 312.4 ms to 71.5 ms per chunk, a 4.37× speedup, while decoded throughput with the original VAE increases from 27.4 to 61.1 FPS. Operator fusion, prompt caching, synchronization removal, and Triton RoPE account for the largest share of the gain; per-frame AdaLN, graph execution, and AMD-specific AITER and hipBLASLt kernels provide the remaining reduction. The optional lightweight VAE increases throughput to 116.4 FPS with a quality trade-off.

## Experimental Results

We evaluate Micro-World 2 on Mind Benchmark, WBench, and GameWorld Score. Table 4 compares our 5B model with seven recent video action models and interactive world models, including LingBot-World-V2-1.3B and Sana-WM, with model sizes ranging from 1.3B to 5B parameters. For this comparison, we use the Micro-World 2 checkpoint distilled with 161-frame sequences and the original Wan2.2 VAE decoder. This configuration achieves 61.1 decoded FPS on a single AMD Instinct™ MI350X GPU (Table 3); the optional lightweight decoder raises throughput to 116.4 FPS, with its quality trade-offs evaluated in Table 5.

*Table 4. Quantitative comparison of Micro-World 2 with recent video action models and interactive world models.*

```{figure} ./images/table4.png
:align: center
```

*Boldface indicates the best value in each row and underlining indicates the second-best value.*

Among the models in Table 4, Micro-World 2 ranks first on 8 of the 21 reported metrics, including the WBench average. On WBench, it achieves the highest average score of 76.68, ahead of Sana-WM at 76.00, and leads in Interaction (83.89), Consistency (82.85), and Physical (68.62). On Mind Benchmark, it achieves the lowest GSC (0.0145) and ASG (0.0427), demonstrating strengths in memory consistency and action-space generalization, while ranking second on LCM and rotational RPE. On GameWorld Score, it achieves the highest motion smoothness (0.9935) and scenario consistency (0.9647), alongside second-place results in aesthetic quality, keyboard accuracy, and object consistency.

Visual quality and mouse-control accuracy remain areas for improvement. LingBot-World-V2-1.3B achieves the highest WBench Quality score (81.22), while Sana-WM leads both visual-quality metrics on Mind Benchmark, with an aesthetic score of 0.7006 and an image-quality score of 0.5870. Micro-World 2’s Mind Benchmark image-quality score (0.4244) and GameWorld Score mouse accuracy (0.263) indicate remaining gaps despite its strong results in interaction, consistency, and action generalization.

Table 5 examines training data, DMD sequence length, and the VAE decoder on Mind Benchmark. It compares the DL3DV [15] and Capsaicin data variants, 81- and 161-frame DMD in the Mixture setting, and the original and lightweight decoders. In the Mixture setting, the action-control module is trained on Capsaicin trajectories, while public videos are used for DMD.

*Table 5. Ablation of the data source, DMD sequence length, and VAE decoder for Micro-World 2.*

| Dimension                | Micro-World (DL3DV) | Micro-World (Capsaicin) | Micro-World (Mixture DMD 81 frames) | Micro-World (Mixture DMD 161 frames) | Micro-World (Mixture DMD 161 frames + light VAE) |
| ------------------------ | ------------------- | ----------------------- | ----------------------------------- | ------------------------------------ | ------------------------------------------------ |
| LCM ↓                    | 0.0592              | **0.0443**              | 0.0495                              | 0.0487                               | 0.0456                                           |
| GSC ↓                    | 0.0283              | 0.0193                  | 0.0154                              | **0.0145**                           | 0.0158                                           |
| ASG ↓                    | 0.0563              | 0.0403                  | 0.0449                              | 0.0427                               | **0.0396**                                       |
| Aesthetic ↑              | 0.5332              | **0.5697**              | 0.5566                              | 0.5666                               | 0.5538                                           |
| Image Quality ↑          | 0.4183              | **0.4486**              | 0.4131                              | 0.4244                               | 0.4128                                           |
| Action RPE ↓ (Trans/Rot) | **0.0436** / 1.508  | 0.046 / 0.94            | 0.0469 / 0.73                       | 0.0465 / **0.6475**                  | 0.0532 / 0.8834                                  |

*Bold marks the best value in each row.*

Training on Capsaicin improves six of the seven reported scalar metrics over DL3DV, with translation RPE increasing from 0.0436 to 0.0460. Extending DMD from 81 to 161 frames improves all seven metrics. The lightweight decoder improves LCM and ASG but worsens GSC, visual quality, and both translation and rotation RPE. We retain the original decoder for the main results to preserve visual quality and pose accuracy relative to the lightweight alternative. The resulting 161-frame Mixture configuration achieves the best GSC and rotational RPE in this ablation.

## Open-Source Availability

We assess openness across four artifacts: weights, inference code, training code, and training data. Table 6 broadens the comparison to 14 model releases selected from WBench with scores comparable to or higher than ours, spanning different backbones and model sizes. Each provides at least one downloadable checkpoint and runnable local inference code.

*Table 6. Open-source availability of Micro-World 2 and 14 recent model releases, grouped by training-code availability (verified September 9, 2026).*

```{figure} ./images/table6.png
:align: center
```

*✅: released and usable as-is. ◐: released only in part. ❌: not found in official channels. In the training-data column, ◐ means the upstream public datasets a project built its data from, or a small sample of its own, can be downloaded, but not the filtered and annotated mixture the released weights were trained on.*

Training-code availability reflects coverage of each project’s training pipeline from its stated base checkpoint. Among the 14 comparison releases, AlayaWorld and Lyra 2.0 provide full training code, while four others release only parts of their training pipelines. Complete training-data releases remain the largest gap: public source datasets or small samples do not provide the actual filtered and annotated corpus. This matters especially for action control, where action labels cannot be reliably recovered from video alone.

Micro-World 2 releases its weights, streaming inference pipeline, five-stage training scheduler, and all 440 Capsaicin trajectories, including per-frame keyboard, mouse, and camera states. These resources support further training and evaluation on ROCm. The repository and Capsaicin dataset card are linked under Additional Resources.

## Visual Performance

To demonstrate the generation quality of Micro-World 2, we present several sample videos in Figure 5:
<table border="0" style="width: 100%; table-layout: fixed; text-align: center; margin-top: 20px;">
  <tr>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_001_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 001
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_002_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 002
        </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_004_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 003
        </div>
     </td>
  </tr>
  <tr>
     <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_007_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 004
        </div>
     </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_012_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 005
        </div>
      </td>
      <td style="vertical-align: top; width: 33%;">
          <video src="https://raw.githubusercontent.com/AdvancedHe/preview_blog/main/assets/game_style_015_generated_overlay.mp4" width="100%" controls autoplay loop></video>
          <div style="margin-top: 8px; font-size: 14px;">
        Case 006
        </div>
     </td>
  </tr>
</table>
<p align="center">
  Figure 5. Samples generated by Micro-World 2.
</p>

## Summary

In this blog, we introduced **Micro-World 2**, a real-time interactive world model for action-controllable causal video generation. Its end-to-end workflow combines frame-aligned action data collected with Capsaicin, a five-stage training scheduler that transforms Wan2.2 5B into a four-step action-controllable causal autoregressive model, and a ROCm-optimized streaming inference pipeline. Our experiments demonstrate that precisely annotated trajectories improve action generalization and control, while longer-sequence distillation strengthens interaction response and long-horizon scene consistency.

On a single AMD Instinct™ MI350X GPU, AMD GEAK-guided optimization reduces generation latency from 312.4 ms to 71.5 ms per chunk, a 4.37× speedup, and raises decoded-video throughput from 27.4 to 61.1 FPS with the original Wan2.2 VAE. The optional lightweight VAE increases throughput to 116.4 FPS with a trade-off in visual quality. By open-sourcing the dataset, training pipeline, and inference code, we aim to provide a practical foundation for further research and development of real-time interactive world models on AMD ROCm.

```{update} Nov 18, 2026
Updated performance results and figures for Micro-World 2 based on the latest optimizations.
```

```{update} Nov 18, 2026
Updated performance results and figures for Micro-World 2 based on the latest optimizations.
```

```{Note}
This blog was originally uploaded on October 21, 2026
```

## Additional Resources

**Hugging Face Model Card:**

- [amd/Micro-World-2](https://huggingface.co/amd/Micro-World-2)

**Hugging Face Dataset Card:**

- [amd/Micro-World-Capsaicin](https://huggingface.co/datasets/amd/Micro-World-Capsaicin)

**Code:**

- [AMD-AGI/Micro-World](https://github.com/AMD-AGI/Micro-World)
- [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK)

Related Work on Diffusion Models by the AMD Team:

- [Micro-World: First AMD Open-Source World Models for Interactive Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html)
- [Bridging the Last Mile: Deploying Hummingbird-XT for Efficient Video Generation on AMD Consumer-Grade Platforms](https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html)
- [AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [Nitro-E: A 304M Diffusion Transformer Model for High Quality Image Generation](https://rocm.blogs.amd.com/artificial-intelligence/nitro-e/README.html)

## Acknowledgments

We thank the AMD Capsaicin team for developing and open-sourcing Capsaicin, a real-time rendering research framework that provided an important foundation for Micro-World 2’s data collection pipeline. Building on this framework, we collected and curated 440 interactive navigation sequences with synchronized per-frame keyboard inputs, mouse inputs, and camera poses, providing precisely annotated data for action-control training and evaluation. We appreciate the team’s contributions to open-source graphics research, which enabled us to build on their work in exploring controllable, interactive world models. For more information, please visit the [official Capsaicin website](https://gpuopen.com/capsaicin/) and the [GitHub repository](https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin).

## References

1. Mao X, Li Z, Li C, et al. Yume 1.5: A text-controlled interactive world generation model[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2026: 7752-7761.
2. Team R, Gao Z, Wang Q, et al. Advancing open-source world models[J]. arXiv preprint arXiv:2601.20540, 2026.
3. Zhu H, Liu H, Zhao Y, et al. Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer[J]. arXiv preprint arXiv:2605.15178, 2026.
4. World Labs Team. Marble: A multimodal world model[EB/OL]. World Labs Blog, 2025-11-12. [https://www.worldlabs.ai/blog/marble-world-model](https://www.worldlabs.ai/blog/marble-world-model)
5. World Labs Team. Atlas: A world model for spatial intelligence[EB/OL]. World Labs Blog, 2026-09-01. [https://www.worldlabs.ai/blog/atlas](https://www.worldlabs.ai/blog/atlas)
6. Geng Y, Chan W, Zhou D, et al. Micro-World: First AMD open-source world models for interactive video generation[EB/OL]. AMD ROCm Blogs, 2026-02-05. [https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html](https://rocm.blogs.amd.com/artificial-intelligence/micro-world/README.html)
7. Advanced Micro Devices, Inc. AMD Capsaicin Framework: a Direct3D 12 framework for real-time graphics research[EB/OL]. AMD GPUOpen, 2025-11. [https://gpuopen.com/capsaicin/](https://gpuopen.com/capsaicin/)
8. AMD-AGI. GEAK: Generating Efficient AI-Centric Kernels[EB/OL]. GitHub. [https://github.com/AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK).
9. Wan T, Wang A, Ai B, et al. Wan: Open and advanced large-scale video generative models[J]. arXiv preprint arXiv:2503.20314, 2025.
10. Zhu H, Zhao M, He G, et al. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation[J]. arXiv preprint arXiv:2602.02214, 2026.
11. Lin B, et al. Open-Sora Plan: Open-source large video generation model[J]. arXiv preprint arXiv:2412.00131, 2024.
12. Hu E J, et al. LoRA: Low-rank adaptation of large language models[C]//International Conference on Learning Representations. 2022.
13. Peebles W, Xie S. Scalable diffusion models with transformers[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
14. Yin T, et al. One-step diffusion with distribution matching distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
15. Ling L, Sheng Y, Tu Z, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision[C]//2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024: 22160-22169.
16. Wu R, He X, Cheng M, et al. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory[J]. arXiv preprint arXiv:2602.02393, 2026.
17. He X, Peng C, Liu Z, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model[J]. arXiv preprint arXiv:2508.13009, 2025.
18. HunyuanWorld T. HY-World 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency[J]. arXiv preprint, 2025.
19. Wang Z, Liu Z, Li J, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory[J]. arXiv preprint arXiv:2604.08995, 2026.
20. Ye Y, Lu X, Jiang Y, et al. Mind: Benchmarking memory consistency and action control in world models[J]. arXiv preprint arXiv:2602.08025, 2026.
21. Ying K, Hu H, Ren S, et al. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation[J]. arXiv preprint arXiv:2605.25874, 2026.
22. Zhang Y, Peng C, Wang B, et al. Matrix-game: Interactive world foundation model[J]. arXiv preprint arXiv:2506.18701, 2025.
23. Team A W, Zhang K, Li C, et al. AlayaWorld: Interactive Long-Horizon World Modeling--Full Technical Report[J]. arXiv preprint arXiv:2607.18367, 2026.
24. Shen T, Bahmani S, He K, et al. Lyra 2.0: Explorable generative 3d worlds[J]. arXiv preprint arXiv:2604.13036, 2026.
25. Yin Y, Wang G, Zhan Y, et al. Alaya-EVOKE: From Linear-Scaling Supervision to Endless World[J]. arXiv preprint arXiv:2608.13546, 2026.
26. Wu B, Zou C, Li C, et al. Hunyuanvideo 1.5 technical report[J]. arXiv preprint arXiv:2511.18870, 2025.
27. Duan N, Huang H, Jin W, et al. Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds[J]. arXiv preprint arXiv:2608.23383, 2026.
28. Seedleap.ai team. Zing-0.5[EB/OL]. 2026. [https://github.com/seedleap/zing-world-model](https://github.com/seedleap/zing-world-model)
29. Gao Z, Wang Q, Zhu J, et al. Infinite Worlds with Versatile Interactions[J]. arXiv preprint arXiv:2607.07534, 2026.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
