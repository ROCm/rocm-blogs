---
blogpost: true
blog_title: "AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation"
date: 3 Aug 2025
author: 'Takashi Isobe, Dong zhou, He Cui,Mengmeng Ge,Dong Li,Emad Barsoum'
thumbnail: 'image-to-video-generation.jpg'
tags: AI/ML
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: The key value proposition is delivering high-performance, cost-efficient AI infrastructure optimized for developers, enabling faster innovation and seamless deployment across diverse workloads.
language: English
myst:
    html_meta:
        "author": "Takashi Isobe, Dong zhou, He Cui,Mengmeng Ge , Dong Li, Emad Barsoum"
        "description lang=en": "We present AMD Hummingbird, offering a two-stage distillation framework for efficient, high-quality text-to-video generation using compact models."
        "keywords": "AI Infrastructure, Training & Inference, Image to Video"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Takashi Isobe, Dong zhou, He Cui,Mengmeng Ge , Dong Li, Emad Barsoum"
---
# AMD Hummingbird Image to Video: A Lightweight Feedback-Driven Model for Efficient Image-to-Video Generation

In this blog, we present AMD Hummingbird-I2V, a lightweight and feedback-driven image-to-video generation model designed to deliver high-quality results efficiently on resource-constrained hardware. Image-to-video (I2V) generation has become a significant challenge in computer vision, driven by the increasing demand for automated content creation in areas such as digital media production, animation, and advertising. While recent advancements have improved video quality, deploying I2V models in practical scenarios remains challenging due to their large model sizes and high inference costs. For example, DynamiCrafter [1] employs a 1.4B-parameter U-Net and typically requires 50 denoising steps to synthesize a single video. Step-Video [2], a DiT-based model with 30B parameters, takes approximately 30 minutes to generate one video on an AMD Instinct ™ MI250 GPU, making it impractical for latency-sensitive or resource-constrained environments, such as gaming-oriented desktop GPUs. In this work, we present AMD Hummingbird-I2V, a compact and efficient diffusion-based I2V model designed for high-quality video synthesis under limited computational budgets. Hummingbird-I2V adopts a lightweight U-Net architecture with 0.9B parameters and a novel two-stage training strategy guided by reward-based feedback, resulting in substantial improvements in inference speed, model efficiency, and visual quality. To further improve output resolution with minimal overhead, we introduce a super-resolution module at the end of the pipeline. Additionally, we leverage ReNeg [3], an AMD proposed reward-guided framework for learning negative embeddings via gradient descent, to further boost visual quality. As a result, Hummingbird-I2V can generate high-quality 4K video in just 11 seconds with 16 inference steps on an AMD Radeon™ RX 7900 XTX GPU.  Quantitative results on the VBench-I2V [4] benchmark show that Hummingbird-I2V achieves state-of-the-art performance among U-Net-based diffusion models and competitive results compared to significantly larger DiT-based models. We provide a detailed analysis of the model architecture, training methodology, and benchmark performance.

## Key Takeaways

- An efficient and lightweight diffusion-based I2V model optimized for high-quality video synthesis on resource-constrained devices.
- A two-stage training strategy combining lightweight U-Net distillation with performance enhancement through reward-based feedback.
- A reward-guided framework that learns negative embeddings via gradient descent to further improve visual fidelity.
- Capable of generating 4K-resolution videos in just 11 seconds on an AMD Radeon RX 7900 XTX GPU.
- Figure 1 below showcases end-to-end vertical integration, combining powerful cloud-based training with efficient edge deployment.

```{figure} ./images/Picture1.png
:align: center
:alt: Scaling performance
Figure 1. Illustration of the key contributions of this work.
```

## Network Design

Most prior works on I2V generation have primarily focused on enhancing visual quality and motion dynamics. These methods typically follow a three-stage design pipeline: (1) constructing a foundation model by extending the U-Net architecture from Stable Diffusion 1.5 with 3D temporal layers to support video latents; (2) introducing advanced attention modules to integrate image, prompt, and motion conditions; and (3) training the model on curated datasets that include quality and motion assessments, prompt recaptions, and reward-based scoring mechanisms. These efforts have led to substantial progress in both perceptual fidelity and temporal coherence.

However, the large model sizes and high inference costs of these systems limit their deployment on resource-constrained devices, especially AI PCs. To address this issue, we conducted a systematic ablation study to analyze the contribution of each component within the U-Net architecture. Specifically, we ablate blocks from the downsampling, middle, and upsampling stages, and we evaluated the resulting models using FID scores. This analysis yielded three key insights:

1. In I2V models, removing the middle blocks leads to a significant drop in both motion consistency and visual quality. In contrast, T2V models can still preserve some degree of semantic structure, texture, and motion even without them.
2. Removing a single block from either the downsampling or upsampling stage causes minor performance drops.
3. Eliminating all blocks from an entire downsampling or upsampling stage, while preserving the other, results in complete failure of video generation.

Based on these empirical observations, we propose a compact architecture that reduces the number of blocks in the downsampling and upsampling stages by half, while preserving the middle blocks. We further analyze the noise distribution across layers and find that our architecture aligns more closely with the original model at corresponding layers compared to other variants.  This indicates that our design preserves the model’s ability to capture both motion dynamics and fine-grained texture. As a result, the optimized model reduces the parameter count from 1.4B to 0.9B and achieves a 1.3x speedup in inference per step.

## Two-Stage Training with Reward-based Feedback

Most previous work relies on large-scale, high-quality datasets to train their models, where both data preparation and model training demand significant human effort and computational resources. In this work, we propose a two-stage training strategy that efficiently aligns a compact model’s distribution with the original high-capacity model, using only limited data and training time. The overall training pipeline is illustrated as shown in Figure 2.

```{figure} ./images/Picture2.png
:align: center
:alt: Scaling performance
Figure 2. Illustration of the proposed two-stage distillation pipeline for I2V diffusion models. The first stage aims to quickly align the feature-level distribution between the pruned compact model and the original model after each corresponding layer. The second stage distills the inference steps, further enhancing visual quality and prompt alignment through feedback learning.
```

### Stage I: Aligning Feature Distributions Between Compact and Original Models

In this stage, we first initialize the compact model by mapping the corresponding parameters from the original U-Net. To align the behavior of the compact model with the original, we minimize a distillation loss that enforces consistency across two aspects: (1) the intermediate feature representations at each corresponding layer, and (2) the final v-prediction outputs. This helps the compact model retain the motion modeling and texture generation capabilities of the original despite its reduced capacity. In Stage I, we distill the knowledge from the original, large U-Net into a smaller, more efficient compact U-Net by minimizing a carefully designed distillation loss function. $F_s^{(l)}$ and $F_t^{(l)}$ denote the features at layer $l$ extracted from the original and pruned and models, respectively. Likewise, $v_s$ and $v_t$ correspond to their final v-prediction outputs. ${L}_{dist}$ represents the distillation loss, where $\lambda_1$ and $\lambda_2$ are the corresponding weighting coefficients.

$$
\mathcal{L}_{dist} = \lambda_1 \sum_l \left| F_s^{(l)} - F_t^{(l)} \right|^2 + \lambda_2 \left| v_s - v_t \right|^2
$$

To accelerate training, we use a curated set of 400,000 high-quality video clips that cover a wide range of motion patterns. Empirically, we observe that high-quality samples significantly enhance convergence speed and training stability.

For image conditioning, we randomly select a frame from each video to generate the image embedding used in the cross-attention modules. This prevents the model from overfitting to a specific frame position and encourages it to learn more flexible, context-aware representations. We also find that this strategy helps the model generate videos with more dynamic motion.

Additionally, we jointly train the image projector with the compact U-Net to ensure effective integration of visual conditioning and maintain strong alignment with the original model’s representation space.

### Stage II: Inference Step Distillation and Reward-Based Feedback Learning

Although the compact model trained in Stage I can produce results comparable to the original model, three key issues remain: (1) the generated videos lack high-frequency details; (2) when given complex prompts, the model struggles to capture the intended semantics, leading to misalignment between the prompt and the generated content; and (3) the model still requires a relatively large number of inference steps to produce visually pleasing results.

To address these challenges, we introduce a second-stage distillation process that reduces inference steps while improving visual quality and prompt alignment using reward-based feedback. Specifically, we follow the Latent Consistency Model (LCM) framework, which minimizes the difference between adjacent points along a probability flow Ordinary Differential Equation (ODE) trajectory using a pre-trained diffusion model. LCM leverages fast ODE-based samplers (e.g., DDIM, DPM-Solver) to construct training pairs $(z_n, z_{n+k})$ by k steps, enabling efficient distillation of the generative trajectory.

However, the performance of the distilled model is inherently constrained by the quality of the teacher. To overcome this limitation and better align generations with human preferences, we further integrate reward-based feedback into the distillation process, following the approach of T2V-Turbo-v2 [6]. Our method incorporates a mixture of reward models, including both image-text and video-text reward models, to jointly enhance prompt alignment, visual quality, and temporal consistency. The stepwise distillation strategy also enables us to reduce the number of inference steps from 50 to 16.

### On Feasibility of Learning Negative Embeddings

Recent advances in diffusion models have significantly improved the quality of image and video generation, with Classifier-Free Guidance (CFG) playing a central role in enhancing text controllability. While CFG typically uses a null-text prompt during inference, replacing it with manually crafted negative prompts has shown superior performance in both text-to-image and text-to-video tasks. However, existing methods for generating negative prompts rely on human heuristics and trial-and-error search, resulting in incomplete coverage of the negative concept space and suboptimal performance. In this work, we use the AMD proposed **ReNeg**, a reward-guided framework for learning negative embeddings through gradient descent. Unlike manual approaches, ReNeg optimizes in the continuous embedding space of a text encoder, allowing for a more expressive and scalable search.

## Super Resolution Model Training

Generating high-resolution videos directly remains a significant challenge due to the substantial computational demands during both training and inference, as well as the requirement for high-quality training data. Moreover, preserving complex and dynamic motion in high-resolution video generation adds to the difficulty.

To overcome these limitations, we avoid training a 4K generative model in an end-to-end manner on resource-constrained devices. Instead, we append a lightweight super-resolution model, trained on generated data, to the end of the pipeline. This approach is flexible and provides a more practical solution for high-resolution video synthesis. As shown in Figure 3, we build our model upon Real-ESRGAN [7,8] with two key modifications: (1) applying pixel-unshuffle [10] to enable feature extraction at a reduced spatial resolution, which improves computational efficiency. (2) reducing the overall model size from 16.7M to 1.84M.

```{figure} ./images/Picture3.png
:align: center
:alt: Scaling performance
Figure 3. Illustration of the SR framework.
```

### Experimental Setting

#### Data Processing

We use the processed **WebVid** [12] and **VidGen-1M** [13] datasets for training. **WebVid** is a publicly available, large-scale dataset consisting of 10 million video-text pairs collected from stock footage platforms. While it offers broad visual diversity, it suffers from noisy captions, inconsistent video quality, and limited temporal coherence. To complement this, we incorporate VidGen-1M, a dataset specifically curated for I2V generation. Curated through a coarse-to-fine strategy, **VidGen-1M** ensures high-quality videos paired with detailed captions and excellent temporal consistency.

Both datasets are further refined by our data processing pipeline, which improves overall data quality through **video quality assessment, motion estimation,** and **prompt re-captioning**. Specifically, we first apply a Video Quality Assessment (VQA) model to score videos based on aesthetic and compression-related metrics. A motion estimation module is then used to remove unstable motion patterns, such as dolly zooms, from the high-quality set. Finally, to enhance textual alignment, we leverage the prior knowledge of a large language model, **LLaMA-8B** [14], to perform prompt re-captioning. Experimental results show that training on videos with both high visual and textual quality significantly enhances the model’s ability to generate detailed textures and accurately follow input prompts.

As for the super-resolution model training, we utilize JourneyDB [5], a large-scale dataset comprising 4,429,295 high-resolution images generated by Midjourney, each annotated with a corresponding text prompt, image caption, and visual question-answering data. We use the high-resolution images as ground truth and generate low-resolution counterparts by applying a set of multi-path degradation operations.

### Training Details

The first stage is trained for 200K steps with a learning rate of 1×10⁻⁴ and a batch size of 16. The second stage is trained for 50K steps with a learning rate of 1×10⁻⁵ and a batch size of 3. During training, we randomly sample 16 frames per video at a resolution of 320×512 using a dynamic frame stride. The frame rate condition is adapted based on the codec of the original video in the first stage and fixed at 24 fps in the second stage.

All training is conducted on AMD Instinct MI250 GPUs, built on the AMD CDNA™ 2 architecture. Each GPU provides 362.1 TFLOPs of peak FP16 performance, 64 GB of HBM2e memory, 3.2 TB/s memory bandwidth, and a 500W TDP. The software stack includes Python 3.8, ROCm 5.6.0, PyTorch 2.2.0, and FlashAttention 2.2.0.

In the second stage, we incorporate both image and video reward signals to guide training. The image reward weight is set to 0.05, and the video reward weight to 0.5. Image reward is computed with a batch size of 1, while video reward is evaluated using 4 frames per sample. The full set of parameters in both the U-Net and the image projector are optimized during this stage.

We also maintain a privately collected dataset that can further enhance model performance. However, due to licensing and privacy restrictions, this dataset is not publicly released. Therefore, all training for the public model is conducted exclusively on the publicly available WebVid-10M and VidGen datasets.

### Quantitative Comparison

```{figure} ./images/table1.png
:align: center
:alt: Scaling performance
Table 1: VBench-I2V Benchmark Results. Higher scores indicate better performance in each dimension.   The “Total Score” reflects the overall performance across all metrics. CogVideoSFT was implemented by the GaoDeI2V team. The results of HunYuan and Wan-2.1-14B are taken from Sparse VideoGen2 team[9].
```

To evaluate the performance of Hummingbird-I2V, we compare it against ten state-of-the-art models, including both DiT-based and U-Net-based architectures. The evaluation is conducted using text prompts and reference images from the VBench-I2V benchmark, which covers a diverse range of scenes such as natural landscapes, science fiction environments, and urban settings. The first five models in the comparison are DiT-based, while the latter five are U-Net-based.

As shown in Table 1, Hummingbird-I2V achieves the highest total score among all U-Net-based models and delivers competitive performance relative to DiT-based models, despite its smaller model size and fewer inference steps. These results demonstrate the model's effectiveness in generating high-quality video content while maintaining strong computational efficiency.

### Effectiveness of Super-Resolution Model

We evaluate the proposed SR model on video generation tasks. The performance of our super-resolution model was quantitatively assessed using the VBench [4] and DOVER VQA [11] benchmarks. We use the Hummingbird-T2V as the baseline. As shown in Table 2, the results indicate that our method leads to a significant enhancement in the quality score and semantic score of the upscaled videos compared to their low-resolution counterparts.

|               | **VBench**               |                   |                | **DOVER**               |                  |             |
|---------------|--------------------------|-------------------|----------------|-------------------------|------------------|-------------|
|               | Quality Score            | Semantic Score    | Final Score    | Aesthetic Score         | Technical Score  | Final Score |
| **Original Videos** | 81.32                   | 74.51             | 79.96          | 99.86                   | 6.34             | 75.49       |
| **SR Videos**       | **82.38**               | **75.43**         | **80.99**      | **99.98**               | **11.42**        | **89.2**    |

<p align="center">
    Table 2. VBench-T2V Benchmark and DOVER Benchmark. The best result in each column is highlighted in bold. The “Final Score” reflects the overall performance across all metrics.
</p>
Meanwhile, we also evaluated the performance of several text-to-video generation models when enhanced by our super-resolution model. As shown in Table 3, our model consistently improves the results of these generation models on both the VBench-T2V Benchmark and the DOVER Benchmark, demonstrating its effectiveness in enhancing video quality in downstream tasks.

| **Model**          |         | **VBench**             |                   |                | **DOVER**             |                  |             |
|--------------------|---------|------------------------|-------------------|----------------|-----------------------|------------------|-------------|
|                    |         | Quality Score          | Semantic Score    | Final Score    | Aesthetic Score       | Technical Score  | Final Score |
| **tft2v**          | Original Videos | 80.05                 | 56.71             | 75.38          | 98.21                 | 6.51             | 54.11       |
|                    | SR Videos      | **81.47**             | **58.76**         | **76.93**      | **99.81**             | **9.47**         | **79.06**   |
| **lavie**          | Original Videos | 78.72                 | 70.33             | 77.04          | 96.28                 | 7.22             | 57.67       |
|                    | SR Videos      | **80.57**             | **70.75**         | **78.61**      | **78.92**             | **7.92**         | **71.63**   |
| **animatel cm**    | Original Videos | 82.33                 | 67.64             | 79.39          | 99.82                 | 9.47             | 79.53       |
|                    | SR Videos      | **82.63**             | **68.24**         | **79.75**      | **99.94**             | **9.58**         | **84.99**   |
| **videocrafter2**  | Original Videos | 82.19                 | 73.42             | 80.44          | 99.38                 | 7.47             | 69.12       |
|                    | SR Videos      | **83.3**              | **75.37**         | **81.71**      | **99.83**             | **9.22**         | **82.29**   |

<p align="center">
Table 3. Quantitative Evaluation of Super-Resolution Enhancement on Text-to-Video Models.
</p>

### Effectiveness of ReNeg in Enhancing Visual Quality

```{figure} ./images/Picture5.png
:align: center
:alt: Scaling performance
Figure 4. Comparison of video frames generated by the proposed Hummingbird-I2V model. Left: without ReNeg. Right: with ReNeg.
```

## Summary

By open-sourcing the training code, dataset, and model weights for the AMD Hummingbird-I2V diffusion model, we aim to empower the AI developer community to build efficient and high-performing generative video models. Developers are encouraged to download and explore the model on AMD hardware. For details on the training methodology, inference pipeline, and model insights, please visit the [AMD GitHub repository](https://github.com/AMD-AIG-AIMA/AMD-Hummingbird-T2V) to access the source code and the Hugging Face model card to download the model weights. As an added benefit, AMD offers access to a dedicated cloud infrastructure featuring the latest GPU instances for AI development. To request access and learn more, please visit the [AMD developer cloud](https://devcloud.amd.com/). For further inquiries, contact the AMD team at [amd_ai_mkt@amd.com](mailto:amd_ai_mkt@amd.com).

## Resources

Huggingface model cards: [AMD-Hummingbird-I2V](https://huggingface.co/amd/AMD-Hummingbird-I2V)

Full training code: [AMD-AIG-AIMA/Hummingbird](https://github.com/AMD-AIG-AIMA/AMD-Hummingbird-T2V)

Related work on diffusion models by the AMD team:

- [AMD Hummingbird-0.9B: An Efficient Text-to-Video Diffusion Model with 4-Step Inferencing](https://rocm.blogs.amd.com/artificial-intelligence/image-to-video/README.html)
- [ReNeg: Learning Negative Embedding with Reward Guidance](https://arxiv.org/abs/2412.19637)
- [AMD Nitro Diffusion: One-Step Text-to-Image Generation Models](https://www.amd.com/en/developer/resources/technical-articles/amd-nitro-diffusion-one-step-text-to-image-generation-models.html)

Please refer to the following resources to get started with training on AMD ROCm™ software:

- Use the [public PyTorch ROCm Docker images](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html) that enable optimized training performance out-of-the-box
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)

### References

1. Xing J, Xia M, Zhang Y, et al. Dynamicrafter: Animating open-domain images with video diffusion priors[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 399-417.
2. Ma G, Huang H, Yan K, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model[J]. arXiv preprint arXiv:2502.10248, 2025.
3. Li X, Liu Y, Isobe T et al. "Reneg: Learning negative embedding with reward guidance." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 23636-23645. 2025.
4. Huang Z, He Y, Yu J, et al. Vbench: Comprehensive benchmark suite for video generative models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 21807-21818.
5. Sun K, Pan J, Ge Y, et al. Journeydb: A benchmark for generative image understanding[J]. Advances in neural information processing systems, 2023, 36: 49659-49678.
6. Li, Jiachen, et al. "T2v-turbo-v2: Enhancing video generation model post-training through data, reward, and conditional guidance design." arXiv preprint arXiv:2410.05677 (2024).
7. Wang X, Xie L, Dong C, et al. Real-esrgan: Training real-world blind super-resolution with pure synthetic data[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 1905-1914.
8. Wang X, Yu K, Wu S, et al. Esrgan: Enhanced super-resolution generative adversarial networks[C]//Proceedings of the European conference on computer vision (ECCV) workshops. 2018: 0-0.
9. Yang, Shuo, et al. "Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation." arXiv preprint arXiv:2505.18875 (2025).
10. Shi W, Caballero J, Huszár F, et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 1874-1883.
11. Wu H, Zhang E, Liao L, et al. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 20144-20154.
12. Bain M, Nagrani A, Varol G, et al. Frozen in time: A joint video and image encoder for end-to-end retrieval[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 1728-1738.
13. Tan Z, Yang X, Qin L, et al. Vidgen-1m: A large-scale dataset for text-to-video generation[J]. arXiv preprint arXiv:2408.02629, 2024.
14. Grattafiori A, Dubey A, Jauhri A, et al. The llama 3 herd of models[J]. arXiv preprint arXiv:2407.21783, 2024.

**SYSTEM CONFIGURATION:**

The system is configured with an AMD Instinct MI250 GPU, as tested by AMD on May 17, 2025, results may vary based on configuration, usage, software version, and optimizations.

AMD INSTINCT MI250 (MCM) OAM AC MBA
CPU: 1x AMD EPYC 73F3 16-Core Processor
GPU: 8x AMD Instinct MI250 GPU
OS: Ubuntu 22.04
System BIOS: 2.7a
System BIOS Vendor: American Megatrends International, LLC.
GPU Driver (amdgpu version): ROCm 6.1.1

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

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
