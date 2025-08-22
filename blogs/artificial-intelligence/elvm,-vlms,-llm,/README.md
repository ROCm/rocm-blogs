---
blogpost: true
blog_title: "Introducing AMD EVLM: Efficient Vision-Language Models with Parameter-Space Visual Conditioning"
date: 22 Aug 2025
author: 'Zhenhua Liu, Xuanwu Yin, Dong Li, Emad Barsoum'
thumbnail: 'EVLM_thum.png'
tags: Multimodal, AI/ML
category: Applications & models
target_audience: AL DEVELOPERS AND ENTHUSIAST
key_value_propositions: Injecting visual features into model parameters boosts multimodal efficiency—no visual tokens, lower compute, same benchmark performance.
language: English
myst:
    html_meta:
        "author": "Zhenhua Liu, Xuanwu Yin, Dong Li, Emad Barsoum"
        "description lang=en": "A novel approach that replaces visual tokens with perception-conditioned weights, reducing compute while maintaining strong vision-language performance."
        "keywords": "VLM, EVLM, LLM, LoRA"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Zhenhua Liu, Xuanwu Yin, Dong Li, Emad Barsoum"
---

# Introducing AMD EVLM: Efficient Vision-Language Models with Parameter-Space Visual Conditioning

This blog introduces a novel and computationally efficient paradigm for Vision-Language Models (VLMs), which diverges from the conventional method of prepending visual tokens to textual input. Instead of elongating the input sequence, this approach injects visual information directly into the Large Language Model's (LLM) parameters. It achieves this by using a vision encoder to extract image features and then employing a perceptual weight generator to transform these features into dynamic, low-rank adapter weights. These weights are temporarily integrated with the LLM's parameters, effectively conditioning the model on the image without increasing the input length. This mechanism allows the model to achieve performance comparable to traditional VLMs on standard benchmarks while significantly reducing computational costs during inference.

## Rethinking the Vision-Language Alignment in VLMs

Large Language Models (LLMs) have demonstrated remarkable success across a spectrum of natural language tasks, exhibiting strong generalization capabilities. As a natural progression, Vision-Language Models (VLMs) aim to extend these capabilities towards multi-modal artificial intelligence by incorporating visual perception. The predominant paradigm for existing VLMs involves aligning visual information within the input space of the LLM. This is typically achieved by encoding an image into a sequence of visual tokens and prepending them to the textual input to form a unified sequence. While this approach has yielded compelling results on various vision-language benchmarks, it is inherently limited by the substantial computational overhead incurred from processing elongated input sequences.

In this work, we diverge from this convention and introduce a novel parameter space alignment paradigm. Instead of representing visual information as input tokens, we propose to inject it directly into the model's parameters as dynamic, perception-conditioned weights. Specifically, for a given input image, a vision encoder first extracts high-level visual features. Then, a dedicated perceptual weight generator transforms these features into a set of low-rank adapter weights, structurally analogous to those used in Low-Rank Adaptation (LoRA). These generated weights are subsequently integrated with the LLM's original parameters.

This mechanism effectively conditions the model on the visual input without altering the input sequence length. Consequently, our method obviates the need for visual tokens in the input stream, leading to a significant enhancement in computational efficiency. Comprehensive experiments demonstrate that our proposed method achieves performance comparable to conventional input-space alignment techniques across multiple standard benchmarks, while substantially reducing computational costs during inference.

## Challenges of Input-Space Visual Alignment

As shown in Figure 1, previous VLMs align the visual features with the input space of LLM and require additional visual tokens as input for the LLM. This approach can lead to computational inefficiency, which becomes more severe when processing high-resolution or multiple images due to the drastically increasing number of tokens.

```{figure} ./images/Picture1.png
:align: center
:alt: Scaling performance
Figure 1. The training paradigm of traditional VLMs
```

To address this issue, we propose a novel approach that directly aligns visual features with the parameter space of the Large Language Model (LLM), thereby avoiding the introduction of extra visual tokens into the input sequence. Specifically, our method encodes the visual information of an input image into a set of "perceptual weights," which are then integrated directly into the LLM's original parameters. This parameter-space integration enables the model to perceive visual content without altering its input structure.

Let **W** denote the weight matrix of LLMs and **I** represent the input image. The vision encoder ξ(⋅) is introduced to extract the visual features **F**=ξ(**I**). Subsequently, we introduce a perceptual weights generator, denoted as φ(⋅), which maps visual features to a set of perceptual weights **ΔW** . Critically, to enable the LLM to perceive visual information without compromising its inherent language capabilities, we constrain **ΔW** to be a low-rank matrix. This low-rank constraint serves a dual purpose: it helps preserve the model's original linguistic functions while also significantly reducing the computational overhead of the weight generation process.

The generated perceptual weights, **ΔW**, are directly integrated into the parameters of the Large Language Model (LLM). As illustrated in Figure 2, this direct parameter fusion endows the model with visual perception capabilities. A key advantage of this approach is that it incurs no additional computational overhead during the LLM's inference phase, as the weight merging is completed beforehand. Furthermore, this integration mechanism is general and can be applied to any target weight matrix within each decoder block of the LLM.

```{figure} ./images/Figure2.png
:align: center
:alt: Scaling performance
Figure 2. The training paradigm of proposed EVLM without visual tokens
```

We train the model via a curriculum learning principle, where training objectives and examples of increasing difficulty are observed in a stage-wise manner. With a fixed computation budget, this strategy helps decompose the training process and produces immediate checkpoints that can be reused in more experimental trials.

Stage-1: Language-Image Alignment. In this stage, only the Multi-Layer Perceptron (MLP) module is pre-trained and the goal is to well-align the visual features into the word embedding space of LLMs.

Stage-2: High-Quality Knowledge Learning. In this stage, the whole model is trained under a next-token prediction.  To strike a balance between computation-efficiency and injecting new knowledge into VLMs, we utilize high-quality knowledge for VLMs' learning. The training configuration mirrors the settings used in stage-1, ensuring consistency and allowing the model to integrate new information seamlessly.

Stage-3: Visual Instruction Tuning. To teach VLM to solve a diverse set of visual tasks with preferred responses, we organize the instruction data into different groups. The model is scheduled to train on these groups in order.

Following the llava-onevision, we utilize high-quality knowledge for the pre-training of stage-1 and stage-2, as well as visual instruction tuning data for stage-3.

## Performance and Efficiency Evaluation

| **Model Name**     | **Vision Encoder** | **Text Encoder**        | **Speed on Mi-250** | **GQA** | **SQA** | **VQA** | **PQE** | **MM-Bench** | **MM MU** | **Real World QA** | **MMStar** |
|--------------------|--------------------|------------------------|---------------------|---------|---------|---------|---------|--------------|-----------|-------------------|------------|
| MobileVL M-3B      | CLIP               | MobileLLaMA-2.7B       | 1.1s/token          | 59.00   | 61.00   | 47.50   | 84.90   | 59.60        | --        | --                | --         |
| DeepSeek-VL-1.3B   | SigLIP             | DeepSeek-LLM-1B        | 1.9 s/token         | --      | 64.52   | 56.40   | 85.80   | 64.34        | 28.67     | 50.20             | 38.30      |
| TinyLLaVA-3B       | SigLIP             | Phi2-2.7B              | 1.0 s/token         | 61.00   | 70.10   | 53.50   | 86.30   | 68.30        | --        | --                | --         |
| MiniCPM-V-2        | SigLIP             | MiniCPM-2.4B           | 0.8 s/token         | --      | 76.10   | 57.60   | 86.56   | 70.44        | 38.55     | 55.03             | 40.93      |
| Ours               | SigLIP             | Qwen2.5-3B             | 0.3 s/token         | 63.89   | 76.20   | 56.90   | 86.90   | 72.40        | 39.10     | 56.40             | 43.69      |

Table 1. The comparison of our method with other large Vision-Language Models.

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate model performance. As shown in Table 1, compared to the open-weight VLMs such as MobileVLM, DeepSeekVL, TinyLLaVA, MiniCPM, the proposed method achieves comparable performance with faster generation speed, i.e., lower computational costs.

## Summary

We presented a new method for aligning visual features with the parameter space of LLMs rather than the input space. This allows us to completely remove visual tokens from the input and instead use the LLM to condition on images via visual feature-derived dynamic low-rank perceptual weights. We show that our method, which requires no additional computation during inference, matches the performance of input-based models on a variety of benchmarks with a much lower computational cost. This opens a new door to efficiently scaling vision-language methods on AMD hardware.

## Endnotes

You are invited to download and test this model on AMD hardware. For more information on the training, inference, and insights of this model, visit the [AMD GitHub repository](https://github.com/AMD-AIG-AIMA/AMD-EfficientVLM), where you can access the code, as well as the [Hugging Face Model Card](https://huggingface.co/amd/EVLM-3B) to download the model file. Additionally, AMD has established a dedicated cloud infrastructure featuring AMD Instinct™ GPU instances for AI developers. Visit [AMD Developer Cloud](https://www.amd.com/en/forms/registration/developer-cloud-application.html) for specific access requests and usage guidelines. If you have any questions, feel free to reach out to the AMD team at [amd_ai_mkt@amd.com](mailto:amd_ai_mkt@amd.com).

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
