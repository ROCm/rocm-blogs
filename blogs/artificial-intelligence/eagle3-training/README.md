---
blogpost: true
blog_title: "Training EAGLE3 Speculative Decoding Draft Models on AMD Instinct MI350X with AMD Quark"
date: "27 Oct 2026"
author: "Larry Li, Haichen Zhang, Chao Li, Chun Fang, Andy Luo, Zihao Wang, Xikai Meng, Spandan Tiwari, Ashish Sirasao"
thumbnail: 'eagle3-training-thumbnail.jpg'
tags: "LLM, AI/ML, Serving, Performance, Benchmarking"
category: "Applications & models"
target_audience: "AI developers, AI practitioners, model training engineers, inference engineers"
key_value_propositions: "Train and evaluate EAGLE3.1 speculative decoding draft models on AMD Instinct MI350X GPUs with a ROCm-native workflow."
language: English
myst:
    html_meta:
        "author": "Larry Li, Haichen Zhang, Chao Li, Chun Fang, Andy Luo, Zihao Wang, Xikai Meng, Spandan Tiwari, Ashish Sirasao"
        "description lang=en": "Learn how the AMD Quark team trains EAGLE3.1 draft models from scratch on AMD Instinct MI350X GPUs using ROCm, vLLM, TorchSpec, and SPEED-Bench."
        "keywords": "LLM, Speculative Decoding, EAGLE3, EAGLE3.1, AMD Quark, vLLM, ROCm, TorchSpec, FSDP2, SPEED-Bench, MI350X, Model Training"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Training, AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
        "amd_blog_authors": "Larry Li, Haichen Zhang, Chao Li, Chun Fang, Andy Luo, Zihao Wang, Xikai Meng, Spandan Tiwari, Ashish Sirasao"
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

# Training EAGLE3 Speculative Decoding Draft Models on AMD Instinct MI350X with AMD Quark

In a [previous blog](https://rocm.blogs.amd.com/artificial-intelligence/eagle3-speculative-decoding/README.html), we looked at the serving side of EAGLE3 speculative decoding for large language models on AMD Instinct™ GPUs: ROCm and vLLM backend enablement, and [AMD Quark](https://quark.docs.amd.com/latest/) FP8 draft-model quantization. Serving is only half of the picture, though. Speculative decoding also depends on having a draft model worth serving. A serving stack can only accelerate a target model as far as the draft model allows.

This post covers the training half: how the [AMD Quark](https://quark.docs.amd.com/latest/) team trained an **EAGLE3 draft model from scratch** for [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4), a large mixture-of-experts (MoE) target, entirely on AMD Instinct MI350X GPUs with a ROCm-native stack of ROCm, vLLM, and [TorchSpec](https://github.com/lightseekorg/TorchSpec). The draft uses **EAGLE3.1**, a refinement of EAGLE3; we describe that distinction below. We then walk through the training pipeline, the ROCm enablement behind it, the training choices that mattered most, and draft quality measured by acceptance length on the official SPEED-Bench benchmark.

The broader point is simple: the [AMD Quark](https://quark.docs.amd.com/latest/) team can now train EAGLE3 drafts for newly released LLMs on AMD Instinct.

## Why Draft-Model Quality Matters

Standard autoregressive decoding produces one token per target model step. Speculative decoding breaks that bottleneck by letting a small draft model propose several next tokens, which the target model verifies in a single forward pass. Tokens accepted by the target are emitted together, and generation resumes from the first rejection. Since every emitted token is still verified by the target, speculative decoding is lossless: it preserves the target model’s output distribution.

Acceptance length (AL) is the metric that tells us whether the draft is doing useful work: it is the average number of tokens emitted per target model verification step. An AL of 1.0 means the draft adds no acceleration beyond the guaranteed token. A higher acceptance length means the target confirms more tokens per verification step, which is why the draft deserves careful training.

EAGLE has moved through several generations: feature-level drafting in EAGLE, better acceptance in EAGLE2, and, in EAGLE3, low-, mid-, and high-level target model features paired with a training-time-test objective. That design keeps the draft closely aligned with the verifier it must satisfy. An EAGLE3 draft is not a generic small language model attached to the side; it is trained against the target model’s internal representations, making its proposals much easier for the verifier to accept.

**EAGLE3 and EAGLE3.1.** The draft in this work is an **EAGLE3.1** draft. EAGLE3.1 is a small, backward-compatible refinement of EAGLE3 rather than a new method: it keeps the same feature-level drafting design and the same vLLM serving format (LlamaForCausalLMEagle3), while adding normalization inside the draft head to make training more stable. That matters here because the draft is trained from scratch. In this post, “EAGLE3” refers to the method and stack in general, while “EAGLE3.1” refers to the specific draft variant we trained.

## The EAGLE3.1 Training Pipeline

At a high level, EAGLE3.1 draft training follows a five-stage pipeline. During training, the target model supplies the hidden states used to supervise the draft. During serving, it verifies the draft’s proposed tokens.

```{figure} ./images/eagle3-training-pipeline.png
:align: center
:alt: EAGLE3.1 draft training pipeline on AMD Instinct MI350X
EAGLE3.1 draft training pipeline on AMD Instinct MI350X
```

1. **On-policy data.** The target model regenerates responses for a broad, multi-domain prompt pool. Training on the target’s own outputs, rather than off-the-shelf reference answers, aligns the draft with the distribution it will actually need to predict at serving time.

2. **Hidden-state extraction.** Live vLLM-ROCm engines run the target model and stream its multi-layer hidden states to the trainer, giving the draft the low/mid/high-level features EAGLE3 uses.

3. **Cold-start EAGLE3.1 training.** The draft head starts from random initialization and is trained with the training-time-test (TTT) objective, sharded across GPUs with FSDP2.

4. **Export.** The trained draft is converted into a vLLM-loadable LlamaForCausalLMEagle3 checkpoint that can be dropped into a speculative-decoding server.

5. **Serving and evaluation.** The exported draft is deployed with vLLM, and SPEED-Bench measures its acceptance length.

The full loop runs on a single 8-GPU AMD Instinct MI350X node. Some GPUs host the vLLM target engines that produce hidden states; the remaining GPUs run the FSDP2 draft trainer.

## User-Friendly YAML and CLI Workflows for EAGLE3 Training

AMD Quark 0.13 packages EAGLE3 training into reusable YAML recipes and a small CLI surface. Prepare the ROCm environment once, then choose the workflow that fits your target:

- The [Qwen3-8B Quick Start](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_quick_start.rst) is the fully GPU-validated, reproducible path.
- The [EAGLE-3 Large-Model Best Recipe](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_best_recipe.rst) uses `amd/MiniMax-M3-MXFP4` as an example target for adapting the workflow to large, quantized models.

### [Qwen3-8B Quick Start](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_quick_start.rst)

Qwen3-8B is the validated AMD Quark 0.13 workflow. The CLI generates on-policy data, trains across eight AMD Instinct GPUs, exports a vLLM-compatible draft, and benchmarks baseline and speculative serving. Use `--base_model` to select the target model.

**Prepare the environment once.** The setup command is idempotent: it checks Docker and ROCm GPU access, prepares the validated training and serving dependencies and ROCm image, downloads Qwen3-8B, and records an environment manifest for reproducibility.

```console
python3 -m quark.experimental.speculative_decoding.setup --base_model Qwen/Qwen3-8B
```

**Run the validated default recipe.** This is the shortest end-to-end form. The packaged recipes/eagle3_default.yaml supplies the data, training, export, and benchmark defaults, so the user only needs to name the target model.

```console
python3 -m quark.experimental.speculative_decoding.run --base_model Qwen/Qwen3-8B
```

**Use the explicit recipe form when needed.** Set the target with `--base_model`.

```console
python3 -m quark.experimental.speculative_decoding.run \
     --config recipes/eagle3_default.yaml \
     --base_model Qwen/Qwen3-8B \
     training.output_dir=ckpts/qwen3-8b-eagle3 training.num_epochs=2
```

### [EAGLE-3 Large-Model Best Recipe](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_best_recipe.rst): MiniMax-M3 Example

The Best Recipe is a portable baseline rather than a configuration tuned for a specific LLM. For a full run, create a manifest from `domain_manifest.full.example.yaml`, replace its placeholders with reviewed data sources and license information, adjust the domain caps and held-out policy, and use a fresh output directory. The selected target regenerates responses on-policy, while runtime and measured results depend on the manifest and serving configuration.

```console
python3 -m quark.experimental.speculative_decoding.setup \
    --base_model amd/MiniMax-M3-MXFP4

python3 -m quark.experimental.speculative_decoding.run \
    --config recipes/eagle3_mxfp4_moe.yaml \
    --base_model amd/MiniMax-M3-MXFP4 \
    execution.profile=full \
    data.domain_manifest=/absolute/path/to/domain_manifest.yaml \
    training.output_dir=ckpts/minimax-m3-eagle3-full
```

### The Packaged Default Recipe

Most users do not need to edit the YAML. Its core user-facing fields are shown below; the packaged file also retains lower-level draft-architecture and native-reference settings. The defaults select the validated eight-GPU full pipeline while keeping lower-level implementation details out of the user workflow.

```yaml
# recipes/eagle3_default.yaml
method: eagle3

execution:
  backend: torchspec

model:
  target_model_path: Qwen/Qwen3-8B

data:
  auto_generate: true
  prompt_dataset: allenai/tulu-3-sft-mixture
  num_prompts: 150000
  eval_size: 256
  generation_max_tokens: 4096

training:
  num_epochs: 2
  num_gpus_per_node: 8
  output_dir: ckpts/eagle3

benchmark:
  num_prompts: 40
  rounds: 3
```

With these defaults, a successful full run leaves the vLLM-loadable draft and report.json under training.output_dir. Reusable model and on-policy data assets are cached, making subsequent recipe runs faster while preserving the exact configuration and environment needed to reproduce the result.

## AMD ROCm Training-Stack Enablement

Running this pipeline natively on ROCm required several pieces to work together, so training, extraction, and serving could stay on one AMD stack:

- **vLLM-ROCm hidden-state extraction.** EAGLE3 training needs the target model’s intermediate hidden states, not just output tokens. We use a vLLM-ROCm build whose worker exposes a hidden-state extraction path, allowing a standard vLLM serve process to act as the feature source for training on AMD Instinct GPUs.

- **MXFP4 MoE target loading.** The target, [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4), is a large MXFP4-quantized MoE model. Once it can load and run correctly under vLLM on ROCm, using the AITER MoE fast path and the appropriate attention backend, the same quantized model used in production can also serve as the verifier and hidden-state source during training.

- **TorchSpec streaming trainer.** TorchSpec’s disaggregated EAGLE3 trainer pairs several vLLM extraction engines with FSDP2 training ranks and streams hidden states between them. That lets a very large MoE target supervise the draft on a single node without writing terabytes of features to disk.

Together, these pieces make the train-to-serve loop a single ROCm-native workflow: [AMD Quark](https://quark.docs.amd.com/latest/) for the quantized target, ROCm and vLLM for extraction and serving, and TorchSpec for the training core.

## Training Strategy

The following choices, more than any fixed recipe, had the biggest impact on draft quality:

- **Cold-start, from scratch.** The draft head starts from random initialization and is never warm-started from an existing draft, keeping the result cleanly and independently trained.

- **On-policy data is the most important factor.** The most important choice was training on data drawn from the target’s own output distribution. The target regenerates responses across a broad domain mix: general chat and instruction following, multilingual content, and a substantial slice of code and math. As a result, the draft learns to predict what the target actually produces.

- **Training-time-test (TTT) objective.** Instead of only predicting the immediate next token, the draft is unrolled for several steps during training. This teaches it to propose multiple tokens in a row, which matches how it is used at inference.

- **Chat-template fidelity.** Training data uses the target’s exact chat template. If the training template drifts from the serving template, acceptance length can drop quietly; keeping them identical protects draft quality.

- **EAGLE3.1 normalization.** The added normalization in the EAGLE3.1 draft head makes cold-start training more stable, especially for a large draft trained from scratch.

## Results

We evaluate the trained draft with the official SPEED-Bench harness (specdec_bench) with tensor parallelism of eight (TP=8) on vLLM ROCm v0.27.1 with native AITER MXFP4 MoE kernels, using num_speculative_tokens=3 and temperature 0. Qualitative results are the mean of two complete runs; each fixed-context Throughput split is one complete run. Acceptance length is the average number of tokens emitted per target-model verification step: one guaranteed token plus the accepted draft tokens.

The figure below shows that acceptance length remains stable across the fixed-context SPEED-Bench Throughput splits spanning 1K–32K tokens:

```{figure} ./images/eagle3-acceptance-length-context.png
:align: center
:alt: Acceptance length across SPEED-Bench Throughput splits
Acceptance length across SPEED-Bench Throughput splits
```

The qualitative-suite breakdown shows where the draft is strongest. Code, math, and retrieval-augmented generation lead the set; multilingual, reasoning, and STEM tasks also maintain acceptance lengths above 2.8. Across all eleven categories, the draft achieves an average acceptance length of 2.81.

| **Domain** | **AL** | **Domain**    | **AL**   |
|------------|--------|---------------|----------|
| Coding     | 3.33   | Math          | 3.17     |
| RAG        | 3.14   | Multilingual  | 3.07     |
| Reasoning  | 2.91   | Summarization | 2.88     |
| STEM       | 2.84   | Humanities    | 2.70     |
| QA         | 2.55   | Writing       | 2.33     |
| Roleplay   | 2.02   | **Overall**   | **2.81** |

<em>MiniMax-M3 EAGLE3.1 draft acceptance length by domain</em>

***Native AITER serving metrics on SPEED-Bench Qualitative***

| **Metric**            | **EAGLE3 avg.** | **Non-spec** | **Change**     |
|-----------------------|-----------------|--------------|----------------|
| Output throughput/GPU | 415.2 tok/s     | 281.7 tok/s  | 1.47×          |
| Median E2E latency    | 9.68 s          | 13.15 s      | 1.36× faster   |
| Median TTFT           | 154 ms          | 130 ms       | +18.5% latency |

*The EAGLE3 values are means of two complete runs. EAGLE3 and the matched non-speculative baseline use vLLM ROCm v0.27.1, native AITER MXFP4, TP=8, concurrency=48, temperature=0, a 12,288-token maximum model length, and a 2,048-token output cap. Change is relative to the non-speculative baseline; TTFT shows the first-token latency trade-off.*

The figure below shows that EAGLE3 improves per-GPU output throughput at every tested concurrency from 1 to 128; the relative gain narrows from 1.77× at concurrency 1 to 1.40× at concurrency 128 as the non-speculative baseline approaches saturation:

```{figure} ./images/eagle3-throughput-concurrency.png
:align: center
:alt: Native AITER per-GPU output throughput across concurrency on SPEED-Bench Qualitative
Native AITER per-GPU output throughput across concurrency on SPEED-Bench Qualitative
```

The trained draft is published as [amd/MiniMax-M3-EAGLE3.1](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1) and can be used with vLLM speculative decoding for the [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) target:

```console
export VLLM_ROCM_USE_AITER=1
vllm serve amd/MiniMax-M3-MXFP4 --trust-remote-code --tensor-parallel-size 8 \
    --block-size 128 --attention-backend TRITON_ATTN --moe-backend aiter \
    --speculative-config '{"method":"eagle3","model":"amd/MiniMax-M3-EAGLE3.1","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

## What Drives Acceptance Length

Two observations from this run should carry over to EAGLE3 draft training on AMD Instinct more broadly:

- **Domain coverage shows up directly in the per-domain acceptance length.** Structured, high-signal domains such as code and math produce the highest acceptance lengths. Open-ended, high-entropy domains such as roleplay and creative writing are inherently harder to predict and land lower, a pattern that holds regardless of the draft.

- **Acceptance length holds up at long context. The draft’s acceptance length remains essentially flat from 1K to 32K tokens, so draft quality does not erode as context grows.**

## AMD Quark 0.13 Support and Roadmap

AMD Quark 0.13 now provides experimental EAGLE-3 workflows for on-policy data generation, draft training, vLLM export and deployment, and acceptance-length and throughput evaluation.

The release includes a validated [Qwen3-8B Quick Start](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_quick_start.rst) and an [EAGLE-3 Large-Model Best Recipe](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_best_recipe.rst) that uses `amd/MiniMax-M3-MXFP4` as an example target for large, quantized models. We also plan to expand model coverage and add more speculative decoding methods over time.

Quark also ships AI Agent Skills that help you get started. See: [How an AI Agent Improves the AMD Quark User Experience](https://www.amd.com/en/developer/resources/technical-articles/2026/how-an-ai-agent-improves-the-amd-quark-user-experience.html) to learn how to use them to install Quark and run your first quantization workflow.

## Summary

In this blog, you explored how to train an EAGLE3.1 speculative-decoding draft model from scratch for amd/MiniMax-M3-MXFP4 on AMD Instinct MI350X GPUs. You learned how the ROCm-native workflow combines on-policy data generation, vLLM hidden-state extraction, FSDP2 training, model export, and serve-based evaluation.

You also saw how to evaluate the draft with SPEED-Bench and deploy it with vLLM. The resulting model achieves an overall acceptance length of 2.81, remains stable at 2.71–2.72 from 1K to 32K context, and delivers 1.40×–1.77× higher per-GPU output throughput than the matched non-speculative baseline. Together with AMD Quark, this workflow provides an end-to-end path for training and serving EAGLE3 drafts on AMD GPUs.

## Acknowledgements

We would like to thank the [AMD Quark](https://quark.docs.amd.com/latest/) team, the AMD ROCm and vLLM contributors, the TorchSpec maintainers, and the EAGLE3 research community for their work and feedback.

## Additional Resources

- AMD Quark 0.13 speculative decoding overview: [Speculative Decoding](https://github.com/amd/Quark/blob/release/0.13/docs/source/speculative_decoding.rst)

- AMD Quark 0.13 EAGLE-3 documentation: [EAGLE-3](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3.rst)

- AMD Quark 0.13 validated workflow: [Qwen3-8B Quick Start](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_quick_start.rst)

- AMD Quark 0.13 portable large-model workflow: [EAGLE-3 Large-Model Best Recipe](https://github.com/amd/Quark/blob/release/0.13/docs/source/eagle3_best_recipe.rst)

- Draft model: [amd/MiniMax-M3-EAGLE3.1](https://huggingface.co/amd/MiniMax-M3-EAGLE3.1)

- Target model: [amd/MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4)

- EAGLE3 project: [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

- EAGLE3.1 blog: [EAGLE 3.1: Advancing Speculative Decoding Through Collaboration Between the EAGLE Team, vLLM, and TorchSpec](https://blog.vllm.ai/2026/05/26/eagle-3-1.html)

- TorchSpec: [lightseekorg/TorchSpec](https://github.com/lightseekorg/TorchSpec)

- SPEED-Bench paper (NVIDIA): [SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](https://research.nvidia.com/publication/2026-02_speed-bench-unified-and-diverse-benchmark-speculative-decoding)

- SPEED-Bench dataset: [nvidia/SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)

- Companion inference blog: [Accelerating Large-Scale LLM Inference on AMD Instinct with EAGLE3 and AMD Quark](https://rocm.blogs.amd.com/artificial-intelligence/eagle3-speculative-decoding/README.html)

- AMD Quark AI Agent blog: [How an AI Agent Improves the AMD Quark User Experience](https://www.amd.com/en/developer/resources/technical-articles/2026/how-an-ai-agent-improves-the-amd-quark-user-experience.html)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
