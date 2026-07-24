---
blogpost: true
blog_title: "Enabling Language-specific Reasoning in Multilingual Models with Reinforcement Learning"
date: "24 Jul 2026"
author: 'Kai Hakala, Jouni Luoma, Daniel Zautner, Adam Hrin, Elaine Zosa, Maria Barrett, Mika Koistinen, Jonathan Burdge'
thumbnail: 'poro2-playbook.png'
tags: "AI/ML, Reinforcement Learning, Fine-Tuning, LLM"
category: "Applications & models"
target_audience: "LLM researchers and developers"
key_value_propositions: "Step-by-step approach to training a reasoning model with reinforcement learning on AMD hardware. This lets the audience know how they can use the latest RL methods on AMD."
language: English
myst:
    html_meta:
        "author": "Kai Hakala, Jouni Luoma, Daniel Zautner, Adam Hrin, Elaine Zosa, Maria Barrett, Mika Koistinen, Jonathan Burdge"
        "description lang=en": "Learn how to train multilingual reasoning models with reinforcement learning and extend context windows on AMD Instinct GPUs."
        "keywords": "LLM, reinforcement learning, reasoning, long context, finetuning"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_author": "Kai Hakala, Jouni Luoma, Daniel Zautner, Adam Hrin, Elaine Zosa, Maria Barrett, Mika Koistinen, Jonathan Burdge"
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

# Enabling Language-specific Reasoning in Multilingual Models with Reinforcement Learning

We introduce the **Poro 2 Long** family of models, a follow-up to the [Poro 2](https://huggingface.co/collections/LumiOpen/poro-2) family that demonstrated exceptional performance in Finnish and English instruction-following and conversational tasks. This model family demonstrates our progress in developing models capable of reasoning in the same language as the user. A prerequisite for an effective reasoning model is the ability to process longer sequences beyond the sequence lengths used in pretraining; therefore, the Poro 2 Long models feature a context window of 128k tokens.

Figure 1 illustrates the training pipeline for the Poro 2 Long family of models. We show that the context-extension stage maintains the short-context performance of Poro 2 in English and Finnish while achieving strong performance in long-context benchmarks for both languages. We demonstrate how to finetune a model that can reason in English and Finnish in a specialised domain. We also contribute new Finnish language benchmarks for long-context and reasoning. After the context-extension stage, we train two types of models: reasoning and instruction. The reasoning model goes through supervised finetuning (SFT) followed by reinforcement learning (RL) with group relative policy optimization (GRPO) [[Shao et al.](https://arxiv.org/abs/2402.03300)]. This produces the reasoning RL checkpoint. For the instruction model, we run SFT using the same instruction dataset as [Poro 2 Instruct](https://huggingface.co/LumiOpen/Llama-Poro-2-70B-Instruct). At each stage of our pipeline we use a mix of carefully curated English and Finnish datasets to enhance Poro 2 Long's proficiency in both languages.

![Training pipeline diagram for Poro 2 Long. Llama 8B base undergoes continued pretraining to become Poro 2 8B Base, then context extension to Poro 2 8B Long Base. This branches into two paths within a 128k context window: a math reasoning path and an instruct path.](images/image6.png)
Figure 1: Training pipeline for the Poro 2 Long family of models.

In this playbook, we will go through our end-to-end training pipeline for extending the context window of a multilingual base model and training a reasoning model with SFT then RL. Specifically, we will cover these topics:

- Extending the context window of a multilingual base model
- Training a multilingual model to reason in the same language as the user prompt
- Finnish benchmarks for long-context and reasoning models

## Poro 2 Long Family

| Stage | Checkpoint |
| --- | --- |
| Context extension | [Poro 2 8B Long Base](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-Long-Base) |
| Reasoning SFT | [Poro 2 8B Long Math Reasoning SFT](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-Long-Math-Reasoning-SFT-Preview) |
| Reasoning RL | [Poro 2 8B Long Math Reasoning RL](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-Long-Math-Reasoning-RL-Preview) |
| Instruct SFT | [Poro 2 8B Long Instruct](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-Long-Instruct) |

## Collaboration and resources

This work was done on the [LUMI](https://lumi-supercomputer.eu/) supercomputer with AMD Instinct MI250X processors and on a small private cluster with 128 AMD Instinct MI325X processors. This work is a collaboration with [TurkuNLP](https://turkunlp.org/) from the University of Turku and the [OpenEuroLLM](https://openeurollm.eu/) project.

## Context Extension

### Background and Datasets

Long context models are a prerequisite for many tasks expected of modern language models such as reasoning on complex problems, interacting with long documents, long trajectory agentic workflows, and many others. In this section, we discuss our recipe for extending the context window of our existing English-Finnish base model [Poro 2 8B base](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-base).

[Poro 2 8B base](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-base) is a continued pretraining (CPT) checkpoint of [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B). We discussed our CPT playbook in an earlier [post](https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html). While [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) has a context length of 128k tokens, the CPT stage for Poro 2 used an 8k context length and thus lost its long-context capability in the process. Therefore our goal in this section is to regain the long-context capability of [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) in English, retain the short-context Finnish capability of [Poro 2 8B base](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-base) and acquire long-context capability in Finnish.

We implement context extension with RoPE scaling-based methods as we have a limited amount of long context data in Finnish readily available. We experimented with YaRN [[Peng et al.](https://arxiv.org/abs/2309.00071)], LongRoPE [[Ding et al.](https://arxiv.org/abs/2402.13753)], and also in initial tests tried Llama3-style RoPE scaling that was readily supported by Megatron-LM. In our experiments, we found LongRoPE to perform the best against our evals with YaRN coming a close second.

LongRoPE uses evolutionary search to find optimal, non-uniform rescale factors for RoPE rotation angles in different RoPE dimensions. We modify the [LongRoPE code](https://github.com/microsoft/LongRoPE) for use in a slurm-based multinode environment. We make this implementation available [here](https://github.com/LumiOpen/poro2-longrope-search). We use a mix of English and Finnish long-context samples for the LongRoPE search. Specifically, we select samples longer than 128k tokens  from [PG19](https://huggingface.co/datasets/deepmind/pg19) for English and and out-of-copyright books from [Project Lönnrot](https://www.lonnrot.net/index.html) for Finnish.

We implement LongRoPE and YaRN on [our fork](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM) of Megatron-LM and use this fork for LongRoPE context extension. For data, we use the English and Finnish subsets of [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu), supplemented with data from [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) and [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) to retain the code and math capabilities of the Poro 2 base model.

We sort the English and Finnish data into four bins by token lengths (under-4k, 4k-16k, 16k-64k, and 64k-256k)  and prepare data mixes with varying proportions of samples from each bin combined with code and math data. The final data mix is composed of 10% (under-4k bin), 10% (4k-16k bin), 20% (16k-64k bin), 40% (64k-256k bin), 10% StarCoder, 10% FineMath (Table 1). We perform context extension for 4B tokens (1000 steps with global batch size of 4M tokens) using LongRoPE with direct16x extension (from 8k tokens to 128k tokens). Table 2 shows our context extension hyperparameters.

| Proportion | Tokens | Bin | Language/Domain |
| --- | --- | --- | --- |
| 5% | 200M | under-4k | English |
| 5% | 200M | under 4k | Finnish |
| 5% | 200M | 4k-16k | English |
| 5% | 200M | 4k-16k | Finnish |
| 10% | 400M | 16k-64k | English |
| 10% | 400M | 16-64k | Finnish |
| 20% | 800M | 64k-256k | English |
| 20% | 800M | 64k-256k | Finnish |
| 10% | 800M | StarCoder | code |
| 10% | 800M | FineMath | math |

Table 1: Data mix for long context extension

| Hyperparameters |  |
| --- | --- |
| context_parallel_size | 4 |
| tensor_model_parallel_size | 2 |
| pipeline_model_parallel_size | 4 |
| max_position_embeddings | 131072 (16x extension directly) |
| longrope_original_max_position_embeddings | 8192 |
| lr | 1e-5 |
| min-lr | 1e-7 |
| warmup | 100 steps |
| steps | 1000 |
| Global batch size | 4M tokens (32 examples) |
| Total tokens | 4B |

Table 2: Hyperparameters for long context extension

### Evaluation Results

We evaluate the long-context base model, Poro 2 8B Long base, on a standard set of English and Finnish short-context benchmarks and compare the results with Poro 2 8B base with a context window of 8k tokens and Llama 3.1 8B base which has a context window of 128k. Tables 3 and 4 show that the context-extension process retains the short-context English performance of Llama 3.1 base and the exceptional Finnish performance of Poro 2 8B base, respectively.

For long-context benchmarks, we evaluate using HELMET (English) and OneRuler (English and Finnish). Tables 5 and 6 show the HELMET and OneRuler scores, respectively.

|  | arc_challenge | goldenswag | mmlu | truthfulqa_mc | gsm8k | english avg |
| --- | --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long base (ours)** | **61.6** | 76.5 | 62.58 | **46.82** | 53.9 | 60.28 |
| Poro 2 8B base (ours) | 61.01 | 75.8 | 63.91 | 46.47 | **54.59** | **60.36** |
| Llama-3.1-8B | 58.19 | **76.72** | **65.41** | 44.26 | 50.27 | 58.97 |

Table 3: Performance on short-context English benchmarks

|  | arc_challenge_fi | goldenswag_mt_fi | mmlu_mt_fi | truthfulqa_mc_mt_fi | gsm8k_mt_fi |
| --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long base (ours)** | 48.63 | **57.05** | 55.93 | 48.47 | **44.81** |
| Poro 2 8B base (ours) | **48.98** | 56.72 | **56.2** | **48.76** | 42.68 |
| Llama 3.1 8B | 38.48 | 41.84 | 50.21 | 44.69 | 31.31 |

Table 4: Performance on short-context Finnish benchmarks

|  | 8k | 16k | 32k | 64k | 128k |
| --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long Base (ours)** | 58.67 | **57.63** | **58.24** | **53.62** | 48.97 |
| Llama 3.1 8B | **60.45** | 57.54 | 56.75 | 52.18 | **49.65** |

Table 5: HELMET scores of Poro 2 Long base vs. Llama 3.1 8B base.

|                                | **OneRuler EN** |        |           |           | **OneRuler FI** |           |           |           |
| ------------------------------ | --------------- | ------ | --------- | --------- | --------------- | --------- | --------- | --------- |
|                                | **8k**          | **32k** | **64k**   | **128k**  | **8k**          | **32k**   | **64k**   | **128k**  |
| **Poro 2 8B Long Base (ours)** | **48.57**       | 52.27  | 49.43     | 56.57     | **60.86**       | **55.14** | **52.86** | **51.43** |
| Llama 3.1 8B                   | 40.86           | **56** | **48.29** | **63.14** | 55.57           | 54.43     | 45.43     | 40.86     |

Table 6: OneRuler scores (English and Finnish) of Poro 2 Long base vs. Llama 3.1 8B base.

## Language-specific Reasoning

Our reasoning pipeline has two stages: supervised finetuning (SFT) followed by reinforcement learning (see Figure 1). The major challenge we face in training a reasoning model that generates reasoning traces and responses in the same language as the user prompt is the lack of openly available reasoning datasets in Finnish. In this work we translate existing reasoning traces. We use DeepSeek-V3 as the translation model and translate datasets for the SFT and GRPO stages into Finnish.

### Translation pipeline

We evaluate the translation quality of two models, [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct), by using these models to translate 100 math problems from the [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) dataset into Finnish. We use [Qwen3-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) to provide solutions to both sets of translated problems. We use the accuracy of the solution as a proxy for the translation quality. Qwen3 correctly solved 79.25% of the problems translated by DeepSeek versus 77.75% of the problems translated by Qwen2.5. We therefore use DeepSeek-V3 to translate the reasoning datasets.

We also explored generating reasoning traces in Finnish by prompting [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) to provide step-by-step solutions to the translated math problems. We use Qwen3 to solve the problem by prefilling reasoning with the step-by-step solutions generated by DeepSeek. Qwen3 solved only 69.5% of the problems with these reasoning traces compared to 87.75% when using translated traces.

We translate 2.2 million samples from the math subset of [Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset). We use the same inference framework from our previous work to efficiently produce translations for large datasets. We make our translation pipeline available [here](https://github.com/LumiOpen/dispatcher/tree/traces-eval/examples/reasoning_traces). After post-translation heuristic filtering, we end up with 1.4 million Finnish samples.

### Supervised Finetuning (SFT)

Supervised finetuning (SFT) has been shown to be an essential step in training reasoning models in priming the model with reasoning behaviour before the RL phase especially for small models [[Team Olmo](https://arxiv.org/abs/2512.13961)]. We use the Finnish translation of the math subset of the [Llama-Nemotron-Post-Training dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) combined with the original English dataset for SFT.

In prior studies varying hyperparameters have been selected for reasoning SFT. For example, [Olmo 3](https://arxiv.org/abs/2512.13961) trained for 2 epochs with a learning rate of 5.0 × 10^−5, whereas [Klear reasoner](https://arxiv.org/abs/2508.07629) uses a more aggressive approach by training for 4 epochs with an initial learning rate of 8.0 × 10^−5. We follow Olmo 3’s approach with a learning rate of 5.0 × 10^−5, but continue training for 3 epochs to ensure convergence. In the final training run we use the 2.2 million English and 1.4 million Finnish samples described above. We use a maximum sequence length of 32k tokens and pack the samples. We use the same [Megatron-LM fork](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM) as in the context extension stage.

#### Data mix ablations

We conduct ablation experiments on the relative proportions of the English and Finnish data in SFT since translating large reasoning datasets is computationally expensive. We start by training a model with 400k English and 400k Finnish traces, and progressively increase the amount of English data to 800k and 1600k samples. As this work was conducted in parallel with the context extension work, these experiments are done with the original [Poro-2-8B-base](https://huggingface.co/LumiOpen/Llama-Poro-2-8B-base) model.

Table 7 shows the results of the ablation experiments on math benchmarks. The results show that although there is a steady performance improvement on English AIME2025 as the amount of English data increases, these improvements do not transfer to Finnish math capabilities.

| Training data | MATH-500 en | MATH-500 fi | AIME 2025 en | AIME 2025 fi |
| --- | --- | --- | --- | --- |
| 400k en, 400k fi | 0.942 | 0.904 | 0.410 | 0.315 |
| 800k en, 400k fi | 0.946 | 0.906 | 0.457 | 0.355 |
| 1600k en, 400k fi | 0.960 | 0.916 | 0.495 | 0.333 |

Table 7: Results for the reasoning SFT data mix ablations. AIME scores are avg@64.

We notice that Finnish reasoning traces tend to be longer than their English counterparts, most likely due to inferior tokenizer fertility. A common practice in math evaluations is to limit the generation budget to 32k tokens, which results in many of the Finnish traces being truncated. We tried to address this issue by filtering out Finnish training samples that considerably exceed the length of their English counterparts. This however did not result in consistent performance gains and was not included in the final training.

Finally, we ran early experiments mixing standard instruction data with reasoning data. We noticed a drop in math capabilities as well as instability in generating reasoning traces, forcing us to prefill the reasoning start token <think> in the chat template. Thus, only the  math reasoning data is used in the final checkpoint.

#### Instruct SFT Results

Table 8 shows the results of our final reasoning SFT checkpoint on math benchmarks.  We reach 96% accuracy on English Math-500 and 49.9% on English AIME2025, surpassing a similar class of models such as [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B). The weak Finnish performance observed with [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) underscores the necessity for robust multilingual capabilities. Compared against Poro2 8B Instruct, we increase math performance substantially, with Finnish AIME 2025 scores improving by orders of magnitude.

We also benchmark our model against [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) as it is one of the few open multilingual reasoning models out there. [Ministral](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) has significantly weaker math performance on English benchmarks. Although [Ministral](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) has multilingual capabilities, Finnish is not one of the supported languages. This is visible in the Finnish evaluations where it is significantly weaker than our SFT checkpoint but it does demonstrate stronger Finnish performance than [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).

We also observe that our model consistently uses the targeted language in the reasoning traces in MATH-500 and AIME2025 evaluations. [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) on the other hand fails to reason in Finnish on 4-10% of the evaluation samples, depending on the dataset.

|  | Math-500 EN | Math-500 FI | AIME25 EN | AIME25 FI | AIME26 EN | AIME26 FI |
| --- | --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long Math Reasoning SFT (ours)** | **0.960** | **0.928** | **0.499** | **0.403** | **0.604** | **0.478** |
| DeepSeek-R1-Distill-Llama-8B | 0.942 | 0.352 | 0.356 | 0.051 | 0.389 | 0.024 |
| Ministral-3-8B-Reasoning-2512 | 0.906 | 0.658 | 0.299 | 0.074 | 0.294 | 0.04 |

Table 8: Reasoning SFT results on math benchmarks. AIME scores are avg@64.

### Reinforcement Learning

We use the [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) framework for running GRPO-like verifiable reinforcement learning. PrimeRL implements their own variant of GRPO. In our training, we use the existing [Prime Intellect math environment](https://app.primeintellect.ai/dashboard/environments/primeintellect/math-env), which relies on the [math-verify](https://github.com/huggingface/Math-Verify) library for validating the generated solutions.

Math-verify is an open-source symbolic verification library that checks the equivalence of mathematical expressions, enabling deterministic reward computation without relying on LLM-based judges. We also explore using an LLM judge as a fallback for cases where math-verify fails to parse the model’s output. We compare math-verify against Qwen3.5 judges of varying sizes on our training rollouts, using Claude Opus 4.6 as ground truth. The agreement rate between math-verify and the LLM judges is sufficiently high on our dataset that the additional cost and latency of running a judge model during training is not justified.

[Prior studies](https://arxiv.org/abs/2510.26788) have shown that numerical mismatches between the inference and training engines are a common cause of instability in RL training. Some suggested solutions include using fp16 and gradient scaling instead of bf16, or using fp32 in the output layer. In our experiments, we detect considerable numerical mismatches and trace the issue to misaligned fp32 matrix multiplication precision configuration, which leads to PyTorch internally using bf16 inconsistently across inference and training. The GRPO implementation in Prime-RL directly uses the inference engine’s log probabilities as the reference model. This results in two issues: not only are the gradients noisy, but a significant number of them are also clipped because the importance ratio computation is inaccurate. We resolve this by setting matmul_precision to "highest" in the trainer configuration, ensuring consistent fp32 precision across both engines. This eliminates the gradient clipping issue and stabilizes training.

We train with a constant learning rate of 1e-6, KL penalty of 0.01, IPO masking of 0.2 on both tails, advantage temperature of 2.0, rollout temperature of 1.0, batch size of 128, and 8 rollouts per example. Due to memory constraints, we use a shorter sequence length for training (24,576 tokens) than for inference (32,768 tokens). We find that this mismatch does not hurt performance: correct solutions tend to be completed within 24k tokens, so the winning trajectories that drive the policy gradient are captured at the shorter training length. The longer inference budget primarily benefits exploration during rollout generation. We use 4 nodes: 1 trainer node (8 MI325X GPUs) and 3 inference nodes (24 MI325X GPUs, TP=1, 24-way data parallel). Table 9 shows the full list of hyperparameters.

| Hyperparameter | Value |
| --- | --- |
| Learning rate | 1e-6 (constant, no warmup) |
| KL penalty | 0.01 |
| IPO mask (low / high) | 0.2 / 0.2 |
| Advantage temperature | 2.0 |
| Rollout temperature | 1.0 |
| Batch size | 128 |
| Rollouts per example | 8 |
| Inference sequence length | 32,768 |
| Trainer sequence length | 24,576 |
| Max generation tokens | 24,576 |
| Max gradient norm | 1.0 |
| Matmul precision | highest |
| Infrastructure | 4 nodes: 1 trainer (8 × MI325X) + 3 inference (24 × MI325X, TP=1, 24-way DP) |

Table 9: Hyperparameters for RL training

#### Dataset

We use [KlearReasoner-MathSub-30K](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-MathSub-30K), a curated dataset of 30,000 competition-level math problems. We translate the prompts into Finnish using DeepSeek-V3 and translate the prompt templates used by the math environment into the target language so that the model receives instructions in Finnish when training on Finnish problems.

We apply offline difficulty filtering before training to focus on problems at the frontier of the model’s capability. The filtering process generates 16 completions per problem at temperature 1.0, scores each with math-verify, and computes a pass rate. We keep only problems where the pass rate falls between 25% and 75%, removing problems the model always solves (no learning signal) and problems it never solves (only negative rewards). We estimate difficulty separately for each language using the SFT checkpoint. We also enable online difficulty filtering in the orchestrator, which dynamically adjusts the problem distribution during training as the model’s solve rates evolve.

#### Experiments

We train with two separate environments from step 0, one for English and one for Finnish math problems. Each environment has its own difficulty-filtered dataset, allowing online difficulty filtering to operate independently per language while sharing a single policy. Both English and Finnish performance improve steadily throughout training.

In ablation experiments, we also try combining both languages in a single mixed-language environment where English and Finnish problems share one dataset. This underperforms relative to the two-environment configuration. We attribute this to the interaction with online difficulty filtering: when two languages with different difficulty distributions share a single environment, the filtering cannot adapt to each language independently. The two-environment setup avoids this problem entirely.

Due to memory constraints, we use a shorter sequence length for training (24,576 tokens) than for inference (32,768 tokens). We find that this mismatch does not hurt performance: correct solutions tend to be completed within 24k tokens, so the winning trajectories that drive the policy gradient are captured at the shorter training length. The longer inference budget primarily benefits exploration during rollout generation.

We train for 1,350 steps over approximately 8 days on 32 MI325X GPUs. We select the best checkpoint by computing the harmonic mean across all evaluation benchmarks and both languages. The best checkpoint is at step 900, reached after approximately 110 hours of training (~3,500 GPU-hours).

#### RL Results

Table 10 shows the results of the RL checkpoint compared against the SFT checkpoint and baseline models. RL training yields consistent improvements over SFT across all benchmarks and both languages. On English MATH-500, accuracy improves from 96.0% to 97.0%. On AIME 2025, English performance increases from 49.9% to 60.7% and Finnish from 40.3% to 52.1%. On the held-out AIME 2026, English improves from 60.4% to 67.8% and Finnish from 47.8% to 57.2%.

Our RL checkpoint, like our SFT checkpoint, also outperforms [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) and [Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512) by a substantial margin.

|  | Math-500 EN | Math-500 FI | AIME25 EN | AIME25 FI | AIME26 EN | AIME26 FI |
| --- | --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long Math Reasoning RL (ours)** | **0.97** | **0.95** | **0.607** | **0.521** | **0.678** | **0.572** |
| Poro 2 8B Long Math Reasoning SFT (ours) | 0.96 | 0.928 | 0.499 | 0.403 | 0.604 | 0.478 |
| DeepSeek-R1-Distill-Llama-8B | 0.942 | 0.352 | 0.356 | 0.051 | 0.389 | 0.024 |
| Ministral-3-8B-Reasoning-2512 | 0.906 | 0.658 | 0.299 | 0.074 | 0.294 | 0.04 |

Table 10: Reasoning RL performance on math benchmarks. AIME scores are avg@64.

## Instruction Tuning

We finetune a general-purpose instruction checkpoint separate from the reasoning checkpoint. For instruction tuning, we run SFT on Poro 2 8B Long base using the same instruction dataset as [Poro 2 Instruct](https://huggingface.co/LumiOpen/Llama-Poro-2-70B-Instruct). We pack the samples and use a sequence length of 32k.

We evaluate the instruction SFT checkpoint on two short-context benchmarks in English and Finnish: MTBench and IFEval. For long-context evaluation, we evaluate on HELMET (English) and OneRuler (English and Finnish). We notice that the long-context performance of the SFT checkpoint drops considerably compared to Poro 2 8B Long base. Other works have reported long-context degradation after post-training, for example in [SmolLM3](https://huggingface.co/blog/smollm3). SmolLM3 suggests model merging as a solution to recover the long-context performance of the base model. We follow this solution and merge the SFT checkpoint with the base model using MergeKit. We use merge weights of 0.8 and 0.2 for the SFT checkpoint and base checkpoints, respectively. This merged checkpoint is our Poro 2 8B Long Instruct model.

We present the short-context results of Poro 2 8B Long Instruct in Table 11 and long-context performance in Tables 12 (HELMET) and 13 (OneRuler). Poro 2 8B Long Instruct outperforms Llama 3.1 8B Instruct on Finnish short-context by a healthy margin but underperforms Llama on English benchmarks. In long-context performance,  Poro 2 8B Long Instruct performs close to Poro 2 8B Long base but underperforms compared to Llama Instruct. We also note that in the Llama models, the instruct model improves considerably over its base counterpart. In the future, we plan to improve long-context performance throughout the post-training stages.

|  | MTBench EN | IFEval EN | MTBench FI | IFEval FI |
| --- | --- | --- | --- | --- |
| **Poro 2 8B Long Instruct (ours)** | 6.84 | 76.16 | 6.18 | 63.4 |
| Poro 2 8B Long SFT (before merging, ours) | 7.04 | 78.93 | **6.25** | **67.1** |
| Llama 3.1 8B Instruct | **7.7** | **79.48** | 4.1 | 47.31 |

Table 11: Short-context performance of Poro 2 8B Long Instruct compared to Llama 3.1 8B Instruct.

|  | 8k | 16k | 32k | 64k | 128k |
| --- | --- | --- | --- | --- | --- |
| **Poro 2 8B Long Instruct (ours)** | 60.63 | 58.83 | 55.84 | 51.18 | 43.02 |
| Poro 2 8B Long SFT (before merging, ours) | 45.62 | 45.64 | 40.99 | 34.62 | 27.78 |
| Poro 2 Long Base (base, ours) | 58.67 | 57.63 | 58.24 | 53.62 | 48.97 |
| Llama 3.1 8B Instruct | **64.4** | **64.27** | **62.63** | **62.82** | **56.45** |
| Llama 3.1 8B (base) | 60.45 | 57.54 | 56.75 | 52.18 | 49.65 |

Table 12: HELMET scores for Poro 2 8B Long Instruct compared to Llama 3.1 8B Instruct and two base models.

|                         | **OneRuler EN** |           |           |           | **OneRuler FI** |           |           |           |
| ----------------------- | --------------- | --------- | --------- | --------- | --------------- | --------- | --------- | --------- |
|                         | **8k**          | **32k**   | **64k**   | **128k**  | **8k**          | **32k**   | **64k**   | **128k**  |
| **Poro 2 8B Long Instruct (ours)** | 53.74           | 51.03     | 51.66     | 46.17     | 67.2            | 46.63     | **55.29** | 45.34     |
| Poro 2 8B Long base (base, ours)     | 48.57           | 52.57     | 49.43     | **56.57** | 60.86           | 55.14     | 52.86     | **51.43** |
| Llama 3.1 8B Instruct   | **72.43**       | **67.26** | **64.51** | 55.57     | **77.11**       | **67.54** | 54.54     | 44.84     |
| Llama 3.1 8B (base)                  | 40.86           | 56 | 48.29 | 63.14 | 55.57           | 54.43     | 45.43     | 40.86     |

Table 13: OneRuler scores for Poro 2 8B Long Instruct compared to Llama 3.1 8B Instruct and two base models.

## Summary

This playbook presents a complete training pipeline for the Poro 2 8B Long family of models, demonstrating cost-effective context extension from 8k to 128k tokens using LongRoPE and building strong reasoning capabilities in English and Finnish. Readers learn practical techniques for extending context windows with limited compute, strategies for creating multilingual reasoning datasets through translation, and critical insights for stable bilingual reinforcement learning with GRPO. The final reasoning model achieves 60.7% on AIME 2025 in English and 52.1% in Finnish, significantly outperforming comparable models like DeepSeek-R1-Distill-Llama-8B and Ministral-3-8B-Reasoning, demonstrating that world-class multilingual AI is achievable through careful engineering and strategic data curation.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
