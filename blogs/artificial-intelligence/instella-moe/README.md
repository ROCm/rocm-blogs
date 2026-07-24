---
blogpost: true
blog_title: "Introducing Instella-MoE: A State-of-the-Art Fully Open Mixture-of-Experts Language Model"
date: "24 Jul 2026"
author: "Jiang Liu, Sudhanshu Ranjan, Prakamya Mishra, Yonatan Dukler, Gowtham Ramesh, Zicheng Liu, Jialian Wu, Ximeng Sun, Wen Xie, Chaojun Hou, Vikram Appia, Zhenyu Gu, Emad Barsoum"
thumbnail: 'instella-moe-thumbnail.png'
tags: "AI/ML"
category: "Applications & models"
target_audience: "AI developers, AI researchers, and AI enthusiasts"
key_value_propositions: "AMD's fully open 16B MoE Model"
language: English
myst:
    html_meta:
        "author": "Jiang Liu, Sudhanshu Ranjan, Prakamya Mishra, Yonatan Dukler, Gowtham Ramesh, Zicheng Liu, Jialian Wu, Ximeng Sun, Wen Xie, Chaojun Hou, Vikram Appia, Zhenyu Gu, Emad Barsoum"
        "description lang=en": " Explore Instella-MoE-16B-A3B, AMD’s fully open 16B MoE LLM with 2.8B active params per token, trained on AMD Instinct™ MI300 & MI325 GPUs."
        "keywords": "Instella, MoE, LLM, RL, MI325, MI300, Training"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Jiang Liu, Sudhanshu Ranjan, Prakamya Mishra, Yonatan Dukler, Gowtham Ramesh, Zicheng Liu, Jialian Wu, Ximeng Sun, Wen Xie, Chaojun Hou, Vikram Appia, Zhenyu Gu, Emad Barsoum"
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

# Introducing Instella-MoE: A State-of-the-Art Fully Open Mixture-of-Experts Language Model

AMD is excited to introduce Instella-MoE, a state-of-the-art fully open Mixture-of-Experts (MoE) language model with 16 billion total parameters and 2.8 billion active parameters. Trained from scratch on AMD Instinct™ MI300X and MI325X GPUs with AMD ROCm™ software stack, Instella-MoE combines a sparsely activated MoE design with architectural innovations such as Gated Multi-head Latent Attention (Gated MLA) and [FarSkip-Collective](https://rocm.blogs.amd.com/artificial-intelligence/farskip-collective-moe/README.html) connectivity. Instella-MoE delivers competitive performance across a broad suite of benchmarks against both dense and MoE baselines (as shown in Figure 1) including models with comparable or larger active parameter counts, establishing it as one of the strongest fully open language models at its scale.

Instella-MoE advances AMD’s broader effort to train advanced language models fully on AMD hardware and software with open tools and open research practices. Trained end-to-end with the open-source [Primus](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html) and [Miles](https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/) frameworks, this model highlights the scalability and efficiency of AMD hardware and software stack for large-scale MoE training and inference, spanning from pre-training to post-training. Beyond strong model quality, it incorporates architecture- and system-level innovations that improve both training and serving efficiency.

```{figure} ./images/fig_cost_performance.png
:align: center

*Figure 1: Pre-trained and Post-trained Instella-MoE model performance compared with other similar size state-of-the-art models.*
```

In line with AMD's commitment to open source, we are releasing all artifacts for the Instella-MoE model here, including model weights from every training stage, training configurations, data mixtures, and code. By making these resources openly available, we hope to help the AI community collaborate, reproduce results, and accelerate innovation in open language models.

This blog introduces the Instella-MoE model and then describes the architecture and multi-stage training pipeline behind the model. The release includes checkpoints from the following Instella-MoE training pipeline stages as shown in Table 1 below:

| Model | Stage | Description |
| ----- | ----- | ----------- |
| **Instella-MoE-16B-A3B-Pretrain** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-Pretrain)) | Pre-training | MoE base model trained from scratch on a large and diverse training corpus. |
| **Instella-MoE-16B-A3B-Midtrain** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-Midtrain)) | Mid-training | Pretrained model further trained on high-quality data mixtures to refine key capabilities. |
| **Instella-MoE-16B-A3B-Base** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-Base)) | Long-context | Long-context training to extend the model’s ability to process and reason over longer sequences. **We use this as our final base checkpoint**. |
| **Instella-MoE-16B-A3B-SFT** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-SFT)) | SFT | Base checkpoint extended via supervised fine-tuning (SFT) to enable instruction following and chain-of-thought reasoning capabilities. |
| **Instella-MoE-16B-3AB-DPO** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-DPO)) | DPO | Direct preference optimization (DPO) on contrastive preference data to improve model performance. |
| **Instella-MoE-16B-3AB-Think** ([Link](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think)) | RL | Final thinking checkpoint refined with reinforcement learning (RL) to further strengthen instruction following and overall response quality. |

<em><strong>Table 1:</strong> Instella-MoE-16B-A3B models and training stages.</em>

## Takeaways

- Instella-MoE is a new state-of-the-art fully open Mixture-of-Experts language model developed by AMD, with 16 billion total parameters and 2.8 billion active parameters per token, trained from scratch on AMD Instinct™ MI300X and MI325X GPUs.

- The Instella-MoE model checkpoint release spans every stage of the model training pipeline, including pre-training, mid-training, long-context extension, SFT, DPO and RL.

- Built entirely on the AMD ROCm™ software stack on top of the Primus training and Miles RL frameworks, Instella-MoE incorporates cutting-edge architecture and systems innovations—including Gated Multi-head Latent Attention (Gated MLA) and extreme communication-computation overlap through FarSkip-Collective—for efficient large-scale training and inference on AMD hardware.

- Fully open and accessible: we provide our complete training recipe across all training stages, including training frameworks, data mixtures, intermediate checkpoints and inference code.

## Instella-MoE Model Architecture

The Instella-MoE architecture is a decoder-only MoE design built with Gated Multi-head Latent Attention (Gated MLA) and FarSkip-Collective connectivity. The model consists of 27 decoder layers with a hidden size of 2048, totaling 16B parameters, with 2.8B active per token. In the MoE layers, a shared-plus-routed expert design is used, with 2 shared experts and 6 routed experts selected from 64 available routed experts per token. We use a Multi-Token Prediction (MTP) objective [^1],[^2] during pre-training and mid-training.

We augment Multi-head Latent Attention (MLA) [^3] with a lightweight learned output gate [^4], a design we refer to as Gated MLA. Each Gated MLA layer derives an input-conditioned gate through a dedicated linear projection and applies it multiplicatively to the MLA output prior to the output projection. By introducing a data-dependent non-linearity, Gated MLA enables the model to selectively attenuate low-utility attention responses for each token and increases model expressivity at modest cost.

In addition, we apply FarSkip-Collective [^5] which modifies the standard MoE connectivity to better overlap communication with computation during expert-parallel training. By passing outdated and partial activations into the MoE and attention layers, FarSkip-Collective reduces communication bubbles and improves overall training and inference efficiency while maintaining model accuracy.

During pre-training, FarSkip-Collective sped up model pre-training by 12.7% by overlapping communication arising from expert-parallelism, as shown in Figure 2. To realize these savings, we implemented an optimized communication-overlapped algorithm in the Primus framework that allows us to overlap both the forward and backward communication of model training. For more details see [^5], [^6]. For inference, we implement Instella-MoE using the SGLang inference framework. By overlapping communication with computation, this implementation delivers up to a 39.2% reduction in Time to First Token (TTFT) when the model is served with expert parallelism [^7].

```{figure} ./images/training_inference_efficiancy.png
:align: center

*Figure 2: Instella-MoE-16B-A3B training and inference throughput efficiency.*
```

## Training Pipeline

The Instella-MoE model is trained through a multi-stage pipeline (Figure 3), with each stage progressively enhancing the model’s capabilities from broad foundational language understanding to long-context processing, instruction following, alignment, and reasoning. The full training pipeline is shown in Figure 2.

```{figure} ./images/training_pipeline.png
:align: center

*Figure 3: Instella-MoE-16B-A3B training pipeline.*
```

### Pretraining

In the first training stage, Instella-MoE is trained from scratch on a large and diverse corpus including general web text, code, mathematics, and science data with 7.1T tokens. The pretraining mixture is drawn primarily from recent high-quality open datasets and is designed to establish broad foundational capabilities across natural language understanding, world knowledge, reasoning, coding, and mathematics. The corpus includes sources such as [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) for web data, [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1), [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) for mathematics, [RefineCode](https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-code-corpus-meta) and [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) for code, [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) for curated pretraining SFT-style data, and [TxT360](https://huggingface.co/datasets/LLM360/TxT360) for other domains.

### Mid-training

A mid-training stage is performed on top of the pretrained model to further sharpen key capabilities such as math, coding, and reasoning. This mid-training stage emphasizes curated math, code, science, instruction-following, and reading comprehension data, using [Dolma3 Dolmino 100B](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125) as the core data mixture. To improve robustness and final model quality, this stage is run with multiple data mixture variants to emphasize STEM and reasoning domains.  We combine the resulting checkpoints via model merging to produce the Mid-training checkpoint, Instella-MoE-16B-A3B-Midtrain.

### Long-Context Extension

Starting from the mid-training checkpoint, a dedicated long-context extension stage is performed to produce the Instella-MoE-16B-A3B-Base model, extending the context window from 4K to 64K tokens during training. This stage is trained on [Longmino 100B](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125), a long-context corpus of coherent long documents. To facilitate longer document support, we incorporate YaRN positional encoding and increase the RoPE theta parameter from the pretraining stage value. To prevent attention leakage across packed document boundaries, we also introduce document masking during training.

The long-context stage follows a two-phase recipe. The first phase teaches the model to attend to information across the full 64K window using a long-context general mixture. The second phase further sharpens long-context performance on math, code, and reasoning-heavy tasks. We use data sourced from the math and coding subsets of the [Dolma3 Dolmino](https://huggingface.co/datasets/allenai/dolma3_dolmino_pool) dataset, along with other datasets like [Instella GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic) and [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath).

Both standard and long-context results for our final base checkpoint are presented in the [Results](#results) section (Table 3 and Table 4).

### Supervised Fine-Tuning

The SFT stage is built on top of the Instella-MoE-16B-A3B-Base model. The SFT mixture uses [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) as a foundation and is augmented with additional data from the [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) and [Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) pool spanning mathematics, code, and science to further improve reasoning and STEM performance.

#### Feedback-Driven Data Curation

For the final phase of SFT, we use fine-grained error analysis to identify areas that remain difficult for the model and retrieve training examples that directly target those weaknesses. The pipeline begins by generating responses with an intermediate SFT checkpoint on held-out validation data. A strong judge model then analyzes these responses and provides detailed feedback, which a reflection model uses to identify recurring error patterns and generate a weighted set of retrieval queries. Based on these queries and their associated weights, training examples are selected from the large embedded SFT data pool through similarity-based retrieval. The retrieved examples are combined with uniformly sampled examples from each domain to maintain broad capability coverage. This process yields a targeted mixture of 512K examples for the final stage of SFT. Compared with uniform sampling from the SFT dataset, training on this curated mixture notably improves model performance, especially on mathematics and coding benchmarks.

#### Preference Tuning

We then apply DPO on top of the SFT model on contrastive preference data to exploit capability-relevant differences between chosen and rejected responses. This provides an additional training signal beyond imitation-based SFT and further improves the model’s reasoning, instruction-following, and general task performance [^8] . In our preliminary experiments, we found that directly applying DPO to the MoE model led to performance degradation. We hypothesize that this behavior is related to the load-balancing objective, which can rapidly change the experts’ router affinity. To improve training stability, during DPO we disable router bias updates and the auxiliary load-balancing loss. With these adjustments, we observe no further degradation during DPO training. The resulting model retains the reasoning and instruction-following capabilities acquired during SFT while improving performance across a broad suite of tasks.

### Reinforcement Learning

On top of the DPO model, a final reinforcement learning (RL) stage is used to further improve model capability. RL runs natively on AMD Instinct GPUs using the [Miles RL framework with ROCm support](https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/). Miles is an open-source framework for large-scale post-training of language and multimodal models, built with SGLang and the Slime RL ecosystem. It provides the distributed infrastructure we use for our RL post-training, including distributed rollout generation, GRPO/PPO policy optimization, on-policy training loops, and Ray-based orchestration across multi-node clusters with native integration for Megatron-LM and SGLang.

```{figure} ./images/rl_training_pipeline.png
:align: center

*Figure 4: Instella-MoE-16B-A3B RL training pipeline.*
```

We focus our RL efforts primarily on improving instruction following (IF) capabilities because our DPO model already scores highly on other domains. To achieve this, as shown in Figure 4, we first perform IF-specialized RL training to obtain an IF expert, and then perform [Multi-Teacher On-Policy Distillation (MOPD)](https://arxiv.org/abs/2606.30406) to integrate the enhanced IF capabilities into the DPO model while maintaining the performance on all other domains. The final Instella-MoE-Think results are reported in the [Results](#results) section.

#### Stage 1: IF-specialized RL Training Setup

For the IF RL training, we use the IF-RLVR subset from the [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) dataset and train for 1,400 steps. We use asynchronous training and a maximum response length of up to 16K tokens. We incorporate several improvements [^9] [^10] to GRPO [^11] such as zero-gradient signal filtering, active sampling, token-level loss, no KL loss, clip-higher, no standard-deviation normalization, and Rollout Routing Replay (R3) [^12].

#### Stage 2: MOPD

We apply [Multi-Teacher On-Policy Distillation (MOPD)](https://arxiv.org/abs/2606.30406) to integrate the IF capability of the teacher model into the DPO model. Specifically, we distill on the student's own on-policy rollouts and route each rollout to a teacher by its prompt domain: instruction-following prompts are scored by the IF-RL teacher, while all remaining prompts (math, code, and general) are scored by the frozen DPO model itself. The student is initialized from the DPO model, so on the non-IF domains the teacher is the student's own initialization and acts as a self-anchoring regularizer that preserves math and general capabilities while the IF-domain signal does the lifting. As shown in Table 2, MOPD recovers most of the IF expert’s instruction-following gain on IFEval while maintaining math, code, MMLU, and AGIEval at the DPO level. The final results for our post-trained Instella-MoE-16B-A3B-Think model are presented in the [Results](#results) section (Table 5).

```{figure} ./images/result_table_rl.png
:align: center

*Table 2: Instella-MoE-Think RL results.*
```

## Results

We evaluate Instella-MoE at two key stages of the pipeline: the final base model (Instella-MoE-16B-A3B-Base) and the final post-trained thinking model (Instella-MoE-16B-A3B-Think). Across both, Instella-MoE is competitive with or outperforms open dense and MoE models of comparable or larger active parameter counts, while activating only 2.8B parameters per token.

### Base Model Results

On standard benchmarks (Table 3), our final base checkpoint (Instella-MoE-16B-A3B-Base) attains an average score of 76.7, the strongest among fully open models—well ahead of SmolLM3-3B-Base (70.5), OLMo-3-7B (70.1), and OLMoE-1B-7B (61.9). It is also highly competitive with leading open weight models, surpassing Moonlight-16B-A3B (76.2) on average and trailing only Qwen3.5-4B-Base (79.5). Instella-MoE-16B-A3B-Base leads all evaluated models on WinoGrande (86.5) and delivers strong coding results (HumanEval+ 65.7), with balanced performance across knowledge, reasoning, math, and code. Notably, it achieves these results while activating only 2.8B parameters per token, outperforming fully open dense models such as OLMo-3-7B that activate more than twice as many parameters.

```{figure} ./images/result_table_base_standard.png
:align: center

*Table 3: Instella-MoE-16B-A3B-Base Results on Standard Benchmarks.*
```

The base model also demonstrates strong long-context capability on the HELMET and RULER benchmarks (Table 4), sustaining performance out to 64K tokens with a HELMET average of 41.5 and a RULER average of 79.4. Among fully open models, it is highly competitive with the dense OLMo-3-7B (43.1 / 80.2) and clearly ahead of SmolLM3-3B-Base (37.6 / 78.6). This confirms that the long-context extension stage successfully extends the usable context window while maintaining strong performance on standard benchmarks.

```{figure} ./images/result_table_base_long.png
:align: center

*Table 4: Instella-MoE-16B-A3B-Base Results on Long Context HELMET and RULER Benchmarks.*
```

### Post-trained Model Results

Model quality improves steadily through post-training (Table 5), from Instella-MoE-16B-A3B-SFT (71.58) to DPO (72.67) to the final Instella-MoE-16B-A3B-Think (73.22). The Think model achieves the highest average among all fully open models, ahead of Olmo3-7B-Think (71.97), and also outperforms strong open-weight models such as Gemma-4-E4B (think) (70.47) and Qwen3.5-4B (69.73). Consistent with our RL focus on instruction following, the final stage delivers its largest gains on IFEval—raising it from 77.08 (DPO) to 83.70—while preserving math, code, and MMLU performance, with the Think model also leading on AGIEval (82.50) and AIME25 (73.40).

```{figure} ./images/result_table_instruct.png
:align: center

*Table 5: Instella-MoE-Think results. We evaluate all models using the OLMES framework, generating up to a maximum of 32768 tokens.*
```

## Summary

In this blog, we introduced Instella-MoE, a fully open Mixture-of-Experts language model that advances both model quality and efficiency in the open LLM ecosystem. We showed that a 16B-parameter MoE model with 2.8B active parameters per token can deliver strong performance while keeping compute costs closer to those of a much smaller dense model. We also demonstrated that AMD hardware and software can support large-scale end-to-end MoE development, from pretraining through post-training, using an entirely open stack built on AMD Instinct™ MI300X and MI325X GPUs, ROCm™, Primus, and Miles.

We used this blog to present the main technical contributions behind Instella-MoE, including Multi-head Gated Latent Attention (MGLA) and FarSkip-Collective, and to describe the full training pipeline: pretraining, mid-training, long-context extension to 64K context, supervised fine-tuning, DPO, and reinforcement learning. We also highlighted the main deliverables of this release: model weights for every stage, training configurations, data mixtures, intermediate checkpoints, and code.

Through this release, we provide the blog’s key deliverables: model weights for every training stage, training configurations, data mixtures, intermediate checkpoints, and code. We believe these resources will help researchers and developers reproduce our results, study open MoE training recipes in detail, and build new models and systems on top of the Instella-MoE foundation. We view Instella-MoE as an important step toward more transparent, reproducible, and high-performance open language models, and we plan to continue extending this line of work with larger models, better reasoning, stronger instruction following, and further efficiency improvements.

## Acknowledgements

We are deeply grateful to the LLM360 team and the Miles team for their invaluable support throughout the development of our model.

## Additional Resources

### Huggingface Models

- The Instella-MoE-16B-A3B model collection is available under [https://huggingface.co/collections/amd/instella-moe](https://huggingface.co/collections/amd/instella-moe)

### Code

- The complete training code is available here: [https://github.com/AMD-AGI/Instella-MoE](https://github.com/AMD-AGI/Instella-MoE)

Please refer to the following blogs to dive deeper into AMD’s training frameworks we use:

- [Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)
- [ROCm Support for Miles: Large-Scale RL Post-Training on AMD Instinct™ GPUs](https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/)
- [Introducing Instella: New State-of-the-art Fully Open 3B Language Models](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html)

## Bias, Risks, and Limitations

- The models are being released for research purposes only. They are not intended for use cases requiring high levels of factual accuracy, safety-critical applications, or health and medical applications. They must not be used to generate false information or facilitate toxic conversations.

- Model checkpoints are made accessible without any safety promises. It is crucial for users to conduct comprehensive evaluations and implement safety filtering mechanisms as per their respective use cases.

- It may be possible to prompt the model to generate content that may be factually inaccurate, harmful, violent, toxic, biased, or otherwise objectionable. Such content may also be generated in response to prompts that were not intended to elicit it. Users are thus requested to be aware of this and exercise caution and responsible thinking when using the model.

- The multilingual abilities of the models have not been tested and thus may misunderstand and generate erroneous responses across different languages.

## License

- The Instella-MoE-16B-A3B models are licensed for academic and research purposes under a Research RAIL license.

- Refer to the [LICENSE](https://huggingface.co/amd/Instella-MoE-16B-A3B-Think/blob/main/LICENSE) for more information.

## Contributors

> **Core contributors**:
> Jiang Liu, Sudhanshu Ranjan, Prakamya Mishra, Yonatan Dukler, Gowtham Ramesh, Zicheng Liu
>
> **Contributors:**
> Jialian Wu, Ximeng Sun, Wen Xie, Chaojun Hou, Vikram Appia, Zhenyu Gu, Emad Barsoum

## Citation

The Instella-MoE technical report is coming soon. Feel free to cite our Instella-3B model and FarSkip paper:

```text
@article{instella,
  title={Instella: Fully Open Language Models with Stellar Performance},
  author={Liu, Jiang and Wu, Jialian and Yu, Xiaodong and Su, Yusheng and Mishra, Prakamya and Ramesh, Gowtham and Ranjan, Sudhanshu and Manem, Chaitanya and Sun, Ximeng and Wang, Ze and Brahma, Pratik Prabhanjan and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2511.10628},
  year={2025}
}

@inproceedings{
dukler2026farskipcollective,
title={FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models},
author={Yonatan Dukler and Guihong Li and Deval Shah and Jiang Liu and Vikram Appia and Emad Barsoum},
booktitle={Ninth Conference on Machine Learning and Systems},
year={2026},
url={https://openreview.net/forum?id=ruOpvLzsGV}
}

```

[^1]: Gloeckle, F., et al. Better & Faster Large Language Models via Multi-token Prediction. Advances in Neural Information Processing Systems (NeurIPS), 2024.

[^2]: DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437, 2024.

[^3]: DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv preprint arXiv:2405.04434, 2024.

[^4]: Qiu, Z., et al. Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[^5]: Dukler, Y., et al. FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models. Proceedings of the 9th MLSys Conference, 2026.

[^6]: github.com/AMD-AGI/Primus/blob/dev/farskip/README.md

[^7]: https://github.com/AMD-AGI/FarSkip-Collective

[^8]: Olmo Team. "Olmo 3." arXiv preprint arXiv:2512.13961, 2025.

[^9]: Yu, Q., et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[^10]: Liu, Z., et al. Understanding R1-Zero-Like Training: A Critical Perspective. Conference on Language Modeling (COLM), 2025.

[^11]: Shao, Zhihong, et al. "Deepseekmath: Pushing the limits of mathematical reasoning in open language models." arXiv preprint arXiv:2402.03300 (2024).

[^12]: Ma, W., et al. Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers. arXiv preprint arXiv:2510.11370, 2025.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, ROCm and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
