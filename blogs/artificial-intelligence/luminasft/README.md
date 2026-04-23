---
blogpost: true
blog_title: "LuminaSFT: Generating Synthetic Fine-Tuning Data for Small Language Models"
date: 24 Feb 2026
author: 'Sudhanshu Ranjan, Jiang Liu, Gowtham Ramesh, Prakamya Mishra, Jialian Wu, Xiaodong Yu,  Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum'
thumbnail: 'lumina-sft-blog.png'
tags: AI/ML, GenAI, Fine-Tuning
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: LuminaSFT is a supervised fine-tuning (SFT) dataset designed to advance the development of small language models (SLMs). While prior work has largely emphasized general-purpose data creation or targeted domains such as mathematics and coding, LuminaSFT focuses on evaluating the impact of task-specific data generation. The dataset is released publicly where possible. The codebase has been run on AMD GPUs and ROCm, showcasing the AMD GPUs.
language: English
myst:
    html_meta:
        "author": "Sudhanshu Ranjan, Jiang Liu, Gowtham Ramesh, Prakamya Mishra, Jialian Wu, Xiaodong Yu,  Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum"
        "description lang=en": "Learn how task-specific synthetic data can improve small language model performance and explore results from the LuminaSFT study."
        "keywords": "SLM, LLM, SFT"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Sudhanshu Ranjan, Jiang Liu, Gowtham Ramesh, Prakamya Mishra, Jialian Wu, Xiaodong Yu,  Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum"
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

# LuminaSFT: Generating Synthetic Fine-Tuning Data for Small Language Models

Small language models (SLMs) are emerging as a lightweight and cost-efficient alternative to large language models (LLMs). They significantly reduce inference costs and latency, and when carefully optimized for specific tasks, can approach—or even match—the performance of larger models. However, due to their limited parameter capacity, SLMs typically require stronger supervision to reach their full potential. Supervised fine-tuning (SFT) therefore plays a crucial role in enhancing their performance.

In this blog, we introduce [LuminaSFT](https://huggingface.co/collections/amd/luminasft), a synthetic SFT dataset specifically designed to improve both general-purpose and task-specific SLMs. LuminaSFT consists of multiple curated splits that target diverse capabilities:

- **[UltraChat200K-DeepSeek](https://huggingface.co/datasets/amd/UltraChat200K-regenerated)** – A regenerated base SFT dataset for broad instruction-following.
- **[InstructGPT-NaturalQA](https://huggingface.co/datasets/amd/InstructGpt-NaturalQa)** and **[InstructGPT-TriviaQA](https://huggingface.co/datasets/amd/InstructGpt-TriviaQa)** – Factual question answering datasets to strengthen knowledge recall and answer accuracy.
- **[CoT-Drop](https://huggingface.co/datasets/amd/Cot-Drop)** – A reading comprehension dataset with detailed reasoning traces to enhance multi-step reasoning.
- **[InstructGPT-Educational](https://huggingface.co/datasets/amd/InstructGpt-educational)** – A pedagogical QA dataset with step-by-step explanations to improve educational assistance.

Our experiments demonstrate that LuminaSFT supports both general-purpose fine-tuning and targeted task adaptation, offering measurable gains across multiple downstream benchmarks.

We also detail our data generation methodology and publicly release LuminaSFT where licensing permits. All experiments were conducted on AMD Instinct™ MI300X and MI250 GPUs using the AMD ROCm™ software stack.

## Key takeaways

- **We release [LuminaSFT](https://huggingface.co/collections/amd/luminasft)** — a supervised fine-tuning dataset comprising both general-purpose and multiple task-specific splits designed to improve small language model (SLM) performance.
- **Stronger teacher models can improve existing SFT data, but gains are task-dependent.** Regenerating responses with a stronger teacher model (e.g., DeepSeek-V3) yields measurable improvements—up to ~4% on average across multiple tasks. However, the magnitude of improvement depends heavily on the downstream task and the quality of the original prompts.
- **Task-specific data generation can deliver substantial performance gains.** In domains such as reading comprehension, targeted dataset construction boosts performance by as much as ~41%, highlighting the value of specialization for SLMs.
- **High-quality data can be generated even without seed datasets.** When no existing seed data is available, detailed task-specific prompts combined with multi-step generation strategies can still produce effective supervision signals.

## Dataset

LuminaSFT contains data spanning general-purpose instruction following, factual QA, reading comprehension, and educational QA. For general-purpose instruction following, [UltraChat200K-DeepSeek](https://huggingface.co/datasets/amd/UltraChat200K-regenerated) preserves the original UltraChat200K prompts and regenerates responses using DeepSeek-V3 as the teacher, yielding improvement in 5 out of 7 standard benchmarks. For general-purpose QA, [InstructGPT-NaturalQA](https://huggingface.co/datasets/amd/InstructGpt-NaturalQa) and [InstructGPT-TriviaQA](https://huggingface.co/datasets/amd/InstructGpt-TriviaQa) are each ~1M-sample datasets produced via self-instruct from the NaturalQA and TriviaQA train splits respectively, with DeepSeek-V3 as the teacher; when combined with a general-purpose SFT dataset, they improve accuracy by 2-4%. For reading comprehension, [CoT-Drop](https://huggingface.co/datasets/amd/Cot-Drop) augments the DROP train split with chain-of-thought reasoning chains generated by Qwen3-30B-A3B, boosting performance by up to +41.6%. For educational QA, [InstructGPT-Educational](https://huggingface.co/datasets/amd/InstructGpt-educational) is a fully synthetic dataset created through a multi-step pipeline (exams or tracks → topics → questions) using Qwen3-30B-A3B with no seed training data, achieving ~2.4% average improvement on MMLU, AGIEval, and MMLU-Pro.

## Method

Recent research has largely focused on building broad, general-purpose SFT datasets or datasets tailored to domains such as math and coding. With LuminaSFT, we explore two complementary research questions:

1. **How can stronger teacher models improve existing SFT datasets?**
2. **How can task-specific data generation enhance downstream performance?**

To address the first question, we investigate data regeneration. We retain the original prompts from widely used SFT datasets and regenerate responses using a stronger teacher model. This controlled setup allows us to isolate the impact of teacher quality, quantify performance improvements, and identify which task categories benefit most from regeneration.

To address the second question, we generate new task-specific datasets spanning general question answering, reading comprehension, and educational QA. For each task, we begin with a seed dataset and apply task-specific prompting strategies to synthesize high-quality supervision signals. This enables us to evaluate the effectiveness of targeted dataset construction for specialized SLM training.

## Data regeneration

We regenerate data for some commonly used SFT datasets to fine-tune a general purpose SLM. We use DeepSeek-V3 as the teacher model and Instella-3B-base as the student model for our experiments. We considered four popular open-source datasets: TuluV3 [1], UltraChat200K [2], MagpiePro-1M [3], and SmolTalk [4]. We use the same prompts as these datasets. For samples with multiple turns, we first generate the response to the initial prompt from the user and then use the first prompt from the user along with the generated answer and the next prompt to get the follow-up response from the teacher model. We do this for up to 3 turns. We set the maximum sequence length of samples while generation to be 4096. During preliminary experiments, we found that performance did not improve for MagpiePro-1M and SmolTalk, so we continued with the other datasets in subsequent experiments.

For evaluation, we consider 7 standard tasks: MMLU, GSM8k, IFEVAL, TruthfulQA, BBH, GPQA and MATH. The results are presented in table 1.

- We observe that average performance improves by ~4% in the case of the TuluV3 dataset and a minor improvement in the case of UltraChat200K.
- In the case of TuluV3, we observe a performance improvement in 6 out of 7 tasks, and in the case of UltraChat200K, we observe performance improvement in 5 out of 7 tasks.
- We observe that for mathematics and science related datasets (MATH, GSM8K, GPQA), there is usually an improvement, with the maximum improvement observed on GSM8K (~7%).

Overall, data regeneration methods can help improve the average performance, but the improvement relies on the downstream task as well as the prompts used while regenerating the data.

**Table 1: Results for data regeneration using DeepSeek-V3 as the teacher model.**
| Model                     | MMLU  | GSM8k | IFEval | TruthfulQA | BBH   | GPQA  | MATH  | Average |
|---------------------------|-------|-------|--------|------------|-------|-------|-------|---------|
| TuluV3                    | **57.91** | 59.14 | 56.56 | 46.27 | 39.33 | 20.54 | 13.64 | 41.91  |
| TuluV3-DeepSeek           | 57.61 | **66.19** | **58.96** | **51.54** | **41.88** | **24.11** | **20.64** | **45.85** |
| Improvement               | -0.30 | 7.05 | 2.40 | 5.27 | 2.55 | 3.57 | 7.00 | 3.94 |
| UltraChat200K             | 57.39 | 62.70 | 37.15 | 51.19 | 40.53 | 20.54 | 12.66 | 40.31 |
| UltraChat200K-DeepSeek    | 57.53 | 63.91 | 38.63 | 47.81 | 38.40 | 23.66 | 13.76 | 40.53 |
| Improvement               | 0.13 | 1.21 | 1.48 | -3.38 | -2.13 | 3.13 | 1.10 | 0.22 |

## Task-specific data generation

General-purpose datasets improve overall averages, but when the downstream task is known in advance, generating targeted datasets can lead to far larger improvements. In this section, we evaluate the benefits of task-specific data generation across several benchmarks. For the general purpose QA experiments, we use DeepSeek-V3 as the teacher and Instella-3B-base as the student. For the reading comprehension and educational QA experiments, we use Qwen3-30B-A3B as the teacher and Llama-1B as the student.

### Data generation for general purpose QA tasks

We consider two benchmarks for general purpose QA tasks: NaturalQA and TriviaQA.  We use the train split of NaturalQA and TriviaQA as the seed dataset and follow the self-instruct method for data generation. We generate ~1M questions for both. We use Instella-3B-base fine-tuned with TuluV3 as the baseline. During our initial experiments, we found that using only our synthetic data underperforms the model trained on Tulu-V3. When we fine-tune the student model on our dataset together with TuluV3, we observe improvements in performance. Table 2 shows that augmenting Tulu-V3 with our synthetic data improves the student model’s accuracy by ~2% on NaturalQA and ~4% on TriviaQA.

**Table 2: Results for general purpose QA tasks.**
| Model                 | NaturalQA | TriviaQA |
|-----------------------|-----------|----------|
| Instella-TuluV3       | 18.0      | 56.5     |
| Instella-NaturalQA    | **20.3**  | 55.5     |
| Instella-TriviaQA     | 15.8      | **60.5** |
| Max Absolute Improvement | 2.3    | 4.0      |
| Max Relative Improvement | 12.8   | 7.1      |

We further investigated the seed corpus for the NaturalQA task. More specifically, we use an embedding model to search for the answers to the test questions in the seed corpus. Once we have retrieved the top k documents, we query DeepSeek-V3 to check if the answer to the question is present in the seed corpus. We compare two seed datasets for the task: seed-1[5] and seed-2[6]. We find that seed-1 has answers to ~49% of the questions and seed-2 has answers to ~35% of the questions. Despite that, we observe performance improvement when using seed-2, but no performance gains when using seed-1. This might be due to a format mismatch between the seed dataset and test set. We leave further exploration of this topic to future work.

### Data generation for reading comprehension tasks

We consider two reading comprehension tasks: DROP and RACE. For both the datasets, we take the train split and generate chain-of-thought reasoning chains. We fine-tune Llama-1B for the two datasets and obtain Llama-Drop and Llama-Race. We observe that using dedicated reasoning chains can yield significant performance improvements over the baselines. Table 3 highlights the impact of dedicated reasoning chains: performance improves by as much as ~41% on DROP and ~9% on RACE compared to the baseline.

**Table 3: Results for reading comprehension tasks.**
| Model              | DROP   | RACE   |
|--------------------|--------|--------|
| Llama-TuluV3       | 23.30  | 70.43  |
| Llama-Drop         | **64.90** | 14.16  |
| Llama-Race         | 25.10  | **79.43** |
| Max improvement    | 41.60  | 9.00   |
| Max relative improvement | 178.54 | 12.77 |

### Data generation for educational QA tasks

We consider three educational tasks: MMLU, AGIEval (English) and MMLU-Pro. These benchmarks focus on educational tasks and don’t have corresponding train splits. We follow a data generation approach similar to [7], which uses publicly sourced material to build seed datasets. However, we rely on task-specific prompting rather than external data scraping.

We generate data using a multi-step pipeline and consider two possibilities:

Exams -> Topics -> QA

Tracks -> Subjects -> Topics -> QA

For exams, we consider two possibilities. The list of exams only includes competitive exams like SAT, GRE, etc. We denote the dataset generated using this method as competitive. For the next iteration, we consider all the competitive exams we considered previously along with AP and IB exams. We denote this dataset as an exam-all. For tracks, we specify the following tracks: STEM, humanities, and social sciences. We then generate data for each track at the high school and college levels. We denote this dataset as a track. We fine-tune Llama-1B for the three datasets and obtain Llama-competitive, Llama-exam-all, and Llama-track respectively. The results are presented in table 4, below.

**Table 4: Results for educational QA tasks.**
| Model               | AGIEval (English) | MMLU  | MMLU-Pro | Average |
|---------------------|-------------------|-------|----------|---------|
| Llama-TuluV3        | 46.20             | 54.16 | 24.13    | 41.50   |
| Llama-exam-all      | 46.22             | 55.48 | **29.36** | 43.69   |
| Llama-competitive   | **48.24**         | 54.29 | 29.14    | **43.89** |
| Llama-track         | 45.16             | **55.79** | 29.14 | 43.36   |
| Max improvement     | 2.04              | 0.13  | 5.01     | 2.39    |
| Relative improvement | 4.41             | 0.24  | 20.78    | 5.77    |

When comparing the three dataset variants, we observe consistent gains over the baseline. Table 4 highlights these improvements, showing an average gain of up to ~2.4% and the maximum improvement on MMLU-Pro. We observe that for MMLU, the best performing model is Llama-track. This might be because the data generation process for the corresponding dataset is closest to how the MMLU benchmark was initially constructed. Furthermore, Llama-competitive performs the best on AGIEval. This might be because the starting seed dataset (i.e., the list of competitive exams) is closest to how the benchmark was constructed. These results show that in the absence of seed datasets, final performance is highly sensitive to the choice of starting prompts and the design of the data generation pipeline.

## Summary

In this blog, we introduced LuminaSFT, an SFT dataset for SLMs. We conducted a study of synthetic data regeneration and data generation over several datasets spanning different types of tasks. Our results confirm that gains from teacher regeneration depend on both task and prompt design. Task-specific data generation yields much larger gains, up to 40%. Even without seed data, structured pipelines can bootstrap useful datasets. We invite the community to explore, experiment with, and build upon LuminaSFT. The [amd/LuminaSFT](https://huggingface.co/collections/amd/luminasft) dataset is open-sourced where possible, and we welcome feedback, benchmarking results, and contributions.

## Bias, Risks, and Limitations

- LuminaSFT is not intended for use cases that require high levels of factual accuracy, safety-critical decision making, or applications in health, medical, or legal domains. As the dataset is generated using large language models, it may contain factual inaccuracies, incomplete reasoning, or inconsistencies.
- The LuminaSFT datasets are released without any explicit safety guarantees. Users are responsible for conducting thorough evaluations, applying appropriate filtering, and performing task-specific risk assessments before using the data for training or fine-tuning models in downstream applications.
- Due to the nature of large-scale synthetic data generation, the dataset may include biased, misleading, toxic, harmful, or otherwise undesirable content. Such content may appear even when the generation prompts were not explicitly designed to produce it. Users are therefore encouraged to exercise caution and responsible judgment when using or redistributing the dataset.

## Contributors

**Core contributors:** Sudhanshu Ranjan, Jiang Liu, Gowtham Ramesh, Prakamya Mishra, Zicheng Liu, Emad Barsoum

**Contributors:** Jialian Wu, Xiaodong Yu, Yusheng Su, Ximeng Sun, Ze Wang

## Citations

Feel free to cite the Instella paper if you find it helpful to your work:

```text
@article{liu2025instella,
  title={Instella: Fully open language models with stellar performance},
  author={Liu, Jiang and Wu, Jialian and Yu, Xiaodong and Su, Yusheng and Mishra, Prakamya and Ramesh, Gowtham and Ranjan, Sudhanshu and Manem, Chaitanya and Sun, Ximeng and Wang, Ze and others},
  journal={arXiv preprint arXiv:2511.10628},
  year={2025}
}
```

## Endnotes

For experiments related to data regeneration and general-purpose QA, we use [DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) as the teacher model and [Instella-3B-base](https://huggingface.co/amd/Instella-3B) as the student model. For generation, we use the parameters recommended in the official release. For QA generation, we use a higher temperature of 1.0 for better diversity. All SFT experiments were done using the [official Instella training codebase](https://github.com/AMD-AGI/Instella/blob/main/configs/instella-3b-sft.yaml) with a context length of 4096. We use the [OLMES](https://github.com/allenai/olmes) framework for all evaluations.

AMD system configuration for above experiments: 2x Intel® Xeon® Platinum 8480C 48-core Processor (2 sockets, 48 cores per socket, 1 thread per core), AMD Instinct™ MI300X 8x GPU platform (192GB HBM3, 750W), 1.8 TiB RAM, 8x 3.5TB local SSD, 1 NUMA node per socket, Host OS Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-1086-azure, Host GPU driver amdgpu 6.12.12.

For experiments related to reading comprehension and educational QA, we use [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) as the teacher model and [Llama-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) as the student model. For generation, we use the parameters recommended in the official release. All SFT experiments were done using [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) with full SFT [[1](https://github.com/hiyouga/LlamaFactory/blob/v0.9.3/examples/train_full/llama3_full_sft.yaml), [2](https://github.com/hiyouga/LlamaFactory/blob/main/examples/deepspeed/ds_z2_config.json)] with a context length of 4096. We use the [OLMES](https://github.com/allenai/olmes) framework for all evaluations.

AMD system configuration for the above experiments: 2x AMD EPYC 7713 64-Core Processor, AMD Instinct MI250 8x GPU platform (64GB HBM2e per GCD, 560W TDP), 1.0 TiB RAM, 1 NUMA node per socket, Host OS Ubuntu 22.04.5 LTS with Linux kernel 6.2.0-39-generic, Host GPU driver amdgpu 6.3.6.

## References

[1] Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., et al. (2024). Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124. https://arxiv.org/abs/2411.15124

[2] Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M., & Zhou, B. (2023). Enhancing Chat Language Models by Scaling High-quality Instructional Conversations. arXiv:2305.14233. https://arxiv.org/abs/2305.14233

[3] Xu, Z., Jiang, F., Niu, L., Deng, Y., Poovendran, R., Choi, Y., & Lin, B. Y. (2025). Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing. Proceedings of the Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=Pnk7vMbznK

[4] Ben Allal, L., Lozhkov, A., Bakouch, E., Martín Blázquez, G., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Piqueres Lajarín, A., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., Raffel, C., von Werra, L., & Wolf, T. (2025). SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model. arXiv:2502.02737. https://arxiv.org/abs/2502.02737

[5] Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., & Petrov, S. (2019). Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics (TACL), 7, 453–466. https://doi.org/10.1162/tacl_a_00276

[6] Google Research. NQ-Open: Natural Questions Open Dataset. Hugging Face Datasets. https://huggingface.co/datasets/google-research-datasets/nq_open

[7] Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction Tuning with Human Curriculum. arXiv. https://arxiv.org/abs/2310.09518

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
