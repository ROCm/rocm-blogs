---
blogpost: true
blog_title: "Introducing Instella: New State-of-the-art Fully Open 3B Language Models"
date: 5 Mar 2025
author: 'Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu, Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum'
thumbnail: 'PR677_thumbnail_7-5.JPG'
tags: AI/ML, Fine-Tuning, Hardware
category: Applications & models
target_audience: AI developers, AI researchers, and AI enthusiasts
key_value_propositions: AMD's fully open 3B MODEL
language: English
myst:
    html_meta:
        "author": "Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu, Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum"
        "description lang=en": "AMD is excited to announce Instella, a family of fully open state-of-the-art 3-billion-parameter language models (LMs). , In this blog we explain how the Instella models were trained, and how to access them."
        "keywords": "Instella , MI300X, LLMs, ROCm"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and Models"
        "amd_developer_type": "ML/AI Developer"
        "amd_deployment": "Servers, Workstations"
        "amd_product_type": "Accelerators"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "Large Language Model (LLM)"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Thu Mar 5, 12:00:00 PST 2025
---

# Introducing Instella: New State-of-the-art Fully Open 3B Language Models

AMD is excited to announce Instella, a family of fully open state-of-the-art 3-billion-parameter language models (LMs) trained from scratch on AMD Instinct™ MI300X GPUs. Instella models outperform existing fully open models of similar sizes and achieve competitive performance compared to state-of-the-art open-weight models such as Llama-3.2-3B, Gemma-2-2B, and Qwen-2.5-3B, including their instruction-tuned counterparts.

Our journey with Instella builds upon the foundation laid by our previous 1-billion-parameter LMs, [AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html) which helped showcase the feasibility of training LMs end-to-end on AMD GPUs. With Instella, we have scaled our efforts by transitioning from a 1-billion-parameter model trained on 64 AMD Instinct MI250 GPUs using 1.3T tokens to a 3-billion-parameter model trained on 128 Instinct MI300X GPUs using 4.15T tokens. While we compared our previous model with similarly sized fully open models only, Instella not only surpasses existing fully open models but also achieves overall competitive performance as compared to state-of-the-art open-weight models (Figure 1 [^1].), marking a significant step in bridging this gap.

```{figure} ./images/scaling_perf_instruct.png
:align: center
:alt: Scaling performance
Figure 1: Comparing Instella Performance: Pareto frontier of pre-training tokens vs average performance for pre-trained and instruction-tuned models.
```

By training Instella from scratch on Instinct MI300X GPUs, we highlight our hardware’s capability and scalability in handling demanding large-scale AI training workloads, offering a viable alternative in the AI hardware landscape. In line with AMD's commitment to open source, we are releasing all artifacts related to Instella models [here](#additional-resources), including the model weights, detailed training configurations, datasets, and code, enabling the AI community to collaborate, replicate, and innovate, thereby accelerating progress.

This blog will introduce you to our new family of Instella LMs. You will find out how to access these new models, learn, in details, how we trained them, and see how AMD's new Instella LMs benchmark with other models. Follow the [Additional Resources](#additional-resources) section to get started with using Instella models.

## Takeaways

- **Announcing Instella**, a series of 3 billion parameter language models developed by AMD, trained from scratch on 128 Instinct MI300X GPUs.
- **Instella models significantly outperform existing fully open LMs** (Figure 1) of comparable size, as well as bridge the gap between fully open and open weight models by achieving competitive performance compared state-of-the-art open weight models and their instruction-tuned counterparts.
- Fully open and accessible: **Fully open-source release of model weights, training hyperparameters, datasets, and code**, fostering innovation and collaboration within the AI community.  
- Supported by the AMD ROCm software stack, Instella employs efficient training techniques such as **FlashAttention-2, Torch Compile, and Fully Sharded Data Parallelism (FSDP)** with hybrid sharding to **scale model training over a large cluster.**

## Instella Models

In this release, we introduce the following Instella models (Table 2):

<div align="center">
Table 1: Instella models and training stages.
</div>

| Model  | Stage | Training Data (Tokens) | Description |
| :----: | :----: | :----: | :---- |
| [Instella-3B-Stage1](https://huggingface.co/amd/Instella-3B-Stage1)  | Pre-training (Stage 1) | 4.065 Trillion | First stage pre-training to develop proficiency in natural language. |
| [Instella-3B](https://huggingface.co/amd/Instella-3B)  | Pre-training (Stage 2) | 57.575 Billion | Second stage pre-training to further enhance problem solving capabilities. |
| [Instella-3B-SFT](https://huggingface.co/amd/Instella-3B-SFT)  | SFT | 8.902 Billion (x3 epochs) | Supervised Fine-tuning (SFT) to enable instruction-following capabilities. |
| [Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-instruct)  | DPO | 760 Million | Alignment to human preferences and strengthen chat capabilities with direct preference optimization (DPO). |
|  | **Total:** | **4.15 Trillion** |  |

The Instella models are text-only, autoregressive transformer-based LMs having 3 billion parameters. Architecture-wise, Instella is packed with 36 decoder layers, each having 32 attention heads. These models support a sequence length of up to 4,096 tokens and have a vocabulary size of ~50,000 tokens using the OLMo tokenizer[^2]. During both pre-training and fine-tuning, we utilized FlashAttention-2[^3], Torch Compile, and bfloat16 mixed-precision training to reduce memory usage, leading to computational speedups and optimal resource utilization. To balance inter-node memory efficiency and intra-node communication overhead within our cluster, we employed fully sharded data parallelism (FSDP) with hybrid sharding, with model parameters, gradients, and optimizer states sharded within a node and replicated across the nodes.

Our training pipeline is based on the open-sourced OLMo codebase, adapted, and optimized for our hardware and model architecture. For pre-training we used a total of 128 Instinct MI300X GPUs distributed across 16 nodes with each node having 8x Instinct MI300X GPUs. We evaluated our models and baselines using standard tasks from [OLMES](https://github.com/allenai/olmes/tree/main), [FastChat MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md), and [Alpaca](https://github.com/tatsu-lab/alpaca_eval/tree/main). For more details about the architecture, training hyperparameters and evaluations, please refer to our [huggingface model card](https://huggingface.co/amd/Instella-3B) and our [Github repository](https://github.com/AMD-AIG-AIMA/Instella).

## Training Pipeline

```{figure} ./images/Instella_Train_Pipeline.png
:align: center
:alt: Training pipeline
Figure 2: Instella model training pipeline.
```

The training of the Instella models comprised of four stages (Figure 2), where each stage incrementally enhanced the model’s capabilities from fundamental natural language understanding to instruction following and alignment towards human preferences. In this section we will briefly present Instella's two pre-training stages and two instruction tuning & alignment stages, and their benchmark results.

### Two stage pre-training

In the first pre-training stage, we trained the model from scratch on **4.065 trillion tokens** sourced from [OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924)[^4], which is a diverse mix of two high-quality datasets [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)[^5] and [Dolma 1.7](https://huggingface.co/datasets/allenai/dolma)[^6] covering domains like coding, academics, mathematics, and general world knowledge from web crawl. This extensive first stage pre-training established a foundational understanding of general language in our Instella model.

For our final pre-trained checkpoint, **[Instella-3B](https://huggingface.co/amd/Instella-3B)**, we conducted a **second stage pre-training on top of the first-stage [Instella-3B-Stage1](https://huggingface.co/amd/Instella-3B-Stage1) model** to further enhance its capabilities specifically in MMLU, BBH, and GSM8k. To accomplish this, we further trained the model on an **additional 57.575 billion tokens** sourced from high-quality and diverse datasets, specifically from [Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)[^2], [SmolLM-Corpus (python-edu)](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/viewer/python-edu)[^7], the [Deepmind Mathematics](https://github.com/google-deepmind/mathematics_dataset)[^8], and conversational datasets including [Tülu-3-SFT-Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)[^9], [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)[^10], [WebInstructSub](https://huggingface.co/datasets/TIGER-Lab/WebinstructSub)[^11], [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)[^12], and [Ultrachat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)[^13].

In addition to these publicly available datasets, **28.5 million tokens** out of our second stage pre-training data-mix were derived **from our in-house synthetic dataset focusing on mathematical problems**. This synthetic dataset was generated using the training set of GSM8k dataset, where we first used Qwen2.5-72B-Instruct to 1) Abstract numerical values as function parameters and generate a Python program to solve the math question, 2) Identify and replace numerical values in the existing question with alternative values that are still answerable with the same python program solution as the original question. Next, by assigning different new values to these Python parameters and using the abstract solution program to compute the corresponding answers, we expanded our synthetic dataset with new and reliable question-answer pairs[^14]. The conversational datasets in this second stage pre-training data-mix were reformatted by concatenating question-answer pairs to be used for pre-training. We trained the model on this data-mix three times with different random seeds and combined the model weights to obtain the final pre-trained model, [Instella-3B](https://huggingface.co/amd/Instella-3B).

<html>
<head>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      padding: 20px;
      max-width: 100%;
      overflow-x: auto;
    }
    .table-wrapper {
      max-width: 100%;
      overflow-x: auto;
      margin: 20px 0;
      box-shadow: 0 4px 6px light-dark(rgba(0,0,0,0.1), rgba(255,255,255,0.1));
      border-radius: 6px;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      background-color: light-dark(white, black);
      border: 1px solid light-dark(#ddd, #ffffff);
    }
    th, td {
      padding: 12px 15px;
      text-align: center;
      border: 1px solid light-dark(#ddd, #f4f4f4);
      font-size: 14px;
    }
    thead {
      background-color: light-dark(#f5f7fa, #37383a);
    }
    thead th {
      position: sticky;
      top: 0;
      background-color: light-dark(#f5f7fa, #5b5b5b);
      font-weight: 600;
      color: light-dark(#000000, #ffffff);
      border: 1px solid light-dark(#ddd, #ffffff);
    }
    tbody th[colspan] {
      background-color: light-dark(#edf2f7, #383D46);
      font-weight: 600;
      text-align: left;
    }
    td strong {
      font-weight: bold;
      color: light-dark(#000000, #ffffff);
    }
    td ins {
      text-decoration: underline;
      color: light-dark(#000000, #ffffff);
    }
    tr:hover {
      background-color: light-dark(#f9fafb, #878a90);
    }
    tr {
      border-bottom: 1px solid light-dark(#ddd, #ffffff);
    }
    td, th {
      border-right: 1px solid light-dark(#ddd, #ffffff);
    }
    td:last-child, th:last-child {
      border-right: 1px solid light-dark(#ddd, #ffffff);
    }
  </style>
</head>
<body>
<div class="table-wrapper" align="center">
    <em><strong>Table 2:</strong> Pre-trained model performance on standard benchmarks. Here <strong>Bold</strong> represents the best performance, and <ins>Underscore</ins> represents the second best performance.</em>
  <table>
    <thead>
      <tr>
        <th>Models</th>
        <th>Size</th>
        <th>Training Tokens</th>
        <th>Avg</th>
        <th>ARC Challenge</th>
        <th>ARC Easy</th>
        <th>BoolQ</th>
        <th>Hellaswag</th>
        <th>PiQA</th>
        <th>SciQ</th>
        <th>Winnograde</th>
        <th>OpenBookQA</th>
        <th>MMLU</th>
        <th>BBH (3-shot)</th>
        <th>GSM8k (8-shot)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th colspan="15">Open Weight Models</th>
      </tr>
      <tr>
        <td>Gemma-2-2B</td>
        <td>2.61B</td>
        <td>~2T</td>
        <td>59.34</td>
        <td>39.46</td>
        <td>59.30</td>
        <td>74.50</td>
        <td>70.50</td>
        <td>76.40</td>
        <td><strong>96.60</strong></td>
        <td>69.80</td>
        <td>44.80</td>
        <td>53.28</td>
        <td>40.75</td>
        <td>27.37</td>
      </tr>
      <tr>
        <td>Llama-3.2-3B</td>
        <td>3.21B</td>
        <td>~9T</td>
        <td>62.51</td>
        <td>47.16</td>
        <td>64.91</td>
        <td>74.80</td>
        <td>73.10</td>
        <td>75.90</td>
        <td>95.30</td>
        <td>70.30</td>
        <td>51.20</td>
        <td>57.81</td>
        <td><ins>47.00</ins></td>
        <td>30.10</td>
      </tr>
      <tr>
        <td>Qwen2.5-3B</td>
        <td>3.09B</td>
        <td>~18T</td>
        <td><strong>68.30</strong></td>
        <td>51.51</td>
        <td>67.19</td>
        <td><strong>79.10</strong></td>
        <td>72.10</td>
        <td>77.40</td>
        <td>95.50</td>
        <td>69.30</td>
        <td><ins>51.40</ins></td>
        <td><strong>67.22</strong></td>
        <td><strong>56.69</strong></td>
        <td><strong>63.84</strong></td>
      </tr>
      <tr>
        <th colspan="15">Fully Open Models</th>
      </tr>
      <tr>
        <td>Pythia-2.8b</td>
        <td>2.91B</td>
        <td>300B</td>
        <td>49.83</td>
        <td>40.47</td>
        <td>60.70</td>
        <td>64.80</td>
        <td>60.10</td>
        <td>72.50</td>
        <td>89.70</td>
        <td>60.80</td>
        <td>42.60</td>
        <td>26.09</td>
        <td>27.69</td>
        <td>2.73</td>
      </tr>
      <tr>
        <td>GPTNeo-2.7B</td>
        <td>2.72B</td>
        <td>~420B</td>
        <td>47.96</td>
        <td>38.46</td>
        <td>54.56</td>
        <td>62.70</td>
        <td>55.20</td>
        <td>70.80</td>
        <td>88.00</td>
        <td>58.30</td>
        <td>40.80</td>
        <td>27.83</td>
        <td>27.25</td>
        <td>3.71</td>
      </tr>
      <tr>
        <td>OpenELM-3B</td>
        <td>3.04B</td>
        <td>~1.5T</td>
        <td>52.28</td>
        <td>37.46</td>
        <td>58.42</td>
        <td>68.60</td>
        <td>71.70</td>
        <td>75.60</td>
        <td>92.50</td>
        <td>65.40</td>
        <td>46.40</td>
        <td>26.69</td>
        <td>29.40</td>
        <td>2.96</td>
      </tr>
      <tr>
        <td>StableLM-3B-4E1T</td>
        <td>2.8B</td>
        <td>~4T</td>
        <td>58.51</td>
        <td>44.82</td>
        <td>67.02</td>
        <td>75.40</td>
        <td><ins>74.20</ins></td>
        <td><strong>78.40</strong></td>
        <td>93.40</td>
        <td>68.40</td>
        <td>48.60</td>
        <td>45.19</td>
        <td>37.33</td>
        <td>10.84</td>
      </tr>
      <tr>
        <td><strong><a href="https://huggingface.co/amd/Instella-3B-Stage1">Instella-3B-Stage1</a></strong></td>
        <td>3.11B</td>
        <td>~4T</td>
        <td>61.33</td>
        <td><strong>53.85</strong></td>
        <td><strong>73.16</strong></td>
        <td><ins>78.70</ins></td>
        <td><ins>74.20</ins></td>
        <td>77.50</td>
        <td>94.90</td>
        <td><ins>71.20</ins></td>
        <td><ins>51.40</ins></td>
        <td>54.69</td>
        <td>34.30</td>
        <td>10.77</td>
      </tr>
      <tr>
        <td><strong><a href="https://huggingface.co/amd/Instella-3B">Instella-3B</a></strong></td>
        <td>3.11B</td>
        <td>~4T+60B</td>
        <td><ins>66.59</ins></td>
        <td><ins>52.84</ins></td>
        <td><ins>70.53</ins></td>
        <td>76.50</td>
        <td><strong>75.00</strong></td>
        <td><ins>77.80</ins></td>
        <td><ins>96.40</ins></td>
        <td><strong>73.10</strong></td>
        <td><strong>52.40</strong></td>
        <td><ins>58.31</ins></td>
        <td>39.74</td>
        <td><ins>59.82</ins></td>
      </tr>
    </tbody>
  </table>
</div>
</body>
</html>

#### Pre-training Results

- Both Instella-3B-Stage1 & Instella-3B models outperform all the other fully open models over all the benchmarks individually (except PIQA) (Table 2). **Our final pre-trained checkpoint Instella-3B outperforms the existing top performant fully open pre-trained models by a lead of ⬆️8.08% on average**, with significant improvements in `ARC Challenge [+8.02%], ARC Easy [+3.51%], Winnograde [+4.7%], OpenBookQA [+3.88%], MMLU [+13.12%] and ️GSM8K [+48.98%]`.  
- **Second stage pre-training elevated the overall average performance relative to stage-1 by ⬆️5.26%**, substantially narrowing the performance gap between Instella-3B model vs the closed-source models, and **outperforming Llama-3.2-3B by ⬆️4.08% on average** (`+5.69% [ARC Challenge], +5.61% [ARC Easy], and +29.72% [GSM8k]`), **Gemma-2-2B by ⬆️7.25% on average** (`+13.38% [ARC Challenge], +11.23% [ARC Easy], +4.5% [Hellaswag], +7.6% [OpenBookQA], +5.03% [MMLU], and +32.45% [GSM8k]`), and is **competitive with Qwen-2.5-3B** on the majority of the benchmarks.  
- The multi-stage pre-training with diverse and high-quality data mix significantly enhanced Instella-3B’s capabilities, establishing it as a competitive and open alternative in the landscape of comparable size language models.

### Instruction Tuning & Alignment

The supervised fine-tuning stage was done to enhance the Instella-3B base pre-trained model’s ability to follow instructions and respond to user queries. **Instella-3B-SFT was supervised fine-tuned using Instella-3B as the base model** and training with 8.9 billion tokens of high-quality instruction-response pairs data for three epochs. The primary objective was to improve the base model’s performance in interactive settings, making it better suited for tasks requiring understanding and executing user commands. During this phase, we utilized curated datasets that spanned across a broad spectrum of tasks and domains, ensuring that the model could generalize across various instruction types. This data-mix was selectively soured from [SmolTalk (1.04M samples)](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)[^15], [OpenMathinstruct-2 (1M subset)](https://huggingface.co/datasets/nvidia/OpenMathinstruct-2)[^16], [Tulu 3 Instruction Following (30k samples)](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)[^9], [MMLU auxiliary train set](https://huggingface.co/datasets/cais/mmlu)[^17], and [o1-journey](https://huggingface.co/datasets/GAIR/o1-journey)[^18].

In the final training stage, we focused on aligning the Instella-3B-SFT model with human preferences to ensure that its outputs are helpful, accurate, and safe. Using Instella-3B-SFT as the base model, **Instella-3B-Instruct was trained with Direct Preference Optimization (DPO)[^19] on 0.76 billion tokens** sourced from [OLMo 2 1124 7B Preference Mix](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix)[^2]. This alignment process was essential for tailoring the model’s responses to be more in line with human values and expectations, thereby enhancing the quality and reliability of its outputs.

<div class="table-wrapper" align="center">
    <em><strong>Table 3:</strong> Instruct model performance on standard benchmarks. Here <strong>Bold</strong> represents the best performance, and <ins>Underscore</ins> represents the second best performance.</em>
  <table>
    <thead>
      <tr>
        <th>Models</th>
        <th>Size</th>
        <th>Training Tokens</th>
        <th>Avg</th>
        <th>MMLU</th>
        <th>TruthfulQA</th>
        <th>BBH</th>
        <th>GPQA</th>
        <th>GSM8K</th>
        <th>Minerva MATH</th>
        <th>IFEval</th>
        <th>AlpacaEval 2</th>
        <th>MT-Bench</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th colspan="13">Open Weight Models</th>
      </tr>
        <tr>
        <td>Gemma-2-2B-Instruct</td>
        <td>2.61B</td>
        <td>~2T</td>
        <td>39.04</td>
        <td>58.35</td>
        <td><ins>55.76</ins></td>
        <td>42.96</td>
        <td>25.22</td>
        <td>53.45</td>
        <td>22.48</td>
        <td>55.64</td>
        <td><strong>29.41</strong></td>
        <td><strong>8.07</strong></td>
      </tr>
      <tr>
        <td>Llama-3.2-3B-Instruct</td>
        <td>3.21B</td>
        <td>~9T</td>
        <td><ins>47.53</ins></td>
        <td><ins>61.50</ins></td>
        <td>50.23</td>
        <td><strong>61.50</strong></td>
        <td><ins>29.69</ins></td>
        <td><strong>77.03</strong></td>
        <td><ins>46.00</ins></td>
        <td><strong>75.42</strong></td>
        <td>19.31</td>
        <td>7.13</td>
      </tr>
      <tr>
        <td>Qwen2.5-3B-Instruct</td>
        <td>3.09B</td>
        <td>~18T</td>
        <td><strong>48.72</strong></td>
        <td><strong>66.90</strong></td>
        <td><strong>57.16</strong></td>
        <td><ins>57.29</ins></td>
        <td>28.13</td>
        <td><ins>75.97</ins></td>
        <td><strong>60.42</strong></td>
        <td>62.48</td>
        <td><ins>22.12</ins></td>
        <td><ins>8.00</ins></td>
      </tr>
      <tr>
        <th colspan="13">Fully Open Models</th>
      </tr>
      <tr>
        <td>StableLM-zephyr-3B</td>
        <td>2.8B</td>
        <td>4T</td>
        <td>30.50</td>
        <td>45.10</td>
        <td>47.90</td>
        <td>39.32</td>
        <td>25.67</td>
        <td>58.38</td>
        <td>10.38</td>
        <td>34.20</td>
        <td>7.51</td>
        <td>6.04</td>
      </tr>
      <tr>
        <td>OpenELM-3B-Instruct</td>
        <td>3.04B</td>
        <td>~1.5T</td>
        <td>14.11</td>
        <td>27.36</td>
        <td>38.08</td>
        <td>24.24</td>
        <td>18.08</td>
        <td>1.59</td>
        <td>0.38</td>
        <td>16.08</td>
        <td>0.21</td>
        <td>1.00</td>
      </tr>
      <tr>
        <td><a href="https://huggingface.co/amd/Instella-3B-SFT">Instella-3B-SFT</a></td>
        <td>3.11B</td>
        <td>~4T</td>
        <td>42.05</td>
        <td>58.76</td>
        <td>52.49</td>
        <td>46.00</td>
        <td>28.13</td>
        <td>71.72</td>
        <td>40.50</td>
        <td>66.17</td>
        <td>7.58</td>
        <td>7.07</td>
      </tr>
      <tr>
        <td><a href="https://huggingface.co/amd/Instella-3B-Instruct">Instella-3B-Instruct</a></td>
        <td>3.11B</td>
        <td>~4T</td>
        <td>44.87</td>
        <td>58.90</td>
        <td>55.47</td>
        <td>46.75</td>
        <td><strong>30.13</strong></td>
        <td>73.92</td>
        <td>42.46</td>
        <td><ins>71.35</ins></td>
        <td>17.59</td>
        <td>7.23</td>
      </tr>
    </tbody>
  </table>
</div>

#### Instruction Tuning Results

- **Instella-3B-Instruct model consistently outperforms other fully open models across all evaluated benchmarks with a significant average score lead of ⬆️ 14.37%** w.r.t the next top performing fully open instruction-tuned models (Table 3). With substantial margins across all the chat benchmarks (`+13% [MMLU], 7.57% [TruthfulQA], 7.43% [BBH], +4.46% [GPQA], +37.15 [IFEval], 10.08% [Alpaca 2], and 1.2% [MT-Bench]`).  
- **Instella-3B-Instruct narrows the performance gap with leading open-weight models.** Instella-3B-Instruct performs **on par with or slightly surpasses existing state-of-the-art open weight instruction-tuned models** such as Llama-3.2-3B-Instruct (`+5.24% [TruthfulQA], 0.45% [GPQA], and +0.1% [MT-Bench]`), and Qwen2.5-3B-Instruct (`+2.01% [GPQA] and +8.87% [IFEval]`), while significantly outperforming Gemma-2-2B-Instruct with an average score lead of ⬆️5.83% (`+0.55% [MMLU], +3.79 [BBH], +4.91 [GPQA], +20.47 [GSM8k], +19.98 [Minerva MATH], and +15.17% [IFEval]`).
- **Overall, Instella-3B-Instruct excels in instruction following tasks and multi-turn QA tasks like TruthfulQA, GPQA, IFEval and MT-Bench**, while being highly competitive compared to existing state-of-the-art open weight models on other knowledge recall and math benchmarks, while being trained on significantly fewer training tokens.

## Summary

The release of the Instella family of models represents a significant stride in advancing open-source AI and demonstrating the capabilities of AMD hardware in large-scale language model training. The 3 billion parameter models from Instella family significantly outperform present fully open comparable size models in key benchmarks while also being competitive to comparable open-weight models, which we attribute to the high-quality data-mix selection, multi-stage training pipeline, and the use of high-performance Instinct MI300X GPUs for large scale training.

By fully open sourcing the Instella models, including weights, training configurations, datasets, and code, we aim to foster innovation and collaboration within the AI community. We believe that transparency, reproducibility and accessibility are key drivers of progress in AI research and development. We invite developers, researchers, and AI enthusiasts to explore Instella, contribute to its ongoing improvement, and join us in pushing the boundaries of what is possible with language models.

We will continue enhancing the models across multiple dimensions, including context length, reasoning ability, and multimodal capabilities. Additionally, we will scale up both the model and dataset while exploring diverse architectural approaches. Keep your eyes peeled for more exciting blogs on the Instella LMs family, its features and capabilities!

## Additional Resources

### Hugging face Model Cards

- Pre-trained models:
  - Instella-3B-Stage1: [amd/Instella-3B-Stage1](https://huggingface.co/amd/Instella-3B-Stage1), First stage pre-training checkpoint.
  - Instella-3B: [amd/Instella-3B](https://huggingface.co/amd/Instella-3B), Final pre-training checkpoint.
- Instruction-tuned models:
  - Instella-3B-SFT: [amd/Instella-3B-SFT](https://huggingface.co/amd/Instella-3B-SFT), Supervised fine-tuned checkpoint.
  - Instella-3B-Instruct: [amd/Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct), Final Instruction-tuned checkpoint.

### Datasets

Second stage pre-training GSM8k synthetic dataset: [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic)

- The dataset consists of two splits: “train” and “train_119K”.
- For Instella-3B model second stage pre-training we used the “train_119K” split, which is a subset of the larger “train” split.

### Code

- Github: [https://github.com/AMD-AIG-AIMA/Instella](https://github.com/AMD-AIG-AIMA/Instella)

Please refer to the following blogs to get started with using these techniques on AMD GPUs:

- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)
- [Accelerating Large Language Models with Flash Attention on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)
- [Accelerate PyTorch Models using torch.compile on AMD GPUs with ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)
- [Introducing the First AMD 1B Language Models: AMD OLMo](https://www.amd.com/en/developer/resources/technical-articles/introducing-the-first-amd-1b-language-model.html)

## Bias, Risks, and Limitations

- The models are being released for research purposes only and are not intended for use cases that require high levels of factuality, safety critical situations, health, or medical applications, generating false information, facilitating toxic conversations.
- Model checkpoints are made accessible without any safety promises. It is crucial for users to conduct comprehensive evaluations and implement safety filtering mechanisms as per their respective use cases.
- It may be possible to prompt the model to generate content that may be factually inaccurate, harmful, violent, toxic, biased, or otherwise objectionable. Such content may also get generated by prompts that did not intend to produce output as such. Users are thus requested to be aware of this and exercise caution and responsible thinking when using the model.
- Multi-lingual abilities of the models have not been tested and thus may misunderstand and generate erroneous responses across different languages.

## License

- The Instella-3B models are licensed for academic and research purposes under a ResearchRAIL license.
- The [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic) dataset used in second stage pre-training is built with Qwen2.5-72B-Instruct, and is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic/blob/main/LICENSE) and [NOTICES](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic/blob/main/NOTICES) in the [amd/Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic) dataset card files for more information.
- Refer to the [LICENSE](https://huggingface.co/amd/Instella-3B/blob/main/LICENSE) and [NOTICES](https://huggingface.co/amd/Instella-3B/blob/main/NOTICES) files for more information.

## Contributors

> **Core contributors**:
> Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu
>
> **Contributors:**
> Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum

## Citations

Feel free to cite our Instella-3B models:

```text
@misc{Instella,
    title = {Instella: Fully Open Language Models with Stellar Performance},
    url = {https://huggingface.co/amd/Instella-3B},
    author = {Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu, Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum},
    month = {March},
    year = {2025}
}
```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

[^1]: For instruction-tuned models, we used pre-training tokens for comparison since 1) the exact numbers for instruct models for open weight models are unknown, and 2) adding instruct model training tokens (in billions) leads to marginally insignificant shift in trends.
[^2]: OLMo Team, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia et al. "[2 OLMo 2 Furious.](https://arxiv.org/abs/2501.00656)" arXiv preprint arXiv:2501.00656 (2024).
[^3]: Dao, Tri. "[Flashattention-2: Faster attention with better parallelism and work partitioning.](https://arxiv.org/abs/2307.08691)" arXiv preprint arXiv:2307.08691 (2023).
[^4]: Muennighoff, Niklas, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi et al. "OLMoE: Open mixture-of-experts language models." arXiv preprint arXiv:2409.02060 (2024).
[^5]: Li, Jeffrey, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal et al. "DataComp-LM: In search of the next generation of training sets for language models." arXiv preprint arXiv:2406.11794 (2024).
[^6]: [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://aclanthology.org/2024.acl-long.840/) (Soldaini et al., ACL 2024)
[^7]: Allal, Loubna Ben, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. "[Smollm-blazingly fast and remarkably powerful.](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)" Hugging Face Blog (2024).
[^8]: Saxton, David, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. "[Analysing mathematical reasoning abilities of neural models.](https://arxiv.org/abs/1904.01557)" arXiv preprint arXiv:1904.01557 (2019).
[^9]: Lambert, Nathan, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda et al. "[Tulu 3: Pushing Frontiers in Open Language Model Post-Training.](https://arxiv.org/abs/2411.15124)" arXiv preprint arXiv:2411.15124 (2024).
[^10]: Teknium. "[Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants](https://huggingface.co/datasets/teknium/OpenHermes-2.5)" 2023. URL https://huggingface.co/datasets/teknium/OpenHermes-2.5.
[^11]: Yue, Xiang, Tianyu Zheng, Ge Zhang, and Wenhu Chen. "[Mammoth2: Scaling instructions from the web.](https://proceedings.neurips.cc/paper_files/paper/2024/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)" Advances in Neural Information Processing Systems 37 (2025): 90629-90660.
[^12]: Zheng, Tianyu, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue. "[Opencodeinterpreter: Integrating code generation with execution and refinement.](https://arxiv.org/abs/2402.14658)" arXiv preprint arXiv:2402.14658 (2024).
[^13]: Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://aclanthology.org/2023.emnlp-main.183/). In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3029–3051, Singapore. Association for Computational Linguistics.
[^14]: Yu, Xiaodong, Ben Zhou, Hao Cheng, and Dan Roth. "[ReasonAgain: Using Extractable Symbolic Programs to Evaluate Mathematical Reasoning.](https://arxiv.org/abs/2410.19056)" arXiv preprint arXiv:2410.19056 (2024).
[^15]: Allal, Loubna Ben, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti et al. "[SmolLM2: When Smol Goes Big--Data-Centric Training of a Small Language Model.](https://arxiv.org/abs/2502.02737)" arXiv preprint arXiv:2502.02737 (2025).
[^16]: Toshniwal, Shubham, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. "[Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data.](https://arxiv.org/abs/2410.01560)" arXiv preprint arXiv:2410.01560 (2024).
[^17]: Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. "[Measuring massive multitask language understanding.](https://arxiv.org/abs/2009.03300)" arXiv preprint arXiv:2009.03300 (2020).
[^18]: Qin, Yiwei, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye et al. "[O1 Replication Journey: A Strategic Progress Report--Part 1.](https://arxiv.org/abs/2410.18982)" arXiv preprint arXiv:2410.18982 (2024).
[^19]: Rafailov, Rafael, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. "[Direct preference optimization: Your language model is secretly a reward model.](https://arxiv.org/abs/2305.18290)" Advances in Neural Information Processing Systems 36 (2024).
