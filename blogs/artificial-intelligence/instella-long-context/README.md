---
blogpost: true
blog_title: "Introducing Instella-Long: A Fully Open Language Model with Long-Context Capability"
date: 11 June 2025
author: 'Jialian Wu, Jiang Liu, Sudhanshu Ranjan, Xiaodong Yu, Gowtham Ramesh, Prakamya Mishra, Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum'
thumbnail: 'Image_Instella_Long_Context.png'
tags: AI/ML
category: Applications & models
target_audience: AI enthusiasts and developers
key_value_propositions: long-context language model continually trained from Instella-3B-Instruct on AMD Instinct™ MI300X GPUs
language: English
myst:
    html_meta:
        "author": "Jialian Wu, Jiang Liu, Sudhanshu Ranjan, Xiaodong Yu, Gowtham Ramesh, Prakamya Mishra, Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum"
        "description lang=en": "Learn about Instella-Long: AMD’s open 3B language model supporting 128K context, trained on MI300X GPUs, outperforming peers on long-context benchmarks."
        "keywords": "Instella-Longs, AMD GPU"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Jialian Wu, Jiang Liu, Sudhanshu Ranjan, Xiaodong Yu, Gowtham Ramesh, Prakamya Mishra, Zicheng Liu, Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum"
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

# Introducing Instella-Long: A Fully Open Language Model with Long-Context Capability

AMD is excited to announce
[Instella-Long](https://huggingface.co/amd/Instella-3B-Long-Instruct), a
long-context language model continually trained from
[Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-instruct)
on AMD Instinct™ MI300X GPUs. To our knowledge, Instella-Long makes [Instella](https://huggingface.co/collections/amd/instella-67c8a2c56e9198c85a97dd08) series the first fully open language model trained from scratch that supports long-context. Instella-Long can support 128K context length and achieve competitive performance outperforming open-weights models such as Phi-3.5-mini [^1], Gemma-3-4B [^2], and Qwen2.5-3B [^3] on the long-context benchmark.

By training Instella with long context extension on Instinct MI300X GPUs, we highlight our hardware’s capability and scalability in handling demanding AI training workloads, offering a viable alternative in the AI hardware landscape. In line with the AMD commitment to open source, we are sharing all the model weights, detailed training configurations, datasets, and code, enabling the AI community to collaborate, replicate, and innovate, thereby accelerating progress.

## Key Takeaways

- **Announcing Instella-Long**, a 3B long-context language model with
  128K context length support developed by AMD, trained on 64 Instinct
  MI300X GPUs.

- To our knowledge, Instella-Long makes the Instella series the first **fully open language model**
  trained from scratch that supports long-context. **[Huggingface model](https://huggingface.co/amd/Instella-3B-Long-Instruct), [training data](https://huggingface.co/datasets/amd/Instella-Long), and
  [training code](https://github.com/AMD-AIG-AIMA/Instella/tree/instella-long) are fully open-sourced.**

- Supported by the AMD ROCm software stack, Instella-Long employs efficient training techniques such as **Sequence Parallelism,
  FlashAttention-2 [^4], Torch Compile, and FSDP** to distribute
  **model training over 8 MI300 nodes each with 8 GPUs.**

## Instella-Long

[Instella-Long](https://huggingface.co/amd/Instella-3B-Long-Instruct) is
based on the [Instella
model](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html)
released in March. Specifically, Instella-Long is continually trained
from
[Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-instruct)
and follows the same model architecture. The training of Instella-Long
comprises three stages: 1. Continued Pre-Training, 2. Supervised
Finetuning (SFT), 3. Direct Preference Optimization (DPO).

## Continued Pre-Training

**Training**: We employ a two-phase pre-training starting from Instella-3B-Instruct (4K context length).

**Phase 1**: We extend the context length from 4,096 to 65,536 tokens and train the model using 20B tokens. We follow the [RoPE scaling
law](https://www.gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models)
to increase the base frequency of RoPE [^5] from 10,000 to 514,640.

**Phase 2**: As indicated by [Prolong](https://arxiv.org/pdf/2410.02660) [^6], it is beneficial to train the model with the data whose context length is longer than the target context length. In this phase, we train the model on 20B tokens with a maximum context length of 262,144 - 2× the target context length of 128K. Following the RoPE scaling law, we increase the RoPE base frequency to 3,691,950.

**Data**: Our continued pre-training data originates from the data mix
created by [Prolong](https://arxiv.org/pdf/2410.02660). We use the text
data curated by Prolong and tokenize the data with our tokenizer. In
each phase of the continued pre-training, we train on a mix of long and
short context data. Specific details are outlined as follows:

<div align="center">
    <em><strong>Table 1:</strong> Continued pre-training data.</em>
    <table style="width: 70%">
        <thead>
            <tr>
                <th style="text-align:center">Training Phase</th>
                <th style="text-align:center">64K Long Data</th>
                <th style="text-align:center">256K Long Data</th>
                <th style="text-align:center">Short Data</th>
            </tr>
        </thead>
            <tbody>
                <tr>
                    <td style="text-align:center">Phase 1</td>
                    <td style="text-align:center">Code repos (30%),<br> Books (30%), <br> Textbooks (3%)</td>
                    <td style="text-align:center">-</td>
                    <td style="text-align:center">FineWeb-Edu (10%),<br> FineWeb (10%),<br> StackExchange (4%),<br> Wikipedia (5%),<br> ArXiv (3%),<br> OpenWebMath (5%)</td>
                </tr>
                <tr>
                    <td style="text-align:center">Phase 2</td>
                    <td style="text-align:center">Code repos (10%),<br> Books (15%)</td>
                    <td style="text-align:center">Code repos (20%),<br> Books (15%), <br> Textbooks (2%)</td>
                    <td style="text-align:center">FineWeb-Edu (10%),<br> FineWeb (10%),<br> StackExchange (4%),<br> Wikipedia (5%),<br> ArXiv (4%),<br> OpenWebMath (5%)</td>
                </tr>
            </tbody>
    </table>
</div>

## Supervised Finetuning (SFT)

**Training:** After continued training on the long-context pre-training data, we perform supervised finetuning on long-context instruction data. We train the model using a 1B-token mixture of short- and long-context instruction data.

**Data**: Similar to the continued pre-training stage, we train the model on a mixture of short- and long-context instruction data with a ratio of 4 to 6. For short-context instruction data, we use [Ultrachat
200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) [^7], [OpenMathinstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) [^8], [Tülu-3 Instruction Following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) [^9], and [MMLU auxiliary train set](https://huggingface.co/datasets/cais/mmlu) [^10]. For long-context instruction data, we construct a synthetic long-context instruction dataset due to the lack of long-context SFT data. Specifically, we make use of the long documents from Books, which is part of our continued pre-training data corpus. We select documents with a minimum length of 8K tokens and truncate those exceeding 128K tokens to a maximum length of 128K. Then, we use
[Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)
as a teacher model to synthetically generate question-answer pairs for the documents. To speed up this process, we randomly choose a subpart of the document for the QA generation instead of using the whole document. The length of the subpart is randomly set to be between 2K and 8K tokens. We use NLTK sentence tokenizer to divide documents into sentences to make sure that the selected subpart has complete sentences. The generated question and answer are appended to the end of the long document, serving as a complete single-round instruction-following data sample. In addition, we also generate long-context instruction data using short documents, in order to increase the dataset diversity with more data sources. We use arXiv from our continued pre-training corpus and the DCLM subset from [Dolmino-Mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) [^11]. We first generate QA for each short document following the same pipeline aforementioned. Then, we iteratively concatenate different short documents until it reache 128K tokens. The concatenated document can exceed 128K as we do not truncate the last document. Lastly, we randomly choose one QA corresponding to one of the short documents and append it to the end of the concatenated document. The final data mixture for the SFT stage is shown as follows:

<div align="center">
    <em><strong>Table 2:</strong> Supervised finetuning data.</em>
    <table style="width: 50%">
        <thead>
            <tr>
                <th style="text-align:center">Short Data</th>
                <th style="text-align:center">Long Data</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align:center">Ultrachat 200K (25%), <br> OpenMathinstruct-2, (10%) <br> MMLU auxiliary train set (3%), <br> Tülu-3 Instruction Following (2%)</td>
                <td style="text-align:center">Books (44%), <br> DCLM (10%), <br> ArXiv (6%)</td>
            </tr>
        </tbody>
    </table>
</div>

## Direct Preference Optimization (DPO)

**Training:** At the last training stage, we perform human preference alignment training using Direct Preference Optimization [^12]. We employ the same DPO training as Instella-3B-Instruct using the same data. Unlike previous training stages, in the DPO stage, we train on short data only whose maximum context length is 2K. Consistent with the findings of other open-weights models, we observe that performing DPO on short data alone continues to improve the model performance on long-context tasks.

**Data**: We use the
[OLMo-2-1124-7B-Preference-Mix](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix) [^11] dataset as our DPO data which contains 0.76B tokens.

## Sequence Parallelism

To enable training with extremely long inputs, we implement sequence parallelism based on Deepspeed Ulysses [^13]. The sequence parallelism distributes the attention heads across GPUs during the attention computation. It is more efficient than Ring-Attention [^14] in GPU communications. We use four GPUs as a sequence parallelism group for the Phase 2 continued pre-training and SFT due to the long inputs.

## Results

- We evaluate the long-context performance on [Helmet](https://princeton-nlp.github.io/HELMET/) [^15], a recent comprehensive long-context evaluation benchmark encompassing diverse categories. Helmet demonstrates better consistency of human perception than the previous long-context benchmarks.

- Instella-3B-Long-Instruct outperforms open weights models including Phi-3.5-mini-instruct [^1], Gemma-3-4B-it [^2], Qwen2.5-3B-Instruct [^3], and MiniCPM-2B-128k [^16] on most tasks of the Helmet benchmark (Table 3).

- We performed a side-by-side comparison at 8K, 16K, and 32K context lengths with Qwen2.5-3B-Instruct as its context length is 32K. Instella-3B-Long-Instruct outperforms Qwen2.5-3B-Instruct by 2.75% on average (Table 4).

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
  <em><strong>Table 3:</strong> Long-context evaluation on the Helmet benchmark. The NIAH and RAG tasks are evaluated at five context lengths: 8K, 16K, 32K, 64K, and 128K, and the number is reported by averaging across the five context lengths. The InfiniteBench QA, InfiniteBench MC, and NarrativeQA are evaluated at 128K context length. The InfiniteBench is reimplemented by Helmet.</em>
  <table style="width: 90%">
    <thead>
      <tr>
        <th>Models</th>
        <th>Size</th>
        <th>Training Tokens (from scratch)</th>
        <th>Natural Questions (RAG)</th>
        <th>TriviaQA (RAG)</th>
        <th>HotpotQA (RAG)</th>
        <th>InfiniteBench QA</th>
        <th>InfiniteBench MC</th>
        <th>NarrativeQA</th>
        <th>NIAH (multi value needles)</th>
        <th>Average</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th colspan="15">Open Weight Models</th>
      </tr>
      <tr>
        <td>Llama-3.2-3B-Instruct</td>
        <td style="text-align: center;">3.21B</td>
        <td style="text-align: center;">~9T</td>
        <td style="text-align: center;">51.8</td>
        <td style="text-align: center;">86.2</td>
        <td style="text-align: center;">56.4</td>
        <td style="text-align: center;">38.7</td>
        <td style="text-align: center;">56.0</td>
        <td style="text-align: center;">26.0</td>
        <td style="text-align: center;">99.2</td>
        <td style="text-align: center;"><strong>59.19</strong></td>
      </tr>
      <tr>
        <td>Phi-3.5-mini-instruct</td>
        <td style="text-align: center;">3.82B</td>
        <td style="text-align: center;">-</td>
        <td style="text-align: center;">41.2</td>
        <td style="text-align: center;">78.6</td>
        <td style="text-align: center;">48.6</td>
        <td style="text-align: center;">24.0</td>
        <td style="text-align: center;">55.0</td>
        <td style="text-align: center;">27.7</td>
        <td style="text-align: center;">87.0</td>
        <td style="text-align: center;"><strong>51.73</strong></td>
      </tr>
      <tr>
        <td>gemma-3-4b-it</td>
        <td style="text-align: center;">4.3B</td>
        <td style="text-align: center;">~4T</td>
        <td style="text-align: center;">47.2</td>
        <td style="text-align: center;">76.8</td>
        <td style="text-align: center;">45.2</td>
        <td style="text-align: center;">21.0</td>
        <td style="text-align: center;">49.0</td>
        <td style="text-align: center;">20.7</td>
        <td style="text-align: center;">74.0</td>
        <td style="text-align: center;"><strong>47.70</strong></td>
      </tr>
      <tr>
        <td>Qwen2.5-3B-Instruct</td>
        <td style="text-align: center;">3.09B</td>
        <td style="text-align: center;">~18T</td>
        <td style="text-align: center;">34.6</td>
        <td style="text-align: center;">65.8</td>
        <td style="text-align: center;">41.8</td>
        <td style="text-align: center;">14.7</td>
        <td style="text-align: center;">35.0</td>
        <td style="text-align: center;">21.0</td>
        <td style="text-align: center;">80.4</td>
        <td style="text-align: center;"><strong>41.90</strong></td>
      </tr>
      <tr>
        <td>MiniCPM-2B-128k</td>
        <td style="text-align: center;">2.4B</td>
        <td style="text-align: center;">~1T</td>
        <td style="text-align: center;">28.4</td>
        <td style="text-align: center;">61.6</td>
        <td style="text-align: center;">30.8</td>
        <td style="text-align: center;">3.7</td>
        <td style="text-align: center;">22.0</td>
        <td style="text-align: center;">3.3</td>
        <td style="text-align: center;">46.6</td>
        <td style="text-align: center;"><strong>28.06</strong></td>
      </tr>
      <tr>
        <th colspan="15">Fully Open Models</th>
      </tr>
      <tr>
        <td><strong>Instella-3B-Long-Instruct</strong></td>
        <td style="text-align: center;">3.11B</td>
        <td style="text-align: center;">~4T</td>
        <td style="text-align: center;">43.6</td>
        <td style="text-align: center;">73.0</td>
        <td style="text-align: center;">51.6</td>
        <td style="text-align: center;">30.7</td>
        <td style="text-align: center;">54.0</td>
        <td style="text-align: center;">32.3</td>
        <td style="text-align: center;">84.0</td>
        <td style="text-align: center;"><strong>52.74</strong></td>
      </tr>
    </tbody>
  </table>
</div>
</body>
</html>

<div class="table-wrapper" align="center">
  <em><strong>Table 4:</strong> Comparison with Qwen2.5-3B-Instruct at 8K, 16K, 32K context lengths.</em>
  <table>
    <thead>
      <tr>
        <th rowspan="2" style="text-align: center;">Model</th>
        <th colspan="3" style="text-align: center;">NIAH (multi value needles)</th>
        <th colspan="3" style="text-align: center;">Natural Questions (RAG)</th>
        <th colspan="3" style="text-align: center;">TriviaQA (RAG)</th>
        <th colspan="3" style="text-align: center;">HotpotQA (RAG)</th>
        <th rowspan="2" style="text-align: center;">Average</th>
      </tr>
      <tr>
        <th style="text-align: center;">8K</th><th style="text-align: center;">16K</th><th style="text-align: center;">32K</th>
        <th style="text-align: center;">8K</th><th style="text-align: center;">16K</th><th style="text-align: center;">32K</th>
        <th style="text-align: center;">8K</th><th style="text-align: center;">16K</th><th style="text-align: center;">32K</th>
        <th style="text-align: center;">8K</th><th style="text-align: center;">16K</th><th style="text-align: center;">32K</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="text-align: center;">Instella-3B-Long-Instruct</td>
        <td style="text-align: center;">98</td><td style="text-align: center;">95</td><td style="text-align: center;">87</td>
        <td style="text-align: center;">53</td><td style="text-align: center;">49</td><td style="text-align: center;">46</td>
        <td style="text-align: center;">79</td><td style="text-align: center;">73</td><td style="text-align: center;">75</td>
        <td style="text-align: center;">59</td><td style="text-align: center;">59</td><td style="text-align: center;">51</td>
        <td style="text-align: center;">68.67</td>
      </tr>
      <tr>
        <td style="text-align: center;">Qwen2.5-3B-Instruct</td>
        <td style="text-align: center;">95</td><td style="text-align: center;">94</td><td style="text-align: center;">95</td>
        <td style="text-align: center;">48</td><td style="text-align: center;">42</td><td style="text-align: center;">39</td>
        <td style="text-align: center;">77</td><td style="text-align: center;">78</td><td style="text-align: center;">74</td>
        <td style="text-align: center;">51</td><td style="text-align: center;">50</td><td style="text-align: center;">48</td>
        <td style="text-align: center;">65.92</td>
      </tr>
    </tbody>
  </table>
</div>

**Evaluation Metrics:** We use substring exact match (SubEM) for the RAG tasks including Natural Questions, TriviaQA, and HotpotQA. We use recall for NIAH and exact match for InfiniteBench MC. For InfiniteBench QA and NarrativeQA, where the answers are open-ended, we use gpt-4o-mini to evaluate the answers against the ground truth using the prompt and metric provided by the Helmet.

<div class="table-wrapper" align="center">
    <em><strong>Table 5:</strong> Short-context benchmark comparison with Intsella-3B-Instruct.</em>
  <table>
    <thead>
      <tr>
        <th>Models</th>
        <th>MMLU</th>
        <th>IFEval</th>
        <th>MT-Bench</th>
        <th>TruthfulQA</th>
        <th>Toxigen (↓)</th>
        <th>Crows-Pair</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Instella-3B-Instruct</td>
        <td style="text-align: center;">58.90</td>
        <td style="text-align: center;">71.35</td>
        <td style="text-align: center;">7.23</td>
        <td style="text-align: center;">55.47</td>
        <td style="text-align: center;">57.02</td>
        <td style="text-align: center;">58.86</td>
      </tr>
      <tr>
        <td>Instella-3B-Long-Instruct</td>
        <td style="text-align: center;">57.44</td>
        <td style="text-align: center;">68.76</td>
        <td style="text-align: center;">6.83</td>
        <td style="text-align: center;">55.52</td>
        <td style="text-align: center;">42.34</td>
        <td style="text-align: center;">60.05</td>
      </tr>
    </tbody>
  </table>
</div>

**Short-context results:** We observe performance drops on some short-context benchmarks compared to [Instella-3B-Instruct](https://huggingface.co/amd/Instella-3B-Instruct) (Table 5). Interestingly, TruthfulQA remains stable, while Crows-Pair shows a slight improvement, indicating potential gains in certain responsible AI metrics. The reduction in Toxigen (57.02 → 42.34, lower is better) suggests improved toxicity avoidance in the long-context variant. We hypothesize that these results reflect a trade-off between optimizing for longer context lengths and retaining short-context performance, which may be more pronounced at the 3B parameter scale compared to larger models.

## Summary

In this blog, we introduced Instella-Long, a fully open long-context language model trained from scratch on AMD Instinct™ MI300X GPUs, detailing its training methodology, datasets used and benchmark performance.

The release of the Instella-Long model represents a significant stride in advancing open-source AI and demonstrates the capabilities of AMD hardware in language model training. To our knowledge, Instella-Long makes the Instella series the first fully open language model trained from scratch that supports long-context, while achieving competitive performance compared to open-weights models.

By fully open sourcing the Instella-Long model, including weights, training configurations, datasets, and code, we aim to foster innovation and collaboration within the AI community. We believe that transparency, reproducibility and accessibility are key drivers of progress in AI research and development. We invite developers, researchers, and AI enthusiasts to explore Instella-long, contribute to its ongoing improvement, and join us in pushing the boundaries of what is possible with language models.

## Resources

**Hugging face Model Cards:**
[amd/Instella-3B-Long-Instruct](https://huggingface.co/amd/Instella-3B-Long-Instruct)

**Training data:** [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long)

**Training Code:**
[https://github.com/AMD-AIG-AIMA/Instella/tree/instella-long](https://github.com/AMD-AIG-AIMA/Instella/tree/instella-long)

Please refer to the following blogs to get started with using these
techniques on AMD GPUs:

- [Introducing Instella: New State-of-the-art Fully Open 3B Language Models](https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html)
  
- [PyTorch Fully Sharded Data Parallel (FSDP) on AMD GPUs with
  ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/fsdp-training-pytorch/README.html)

- [Accelerating Large Language Models with Flash Attention on AMD
  GPUs](https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html)

- [Accelerate PyTorch Models using torch.compile on AMD GPUs with
  ROCm™](https://rocm.blogs.amd.com/artificial-intelligence/torch_compile/README.html)

## Bias, Risks, and Limitations

- The models are being released for research purposes only and are not
  intended for use cases that require high levels of factuality, safety
  critical situations, health, or medical applications, generating false
  information, facilitating toxic conversations.

- Model checkpoints are made accessible without any safety promises. It
  is crucial for users to conduct comprehensive evaluations and
  implement safety filtering mechanisms as per their respective use
  cases.

- It may be possible to prompt the model to generate content that may be
  factually inaccurate, harmful, violent, toxic, biased, or otherwise
  objectionable. Such content may also get generated by prompts that did
  not intend to produce output as such. Users are thus requested to be
  aware of this and exercise caution and responsible thinking when using
  the model.

- Multi-lingual abilities of the models have not been tested and thus
  may misunderstand and generate erroneous responses across different
  languages.

## License

The [Instella-3B-Long-Instruct](https://huggingface.co/amd/Instella-3B-Long-Instruct) model is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](https://huggingface.co/amd/Instella-3B-Long-Instruct/blob/main/LICENSE) and [NOTICES](https://huggingface.co/amd/Instella-3B-Long-Instruct/blob/main/NOTICES) files for more information.

The [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long) is a collection of pre-training and instruction following data that is used to train [Instella-3B-Long-Instruct](https://huggingface.co/amd/Instella-3B-Long-Instruct), and is licensed for academic and research purposes under a ResearchRAIL license. Refer to the [LICENSE](https://huggingface.co/datasets/amd/Instella-Long/blob/main/LICENSE) in the [amd/Instella-Long](https://huggingface.co/datasets/amd/Instella-Long) dataset card for more information.

## Contributors

> **Core contributors**:
> Jialian Wu, Jiang Liu, Sudhanshu Ranjan, Xiaodong Yu, Gowtham Ramesh, Prakamya Mishra, Zicheng Liu
>
> **Contributors:**
> Yusheng Su, Ximeng Sun, Ze Wang, Emad Barsoum

Feel free to cite our Instella models:

```text
@misc{Instella,
    title = {Instella: Fully Open Language Models with Stellar Performance},
    url = {https://huggingface.co/amd/Instella-3B},
    author = {Jiang Liu, Jialian Wu, Xiaodong Yu, Prakamya Mishra, Sudhanshu Ranjan, Zicheng Liu, Chaitanya Manem, Yusheng Su, Pratik Prabhanjan Brahma, Gowtham Ramesh, Ximeng Sun, Ze Wang, Emad Barsoum},
    month = {March},
    year = {2025}
}
```

[^1]: Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree et al. "Phi-3 technical report: A highly capable language model locally on your phone." arXiv preprint arXiv:2404.14219 (2024).

[^2]: Gemma Team: Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin et al. "Gemma 3 technical report." arXiv preprint arXiv:2503.19786 (2025).

[^3]: Qwen Team. “Qwen2.5 Technical Report.” arXiv preprint arXiv:2412.15115 (2025).

[^4]: Tri Dao. "Flashattention-2: Faster attention with better parallelism and work partitioning." arXiv preprint arXiv:2307.08691 (2023).

[^5]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.

[^6]: Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. "How to train long-context language models (effectively)." arXiv preprint arXiv:2410.02660 (2024).

[^7]: Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. "Enhancing chat language models by scaling high-quality instructional conversations." arXiv preprint arXiv:2305.14233 (2023).

[^8]: Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. "Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data." arXiv preprint arXiv:2410.01560 (2024).

[^9]: Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda et al. "Tülu 3: Pushing frontiers in open language model post-training." arXiv preprint arXiv:2411.15124 (2024).

[^10]: Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).

[^11]: OLMo Team, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia et al. "2 OLMo 2 Furious." arXiv preprint arXiv:2501.00656 (2024).

[^12]: Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. "Direct preference optimization: Your language model is secretly a reward model." Advances in Neural Information Processing Systems 36 (2023): 53728-53741.

[^13]: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. "Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models." arXiv preprint arXiv:2309.14509 (2023).

[^14]: Hao Liu, Matei Zaharia, and Pieter Abbeel. "Ring attention with blockwise transformers for near-infinite context." arXiv preprint arXiv:2310.01889 (2023).

[^15]: Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. "Helmet: How to evaluate long-context language models effectively and thoroughly." arXiv preprint arXiv:2410.02694 (2024).

[^16]: Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng et al. "Minicpm: Unveiling the potential of small language models with scalable training strategies." arXiv preprint arXiv:2404.06395 (2024).

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
