---
blogpost: true
date: 26 Apr 2024


author: Phillip Dang
tags: PyTorch, AI/ML, Tuning
category: Applications & models
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Table Question-Answering with TaPas">
  <meta name="keywords" content="PyTorch, TaPas, LLM, Bert, AMD, GPU, MI300, MI250">
</head>

# Table Question-Answering with TaPas

Conventionally, the question-answering task is framed as a semantic parsing task where the question is translated to a full logical form that can be executed against the table to retrieve the correct answer. However, this requires a lot of annotated data, which can be expensive to acquire.

In response to this challenge, TaPas opts for predicting a streamlined program by choosing a portion of the table cells along with a potential aggregation operation to apply to them. As a result, TaPas can grasp operations directly from natural language, eliminating the necessity for explicit formal specifications.

The TaPas (Table Parser) model is a BERT-based, weakly supervised question answering model that has been designed and pretrained for answering questions about tabular data. The model is enhanced with positional embeddings  to understand the structure of tables. The input tables are turned into a sequential format of words, segmenting words into tokens, and then integrating question tokens ahead of table tokens. Furthermore, there are two classification layers to facilitate the selection of table cells and aggregation operators, which function on these cells.

TaPas's pretraining data comes from Wikipedia spanning numerous tables, enabling the model to grasp diverse correlations between text and tables, as well as between individual cells and their corresponding headers. The pre-training process involves extracting text-table pairs from Wikipedia, yielding 6.2 million tables with a focus on tables with a maximum of 500 cells, aligning with the structure of their end task datasets, which exclusively feature horizontal tables with header rows containing column names.  

For a deeper dive into the inner workings of TaPas and their performance, refer to [TaPas: Weakly Supervised Table Parsing via Pre-training](https://aclanthology.org/2020.acl-main.398.pdf) by Google Research.

In this blog, we run some inferences with TaPas and demonstrate how it works out-of-the-box with AMD GPUs and ROCm.

## Prerequisites

* Software:
  * [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
  * [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
  * Linux OS

For a list of supported GPUs and OS, please refer to [ROCm's installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). For convenience and stability, we recommend you to directly pull and run the rocm/pytorch Docker image in your Linux system with the following command:

```sh
docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
           --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --name=olmo rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1 /bin/bash
```

* Hardware:

Make sure the system recognizes your GPU:

``` python
! rocm-smi --showproductname
```

```cpp
================= ROCm System Management Interface ================
========================= Product Info ============================
GPU[0] : Card series: Instinct MI210
GPU[0] : Card model: 0x0c34
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] : Card SKU: D67301
===================================================================
===================== End of ROCm SMI Log =========================
```

Let's check if we have the right version of ROCm installed.

```python
!apt show rocm-libs -a
```

```text
Package: rocm-libs
Version: 5.7.0.50700-63~22.04
Priority: optional
Section: devel
Maintainer: ROCm Libs Support <rocm-libs.support@amd.com>
Installed-Size: 13.3 kBA
Depends: hipblas (= 1.1.0.50700-63~22.04), hipblaslt (= 0.3.0.50700-63~22.04), hipfft (= 1.0.12.50700-63~22.04), hipsolver (= 1.8.1.50700-63~22.04), hipsparse (= 2.3.8.50700-63~22.04), miopen-hip (= 2.20.0.50700-63~22.04), rccl (= 2.17.1.50700-63~22.04), rocalution (= 2.1.11.50700-63~22.04), rocblas (= 3.1.0.50700-63~22.04), rocfft (= 1.0.23.50700-63~22.04), rocrand (= 2.10.17.50700-63~22.04), rocsolver (= 3.23.0.50700-63~22.04), rocsparse (= 2.5.4.50700-63~22.04), rocm-core (= 5.7.0.50700-63~22.04), hipblas-dev (= 1.1.0.50700-63~22.04), hipblaslt-dev (= 0.3.0.50700-63~22.04), hipcub-dev (= 2.13.1.50700-63~22.04), hipfft-dev (= 1.0.12.50700-63~22.04), hipsolver-dev (= 1.8.1.50700-63~22.04), hipsparse-dev (= 2.3.8.50700-63~22.04), miopen-hip-dev (= 2.20.0.50700-63~22.04), rccl-dev (= 2.17.1.50700-63~22.04), rocalution-dev (= 2.1.11.50700-63~22.04), rocblas-dev (= 3.1.0.50700-63~22.04), rocfft-dev (= 1.0.23.50700-63~22.04), rocprim-dev (= 2.13.1.50700-63~22.04), rocrand-dev (= 2.10.17.50700-63~22.04), rocsolver-dev (= 3.23.0.50700-63~22.04), rocsparse-dev (= 2.5.4.50700-63~22.04), rocthrust-dev (= 2.18.0.50700-63~22.04), rocwmma-dev (= 1.2.0.50700-63~22.04)
Homepage: https://github.com/RadeonOpenCompute/ROCm
Download-Size: 1012 B
APT-Manual-Installed: yes
APT-Sources: http://repo.radeon.com/rocm/apt/5.7 jammy/main amd64 Packages
Description: Radeon Open Compute (ROCm) Runtime software stack
```

Make sure PyTorch also recognizes the GPU:

``` python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

``` text
number of GPUs: 1
['AMD Radeon Graphics']
```

Let's start testing TaPas. There are 3 different variants to TaPas, corresponding to the different datasets on which TaPas was fine-tuned. In this blog, we'll be focusing on the model that was fine-tuned on the WTQ dataset, which was a `Weak supervision for aggregation` task. This task is not for asking questions in a conversational setup, but for specific questions related to a given table, which might involve aggregation. This is called weak supervision because "the model itself must learn the appropriate aggregation operator (SUM/COUNT/AVERAGE/NONE) given only the answer to the question as supervision".

For more information on other variants, please look at [this doc](https://huggingface.co/docs/transformers/en/model_doc/tapas).

## Libraries

Before you begin, make sure you have all the necessary libraries installed:

``` python
! pip install -q transformers pandas datasets tabulate
```

Next import the modules you'll be working with for this blog:

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
```

### Loading the data

We'll be using a simple dataset about world economy.

```python
from datasets import load_dataset
data = load_dataset("ashraq/ott-qa-20k", split='train')

for doc in data:
    if doc['title'] == 'World economy':
        table = pd.DataFrame(doc["data"], columns=doc['header'])
        break 

print(table.to_markdown())
```

```text
|    |   Rank | Country              | Value ( USD $ )   |   Peak year |
|---:|-------:|:---------------------|:------------------|------------:|
|  0 |      1 | Qatar                | 146,982           |        2012 |
|  1 |      2 | Macau                | 133,021           |        2013 |
|  2 |      3 | Luxembourg           | 108,951           |        2019 |
|  3 |      4 | Singapore            | 103,181           |        2019 |
|  4 |      5 | United Arab Emirates | 92,037            |        2004 |
|  5 |      6 | Brunei               | 83,785            |        2012 |
|  6 |      7 | Ireland              | 83,399            |        2019 |
|  7 |      8 | Norway               | 76,684            |        2019 |
|  8 |      9 | San Marino           | 74,664            |        2008 |
|  9 |     10 | Kuwait               | 71,036            |        2013 |
| 10 |     11 | Switzerland          | 66,196            |        2019 |
| 11 |     12 | United States        | 65,112            |        2019 |
| 12 |     13 | Hong Kong            | 64,928            |        2019 |
| 13 |     14 | Netherlands          | 58,341            |        2019 |
| 14 |     15 | Iceland              | 56,066            |        2019 |
| 15 |     16 | Saudi Arabia         | 55,730            |        2018 |
| 16 |     17 | Taiwan               | 55,078            |        2019 |
| 17 |     18 | Sweden               | 54,628            |        2019 |
| 18 |     19 | Denmark              | 53,882            |        2019 |
| 19 |     20 | Germany              | 53,567            |        2019 |
```

### Loading the model

Let's load the model, its tokenizer and config.

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering, TapasConfig
model_name = "google/tapas-base-finetuned-wtq"
model = TapasForQuestionAnswering.from_pretrained(model_name)
tokenizer = TapasTokenizer.from_pretrained(model_name)
config = TapasConfig.from_pretrained('google/tapas-base-finetuned-wtq')

print(model)

print("Aggregations: ", config.aggregation_labels)
```

```text
TapasForQuestionAnswering(
  (tapas): TapasModel(
    (embeddings): TapasEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(1024, 768)
      (token_type_embeddings_0): Embedding(3, 768)
      (token_type_embeddings_1): Embedding(256, 768)
      (token_type_embeddings_2): Embedding(256, 768)
      (token_type_embeddings_3): Embedding(2, 768)
      (token_type_embeddings_4): Embedding(256, 768)
      (token_type_embeddings_5): Embedding(256, 768)
      (token_type_embeddings_6): Embedding(10, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): TapasEncoder(
      (layer): ModuleList(
        (0-11): 12 x TapasLayer(
          (attention): TapasAttention(
            (self): TapasSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): TapasSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): TapasIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): TapasOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): TapasPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (aggregation_classifier): Linear(in_features=768, out_features=4, bias=True)
)

Aggregation:  {0: 'NONE', 1: 'SUM', 2: 'AVERAGE', 3: 'COUNT'}
```

### Running inference

We are ready to test the model by running some queries on the data. First, let's create a function that takes as input a list of queries, a dataframe, and outputs answers. We adapt our code from this [tutorial](https://huggingface.co/docs/transformers/en/model_doc/tapas) at HuggingFace. The crucial part is the handy method convert_logits_to_predictions, which converts the model's output (or logits) into predicted coordinates—that is, which cell in the table to focus on—and aggregation indices—that is, which aggregation operation should be performed given the question.

```python
def qa(queries, table):    
    inputs = tokenizer(table=table, queries=queries, padding=True, truncation=True, return_tensors="pt") 
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )

    # let's print out the results:
    id2aggregation = config.aggregation_labels
    aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells 
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))

    print("")
    for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
        print(query)
        if predicted_agg == "NONE":
            print("Predicted answer: " + answer)
        else:
            print("Predicted answer: " + predicted_agg + " > " + answer)
        print()
```

Let's try out the model

```python
queries = ["What is the value of Norway?",
           "What is the total value of all countries in 2013?",
           "What is the average value of all countries in 2019?",
           "How many countries are in the data in 2012?",
           "What is the combined value of Sweden and Denmark?"
          ]
qa(queries, table)
```

```text
What is the value of Norway?
Predicted answer: AVERAGE > 76,684

What is the total value of all countries in 2013?
Predicted answer: SUM > 133,021, 71,036

What is the average value of all countries in 2019?
Predicted answer: AVERAGE > 108,951, 83,399, 76,684, 66,196, 65,112, 64,928, 58,341, 56,066, 55,078, 54,628, 53,882, 53,567

How many countries are in the data in 2012?
Predicted answer: COUNT > Qatar, Brunei

What is the combined value of Sweden and Denmark?
Predicted answer: SUM > 54,628, 53,882
```

The model is able to accurately select the relevant cells and aggregation function in the data given the question.
We encourage readers to explore other variants of TaPas as well as fine-tuning it on your own dataset.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
