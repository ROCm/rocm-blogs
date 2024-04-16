---
blogpost: true
date: 16 April 2024

author: Phillip Dang
tags: PyTorch, AI/ML, Tuning
category: Applications & models
language: English
---
<head>
  <meta charset="UTF-8">
  <meta name="description" content="Text Summarization with FLAN-T5 on AMD GPU">
  <meta name="keywords" content="PyTorch, FLAN-T5, LLM, fine-tune, summarization, AMD, GPU, MI300, MI250">
</head>

# Text Summarization with FLAN-T5

In this blog, we showcase the language model FLAN-T5 and how to fine-tune it on a summarization task with HuggingFace in an AMD GPUs + ROCm system.

## Introduction

FLAN-T5 is an open-source large language model published by Google and is an enhancement over the previous T5 model. It is an encoder-decoder model that has been pre-trained on prompting datasets. This means that the model has knowledge of performing specific tasks such as summarization, classification and translation, etc. For more details on FLAN-T5, please refer to the [original paper](https://arxiv.org/pdf/2210.11416.pdf). To see full details of the model's improvement over the previous T5 model, please refer to [this model card](https://huggingface.co/docs/transformers/model_doc/t5v1.1)

## Prerequisites

* [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* [Linux OS](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
* [An AMD GPU](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

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

## Libraries

Before you begin, make sure you have all the necessary libraries installed:

``` python
!pip install -q transformers accelerate einops datasets
!pip install --upgrade SQLAlchemy==1.4.46
!pip install -q alembic==1.4.1 numpy==1.23.4 grpcio-status==1.33.2 protobuf==3.19.6 
```

``` python
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

Next import the modules you'll be working with for this blog:

```python
import time 
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

```

## Loading the model  

Let's load the model and its tokenizer. FLAN-T5 has several variants at different sizes from `small` to `xxl`. We will first run some inferences on the `xxl` variant and demonstrate how to fine-tune Flan-T5 using the `small` variant on a summarization task.

```python
start_time = time.time()
model_checkpoint = "google/flan-t5-xxl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"Loaded in {time.time() - start_time: .2f} seconds")
print(model)
```

```text
Loading checkpoint shards: 100%|██████████| 5/5 [01:23<00:00, 16.69s/it]
Loaded in  85.46 seconds
T5ForConditionalGeneration(
  (shared): Embedding(32128, 4096)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 4096)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
              (relative_attention_bias): Embedding(32, 64)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
              (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
              (wo): Linear(in_features=10240, out_features=4096, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-23): 23 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
              (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
              (wo): Linear(in_features=10240, out_features=4096, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): T5Stack(
    (embed_tokens): Embedding(32128, 4096)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
              (relative_attention_bias): Embedding(32, 64)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
              (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
              (wo): Linear(in_features=10240, out_features=4096, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
      (1-23): 23 x T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerCrossAttention(
            (EncDecAttention): T5Attention(
              (q): Linear(in_features=4096, out_features=4096, bias=False)
              (k): Linear(in_features=4096, out_features=4096, bias=False)
              (v): Linear(in_features=4096, out_features=4096, bias=False)
              (o): Linear(in_features=4096, out_features=4096, bias=False)
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): T5LayerFF(
            (DenseReluDense): T5DenseGatedActDense(
              (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
              (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
              (wo): Linear(in_features=10240, out_features=4096, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): NewGELUActivation()
            )
            (layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (final_layer_norm): FusedRMSNorm(torch.Size([4096]), eps=1e-06, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (lm_head): Linear(in_features=4096, out_features=32128, bias=False)
)
```

## Running inference

Let's run quick inferences since it's worth noting that one can directly use FLAN-T5 without fine-tuning the model first.

We can ask the model a simple question:

```python
inputs = tokenizer("How to make milk coffee", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

```text
['Pour a cup of coffee into a mug. Add a tablespoon of milk. Add a pinch of sugar.']
```

Or we can ask it to summarize a chunk of text

```python
text = """ summarize: 
Amy: Hey Mark, have you heard about the new movie coming out this weekend?
Mark: Oh, no, I haven't. What's it called?
Amy: It's called "Stellar Odyssey." It's a sci-fi thriller with amazing special effects.
Mark: Sounds interesting. Who's in it?
Amy: The main lead is Emily Stone, and she's fantastic in the trailer. The plot revolves around a journey to a distant galaxy.
Mark: Nice! I'm definitely up for a good sci-fi flick. Want to catch it together on Saturday?
Amy: Sure, that sounds great! Let's meet at the theater around 7 pm.
"""
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
tokenizer.decode(outputs[0], skip_special_tokens=True)
```

```text
'Amy and Mark are going to see "Stellar Odyssey" on Saturday at 7 pm.'
```

## Fine-tuning

In this section, we will fine-tune the model for a summarization task. We adapt our code from [this tutorial](https://huggingface.co/docs/transformers/tasks/summarization). As mentioned, we'll be using the `small` variant of the model to do the fine-tuning

```python
start_time = time.time()
model_checkpoint = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"Loaded in {time.time() - start_time: .2f} seconds")
```

### Load Dataset

Our example dataset is the [samsum dataset](https://huggingface.co/datasets/samsum), which contains about 16k messenger-like conversations with summaries.

```python
from datasets import load_dataset
from evaluate import load

raw_datasets = load_dataset("samsum")
```

Below is a sample of our dataset:

```python
print('Dialogue: ')
print(raw_datasets['train']['dialogue'][100])
print() 
print('Summary: ', raw_datasets['train']['summary'][100])
```

```text
Dialogue: 
Gabby: How is you? Settling into the new house OK?
Sandra: Good. The kids and the rest of the menagerie are doing fine. The dogs absolutely love the new garden. Plenty of room to dig and run around.
Gabby: What about the hubby?
Sandra: Well, apart from being his usual grumpy self I guess he's doing OK.
Gabby: :-D yeah sounds about right for Jim.
Sandra: He's a man of few words. No surprises there. Give him a backyard shed and that's the last you'll see of him for months.
Gabby: LOL that describes most men I know.
Sandra: Ain't that the truth! 
Gabby: Sure is. :-) My one might as well move into the garage. Always tinkering and building something in there.
Sandra: Ever wondered what he's doing in there?
Gabby: All the time. But he keeps the place locked.
Sandra: Prolly building a portable teleporter or something. ;-)
Gabby: Or a time machine... LOL
Sandra: Or a new greatly improved Rabbit :-P
Gabby: I wish... Lmfao!

Summary:  Sandra is setting into the new house; her family is happy with it. Then Sandra and Gabby discuss the nature of their men and laugh about their habit of spending time in the garage or a shed.
```

### Set up metric

Next let's load our metric for this task. Typically, in summarization task we use ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics, which measure the similarity between the original document and the summarized one. More specifically, these metrics measure the overlap of n-grams (sequences of n words) between the system and reference summaries. For more details on this metric, see [this link](https://huggingface.co/spaces/evaluate-metric/rouge)

```python
from evaluate import load
metric = load("rouge")
print(metric)
```

```text
EvaluationModule(name: "rouge", module_type: "metric", features: [{'predictions': Value(dtype='string', id='sequence'), 'references': Sequence(feature=Value(dtype='string', id='sequence'), length=-1, id=None)}, {'predictions': Value(dtype='string', id='sequence'), 'references': Value(dtype='string', id='sequence')}], usage: """
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    rouge_types: A list of rouge types to calculate.
        Valid names:
        `"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
        `"rougeL"`: Longest common subsequence based scoring.
        `"rougeLsum"`: rougeLsum splits text using `"
"`.
        See details in https://github.com/huggingface/datasets/issues/617
    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
    use_aggregator: Return aggregates if this is set to True
Returns:
    rouge1: rouge_1 (f1),
    rouge2: rouge_2 (f1),
    rougeL: rouge_l (f1),
    rougeLsum: rouge_lsum (f1)
Examples:

    >>> rouge = evaluate.load('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(results)
    {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
""", stored examples: 0)
```

We need to create a function to compute the rogue metrics.

```python
import nltk
nltk.download('punkt')
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # We need to replace -100 in the labels since we can't decode it 
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Add new line after each sentence for rogue metrics
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # compute metrics 
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # compute the average length of the generated text
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
```

### Process data

Let's create a function to process the data, which includes tokenizing the input and output for each sample document. We also set length thresholds to truncate our input and output.

```python
prefix = "summarize: "

max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["dialogue"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```

### Train the model  

To train our model, we need a few ingredients:

1. A data collator to dynamically pad the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
2. A TrainingArguments class to customize how a model is trained.
3. A Trainer class which is an API for training in PyTorch.

First let's create our data collator.

```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

Next, let's set up our TrainingArgument class

```python
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-samsum",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)
```

`Note`: We have found that since the model was pretrained on Google TPU, not GPU, we need to set `fp16=False` or `bf16=True`. Otherwise we end up with overflow issues which cause NaN values for our loss. This is likely due to the differnces in half-precision floating point format `fp16` and `bf16`.

Finally we need to set up a trainer API

```python
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

With that, we're ready to train our model!

```python
trainer.train()
```

```text
 [1842/1842 05:37, Epoch 2/2]
Epoch Training Loss Validation Loss Rouge1  Rouge2  Rougel  Rougelsum Gen Len
1 1.865700  1.693366  43.551000 20.046200 36.170400 40.096200 16.926700
2 1.816700  1.685862  43.506000 19.934800 36.278300 40.156700 16.837400
```

Running the above trainer should generate a local folder `flan-t5-small-finetuned-samsum` which stores our checkpoints for the model.

### Inference

Once we have fine-tuned model, we can use it for inference! Let's first reload the tokenizer and the fine-tuned model from our local checkpoints.

```python
model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-small-finetuned-samsum/checkpoint-1500")
tokenizer = AutoTokenizer.from_pretrained("flan-t5-small-finetuned-samsum/checkpoint-1500")
```

Next, we come up with some text to summarize. It's important to prefix the input as shown below:

```python
text = """ summarize: 
Hannah: Hey, Mark, have you decided on your New Year's resolution yet?
Mark: Yeah, I'm thinking of finally hitting the gym regularly. What about you?
Hannah: I'm planning to read more books this year, at least one per month.
Mark: That sounds like a great goal. Any particular genre you're interested in?
Hannah: I want to explore more classic literature. Maybe start with some Dickens or Austen.
Mark: Nice choice. I'll hold you to it. We can discuss our progress over coffee.
Hannah: Deal! Accountability partners it is.
"""
```

Finally, we encode the input and generate the summarization

```python
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
tokenizer.decode(outputs[0], skip_special_tokens=True)
```

```text
'Hannah is planning to read more books this year. Mark will hold Hannah to it.'
```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
