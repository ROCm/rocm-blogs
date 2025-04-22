---
blogpost: true
blog_title: "Shrink LLMs, Boost Inference: INT4 Quantization on AMD GPUs with GPTQModel"
date: 9 April 2025
author: 'Fabricio Flores'
thumbnail: 'quantization.png'
tags: GenAI, AI/ML, Performance, Optimization
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Demonstrate how to perform quantization of LLM models on AMD hardware using GPTQModel python library.
language: English
myst:
    html_meta:
        "author": "Fabricio Flores"
        "description lang=en": "Learn how to compress LLMs with GPTQModel and run them efficiently on AMD GPUs using INT4 quantization, reducing memory use, shrinking model size, and enabling fast inference"
        "keywords": "PyTorch, GPTQ, Quantization, Docker, AMD, GPU, MI300X, GPTQModel, INT4"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and models"
        "amd_developer_type": "ML/AI Developer, Data & Research Scientists"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_product_type": "Accelerators"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "Large Language Model (LLM)"
        "amd_industries": "Data Center, Supercomputing & Research"
        "amd_blog_releasedate": Weds Apr 09, 12:00:00 PST 2025
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

# Shrink LLMs, Boost Inference: INT4 Quantization on AMD GPUs with GPTQModel

[GPTQ (Generalized Post Training Quantization)](https://arxiv.org/pdf/2210.17323) is a technique for compressing Large Language Models (LLMs) after they have been fully trained by reducing their numerical precision. The objective of compressing the model is to reduce its memory footprint and computational requirements, making it easier to deploy it on hardware with limited resources.

The decrease in the model's size is accomplished by reducing the numerical precision of the values in the model's weight matrix from formats like `FP32` or `FP16` to quantized integer formats such as `INT4` or `INT8`. This reduction allows for models to run efficiently on systems with limited VRAM (fast memory for quick data storage and retrieval) while maintaining high performance. Unlike traditional quantization methods such as [Post-Training Quantization](https://pytorch.org/TensorRT/tutorials/ptq.html) and [Quantization-Aware Training](https://pytorch.org/blog/quantization-aware-training/) that use scaling, rounding or clamping, GPTQ works by quantizing each row of the weight matrix independently, using a second-order approximation to the loss function that minimizes the overall quantization error. During inference, these quantized weights are dynamically dequantized to the original format, usually `FP16`.

The [GPTQModel](https://github.com/ModelCloud/GPTQModel) Python library is a practical implementation of the [GPTQ](https://arxiv.org/pdf/2210.17323) algorithm. This library enables the efficient quantization of models to lower-bit formats, such as 3-bit or 4-bit, without compromising on accuracy. GPTQModel offers broad support for various models, platforms, and hardware architectures, including AMD ROCm. In particular, starting from ROCm 6.2 and later, GPTQModel provides optimized kernels for accelerated inference, ensuring that quantized models deliver high performance and efficient execution. For more information on the kernels and models supported by `GPTQModel` see: [GPTQModel Model Support](https://github.com/ModelCloud/GPTQModel#model-support) and [GPTQModel Platform and HW Support](https://github.com/ModelCloud/GPTQModel/tree/main?tab=readme-ov-file#platform-and-hw-support).

While quantization is typically aimed at reducing model size and memory bandwidth requirements, it does not always translate to faster inference speeds. This is because quantization can introduce additional overhead during inference, particularly due to the dequantization process. For example, when using quantization formats like `INT4`, computations may incur additional overhead when converting these formats back to the original `FP16` precision.

In this blog we will show you, step-by step, how to use `GPTQModel` for LLMs quantization on AMD GPUs, demonstrating AMD's capabilities for efficient quantization and inference. For the files related to this blog post, see this
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/gptq).

## Requirements

* AMD GPU: See the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems.

* ROCm 6.2+: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) page for installation instructions.

* Docker: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) for installation instructions.

* PyTorch 2.4 Docker Image including `GPTQModel` : We have created a custom Dockerfile to build a Docker image that includes the `GPTQModel` package and additional dependencies. For additional details see the corresponding [Dockerfile](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/gptq/docker/Dockerfile/).

* Hugging Face Access Token: This blog requires a [Hugging Face](https://huggingface.co/) account with a newly generated [User Access Token](https://huggingface.co/docs/hub/security-tokens).

* Access to the `mistralai/Mistral-Large-Instruct-2407` model on Hugging Face. `mistralai` family models are [gated models](https://huggingface.co/docs/hub/models-gated) on Hugging Face. To request access, see: [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407).

## Following along with this blog

* Clone the repo and `cd` into the blog directory:

    ```shell
    git clone https://github.com/ROCm/rocm-blogs.git
    cd rocm-blogs/blogs/artificial-intelligence/gptq
    ```

* Build and start the container. For details on the build process, see the `gptq/docker/Dockerfile`.

    ```shell
    cd docker
    docker compose build
    docker compose up
    ```
  
* Open [http://127.0.0.1:8888/lab/tree/src/gptq.ipynb](http://127.0.0.1:8888/lab/tree/src/gptq.ipynb) in your browser and open the `gptq.ipynb` notebook.

You can follow along with this blog using the `gptq.ipynb` notebook.

## Performing inference with Mistral-Large-Instruct-2407 on a single GPU

`Mistral-Large-Instruct-2407` is a LLM consisting of 123 billion parameters. Due to its size, systems with limited resources face significant challenges in handling this model. Even high-end systems equipped with a single GPU often encounter out-of-memory errors because they cannot provide the necessary memory capacity to load and process all these parameters. This limitation underscores the importance of employing techniques like quantization, which reduce the model's memory footprint. By doing so, `Mistral-Large-Instruct-2407` and similar models can be deployed on systems with lower GPU memory capacities and made accessible in environments with restricted resources.

As an example you can try to load `Mistral-Large-Instruct-2407` in `1` of the `8` GPUs of an AMD MI300X accelerator:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pretrained_model_id = "mistralai/Mistral-Large-Instruct-2407"

# Load the model with Float16 precision on a single GPU
model = AutoModelForCausalLM.from_pretrained(pretrained_model_id, torch_dtype=torch.float16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)

prompt = "In numerical mathematics, quantization is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_new_tokens=50)
tokenizer.decode(output[0], skip_special_tokens=True)
```

You will be presented with the following message:

```text
...
OutOfMemoryError: HIP out of memory. Tried to allocate 672.00 MiB. GPU 0 has a total capacity of 191.98 GiB of which 0 bytes is free. Of the allocated memory 191.53 GiB is allocated by PyTorch, and 520.00 KiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_HIP_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

The model `Mistral-Large-Instruct-2407` with its 123 billion parameters is so large that a single GPU is not enough to load and perform inference with it. This is where quantization can help.

The following sections will walk you through how to apply `INT4` quantization to `Mistral-Large-Instruct-2407` and how to perform inference with the quantized model on a single GPU. This will demonstrate significance and advantages of model quantization.

## Using GPTQModel to quantize Mistral-Large-Instruct-2407

Hugging Face [mistralai/Mistral-Large-Instruct-2407](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) is an open-source LLM from [Mistral AI](https://mistral.ai/) that features 123 billion parameters and 128K token context window. `Mistral-Large-Instruct-2407` utilizes `bfloat16 (BF16)` floating point format (tensor type). `BF16` is a `16-bit` floating-point numeric representation that is designed to balance computational efficiency while maintaining the expansive range of `FP32`.

Quantizing `Mistral-Large-Instruct-2407` with `GPTQModel` involves reducing the numerical precision of the model's weights while optimizing memory usage and preserving accuracy. This quantization process aims to lowers memory usage by compressing the original model, making it ideal for deploying on devices with limited resources.

This section covers the steps needed for quantizing `mistralai/Mistral-Large-Instruct-2407` using `GPTQModel`. The process starts with the model preparation, use of the calibration dataset and model evaluation.

### Model preparation and calibration dataset

This step consists of loading a tokenizer that transforms the raw input data (text) into tokens. GPTQ requires a **calibration dataset** which is a dataset used to adjust the quantization parameters so that reducing the model precision does not significantly affects its accuracy. For more information on the calibration dataset you can see: [Calibration process & calibration dataset used to perform GPTQ](https://github.com/AutoGPTQ/AutoGPTQ/issues/312) and [Importance of dataset used during quantization?](https://github.com/AutoGPTQ/AutoGPTQ/issues/11)

Start by importing the packages:

```python
import os
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import torch
```

and loading the your Hugging Face token and tokenizer

```python
os.environ['HF_TOKEN'] = <YOUR_HF_TOKEN>
```

```python
# The non-quantized model tag:
pretrained_model_id = "mistralai/Mistral-Large-Instruct-2407"

# Load the tokenizer associated to the pretrained model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
```

The **calibration dataset** is a small dataset of unlabeled examples used during the quantization process. The role of the dataset is to guide the adjustment of the model's weights from the original numerical precision to a lower one. This dataset is essential for determining the optimal quantization parameters, ensuring that the reduced precision weights closely mimic the behavior of the original model. The choice of the calibration dataset has direct impact on the performance of the quantized model. Running the calibration dataset through the model ensures the quantized model maintains the performance levels compared to the original model.

The [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) dataset is used in the quantization process. You can explore the dataset as follows:

```python
def get_wikitext2(tokenizer, nsamples, seqlen):
    '''
    Loads the dataset and tokenizes it
    '''
    
    # Select text larger than seqlen
    traindata = load_dataset("Salesforce/wikitext", 
                              "wikitext-2-raw-v1", 
                              split="train").filter(lambda x: len(x["text"]) >= seqlen)
    
    tokenized_data = [(example["text"], tokenizer(example["text"])) for example in traindata.select(range(nsamples))]

    return zip(*tokenized_data)

# Explore the first 3 samples
text_example, tokenized_example = get_wikitext2(tokenizer, nsamples=3, seqlen=5)

for text, tokenized_text in zip(text_example, tokenized_example):
    print(f"EXAMPLE TEXT:\n{text}\nTOKENIZED TEXT:\n{tokenized_text} \n")    
```

The previous code loads a small sample of the dataset and tokenizes it. You will observe the following output:

```text
EXAMPLE TEXT:
 = Valkyria Chronicles III = 

TOKENIZED TEXT:
{'input_ids': [1, 29871, 353, 478, 2235, 29891, 2849, 15336, 4027, 4786, 353, 29871, 13], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]} 

EXAMPLE TEXT:
 Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision (...) serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven " . 

TOKENIZED TEXT:
{'input_ids': [1, 29871, 5811, 29926, 30099, 694, 478, 2235, 29891, 2849, 29871, 29941, 584, 853, 11651, 287, 15336, 4027, 313, 10369, 584, 29871, 30863, 30310, 29941, 1919, 11872, 869, (...), 263, 6584, 284, 9121, 5190, 16330, 278, 5233, 310, 8130, 7035, 4628, 6931, 322, 526, 282, 4430, 2750, 278, 21080, 5190, 376, 3037, 314, 11156, 390, 3496, 376, 869, 29871, 13], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, (...), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]} 

EXAMPLE TEXT:
 The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the (...) II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 

TOKENIZED TEXT:
{'input_ids': [1, 29871, 450, 3748, 4689, 5849, 297, 29871, 29906, 29900, 29896, 29900, 1919, 19436, 975, 263, 2919, 11910, 310, 278, 664, (...), 3517, 9976, 23550, 16459, 278, 2471, 869, 450, 3748, 525, 29879, 8718, 10929, 471, 269, 686, 491, 2610, 525, 29876, 869, 29871, 13], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, (...), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]} 
```

### Loading the original (non-quantized) model and setting the quantization parameters

Quantizing a model using `GPTQModel` requires an instance of the `QuantizeConfig` class. This instance is needed for defining the parameters and behavior of the quantization process (see the [QuantizeConfig](https://github.com/ModelCloud/GPTQModel/blob/d1aea4c38c7946db6ec08053abaf02094e14e171/gptqmodel/quantization/config.py#L139) class). The configuration specifies the quantization level and the group size used among other values. For this example, `4-bit` (`INT4`) and `group_size=128` are used for the quantization level and the group size respectively. The `group_size` is the parameter that determines the number of consecutive weights in the model that are quantized together as a block. For more information about the `group_size` parameter see the *experimental validation* section at [GPTQ (Generalized Post Training Quantization)](https://arxiv.org/pdf/2210.17323) paper.

```python
quantize_config = QuantizeConfig(
    bits=4,  # quantize model to 4-bit (INT4).
    group_size=128  # it is recommended to set the value to 128
)

# Load the original (non-quantized) model.
model = GPTQModel.load(pretrained_model_id, 
                      quantize_config, 
                      trust_remote_code=True # Hugging Face Datasets/load_dataset parameter
                      )
```

You can inspect the loaded (non-quantized) model with:

```python
# Explore the non-quantized model's architecture
model
```

and the output will be:

```text
MistralGPTQ(
  (model): MistralForCausalLM(
    (model): MistralModel(
      (embed_tokens): Embedding(32768, 12288)
      (layers): ModuleList(
        (0-87): 88 x MistralDecoderLayer(
          (self_attn): MistralAttention(
            (q_proj): Linear(in_features=12288, out_features=12288, bias=False)
            (k_proj): Linear(in_features=12288, out_features=1024, bias=False)
            (v_proj): Linear(in_features=12288, out_features=1024, bias=False)
            (o_proj): Linear(in_features=12288, out_features=12288, bias=False)
          )
          (mlp): MistralMLP(
            (gate_proj): Linear(in_features=12288, out_features=28672, bias=False)
            (up_proj): Linear(in_features=12288, out_features=28672, bias=False)
            (down_proj): Linear(in_features=28672, out_features=12288, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): MistralRMSNorm((12288,), eps=1e-05)
          (post_attention_layernorm): MistralRMSNorm((12288,), eps=1e-05)
        )
      )
      (norm): MistralRMSNorm((12288,), eps=1e-05)
      (rotary_emb): MistralRotaryEmbedding()
    )
    (lm_head): Linear(in_features=12288, out_features=32768, bias=False)
  )
)
```

The previous output shows the `mistralai/Mistral-Large-Instruct-2407` architecture with its `Linear` layers. These layers will undergo the quantization process in the next step.

### Quantizing the model

Start with loading a larger portion of the calibration dataset, with a sample size of `nsamples=512`:

```python
# Loading the calibration dataset with 512 samples in the data
_, calibration_dataset = get_wikitext2(tokenizer, nsamples=512, seqlen=1024)
```

Next, call the `quantize` method. The quantization process will take around 3 hours to complete:

```python
# Quantize the original model
model.quantize(calibration_dataset, batch_size=32)

# Save the model locally with a new tag for the new quantized version
quantized_model_id = f"quantized_{pretrained_model_id.split('/')[1]}_4bit"
model.save(quantized_model_id)
```

You will be presented with an output similar to:

```text
INFO  Packing Kernel: Auto-selection: adding candidate `TritonV2QuantLinear`   
INFO  {'process': 'gptq', 'layer': 0, 'module': 'self_attn.k_proj', 'loss': '0.00519', 'damp': '0.01000', 'time': '3.342', 'fwd_time': '4.727'}
INFO  {'process': 'gptq', 'layer': 0, 'module': 'self_attn.v_proj', 'loss': '0.00004', 'damp': '0.01000', 'time': '2.629', 'fwd_time': '4.727'}
...
Quantizing mlp.down_proj in layer     [2 of 87] | 0:02:03 / 1:00:08 [3/88] 3.4%
...
INFO  {'process': 'gptq', 'layer': 87, 'module': 'mlp.down_proj', 'loss': '0.18669', 'damp': '0.01000', 'time': '8.383', 'fwd_time': '5.963'}
INFO  Packing model...                                                         
INFO  Packing Kernel: Auto-selection: adding candidate `TritonV2QuantLinear` 
...
DEBUG Received safetensors_metadata: {'format': 'pt'}                          
INFO  Pre-Quantized model size: 467720.47MB, 456.76GB                          
INFO  Quantized model size: 61923.56MB, 60.47GB                                
INFO  Size difference: 405796.91MB, 396.29GB - 86.76% 
```

The quantization process finalizes with information about the sizes of the original model and its quantized version. These `INFO` messages tell you how much small in `MB` size the quantized model is. In this example we are seeing a reduction of around $87\%$.

You can also explore the architecture of the quantized model by running:

```python
model
```

and the output will be:

```text
MistralGPTQ(
  (model): MistralForCausalLM(
    (model): MistralModel(
      (embed_tokens): Embedding(32768, 12288)
      (layers): ModuleList(
        (0-87): 88 x MistralDecoderLayer(
          (self_attn): MistralAttention(
            (q_proj): TritonV2QuantLinear()
            (k_proj): TritonV2QuantLinear()
            (v_proj): TritonV2QuantLinear()
            (o_proj): TritonV2QuantLinear()
          )
          (mlp): MistralMLP(
            (gate_proj): TritonV2QuantLinear()
            (up_proj): TritonV2QuantLinear()
            (down_proj): TritonV2QuantLinear()
            (act_fn): SiLU()
          )
          (input_layernorm): MistralRMSNorm((12288,), eps=1e-05)
          (post_attention_layernorm): MistralRMSNorm((12288,), eps=1e-05)
        )
      )
      (norm): MistralRMSNorm((12288,), eps=1e-05)
      (rotary_emb): MistralRotaryEmbedding()
    )
    (lm_head): HookedLinear(in_features=12288, out_features=32768, bias=False)
  )
)
```

Comparing the quantized model's architecture with the original architecture, we observe that the `Linear` layers have been replaced with the `TritonV2QuantLinear` quantized layers.

### Performing inference with the quantized model

After quantizing the model and saving it locally, you can test its inference capabilities. The example below loads the quantized model onto a single GPU and performs inference with it:

```python
pretrained_model_id = "mistralai/Mistral-Large-Instruct-2407"
quantized_model_id = f"quantized_{pretrained_model_id.split('/')[1]}_4bit"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_id, use_fast=True,)

# Load the quantized model from local and on a single GPU
device = "cuda:0" 
model = GPTQModel.load(quantized_model_id, device=device, trust_remote_code=True)

# Inference using model.generate
prompt = "In numerical mathematics, quantization is"
output = model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_new_tokens=50)

# Print the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You will be presented with some `INFO` messages and the output will be similar to:

```text
from_quantized: adapter: None
INFO  Loader: Auto dtype (native bfloat16): `torch.bfloat16`                   
INFO  Estimated Quantization BPW (bits per weight): 4.2875 bpw, based on [bits: 4, group_size: 128]
...
INFO  Kernel: loaded -> `[ExllamaQuantLinear]`  

In numerical mathematics, quantization is the process of mapping input values from a large set (often a continuous set) to output values in a (countable) smaller set. Rounding and truncation are typical examples of quantization processes. Quantization is involved to some degree in
```

## Summary

In this blog, you explored GPTQ and the GPTQModel Python package, which efficiently compress Large Language Models using AMD GPUs. By following the step-by-step instructions, you learned how to convert a non-quantized model into an INT4 quantized version with GPTQModel. This process involved preparing a calibration dataset, applying the quantization technique, and achieving significant model size reduction with minimal performance loss. The entire process was completed on AMD GPUs using the AMD ROCm platform, which provided the necessary tools and libraries, making the process accessible and efficient.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
