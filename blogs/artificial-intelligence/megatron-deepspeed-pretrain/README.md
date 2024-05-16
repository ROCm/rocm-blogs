---
blogpost: true
date: 24 Jan 2024
author: Douglas Jia
tags: LLM, AI/ML, Tuning, PyTorch
category: Applications & models
language: English

myst:
  html_meta:
    "description lang=en": "Pre-training a large language model with
  Megatron-DeepSpeed on multiple AMD GPUs"
    "keywords": "Megatron, language model, fine-tuning, DeepSpeed,
  Generative AI, Megatron-DeepSpeed, GPT-3, 3D parallelism, AMD GPU, MI300, MI250,
    flash-attention, AAC, Tuning"
    "property=og:locale": "en_US"
---

# Pre-training a large language model with Megatron-DeepSpeed on multiple AMD GPUs

In this blog, we show you how to pre-train a GPT-3 model using the Megatron-DeepSpeed
framework on multiple AMD GPUs. We also demonstrate how to perform inference on the
text-generation task with your pre-trained model.

## What is Megatron-DeepSpeed?

Microsoft developed Megatron-DeepSpeed by integrating their
[DeepSpeed](https://www.microsoft.com/en-us/research/project/deepspeed/) library into NVIDIA's
[Megatron-LM](https://arxiv.org/abs/1909.08053) framework.

DeepSpeed is Microsoft's optimization library. It was designed to simplify and enhance distributed
training and inference. DeepSpeed introduces a suite of optimizations that streamline processes,
making them efficient and effective.

Megatron-LM is NVIDIA's large and powerful transformer. It can handle massive models and complex
deep-learning tasks, making it an ideal starting point for the advancements brought by DeepSpeed.

What sets Megatron-DeepSpeed apart is its comprehensive support for an array of features, from
mixture-of-experts model training to curriculum learning. This makes it a versatile tool for handling
diverse challenges in the realm of deep learning.

Using Megatron-DeepSpeed, you can train larger models with unprecedented efficiency and scale.

### 3D parallelism

The highlight of Megatron-DeepSpeed is its implementation of 3D parallelism. This approach
combines Zero Redundancy Optimizer (ZeRO) sharding, pipeline parallelism from DeepSpeed, and
Tensor parallelism from Megatron-LM. This combination allows you to efficiently train colossal models,
which opens up new frontiers in model scalability.

ZeRO, as with TensorParallel, performs tensor sharding. What sets ZeRO apart is its ability to
reconstruct the entire tensor in time for computations without any model modification. This innovative
approach also supports various offloading techniques to deal with GPU memory constraints.

Megatron-DeepSpeed introduces three key components of 3D parallelism:

* DataParallel: Replicates setups and processes data slices in parallel, synchronizing at the end
  of each step.
* TensorParallel: Distributes tensor shards across GPUs for independent parallel processing,
  allowing for a horizontal split.
* PipelineParallel: Vertically splits the model across GPUs at the layer level, enabling parallel
  processing of different stages.

## Why use an AMD GPU?

AMD GPUs offer robust open-source support, featuring tools like ROCm and HIP, making them easily
adaptable to AI workflows. Our competitive price-to-performance ratios cater to anyone seeking
cost-effective solutions for AI and deep-learning tasks. As AMD's presence in the market grows, more
machine-learning libraries and frameworks are adding AMD GPU support.

## Hardware and software requirements

To achieve the computational capabilities required for this task, we use the
[AMD Accelerator Cloud (AAC)](https://aac.amd.com/), which is a platform that offers on-demand
cloud computing resources and APIs. On AAC, we use a
[PyTorch Docker container](https://hub.docker.com/r/rocm/pytorch) (version: rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1) with 8 GPUs.

Our methods are hardware-agnostic, meaning that access to AAC is **not** a requirement for
successfully running our code examples. As long as you have access to accelerator devices, such
as GPUs or tensor processing units (TPUs), you should be able to run the code examples with
minimal modification. If you're using AMD GPUs, make sure ROCm and its compatible version of
PyTorch are installed correctly. Refer to the following two tutorials for installation instructions:

* [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
* [PyTorch installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)

## Code example on pre-training of a GPT-3 model

First, install DeepSpeed (and other required packages) and clone the Megatron-DeepSpeed GitHub
repository to your local (or to a server). You then need to download and pre-process the data set
you'll use for pre-training. The cell blocks with `%%sh` represent Linux command line code. We use
`/home/aac` for our home directory; replace this with your home directory when running the code.

```sh
%%sh
python -m pip install --upgrade pip

#Install DeepSpeed
home_dir=$PWD
cd $home_dir
git clone --recursive https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install .[dev,1bit,autotuning]

# Clone Megatron-DeepSpeed

cd $home_dir
git clone https://github.com/microsoft/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
pip3 install pybind11 nltk transformers

# Install libaio-dev

sudo apt-get update
sudo apt-get -y install libaio-dev

# Download data set

cd dataset
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz
bash download_vocab.sh

# Pre-process data for oscar dataset

export BASE_SRC_PATH=$home_dir/Megatron-DeepSpeed
export BASE_DATA_PATH=${BASE_SRC_PATH}/dataset
python3 ${BASE_SRC_PATH}/tools/preprocess_data.py --input ${BASE_DATA_PATH}/oscar-1GB.jsonl --output-prefix ${BASE_DATA_PATH}/my-gpt2 --vocab-file ${BASE_DATA_PATH}/gpt2-vocab.json --dataset-impl mmap --tokenizer-type GPT2BPETokenizer --merge-file ${BASE_DATA_PATH}/gpt2-merges.txt --append-eod --workers 8

# Install FlashAttention (optional). FlashAttention delivers a rapid and memory-efficient
# solution for attention mechanisms. If you don't want to use FlashAttention, remove
# the '--use-flash-attn' flag in the script.

cd $home_dir
git clone --recursive https://github.com/ROCmSoftwarePlatform/flash-attention.git
cd flash-attention
py_version=$(python -V | grep -oP '(?<=[.])\w+(?=[.])')
patch /opt/conda/envs/py_3.${py_version}/lib/python3.${py_version}/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch
python setup.py install
```

Next, train a small GPT-3 model with 8 GPUs in one node. The main training script is
`ds_pretrain_gpt_125M_flashattn.sh`. You must revise several lines of code to match your intended
configuration (e.g., how often to save the model checkpoints, and how to set up the 3D parallelism
configuration). Here is a list of configurations you may need to revise:

* `num_gpus`
* `num_gpus_pernode`
* `num_node`
* `log_interval`
* `eval_iters`
* `eval_interval`
* `num_save`
* `save_interval`
* `vocab_path`
* `merge_path`
* `data_path`
* File paths in `data_options`

Because ROCm doesn't currently support gradient accumulation fusion, you must add
`--no-gradient-accumulation-fusion` to `megatron_options`. You can take a look at the [actual training script](https://github.com/microsoft/Megatron-DeepSpeed/compare/main...jiagaoxiang:Megatron-DeepSpeed:main) we used to gain an understanding of what needs to be revised and how to approach it.

```sh
%%sh
cd /home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase
nohup bash ds_pretrain_gpt_125M_flashattn.sh &
```

Pre-training output is saved in the output folder. You can verify that they're present
if you want to make sure everything is working correctly.

## Convert the DeepSpeed checkpoint to Hugging Face checkpoint

The checkpoint saved by the Megatron-DeepSpeed package is in DeepSpeed format. You can
convert it to Megatron or Hugging Face formats using the functions provided in
`tools/convert_checkpoint` folder. In our inference example, we convert the checkpoint to
Hugging Face format. You may need to modify the
`tools/convert_checkpoint/deepspeed_to_megatron.py` file in order to run the program (change
`from .deepspeed_checkpoint import ARGS_KEY, DeepSpeedCheckpoint` to
`from deepspeed_checkpoint import ARGS_KEY, DeepSpeedCheckpoint`). We convert the checkpoints
from 2,000 and 8,000 iterations so we can compare the performance on inference. You must
modify the paths to the checkpoints in the Python command to match their local paths.

```sh
%%sh
# Install required packages for this step

pip install matplotlib megatron megatron.core transformers

# Convert checkpoint at 8,000 iterations to HF transformers format

python /home/aac/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_transformers.py  \
--input_folder /home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/global_step8000 \
--output_folder /home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/HF/global_step8000

# Convert another checkpoint at 2,000 iterations so we can compare the model performance

python /home/aac/Megatron-DeepSpeed/tools/convert_checkpoint/deepspeed_to_transformers.py  \
--input_folder /home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/global_step2000 \
--output_folder /home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/HF/global_step2000
```

## Load the pre-trained models and perform text generation tasks

Now you can assess the performance of your pre-trained model. While pre-trained models typically
undergo fine-tuning for downstream tasks, you can still gain insights into the capabilities of
your pre-trained model using a text generation task. Loading checkpoints from 2,000 and 8,000
iterations into `model0` and `model1`, respectively, we can evaluate their text generation capabilities
using the prompt "I like to play golf. Today is a sunny day, and I plan to." Each model generates three
samples based on this prompt. Modify the paths `path0` and `path1` to `model0` and `model1`
according to the model you're using.

```python
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import set_seed
import torch

path0 = "/home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/HF/global_step2000/"
path1 = "/home/aac/Megatron-DeepSpeed/examples_deepspeed/rebase/output/checkpoint/gpt_0.125B_tok300B_lr6.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs256_mbs2_g8_z1_mp2_pp2_seed1234_rebase/HF/global_step8000/"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer(vocab_file='/home/aac/Megatron-DeepSpeed/dataset/gpt2-vocab.json', merges_file='/home/aac/Megatron-DeepSpeed/dataset/gpt2-merges.txt')
model0 = GPT2LMHeadModel.from_pretrained(path0, pad_token_id=tokenizer.eos_token_id).to(torch_device)
model1 = GPT2LMHeadModel.from_pretrained(path1, pad_token_id=tokenizer.eos_token_id).to(torch_device)
```

```python
# For more information on how to fine-tune the text generation process,
# see: https://huggingface.co/blog/how-to-generate

# Encode the context to condition the generation

model_inputs = tokenizer('I like to play golf. Today is a sunny day and I plan to', return_tensors='pt').to(torch_device)

# Set the seed to reproduce results (you can change the seed to get different results)

set_seed(1)

# Set top_k = 50, top_p = 0.95, and num_return_sequences = 3

sample_outputs = model0.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("Output with checkpoint from 2000 iterations:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Set top_k = 50, top_p = 0.95, and num_return_sequences = 3

sample_outputs = model1.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("\nOutput with checkpoint from 8000 iterations:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

```

```sh
Output with checkpoint from 2,000 iterations:
----------------------------------------------------------------------------------------------------
0: I like to play golf. Today is a sunny day and I plan to work and get to work with my team. I think that I can make money but I make the effort to get to see this. I know how it works, but I do think that will
1: I like to play golf. Today is a sunny day and I plan to go to the side of my life. It’s really simple! We have been there for a couple of days to try our training program. I have heard the video out there, I think
2: I like to play golf. Today is a sunny day and I plan to get along that summer. A great weekend and a good one can be prepared. I'm a great place to try. It's fun to go and give you the chance to get along with me

Output with checkpoint from 8,000 iterations:
----------------------------------------------------------------------------------------------------
0: I like to play golf. Today is a sunny day and I plan to play some golf in the evening. I have not played my other tournaments until this morning.
1: I like to play golf. Today is a sunny day and I plan to play the whole week of golf. I will be playing in the backyards to play golf. If you are still interested in playing the “American Association” Tournament, please don't hesitate
2: I like to play golf. Today is a sunny day and I plan to get there on Monday morning. You’ll notice me playing in the backyard. My dad bought me the equipment, so I could throw it at home. When we went out to dinner we
```

Our analysis of the generated samples reveals that `model1` produces more logical text and stays
more relevant to the provided context. Note that we achieved this capability with 8 MI210 GPUs
running for less than two days (the time it takes will vary depend on the GPU models you use).

If you prefer to skip the extensive pretraining process, you can directly retrieve these two model
checkpoints from Hugging Face, as shown here:

```python
model3 = GPT2LMHeadModel.from_pretrained('jiagaoxiang/gpt3-125M-2000iter', pad_token_id=tokenizer.eos_token_id).to(torch_device)
model4 = GPT2LMHeadModel.from_pretrained('jiagaoxiang/gpt3-125M-8000iter', pad_token_id=tokenizer.eos_token_id).to(torch_device)

model_inputs = tokenizer('I like to play golf. Today is a sunny day and I plan to', return_tensors='pt').to(torch_device)

# Set seed to reproduce results. You can change the seed to get different results.

set_seed(1)

# Set top_k = 50, top_p = 0.95, and num_return_sequences = 3

sample_outputs = model3.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("Output with checkpoint from 2000 iterations:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

# Set top_k = 50, top_p = 0.95, and num_return_sequences = 3

sample_outputs = model4.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("\nOutput with checkpoint from 8000 iterations:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

```

```sh
Output with checkpoint from 2000 iterations:
----------------------------------------------------------------------------------------------------
0: I like to play golf. Today is a sunny day and I plan to work and get to work with my team. I think that I can make money but I make the effort to get to see this. I know how it works, but I do think that will
1: I like to play golf. Today is a sunny day and I plan to go to the side of my life. It’s really simple! We have been there for a couple of days to try our training program. I have heard the video out there, I think
2: I like to play golf. Today is a sunny day and I plan to get along that summer. A great weekend and a good one can be prepared. I'm a great place to try. It's fun to go and give you the chance to get along with me

Output with checkpoint from 8000 iterations:
----------------------------------------------------------------------------------------------------
0: I like to play golf. Today is a sunny day and I plan to play some golf in the evening. I have not played my other tournaments until this morning.
1: I like to play golf. Today is a sunny day and I plan to play the whole week of golf. I will be playing in the backyards to play golf. If you are still interested in playing the “American Association” Tournament, please don't hesitate
2: I like to play golf. Today is a sunny day and I plan to get there on Monday morning. You’ll notice me playing in the backyard. My dad bought me the equipment, so I could throw it at home. When we went out to dinner we
```
