---
blogpost: true
blog_title: 'Efficient MoE training on AMD ROCm: How-to use MegaBlocks on AMD GPUs'
thumbnail: 'megablocks-moe.jpeg'
date: 23 March 2025
author: Fabricio Flores, Rishi Madduri, Yao Liu, Phani Vaddadi, Vish Vadlamani
tags: PyTorch, AI/ML, LLM
category: Applications & models
language: English
target_audience: AI developers, AI practitioners
key_value_propositions: Demonstrate how to perform Mixture of Experts training on AMD hardware using MegaBlocks.
myst:
    html_meta:
        "author": "Fabricio Flores, Rishi Madduri, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "Learn how to use MegaBlocks to pre-train GPT2 Mixture of Experts (MoE) model, helping you scale your deep learning models effectiveness on AMD GPUs using ROCm"
        "keywords": "MegaBlocks, Megatron-LM, PyTorch, MoE, LLM, Fine-tuning, distributed training, ROCm, AMD, GPU, MI300"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and Models"
        "amd_developer_type": "ML/AI Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_hardware_platforms": 'Instinct GPUs'
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": 'HPC & Scientific Computing'
        "amd_blog_releasedate": "Fri Mar 21 08:52:00 PST 2025"    
---

# Efficient MoE training on AMD ROCm: How-to use MegaBlocks on AMD GPUs

Training massive deep-learning models requires a balance of efficiency and scalability. In the context of the [Transformers architecture](https://arxiv.org/abs/1706.03762), [Mixture of Experts (MoE)](https://huggingface.co/blog/moe) models are massive machine learning architectures characterized for dividing tasks among multiple specialized sub-networks or "experts". A gating network determines the expert to which a given input should be routed, enabling the model to handle complex tasks more efficiently by using the specialized capabilities of each expert. This dynamic routing mechanism allows MoE models to scale efficiently, activating only a subset of the network for each input, therefore reducing computational load while maintaining high model capacity.

Implementing MoE models introduces challenges such as training instability, load imbalance among experts, and increased complexity of routing mechanism management and expert selection. These challenges lead to inefficient memory usage due to the uneven data distribution among the expert sub-networks.

[MegaBlocks](https://github.com/ROCm/megablocks) is a lightweight library for MoE training that uses block-sparse computations to reduce overhead and improve training scalability. Block-sparse computations deal with matrices that are broken down into smaller, denser blocks. Many of these blocks have all their elements set to zero. The main operations happen on the remaining non-zero blocks allowing for efficient storage and computation. MegaBlocks solves the challenges of regular MoE training by reworking computations so that each expert can handle varying amounts of data without discarding or padding. This results in improved training efficiency and performance.

MegaBlocks is highly integrated with [Stanford-Megatron-LM](https://github.com/ROCm/Stanford-Megatron-LM), a scalable framework for training transformer models that implements model and data parallelism strategies. This integration enables MegaBlocks to distribute training workload across multiple GPUs efficiently, ensuring that block-sparse computations are executed with high performance and minimal overhead. To learn more about Megatron-LM on ROCm, see [Training a model with ROCm Megatron-LM](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/train-a-model.html#start-training-on-mi300x-accelerators).

In this blog, you will learn how to use MegaBlocks for Mixture-of-Experts (MoE) training on AMD hardware, gaining insights into its application and effectiveness in training massive transformer-based models. You will also explore the `bigscience/misc-test-data` repository on Hugging Face, which offers datasets like `oscar-1GB.jsonl`, commonly used for pre-training language models and running tests or experiments.

For the files related to this blog post, see this
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/megablocks).

## ROCm MegaBlocks

* [Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/megablocks-install.html)

* [Docker image](https://hub.docker.com/r/rocm/megablocks/tags)

* [GitHub](https://github.com/ROCm/megablocks)

## ROCm Stanford-Megatron-LM

* [Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/stanford-megatron-lm-install.html)

* [Docker image](https://hub.docker.com/r/rocm/stanford-megatron-lm/tags)

* [GitHub](https://github.com/ROCm/Stanford-Megatron-LM)

## Requirements

* AMD GPU: See [System requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems.

* ROCm: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) page for installation instructions.

* Docker: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) for installation instructions.

* PyTorch 2.4 and [ROCm fork of Megablocks](https://github.com/ROCm/megablocks): You can use this [Dockerfile](./docker/Dockerfile/) to prepare a custom Docker image with MegaBlocks pre-installed.

## Getting started with MegaBlocks

* Clone the repo and `cd` into the blog directory:

    ```shell
    git clone https://github.com/ROCm/rocm-blogs.git
    cd rocm-blogs/blogs/artificial-intelligence/megablocks
    ```

* Build the Docker image. For details on the build process, see the `./megablocks/docker/Dockerfile`.

    ```shell
    cd docker
    docker build -t megablocks -f Dockerfile .
    cd ..
    ```

* Start the MegaBlocks container

  ```shell
  docker run -it --rm \
  --privileged -v ./:/app \
  --network=host --device=/dev/kfd \
  --device=/dev/dri --group-add video \
  --name=my_megablocks --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  megablocks
  ```
  
## The oscar-1GB dataset

The [`bigscience/misc-test-data`](https://huggingface.co/bigscience/misc-test-data) repository on Hugging Face contains various datasets intended for testing and experimentation. A dataset commonly used for pre-training language models is the [oscar-1GB.jsonl](https://huggingface.co/bigscience/misc-test-data/tree/main/stas) dataset. This dataset consists of JSON formatted data where each line in the file represents a JSON object consisting of a text sample. One sample of this dataset looks as follows:

```text
{'id': 87297360,
 'text': 'Research on the System Cohesion of Basic Health Care for the Urban '
         "and Rural Residents in Hainan Province - Master's thesis - "
         'Dissertation\n'
         'Research on the System Cohesion of Basic Health Care for the Urban '
         'and Rural Residents in Hainan Province\n'
         'Keywords: The new rural cooperative medical care system the basic '
         'medicalinsurance system for urban residents system cohesion\n'
         'The new health care plan established everyone will have access to '
         'basic medical andhealth services, to establish the basic medical and '
         'health system covering both urban andrural residents, Party’s sixth '
         'plenary of sixteen session and will to2020basic to establishthe '
          ...
         'City from the Perspective of Fairness,F323.89\n'
         'A Study on System and Operation of New Rural Cooperative Medical '
         'System in Linfen City of Shanxi,F323.89\n'
         'Study on the Interface between the New-type Rural Cooperative '
         'Medical System and the Basic Medical Insurance System for Urban '
         'Residents,C913.7\n'
         'Distance from Ensenada to Huntington Beach is 238 kilometers. This '
         'air travel distance is equal to 148 miles.\n'
         'The air travel (bird fly) shortest distance between Ensenada and '
         'Huntington Beach is 238 km= 148 miles.\n'
         'If you travel with an airplane (which has average speed of 560 '
         'miles) from Ensenada to Huntington Beach, It takes 0.26 hours to '
         'arrive.'}

```

## Data preprocessing

When training large-scale language models, data preprocessing is an important step in converting raw text into a suitable format for model training. The oscar-1GB dataset can be processed using the preprocessing script [preprocess_data.py](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/megablocks/third_party/Stanford-Megatron-LM/tools). The script tokenizes the text using a specified tokenizer, such as BERT or GPT tokenizer, and generates a binary file that the model utilizes during training. For the GPT tokenizer, the preprocessing script additionally requires a merge table file [merges.txt](https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt) and a vocabulary file [vocab.json](https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json).

The [merges.txt](https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt) file contains merge operations for the Byte-Pair Encoding (BPE) tokenizer. These operations guide how the tokenizer combines byte pairs to form tokens during text processing. Each line in the [merges.txt](https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt) file specifies a pair of tokens to be merged, ordered by frequency, which helps construct the final vocabulary. For more information see, [Hugging Face Transformers Tokenization GPT2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py) and [Hugging Face Transformers RoBERTa Tokenizer and merges file](https://github.com/huggingface/transformers/issues/1083#issuecomment-524303077).

The [vocab.json](https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json) file contains a mapping of tokens to their corresponding integer IDs. This mapping is needed for the tokenization process, where the text data from `oscar-1GB` dataset is transformed into token IDs that the model can process. Each token, a word, or sub word unit, is assigned a unique identifier in this file.

For the data preprocessing step, you can use this [data_preprocessing.sh](./src/data_preprocessing.sh) bash script. This script contains all the necessary steps for data preprocessing, from downloading the `oscar-1GB` datasets and tokenizer-related files to generating the binary output file needed by the model.

To begin with the preprocessing task, inside the running `megablocks` container, execute each of the following commands:

```bash
cd /app/src
chmod +x data_preprocessing.sh
./data_preprocessing.sh
```

The terminal will display the following output:

```bash
Obtaining dataset, vocabulary, and merge table from HuggingFace:
--2025-02-04 20:19:29--  https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
Resolving huggingface.co (huggingface.co)... 3.167.112.96, 3.167.112.45, 3.167.112.25, ...
Connecting to huggingface.co (huggingface.co)|3.167.112.96|:443... connected.
HTTP request sent, awaiting response... 302 Found
...
Preprocessing training data:
Opening /megablocks/third_party/Stanford-Megatron-LM/tools/oscar-1GB.jsonl
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
> building GPT2BPETokenizer tokenizer ...
Vocab size: 50257
Output prefix: /megablocks/third_party/Stanford-Megatron-LM/tools/my-gpt2
...
Processed 78500 documents (1318.7705640553934 docs/s, 16.986236084312864 MB/s).
Processed 78600 documents (1313.2722290811087 docs/s, 16.93954182020718 MB/s).
Processed 78700 documents (1314.4503426379306 docs/s, 16.955740551038325 MB/s).
Processed 78800 documents (1315.7184569953088 docs/s, 16.976525954590592 MB/s).
Processed 78900 documents (1317.04839458361 docs/s, 16.990241075903537 MB/s).
Processed 79000 documents (1315.9978437097366 docs/s, 16.98728794598181 MB/s).
Done! Now finalizing.

```

This results in two output files `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`, which are located at `/megablocks/third_party/Stanford-Megatron-LM/tools`. These files will be used during the training process.

## Testing MegaBlocks for GPT2-125m distributed training

The ROCm MegaBlocks [GitHub repository](https://github.com/ROCm/megablocks/tree/main/exp) contains several examples that you can test. The following example is based on the [`gpt2_125m_8gpu.sh`](https://github.com/ROCm/megablocks/blob/main/exp/gpt2/gpt2_125m_8gpu.sh) script for pretraining the GPT2-125 million parameter model.

You can use this [megablocks_gpt2_125m_8gpu.sh](./src/megablocks_gpt2_125m_8gpu.sh) script that contains all the necessary instructions and parameters needed to start the training process. Some of the parameters listed in `megablocks_gpt2_125m_8gpu.sh` bash script are:

* The number of training steps is set to 2000 for demonstration purposes and to verify the setup:

  ```bash
    TRAINING_STEPS=2000
  ```

* The model hyperparameters for the GPT2-125m architecture:

  ```bash
  # Model hyperparameters.
  MODEL_ARGUMENTS="\
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --seq-length 1024 \
  --max-position-embeddings 1024"
  ```

  ```{note}
  The `MODEL_ARGUMENTS` values affects the total number of parameters of the model. See [MoE examples](https://github.com/ROCm/megablocks/blob/main/exp/moe) for models with different sizes such as 46M and 356M parameters.
  ```
  
  * The parameters associated with distributed training. The script has a single node with 8 GPUs, the rank (ID) of the current node set to zero, the IP address of the master node set to `localhost`, and the port used for communication between the nodes:

  ```bash
  # Distributed hyperparameters.
  DISTRIBUTED_ARGUMENTS="\
  --nnodes 1 \  
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 6000"
  ```

To begin with the training process, run each of the following commands:

```bash
cd /app/src
chmod +x megablocks_gpt2_125m_8gpu.sh
./megablocks_gpt2_125m_8gpu.sh
```

Initially, some verbose processing information is displayed in the terminal. Then, you will see the following output:

```bash
...

using world size: 8, data-parallel-size: 8, tensor-model-parallel size: 1, pipeline-model-parallel size: 1 
accumulate and all-reduce gradients in fp32 for bfloat16 data type.
using torch.bfloat16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. True
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  barrier_with_L1_time ............................ True

...

training ...
[before the start of training step] datetime: 2025-02-05 21:20:40
 iteration   100/2000 | consumed samples: 51200 | elapsed time per iteration (ms): 660.6 | learning rate: 5.978E-04 | global batch size:   512 | lm loss: 7.643560E+00 | loss scale: 1.0 | grad norm: 0.207 | number of skipped iterations:   0 | number of nan iterations:   0 |

...

 iteration   1000/2000 | consumed samples:       512000 | elapsed time per iteration (ms): 657.0 | learning rate: 3.343E-04 | global batch size:   512 | lm loss: 5.360341E+00 | loss scale: 1.0 | grad norm: 0.783 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 5.314413E+00 | lm loss PPL: 2.032451E+02 | 
------------------------------------------------------------------------------------------------
...
 iteration   2000/2000 | consumed samples:      1024000 | elapsed time per iteration (ms): 657.1 | learning rate: 6.000E-05 | global batch size:   512 | lm loss: 4.640055E+00 | loss scale: 1.0 | grad norm: 0.344 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 4.583530E+00 | lm loss PPL: 9.785927E+01 | 
------------------------------------------------------------------------------------------------

saving checkpoint at iteration    2000 to /megablocks/third_party/Stanford-Megatron-LM/checkpoints
successfully saved checkpoint at iteration    2000 to /megablocks/third_party/Stanford-Megatron-LM/checkpoints
(min, max) time across ranks (ms):save-checkpoint ................................: (1658.80, 1658.85)

[after training is done] datetime: 2025-02-05 21:43:14 
saving checkpoint at iteration    2000 to /megablocks/third_party/Stanford-Megatron-LM/checkpoints
------------------------------------------------------------------------------------------------------------------
validation loss at the end of training for val data | lm loss value: 4.580711E+00 | lm loss PPL: 9.758374E+01 | 
------------------------------------------------------------------------------------------------------------------
successfully saved checkpoint at iteration    2000 to /megablocks/third_party/Stanford-Megatron-LM/checkpoints
-------------------------------------------------------------------------------------------------------------------
validation loss at the end of training for test data | lm loss value: 4.529140E+00 | lm loss PPL: 9.267882E+01 | 
-------------------------------------------------------------------------------------------------------------------
```

The consistent downward trend in the validation loss scores demonstrates that the model was improved by the training process.

## Testing MegaBlocks for GPT2-125m distributed training using MoE

This example demonstrates the training process for a 125-million-parameter GPT-2 Transformer model, utilizing the Mixture of Experts (MoE) architecture and building upon a previous GPT-2 training example. The `MOE_ARGUMENTS` variable defines the MoE architecture that modifies the Transformers section of the GPT-2 model accordingly. Additional parameters included in the `megablocks_moe_gpt2_125m_8gpu.sh` bash script are:

* The number of training steps. For demonstration it is set to 2000:

  ```bash
    TRAINING_STEPS=2000
  ```

* The model hyperparameters for the GPT2-125m architecture:

  ```bash
  # Model hyperparameters.
  MODEL_ARGUMENTS="\
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --seq-length 1024 \
  --max-position-embeddings 1024"
  ```

  * The hyperparameters associated with distributed training. Set as a single node with 8 GPUs:

  ```bash
  # Distributed hyperparameters.
  DISTRIBUTED_ARGUMENTS="\
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 6000"
  ```

  * The Mixture of Expert hyperparameters that specify:
    * The total number of experts in the MoE layer.
    * The capacity of each expert, that is the number of tokens an expert can process.
    * The weigh of the auxiliary loss in MoE training. Helps balancing the expert usage and prevent under-utilization of certain experts.
    * The number of experts selected to process each input token.

  ```bash
  # MoE hyperparameters.
  MOE_ARGUMENTS="\
  --moe-num-experts=64 \
  --moe-capacity-factor=1 \
  --moe-loss-weight=0.1 \
  --moe-top-k=1"  
    ```

To begin with the training process, run each of the following commands:

```bash
cd /app/src
chmod +x megablocks_moe_gpt2_125m_8gpu.sh
./megablocks_moe_gpt2_125m_8gpu.sh
```

After some preprocessing messages, you will see the following output in the terminal:

```bash
...

(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (55.17, 73.82)
    train/valid/test-data-iterators-setup ..........: (1015.72, 1115.27)
training ...
[before the start of training step] datetime: 2025-02-14 15:43:22 
...

 iteration      200/    2000 | consumed samples:       102400 | elapsed time per iteration (ms): 580.2 | learning rate: 1.477E-04 | global batch size:   512 | lm loss: 6.711773E+00 | load balancing loss: 1.010925E-01 | loss scale: 32768.0 | grad norm: 0.428 | number of skipped iterations:   0 | number of nan iterations:   0 |

...

------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 4.735407E+00 | lm loss PPL: 1.139099E+02 | 
------------------------------------------------------------------------------------------------
 iteration     1100/    2000 | consumed samples:       563200 | elapsed time per iteration (ms): 820.3 | learning rate: 7.202E-05 | global batch size:   512 | lm loss: 4.714045E+00 | load balancing loss: 9.930359E-02 | loss scale: 65536.0 | grad norm: 0.681 | number of skipped iterations:   0 | number of nan iterations:   0 |

...

 iteration     2000/    2000 | consumed samples:      1024000 | elapsed time per iteration (ms): 582.8 | learning rate: 1.003E-05 | global batch size:   512 | lm loss: 4.356625E+00 | load balancing loss: 9.936462E-02 | loss scale: 65536.0 | grad norm: 0.406 | number of skipped iterations:   0 | number of nan iterations:   0 |
------------------------------------------------------------------------------------------------
 validation loss at iteration 2000 | lm loss value: 4.342843E+00 | lm loss PPL: 7.692593E+01 | 
------------------------------------------------------------------------------------------------

saving checkpoint at iteration    2000 to ./
  successfully saved checkpoint at iteration    2000 to ./
(min, max) time across ranks (ms):
    save-checkpoint ................................: (293638.70, 293638.73)
[after training is done] datetime: 2025-02-14 16:08:33

------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 4.341037E+00 | lm loss PPL: 7.678711E+01 | 
------------------------------------------------------------------------------------------------------------------
  successfully saved checkpoint at iteration    2000 to ./
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 4.352861E+00 | lm loss PPL: 7.770044E+01 | 
-------------------------------------------------------------------------------------------------------------------
```

From the output, you can see that the pre-training of the GPT2 Mixture of Experts (MoE) model was conducted successfully on a node with eight AMD Instinct™ MI300X GPUs. The training process ran for 2000 steps, during which the loss (`lm loss value`) decreased consistently (from 6.71 at iteration 200, to 4.74 at iteration 1000, to 4.34 at iteration 2000), indicating effective learning.

MegaBlocks makes MoE training easier by letting you set up MoE layers with just a few simple parameters in the `MOE_ARGUMENTS` inside the bash script. This speeds up prototyping and experimentation.

## Summary

This blog shows you how to use MegaBlocks for Mixture of Experts (MoE) training on AMD hardware, with a clear example using the GPT-2 model. It gives a quick overview of MoE and explains how it helps scale deep learning models across multiple GPUs. If you want to boost efficiency or try out MoE training on AMD hardware, this guide will help you get started. Future work will present performance benchmarking of MoE models on AMD GPUs, focusing on metrics like throughput, latency, and power efficiency.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
