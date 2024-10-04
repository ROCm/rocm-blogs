---
blogpost: true
date: 3 October 2024
author: Douglas Jia
blog_title: Leaner LLM Inference with INT8 Quantization on AMD GPUs using PyTorch
tags: AI/ML, GenAI, PyTorch, LLM, Optimization, Performance
category: Applications & models
language: English
thumbnail: './images/image_int8.jpg'
myst:
  html_meta:
    "description lang=en": "This blog demonstrates how to use AMD GPUs to implement and evaluate INT8 quantization, and the derived inference speed-up of Llama family and Mistral LLM models."
    "author": "Douglas Jia"
    "keywords": "LLM inference, INT8 quantization, PyTorch, gpt-fast, torch.compile, optimization, performance, Llama, Mistral, AMD, GPU, MI300, MI250, MI210, ROCm, fp16, fp32"
    "property=og:locale": "en_US"
---

# Leaner LLM Inference with INT8 Quantization on AMD GPUs using PyTorch

With the scale of large language models (LLMs) reaching hundred of billions of parameters, the ways we represent data within these enormous models dramatically impacts the resources required to train them (e.g. the number of GPUs needed for inference).
In our previous blogs ([JAX mixed precision training](https://rocm.blogs.amd.com/artificial-intelligence/jax-mixed-precision/README.html); [PyTorch AMP](https://rocm.blogs.amd.com/artificial-intelligence/automatic-mixed-precision/README.html)), we already demonstrated how mixed precision training can accelerate LLMs training process. In this blog post we will push things further and show you how quantization into an even lower precision data formats can speed up inference, saving time and memory, without sacrificing the overall performance of the model.
Quantization is a technique where the precision of a model’s parameters is reduced from a 32-bit floating point (FP32) or a 16-bit floating point (FP16) to an 8-bit integer (INT8). Standard models typically use 32-bit floating-point (FP32) precision. However, this higher precision is not always necessary for inference tasks. By converting model weights and activations to lower precision formats like INT8 (8-bit integer), we can achieve faster computations and lower memory usage, effectively reducing the model size by three-fourths (from 32-bit) or half (from 16-bit) with only a slight accuracy reduction, which is often outweighed by the speed gains.

In this blog post we will show you, step-by-step, how to implement INT8 quantization on AMD GPUs using ROCm, PyTorch and the [gpt-fast repository](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#amd), and how to evaluate the resulting inference performance. Specifically, we will demonstrate how INT8 quantization dramatically improves the inference speeds of Llama family and Mistral LLM models.

## How to Perform Quantization

Most model quantization techniques fall into the following two categories:

* Post-Training Quantization (PTQ): Applied after the model is fully trained. It is simpler but may result in some performance loss.
* Quantization-Aware Training (QAT): Incorporates quantization during the training process so that the quantized weights can better capture information from the data. It often yields better results but requires more computational resources.

For more information on these two strategies, see [LLM Series - Quantization Overview](https://medium.com/@abonia/llm-series-quantization-overview-1b37c560946b).

For our code example below, we will use PTQ. Here are the general steps of PTQ, though the actual steps may vary greatly in different applications.

### 1. Model Preparation

* **Load Pre-trained Model**: Start with a pre-trained model, typically in FP32, FP16, or BF16 format.
* **Define Quantization Configuration**: Specify the quantization scheme and configurations, such as symmetric or asymmetric quantization, and per-channel or per-tensor quantization.

### 2. Calibration

* **Collect Calibration Data**: Gather a representative dataset that captures the distribution of inputs the model will encounter during inference.
* **Run Calibration**: Use the calibration data to run the model and collect statistics such as the minimum and maximum values for each layer's activations. This step determines the values of the scale and zero-point quantization parameters for the weights and activations. The scale and zero-point parameters are analogous to the standard deviation and mean in normalization.

### 3. Quantization and Model Conversion

* **Quantize Weights and Activations**: Quantize the higher precision weights and activations to INT8 using the values of the quantization parameters determined in the calibration step.
* **Convert Model Format**: Use a framework like PyTorch to convert the model to a quantized format.

For the purposes of our demonstration, we will only be quantizing the weights, and we will be skipping the calibration process. The distribution of the weights is known and fixed, and the quantization scale and zero-point parameters for weights can be computed directly from the weight values themselves.

## Implementation

In the official [gpt-fast repository](https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#amd), the authors measured the inference speed of the `meta-llama/Llama-2-7b-chat-hf` model on a MI-250x GPU, focusing on how quickly the model processes data. However, when evaluating the efficiency of inference in a practical setting, it's important to also consider throughput, which is a measure of how much data (in this case, tokens) can be processed in a given amount of time. Throughput is typically expressed in Tokens/Second and provides a broader understanding of the model's performance in real-world applications.

In our implementation, we will use an MI210 GPU, equivalent to one GCD (Graphics Compute Die) of a MI-250x GPU (one MI-250x GPU can be viewed as two MI210 GPUs). We will measure the inference throughput of Llama-2-7B as a baseline, and then extend our testing to three additional popular models: `meta-llama/Meta-Llama-3-8B` (a newer version of the Llama family models), `mistralai/Mistral-7B-v0.1`, and `meta-llama/Llama-2-13b-chat-hf`. For each model, we will test three modes with different levels of optimization to determine their performance:

1. Eager mode (no optimization)
2. Torch.compile
3. Torch.compile + INT8 quantization

`torch.compile` is a PyTorch feature that optimizes model execution by converting it into a more efficient, compiled form for faster runtime performance.

We will use a ROCm Docker container with a nightly PyTorch build for this demonstration. PyTorch is being improved continuously, and the nightly version often contains the latest optimizations. The Docker container will be run on a server with AMD GPUs running Ubuntu.

See [System requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for the complete list of hardware and operating systems supported by AMD.

* Use the following command in a Linux terminal to pull and run the Docker container:

    ```bash
    docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
              --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
              --name=pt_nightly rocm/pytorch-nightly:latest /bin/bash
    ```

  You can see the number of GPUs detected by your PyTorch framework by running the following code in the Python console. PyTorch needs to detect at least one GPU.
  
    ```python
    import torch
    torch.cuda.device_count()
    ```

* Install the required Python packages:

    ```bash
    python3 -m pip install --upgrade pip
    pip install sentencepiece huggingface_hub tiktoken blobfile
    ```

* Download the `gpt-fast` repository using the following command:

    ```bash
    git clone https://github.com/pytorch-labs/gpt-fast.git
    ```

* Download the `generate.py` and `run_commands.sh` files from [this blog's `src` folder on GitHub](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/int8-quantization).  Place `generate.py` and `run_commands.sh` in the `gpt-fast` folder you downloaded, replacing the original `generate.py` file. The original `generate.py` file in the `gpt-fast` repository calculates the average of tokens per second for the benchmarked models but doesn't exclude the first few warm-up rounds.  This may bias the results. We modified and enhanced the `generate.py` file by:

  * Running 30 iterations in total, with the first 20 iterations as a warm-up, and calculating the average and standard deviation of metrics over the last 10 rounds.
  * Calculating the average and standard deviations of the memory bandwidth.

### Run the Benchmarked Models and Collect Metrics

The `run_commands.sh` file below contains commands to download, quantize, and run the benchmarked models to collect the inference metrics `Tokens/Second` and `Memory Bandwidth (GB/s)`.

You will need to provide your Hugging Face credentials to run `run_commands.sh`. Provide your credentials by running `huggingface-cli login` and following the prompts. For information on how to get a Hugging Face access token, see [User access tokens](https://huggingface.co/docs/hub/security-tokens) in the Hugging Face user documentation.

```bash
#!/bin/bash

# Log file
LOGFILE="output_4_models.log"

# Clear the log file if it exists
> $LOGFILE

# Array of model repositories
MODEL_REPOS=("meta-llama/Llama-2-7b-chat-hf" "meta-llama/Meta-Llama-3-8B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-chat-hf")

# Run commands and log output
{
    echo "Processing models"
    
    # Loop through the model repositories
    for MODEL_REPO in "${MODEL_REPOS[@]}"; do
        # Prepare/download the model
        ./scripts/prepare.sh $MODEL_REPO

        echo -e "\n**************Running baseline with $MODEL_REPO..."
        python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
        echo -e "\n**************Running torch.compile with $MODEL_REPO..."
        python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"

        echo "Setting DEVICE to cuda..."
        export DEVICE=cuda

        echo -e "\n**************Quantizing and running commands with $MODEL_REPO..."
        python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
        echo -e "\n**************Running int8 with $MODEL_REPO..."
        python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
    done
} &> $LOGFILE
```

Run `./run_commands.sh` in the `gpt-fast` folder to collect the metrics.

## Benchmark Results

In the table below, we present metrics for the three inference modes used for each model. In the last column, you can see that INT8 quantization improves throughput by about 25-45% compared to torch compile mode, and even more compared to eager mode.  This confirms that model quantization can be used to boost the inference performance of LLMs.

| Model (mode)                  | Tokens/Second | Memory BW (GB/s) | T/S Ratio to Eager | T/S Ratio to Compile |
|-------------------------------|---------------|------------------|--------------------|----------------------|
| **Llama-2-7B (eager)**        | 33.29         | 439.97           | 1                  | -                    |
| **Llama-2-7B (compile)**      | 88.54         | 1170.01          | 2.66               | 1                    |
| **Llama-2-7B (compile + INT8)**| 112.45        | 743.31           | 3.38               | 1.27                 |
| **Llama-3-8B (eager)**        | 32.52         | 488.06           | 1                  | -                    |
| **Llama-3-8B (compile)**      | 76.88         | 1154.01          | 2.36               | 1                    |
| **Llama-3-8B (compile + INT8)**| 110.35        | 828.50           | 3.39               | 1.44                 |
| **Mistral-7B (eager)**        | 32.68         | 464.77           | 1                  | -                    |
| **Mistral-7B (compile)**      | 81.44         | 1158.20          | 2.49               | 1                    |
| **Mistral-7B (compile + INT8)**| 117.05        | 832.67           | 3.58               | 1.44                 |
| **Llama-2-13B (eager)**       | 21.36         | 549.01           | 1                  | -                    |
| **Llama-2-13B (compile)**     | 44.79         | 1151.30          | 2.10               | 1                    |
| **Llama-2-13B (compile + INT8)**| 59.62        | 766.58           | 2.79               | 1.33                 |

## Summary

In this blog post we showed you, step-by-step, how to use AMD GPUs to implement INT8 quantization, and how to benchmark the resulting inference. We demonstrated the speed-up impact of INT8 quantization on the training of Llama family and Mistral LLM models.

## Acknowledgements

We would like to express our gratitude to PyTorch Labs for developing the [gpt-fast repository](https://github.com/pytorch-labs/gpt-fast), which provided the guidelines for our work.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
