---
blogpost: true
date: 28 Aug 2024
blog_title: "Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission"
thumbnail: "2024-10-03-mlperf.jpeg"
author: Meena Arunachalam, Miro Hodak, Jeremy Arnold, Eliot Li
tags: AI/ML, LLM
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission"
    "keywords": "MLPerf, Inferencing, AMD, GPU, MI300, LLM, Llama2"
    "property=og:locale": "en_US"
---

# Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission

## Introduction

Measuring the performance of new technologies is as old as human history, and often as intriguing (consider for example that we still compare the performance of new electric vehicle motors using horsepower). In the rapidly advancing field of machine learning (ML) MLPerf was established by [MLCommons](https://mlcommons.org/) on May 2nd 2018 and quickly became the golden standard of measuring the accuracy, speed, and efficiency of AI. MLPerf provides benchmarks on training, HPC and Inference performance. Companies across the industry use MLPerf submissions to evaluate the performance of various GPUs and software platforms, and make their technology adoption decisions based on these results.

Recently, two competitive MLPerf Inference submissions were made using AMD’s Instinct TM MI300X GPUs (one by AMD, and the other by Dell), you can find the results
[here](https://mlcommons.org/benchmarks/inference-datacenter/), and read more on how well our GPUs did [here](https://community.amd.com/t5/instinct-accelerators/engineering-insights-unveiling-mlperf-results-on-amd-instinct/ba-p/705623). In this blog post we will show you, step-by-step, how to reproduce the results of AMD’s submission to MLPerf, on your own, using ROCm and AMD Instinct TM MI300X GPU. So, roll up your sleeves and let’s get started!

## MLPerf Submission

The AMD MLPerf Inference v4.1 submission has three entries for Llama 2 70B. The submission used a fully open-source software stack based on the ROCm platform and vLLM inference engine. Because of this, interested users can build on AMD's submissions and customize the software stack for their own high-performance inference workload on MI300X GPUs. The submission entries are:

1. 8xMI300X with 2x AMD EPYC 9374F (Genoa) CPU in the Available category. This entry showcased the best combination of AMD CPU and GPU available on the market for AI tasks.
2. 1xMI300X with 2x AMD EPYC 9374F (Genoa) CPU in the Available category. This entry demonstrated how the memory capacity of the MI300X (192GB) enabled it to run the entire Llama 2 70B model, unlike many of the competing entries, where the task needed to be split between multiple accelerators.
3. 8xMI300X with 2x AMD EPYC Turin CPU in the Preview category. This entry showed how AMD’s next generation of CPU improves performance of AI tasks.

## Setup

### Prerequisites

To follow along with this blog, you will need the following:

- 8 [MI300X AMD GPUs](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html).
- ROCm 6.1.0 or above.
- [Any supported Linux distributions supported by the version of ROCm you are using](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).

See the [ROCm Quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) for information on how to install ROCm.

There are four main steps to set up your own system to try to generate the results of the first entry in the submission.

- Download the Llama 2 70B model.
- Download the dataset specified by MLPerf to run inference.
- Prepare the docker container.
- Quantize the Llama 2 70B model to FP8 format.

We provide detailed instructions for each of these steps below.

### Model Preparation

Download the Llama 2 70B model weight to a location on your file system using the instructions in the [Get Model section](https://github.com/mlcommons/inference/blob/v4.1/language/llama2-70b/README.md#get-model) of the MLcommons github repo.

Set the environment variable, `$LAB_MODEL`, to the path to the model weight directory:

```bash
export LAB_MODEL="<path to model weight>"
```

### Dataset Preparation

Download the preprocessed dataset files associated with the Llama 2 70B model using the instructions in the [Get Dataset section](https://github.com/mlcommons/inference/blob/v4.1/language/llama2-70b/README.md#preprocessed) of the MLCommons github repo.

Set the `$LAB_DATASET` environment variable to point to the `open_orca` directory in the dataset directory.

```bash
export LAB_DATASET="<path to dataset>/open_orca/"
```

### AMD MLPerf Inference Docker Container Setup

To build the docker container to run the inference, clone the repo associated with this blog and cd to the `src/docker` directory:

```bash
git clone https://github.com/ROCm/rocm-blogs.git
cd rocm-blogs/blogs/artificial-intelligence/mlperf-inf-4-1/src/docker
```

Build the Docker image and launch a container using the commands below. Set the environment variable `$LAB_HIST` to point to the directory where benchmark outputs will be stored.

``` bash
# set env variable LAB_HIST
export LAB_HIST="<path to the output>"

# Build the image `mlperf/llama_inference:latest`
bash build_llama2.sh

# Launch a docker container
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v ${LAB_MODEL}:/data/llm/llama2-70b-chat \
    -v ${LAB_DATASET}:/data/open_orca \
    -v ${LAB_HIST}:/lab-hist \
    -e LAB_CLOG=/lab-hist/mlperf-results \
    mlperf/llama_inference:latest
```

### Quantization Preparation

One of the most important components of the submission was to quantize the model to leverage the FP8 support of MI300X.  [Quark](https://quark.docs.amd.com/latest/index.html) was used to quantize the Llama 2 70B chat model to OCP FP8-e4m3 format, using the calibration dataset required by MLPerf. Quark is a deep learning model quantization toolkit developed by AMD for quantizing models from PyTorch, ONNX, and other frameworks.

Quantize the model by running the commands below in the inference container:

``` bash
model_dir=/data/llm/llama2-70b-chat
output_dir=/data/llm/llama2-70b-chat/quantized/quark_share/modelzoo/llama2_70b_wfp8_afp8_ofp8_nomerge/json-safetensors/
calib_dataset=/data/open_orca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

cd /lab-mlperf-inference/code/llama2-70b-99.9/tools/quark-0.1.0+a9827f5-mlperf/examples/torch/language_modeling/

python3 quantize_quark.py --model_dir $model_dir \
    --output_dir $output_dir \
    --quant_scheme w_fp8_a_fp8_o_fp8 \
    --dataset $calib_dataset \
    --num_calib_data 1000 \
    --model_export vllm_adopted_safetensors \
    --no_weight_matrix_merge
```

```{note}
The specific KV cache scales used to quantize the model weights in the container are optimized and different from the mainstream versions in the vLLM repo. It can be found in [this commit in github](https://github.com/vllm-project/vllm/blob/38c4b7e863570a045308af814c72f4504297222e/tests/fp8_kv/llama2-70b-fp8-kv/kv_cache_scales.json).
```

## Generate Results

To generate results for the first entry of our submission, run the command below in an inference container. Logs and results of the inference can be found in the container under the directory `/lab-hist/mlperf-results/<time-stamp>`.

``` bash
cd /lab-mlperf-inference/code/llama2-70b-99.9/test_VllmFp8
bash run_scenarios.sh
```

A summary of the results in the Offline scenario can be found in the `mlperf_log_summary.txt` file under the `Offline/performance/run_1` folder:

```bash
more /lab-hist/mlperf-results/<time-stamp>/Offline/performance/run_1/mlperf_log_summary.txt
```

```text
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 80.2353
Tokens per second: 23545.5
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
...
```

We recorded 23,545.5 tokens per second (unverified) in this particular trial, which matches the result in the submission (23,514.80).

A summary of the result in the Server scenario can be found in `mlperf_log_summary.txt` file under the `Server/performance/run_1/` folder:

```bash
more /lab-hist/mlperf-results/<time-stamp>/Server/performance/run_1/mlperf_log_summary.txt
```

```text
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 69.11
Completed tokens per second: 20360.10
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
TTFT Early Stopping Result:
 * Run successful.
TPOT Early Stopping Result:
 * Run successful.
...
```

We recorded 20,360.10 completed tokens per second (unverified) in this particular trial, which is comparable to the result in the submission under this scenario (21,028.20).

You can also generate results for only the Offline scenario or only the Server scenario.  To run only the Offline or Server scenario only, use `run_tests_Offline.sh` or `run_tests_Server.sh` respectively.

## Summary

In this blog post we showed you how to reproduce, on your own, the results of AMD’s MLPerf Inference submission with Llama 2 70B model Powered by MI300X. You can find the MLPerf results [here](https://mlcommons.org/benchmarks/inference-datacenter/), and read our post discussing the results in-depth, [here](https://community.amd.com/t5/instinct-accelerators/engineering-insights-unveiling-mlperf-results-on-amd-instinct/ba-p/705623). Please note that due to variation in the hardware configuration and condition in each run, specific results may deviate from the submitted result. You are encouraged to build on our effort and optimize your workloads using MI300X and ROCm.

## Disclaimers

This blog includes Unverified MLPerf v4.1 Inference Closed Llama-2-70b results.  These results were not verified by MLCommons Association.  The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See [www.mlcommons.org](http://www.mlcommons.org/) for more information.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
