---
blogpost: true
blog_title: "Reproduce the MLPerf Training v5.0 Submission Result By AMD Using the Instinct™ GPUs"
date: 4 Jun 2025
author: 'Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Su Ann Chong, Sarthak Arora, Sathish Sanjeevi, Karan Verma, Eliot Li'
thumbnail: "MLPerf-Training-v5.0-reproduce.png"
tags: AI/ML, GenAI, Performance, Optimization
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Provide instructions to potential customers and partners to verify our MLPerf Training 5.0  submission result.
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Karan Verma, Su Ann Chong, Sarthak Arora, Sathish Sanjeevi, Eliot Li"
        "description lang=en": "Follow this step-by-step guide to reproduce AMDs MLPerf 5.0 Training Submission with Instinct GPUs using ROCm"
        "keywords": "MLPerf, Training, Optimization"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "Adaptive & Embedded Computing"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Su Ann Chong, Sarthak Arora, Sathish Sanjeevi, Karan Verma, Eliot Li"
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

# Reproduce AMD's MLPerf Training v5.0 Submission Result with Instinct™ GPUs

In recent years, large language models (LLMs) have transformed the landscape of natural language processing, enabling breakthroughs in tasks ranging from code generation to answering complex questions. Among these, the [Llama 2 model family](https://www.llama.com/llama2/) developed by Meta has emerged as a powerful and versatile set of open weight transformer-based models, known for their competitive performance across diverse NLP benchmarks. With model sizes ranging from 7 billion to 70 billion parameters, Llama 2 has quickly become a popular choice for both research and industry after its release in 2023, striking a balance between scalability and efficiency.

In the context of MLPerf Training, a leading benchmark suite that rigorously evaluates the training performance of machine learning models on standardized tasks, Llama 2 has gained notable attention. Its participation highlights the model’s ability to be efficiently trained at scale, pushing forward advances in both hardware utilization and software optimization.

This blog provides a step-by-step guide to reproduce the results of AMD's first MLPerf Training submission. The submission represents a Llama 2 70B LoRA fine-tuning job using the [GovReport](https://gov-report-data.github.io/) dataset. You will learn how to set up the environment, configure the training parameters, and validate the benchmark results.

## Prerequisites

To get started, you will need:

- AMD Instinct MI300X or MI325X platform
- ROCm 6.3.3 or later
- Any Linux distribution supported by the selected ROCm version
- Docker

See the [ROCm Quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) for information on how to install ROCm.

In the following sections, you will first find instructions to set up the environment and preprocess the dataset for training. Then, you’ll walk through the steps to run the model training job and collect benchmark results. Finally, you will be presented the benchmark results submitted to the MLPerf Training v5.0 round by AMD.

## Prepare the Environment, Model, and Dataset

Follow these steps to prepare for running the training benchmark.

### Pull the Docker Container

For the purpose of illustrating the training procedure, it is assumed that the training will be conducted on a MI325X system.
First, pull the docker image `rocm/amd-mlperf:llama2_70b_training_5.0` from docker hub.

```bash
docker pull rocm/amd-mlperf:llama2_70b_training_5.0
```

Copy the following scripts required to run the benchmark to the host machine:

```bash
container_id=$(docker create rocm/amd-mlperf:llama2_70b_training_5.0) && \
docker cp $container_id:/workspace/code/runtime_tunables.sh . && \
docker cp $container_id:/workspace/code/run_with_docker.sh . && \
docker cp $container_id:/workspace/code/config_MI325X_1x8x1.sh . && \
docker rm $container_id
```

If you are running the training procedure on a MI300X system, please copy the script `config_MI300X_1x8x1.sh` instead of `config_MI325X_1x8x1.sh` .

### Prepare the Training Dataset

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

The model used is the Llama 2 70B with fused QKV. You will need 270GB of memory to download and convert the model.

Create the directory `/data/mlperf_llama2` as the host download directory for the training data.

```bash
mkdir -p /data/mlperf_llama2
```

Start the docker container by mounting this directory under `/data` within the container.

```bash
docker run -it -v /data/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_5.0
```

To prepare the dataset for training, start the script for downloading and preprocessing data from within the container:

```bash
bash ./scripts/prepare_data_and_model.sh
```

When the data preprocessing is complete, exit the container.

## Run the Llama 2 70B LoRA Benchmark

### Set Environment

Set the environment variables `DATADIR, CONT` to the directory where the training data is and the container image respectively:

```bash
export DATADIR=/data/mlperf_llama2
export CONT=rocm/amd-mlperf:llama2_70b_training_5.0
```

### Set Configuration

The configuration and system-specific hyperparameters for MI300X and MI325X are available in the scripts `config_MI300X_1x8x1.sh` and `config_MI325X_1x8x1.sh` respectively.

To set the configuration and system-specific hyperparameters for running the Llama 2 70B LoRA benchmark on MI325X, source the `config_MI325X_1x8x1.sh` script that we have copied from the container earlier:

```bash
source config_MI325X_1x8x1.sh
```

### Launch a Single Training Run

To perform a single run of the training, set `NEXP` to `1`, and then run the `run_with_docker.sh` script that we have copied from the container earlier:

```bash
export NEXP=1
bash run_with_docker.sh
```

The output should resemble the following:

```text
+++ readlink -f run_with_docker.sh
++ dirname /home/karverma/run_with_docker.sh
+ SCRIPT_DIR=/home/karverma
+ cd /home/karverma
+ : MI325X_1x8x1
+ : rocm/amd-mlperf:llama2_70b_training_5.0
+ : /data/mlperf_llama2
+ : 1
+ : 1
+ : 1
+ : 5.0.0
+ : ./results
+ : mlperf_llama2sft
+ : 0
+ readonly _config_file=./config_MI325X_1x8x1.sh
+ _config_file=./config_MI325X_1x8x1.sh
+ readonly _cont_name=mlperf_llama2sft
+ _cont_name=mlperf_llama2sft
+ _cont_mounts=("--volume=${DATADIR}/data:/data" --volume=${DATADIR}/model:/ckpt)
+ mkdir -p ./results

...

ENDING TIMING RUN AT 2025-05-30 03:59:18 PM
RESULT,LLM_FINETUNING,,1618,AMD,2025-05-30 03:32:20 PM
```

```{note}
To optimize the machine's performance, the training script will also execute the `runtime_tunables.sh` script automatically before any training run.
```

To prepare for a 10 run training result, simply set the environment variable `NEXP` to `10` before running the `run_with_docker.sh` script:

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, the logs will be available in the `results` folder under the current directory.

Below is the log from one of the training runs in the AMD MI325X submission:

```text
...
:::MLLOG {"namespace": "", "time_ms": 1745926183349, "event_type": "POINT_IN_TIME", "key": "cache_clear", "value": true, "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 231}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "INTERVAL_START", "key": "init_start", "value": null, "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 232}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "POINT_IN_TIME", "key": "submission_benchmark", "value": "llama2_70b_lora", "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 233}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "POINT_IN_TIME", "key": "submission_org", "value": "AMD", "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 233}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "POINT_IN_TIME", "key": "submission_division", "value": "closed", "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 233}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "POINT_IN_TIME", "key": "submission_status", "value": "onprem", "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 233}}
:::MLLOG {"namespace": "", "time_ms": 1745926183350, "event_type": "POINT_IN_TIME", "key": "submission_platform", "value": "1xMI325X", "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 233}}
...
:::MLLOG {"namespace": "", "time_ms": 1745926251617, "event_type": "INTERVAL_START", "key": "run_start", "value": null, "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 177}}
:::MLLOG {"namespace": "", "time_ms": 1745926251617, "event_type": "INTERVAL_START", "key": "block_start", "value": null, "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 178, "samples_count": 0}}
...
:::MLLOG {"namespace": "", "time_ms": 1745927530858, "event_type": "INTERVAL_END", "key": "run_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks/custom_callbacks.py", "lineno": 183, "samples_count": 3072, "status": "success", "duration": "1279.2406167984009 sec -> 21.320676946640013 minutes"}}
```

From the logs, the `run_start` and `run_stop` events happened at timestamp `1745926251617` and `1745927530858` in milliseconds respectively.  The difference between these two timestamps is the time taken for the training to finish, and is equal to `21.321 min`. In most cases, this should serve as a reliable estimate of MLPerf Training score, but getting an MLPerf compliant score is more involved as described below.

## Interpreting Results

Time-to-train is determined by two components: throughput and number of processed samples. Throughput is a measure of hardware performance and should be about 2.73 samples per second for MI325X and about 2.02 samples per second for MI300X. You can find throughput printed in your output file. If your throughput is significantly lower, this indicates misconfiguration or hardware issue that needs to be addressed.

Number of processed samples, given by `samples_count` in the output file, is determined by how fast the training converge to the defined accuracy target. In MLPerf Training, convergence need to match [Reference Converge Checkpoints](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#13-reference-convergence-points-rcps) (RCPs). For hyperparameters in this blog, average number of processed samples should be `3,120` over 10 runs. For individual runs, a value of `3,072` is most common.

To get MLPerf compliant Time-to-Train score, you will need 10 consecutive runs. Calculate timings for each run using the `run_start` and `run_stop` timestamps as described above.  Disregard the slowest and fastest runs and average the remaining 8 runtimes. Next, check convergence of runs by running [RCP Checker Script](https://github.com/mlcommons/logging/tree/44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8/mlperf_logging/rcp_checker) on the directory containing the set of 10 runs. The output should look like this:

```text
RCP Record: {'Benchmark': 'llama2_70b_lora', 'BS': 8, 'Hyperparams': {'opt_base_learning_rate': 0.0004, 'opt_max_grad_norm': 0.3, 'opt_learning_rate_warmup_epochs': 0, 'opt_learning_rate_decay_boundary_epochs': [], 'gradient_accumulation_steps': 1, 'lora_r': 16, 'lora_alpha': 32, 'max_steps': 1024}, 'Epochs to converge': [2688, 2688, 2688, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3456, 3456, 3456, 3456, 3456, 3456, 3456, 3840], 'RCP Mean': 3178.6666666666665, 'RCP Stdev': 299.91358755481554, 'Max Speedup': 1.0736439248523562, 'Min Epochs': 2960.6339616775513}
INFO - Submission mean epochs: 3072.0000
INFO - Submission mean epochs faster than RCP mean but within max speedup range. Score should be normalized by factor of 3178.6666666666665 / 3072.0 = 1.034722222222222
INFO - Results scaling set to normalization factor of 1.0347
INFO - RCP Test Passed: 
```

Make sure there is no error in the output. If RCP checker prints a normalization factor, the average of 8 runs should be multiplied by the factor to arrive at final score.

## Expected Results

For MI325X, the score should be about 22 mins. The AMD MLPerf Training v5.0 submission score is `22.04` mins.

For MI300X, the score should be about 29 mins. The AMD MLPerf Training v5.0 submission score is `29.20` mins.

## Summary

This blog walks you through reproducing and validating AMD’s MLPerf Training v5.0 results with the Llama 2 70B model. You can explore the official MLPerf Training v5.0 results submitted by AMD and other teams on [MLCommons](https://mlcommons.org/benchmarks/training/). Minor differences in hardware setups and conditions may result in small variations from AMD’s results. Check out this [blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.0/README.html) to learn more about the techniques that were used to optimize this training workload. Ready to go further? Apply this foundation to push further by optimizing your AI workloads with AMD Instinct™ GPUs and ROCm.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
