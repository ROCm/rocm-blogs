---
blogpost: true
blog_title: "Reproduce AMD MLPerf Training v5.1 Submission Result"
date: 21 Oct 2025
author: 'Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sarthak Arora, Sathish Sanjeevi, Su Ann Chong, Karan Verma, Eliot Li'
thumbnail: 'mlperf-training5.1-repro-thumbnail.png'
tags: AI/ML, Fine-Tuning, GenAI, Optimization, Performance
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Provide instructions to readers on how to reproduce AMD's MLPerf Training v5.1 submission result.
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sarthak Arora, Sathish Sanjeevi, Su Ann Chong, Karan Verma, Eliot Li"
        "description lang=en": "Learn how to reproduce AMD's MLPerf Training v5.1 submission result."
        "keywords": "MLPerf, Training, Optimization, Llama 2, Llama 3.1, LoRA"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Eliot Li"
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

# Reproducing AMD MLPerf Training v5.1 Submission Result

Building upon the success of the [MLPerf Training v5.0 submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.0/README.html), AMD has not only submitted improved results for the Llama 2 70B LoRA finetuning benchmark for the MI300X and MI325X platforms in the v5.1 round, but also for the MI350X and MI355X platforms. In addition, AMD has submissions for the newly added Llama 3.1 8B pretraining benchmark in this MLPerf Training round. The AMD submissions are summarized in the following table:

| Benchmark | Instinct Platforms with Submission​ |
| :------------------------------: | :-------------------------------------: |
|      Llama 2 70B LoRA finetuning |     MI300X, MI325X, MI350X, MI355X      |
|      Llama 3.1 8B pretraining    |     MI350X, MI355X                      |

This blog provides a comprehensive, step-by-step tutorial on reproducing the results of AMD second MLPerf Training submission. This involves a Llama 2 70B LoRA fine-tuning benchmark utilizing the [GovReport dataset](https://gov-report-data.github.io/), as well as a Llama 3 8B pretraining benchmark using the [C4 (Colossal Cleaned Common Crawl) dataset](https://huggingface.co/datasets/allenai/c4). Readers will gain insight into setting up the environment, adjusting training parameters, and verifying the benchmark outcomes. To learn more about how these training workloads were optimized, visit this [blog post](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.1/README.html).

## Prerequisites

To follow along with this blog, you will need:

- AMD Instinct MI300X, MI325X, MI350X, and MI355X platform
- ROCm 7.0.0 or later
- Any Linux distribution supported by the selected ROCm version
- Docker

See the [ROCm Quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) for information on how to install ROCm.

In the following sections, you will first find instructions to set up the environment and preprocess the dataset for training. Then, you will go through the steps to run the model training job and collect benchmark results. Finally, you will be presented with the benchmark results submitted to the MLPerf Training v5.1 round by AMD.

## Llama 2 70B LoRA finetuning

This benchmark represents a Llama 2 70B LoRA finetuning on the [GovReport](https://gov-report-data.github.io/) dataset.

### Set up docker image

Pull the docker image from the registry

```bash
docker pull rocm/amd-mlperf:llama2_70b_training_5.1
```

### Prepare Dataset

GovReport is a dataset for long document summarization that consists of reports written by U.S. government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has a length of 8,192.

The model used in this submission is the Llama 2 70B with fused QKV. You will need 270GB of disk space to download and convert the model.

#### Download and Preprocess Data & Model

To download the model from Huggingface, you will need to sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT as well as obtain a Hugginface Token.

Copy the folder `/workspace/code`, which contains the scripts needed to run the benchmark, from the docker container to the host. Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/data/mlperf_llama2` as the host download directory:

```bash
container_id=$(docker create rocm/amd-mlperf:llama2_70b_training_5.1) && \
docker cp $container_id:/workspace/code ./code && \
docker rm $container_id

docker run -it -v /data/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_5.1
```

Start the script for downloading and preprocessing data from within the container:

```bash
export HF_TOKEN=<your huggingface token>
bash ./scripts/prepare_data_and_model.sh
```

#### Verify Data

The data and model files are stored under `/data` within the container.
After preprocessing, you should see the following files in the `/data/model` directory:

```
<hash>_tokenizer.model  llama2-70b.nemo  model_config.yaml  model_weights
```

And the following files in the `/data/data` directory:

```
train.npy  validation.npy
```

Exit the container.

### Run finetuning benchmark

#### Set up environment

First, set the environment variables `DATADIR`, `LOGDIR`, and `CONT` to the directory for the data and model, results, and the container respectively. Ensure that the directory set for `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`. In this example we use `/data/mlperf_llama2/results` as the results directory, so please make sure you have created this directory already.

```bash
cd code

export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_5.1

sudo chmod -R 777 $LOGDIR
```

#### Set configuration

To set appropriate configuration and system-specific hyperparameters for each AMD Instinct platform, use the appropriate configuration file listed in the table below:

| Instinct Platform | configuration file​ |
| :----------: | :-----------------: |
|  MI300X      | config_MI300X_1x8x1.sh |
|  MI325X      | config_MI325X_1x8x1.sh |
|  MI350X      | config_MI350X_1x8x1.sh |
|  MI355X      | config_MI355X_1x8x1.sh |

For example, to set the configuration for the MI355X platform on a single node, use the config file `config_MI355X_1x8x1.sh`:

```bash
source config_MI355X_1x8x1.sh
```

**Launch a single finetuning run**

To perform a single run for Llama 2 70B LoRA finetuning, set `NEXP` to `1`, and then run the `run_with_docker.sh` script:

```bash
export NEXP=1
bash run_with_docker.sh
```

**Launch 10 finetuning runs**

To prepare for a 10 run result, simply set the environment variable `NEXP` to `10` before running the `run_with_docker.sh` script:

```bash
export NEXP=10
bash run_with_docker.sh
```

After the benchmarking is completed, the logs will be available under the directory `$LOGDIR`, where you can also find the benchmark results.

## Llama 3.1 8B pretraining

This benchmark represents the pretraining of a (relatively) small Language Model-Llama 3.1 8B

### Setup docker image

Pull the docker image from the registry and copy the `/workspace/code` folder from the container to the host:

```bash
docker pull rocm/amd-mlperf:llama31_8b_training_5.1

container_id=$(docker create rocm/amd-mlperf:llama31_8b_training_5.1) && \
docker cp $container_id:/workspace/code ./code && \
docker rm $container_id
```

### Prepare dataset and model

The dataset used for this benchmark according to MLCommons is the [c4/en/3.0.1 dataset](https://huggingface.co/datasets/allenai/c4) from AllenAI.

#### Download preprocessed data

The pre-tokenized dataset and the tokenizer are available for download. You can navigate to your desired download directory and run the following commands to download the dataset and tokenizer. In this example, the folder `/data/mlperf_llama31_8b` is used as the download directory on the host.

```bash
# desired download directory
cd /data/mlperf_llama31_8b

# download training data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After the download is completed, you should see files with the following naming conventions under the data directory, ending with both `.idx` and `.bin` extensions: 

- Training partitions: `c4-train.en_6_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

The size of the data directory is ~80 GB, and ~30 GB for the model directory.

### Run pre-training benchmark

#### Set up environment

Set environment variables for the directory for the data, model and the container. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`, In this example the folder `/data/mlperf_llama31_8b/results` is used as the results directory, so please make sure to create this directory.

```bash
cd  code

export DATADIR=/data/mlperf_llama31_8b/data
export MODEL=/data/mlperf_llama31_8b/model/
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama31_8b_training_5.1

sudo chmod -R 777 $LOGDIR
```

#### Set configuration

Similar to the Llama 2 70B LoRA finetuning case, there is a configuration file for each Instinct platform submission for the Llama 3.1 8B benchmark:

| Instinct platform | Configuration file        ​ |
| :---------------: | :------------------------: |
|  MI350X           |  config_MI350X_1x8x1_8b.sh |
|  MI355X           |  config_MI355X_1x8x1_8b.sh |

For example, to configure the environment for a MI355X platform, use:

```bash
source config_MI355X_1x8x1_8b.sh
```

**Launch a single training run**

If you want to perform a single run of the pretraining benchmark, use:

```bash
export NEXP=1
bash run_with_docker.sh
```

**Launch 10 training runs**

If you would like to prepare for 10 run submission, use:

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, the logs will be available under the directory `$LOGDIR`.

```{note}
To optimize the machine's performance, the training script will also execute the `runtime_tunables.sh` script before any training run.
```

## Interpreting Results

Below is the log from one of the training runs on the AMD MI355X platform for the Llama 3.1 8B pretraining benchmark:

```text
...

:::MLLOG {"namespace": "", "time_ms": 1760028253483, "event_type": "POINT_IN_TIME", "key": "submission_benchmark", "value": "llama31_8b", "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253483, "event_type": "POINT_IN_TIME", "key": "submission_division", "value": "closed", "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253483, "event_type": "POINT_IN_TIME", "key": "submission_status", "value": "onprem", "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253483, "event_type": "POINT_IN_TIME", "key": "submission_org", "value": "AMD", "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253484, "event_type": "POINT_IN_TIME", "key": "submission_platform", "value": "MI355X", "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253484, "event_type": "POINT_IN_TIME", "key": "max_steps", "value": 1200000, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 341}}
:::MLLOG {"namespace": "", "time_ms": 1760028253484, "event_type": "INTERVAL_END", "key": "init_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 343}}
:::MLLOG {"namespace": "", "time_ms": 1760028253484, "event_type": "INTERVAL_START", "key": "run_start", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 344}}

...

:::MLLOG {"namespace": "", "time_ms": 1760033997670, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 30.694928437391837}, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 350, "step": 172032}}
:::MLLOG {"namespace": "", "time_ms": 1760033997670, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 307, "samples_count": 172032}}
:::MLLOG {"namespace": "", "time_ms": 1760033997670, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 308, "samples_count": 172032}}
:::MLLOG {"namespace": "", "time_ms": 1760034007523, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 3.287374973297119, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 128, "epoch_num": 172000, "samples_count": 172000}}
:::MLLOG {"namespace": "", "time_ms": 1760034007523, "event_type": "INTERVAL_END", "key": "eval_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 318, "samples_count": 172032}}
:::MLLOG {"namespace": "", "time_ms": 1760034007525, "event_type": "INTERVAL_END", "key": "epoch_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 283, "samples_count": 172032}}
:::MLLOG {"namespace": "", "time_ms": 1760034007525, "event_type": "INTERVAL_END", "key": "run_stop", "value": null, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 296, "samples_count": 172032, "status": "success", "duration": "5754.041257619858 sec -> 95.90068762699762 minutes"}}
:::MLLOG {"namespace": "", "time_ms": 1760034007525, "event_type": "POINT_IN_TIME", "key": "train_samples", "value": 172032, "metadata": {"file": "/workspace/code/src/callbacks.py", "lineno": 302}}
```

From the logs, the `run_start` and `run_stop` events happened at timestamp `1760028253484` and `1760034007525` in milliseconds respectively. The difference between these two timestamps is the time taken for the training to finish, and is equal to `95.9 min`. In most cases, this should serve as a reliable estimate of the MLPerf Training score, but obtaining an MLPerf compliant score is more involved, as described below.

Time-to-train is determined by two components: throughput and number of processed samples. Throughput is a measure of hardware performance and from the log above, it is about 30.7 samples per second for the Llama 3.1 8B pretraining benchmark running on MI355X. You can find throughput printed in your output file by searching for the last occurrence of `throughput` in the file. If your throughput is significantly lower, this indicates a misconfiguration or hardware issue that needs to be addressed.

Number of processed samples, given by `samples_count` in the output file, is determined by how fast the training converges to the defined accuracy target. In MLPerf Training, convergence needs to match the [Reference Convergence Checkpoints](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#13-reference-convergence-points-rcps) (RCPs). For hyperparameters used in the AMD submission for an individual run of the Llama 3.1 8B pretraining benchmark on MI355X, a value of `172,032` as in the log shown above is most common.

To obtain an MLPerf compliant Time-to-Train score, you will need 10 consecutive runs. Calculate timings for each run using the `run_start` and `run_stop` timestamps as described above.  Disregard the slowest and fastest runs and average the remaining 8 runtimes. Next, check convergence of runs by running [RCP Checker Script](https://github.com/mlcommons/logging/tree/44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8/mlperf_logging/rcp_checker) on the directory containing the set of 10 runs. The output should look like this:

```text
INFO - ------------------------------
INFO -  Running RCP Checker, pass: pruned_rcps
INFO - ------------------------------
INFO -  RCP Record: {'Benchmark': 'llama31_8b', 'BS': 32, 'Hyperparams': {'opt_base_learning_rate': 0.0008, 'opt_learning_rate_warmup_samples': 4096, 'gradient_accumulation_steps': 1}, 'Epochs to converge': [172032, 172032, 172032, 172032, 172032, 172032, 172032, 172032, 172032, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 196608], 'RCP Mean': np.float64(178858.66666666666), 'RCP Stdev': np.float64(7165.0736883859045), 'Max Speedup': np.float64(1.0299965952657717), 'Min Epochs': np.float64(173649.76494948068)}
INFO -  Submission mean epochs: 178144.0000
INFO -  Submission mean epochs faster than RCP mean but within max speedup range. Score should be normalized by factor of 178858.66666666666 / 178144.0 = 1.004011735824202
INFO - RCP found, RCP test PASSED
** Logging output also at rcp_checker.log
```

Make sure there is no error in the output. If RCP checker prints a normalization factor, the average of 8 runs should be multiplied by the factor to arrive at the final score.

## Expected Results

### Llama 2 70B LoRA finetuning

For MI355X, the score should be about 10 mins. The AMD MLPerf Training v5.1 submission score is `10.18` mins.

For MI350X, the score should be about 12 mins. The AMD MLPerf Training v5.1 submission score is `12.23` mins.

For MI325X, the score should be about 21 mins. The AMD MLPerf Training v5.1 submission score is `21.05` mins.

For MI300X, the score should be about 28 mins. The AMD MLPerf Training v5.1 submission score is `27.95` mins.

### Llama 3.1 8B pretraining

For MI355X, the score should be about 100 mins. The AMD MLPerf Training v5.1 submission score is `99.71` mins.

For MI350X, the score should be about 123 mins. The AMD MLPerf Training v5.1 submission score is `122.93` mins.

## Summary

This blog guides you through the process of replicating and verifying the results submitted by AMD for MLPerf Training v5.1. You can view the official MLPerf Training v5.1 results on [MLCommons](https://mlcommons.org/benchmarks/training/). Keep in mind that slight variations in hardware setups and conditions may lead to minor discrepancies from AMD’s results. For more insights into the techniques used to optimize training workloads in these submissions, check out this [blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.1/README.html). Interested in taking your skills further? Use this foundation to enhance your AI workloads with AMD Instinct™ GPUs and ROCm.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
