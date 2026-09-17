---
blogpost: true
blog_title: "Reproducing AMD MLPerf Inference v6.1 Submission Results"
date: "17 Sep 2026"
author: "Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Lee, Neha Mathews, Rajesh Poornachandran, Karan Verma, Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
thumbnail: 'mlperf_inf_v61_repro_thumbnail.png'
tags: "AI/ML, GenAI, Performance, Optimization, LLM, MLPerf, MLPerf Inference"
category: "Applications & models"
target_audience: "AI developers, AI practitioners"
key_value_propositions: "Share the technical details of how we accomplish the results in our MLPerf Inference v6.1 submission"
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Lee, Neha Mathews, Rajesh Poornachandran, Karan Verma, Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
        "description lang=en": "In this blog, we share the technical details of how we accomplish the results in our MLPerf Inference v6.1 submission."
        "keywords": "MLPerf Inference v6.1, AMD Instinct MI355X, ROCm, reproduce MLPerf results, dlrm-v3, llama2-70b, gpt-oss-120b, MLCommons, MXFP4"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Lee, Neha Mathews, Rajesh Poornachandran, Karan Verma, Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# Reproducing AMD MLPerf Inference v6.1 Submission Results

This blog shows you how to reproduce AMD submission results for [MLPerf Inference v6.1](https://mlcommons.org/benchmarks/inference-datacenter/) on **AMD Instinct MI355X, MI350X, and MI350P** GPUs using self-contained Docker images, publicly available quantized model weights, and a step-by-step benchmark recipe.

**MLPerf Inference** is the industry-standard benchmark suite governed by [MLCommons](https://mlcommons.org/), an open engineering consortium. It covers a representative set of AI workloads (recommendation, language modeling, computer vision) and measures system-level throughput and latency under production-realistic query patterns. The **closed division** — the division in which AMD participates — enforces strict accuracy constraints (99% or 99.9% of the FP32 baseline, depending on the model) and permits quantization, compiler optimization, and hardware-specific tuning, provided accuracy constraints are met. Workloads are tested across multiple **scenarios**: **Offline** (maximize throughput with unconstrained batch size), **Server/Interactive** (maintain tail latency SLO under a Poisson query arrival process) and **SingleStream** (minimize latency for processing 1 sample).

AMD MLPerf Inference v6.1 submission is our **fifth consecutive round** of MLPerf Inference participation. For technical details on the submission see our [companion blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-v6.1/README.html).

## Workloads Summary

The workloads that have been part of the submission are listed in the table below

| Model | Task | Datatype | Scenarios | Platform |
| --- | --- | --- | --- | --- |
| llama2-70b | LLM Summarization | MXFP4 | Offline, Server, Interactive | MI355X, MI350P |
| deepseek-r1 | LLM Reasoning | MXFP4 | Offline, Server | MI355X |
| gpt-oss-120b | MoE LLM | MXFP4 | Offline, Server | MI355X, MI350P |
| dlrm-v3 | Recommendation | FP8 (E4M3) | Offline, Server | MI355X, MI350X, MI350P |
| wan2.2-t2v | Text-to-Video | MXFP4 | Offline, Single Stream | MI355X, MI350X, MI350P |
| llama3.1-8b | LLM Summarization | MXFP4 | Offline, Server, Interactive | MI355X |

## Prerequisites

Before running any workload, confirm your system meets the following requirements.

**Hardware:**

- One or more **AMD Instinct MI355X, MI350X, or MI350P** GPUs. Eight GPUs are required for all single-node submissions. The multi-node GPT-OSS 120B submission requires 72 GPUs across 9 nodes.
- ECC must be enabled on all GPU devices. Verify with:

  ```bash
  rocm-smi --showmeminfo ecc
  ```

  Contact your system administrator if ECC is not enabled; MLCommons rules require it for datacenter submissions.

**Software:**

- **OS:** Ubuntu 22.04 LTS or later (recommended).
- **ROCm:** [PLACEHOLDER: ROCm version, e.g., ROCm 6.4] — install via the [ROCm Quick Start Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).
- **Docker:** Docker Engine with AMD GPU passthrough configured. Verify your installation:

  ```bash
  docker run --rm --device=/dev/kfd --device=/dev/dri rocm/rocm-terminal rocminfo | grep "gfx"
  ```

- **Hugging Face account:** A free account at [huggingface.co](https://huggingface.co) and a user access token are required to download the gated model weights for llama2-70b and gpt-oss-120b. Generate a token at `https://huggingface.co/settings/tokens`.

**Storage:**

Total disk space required across all workloads is approximately 5GB. See each model section for per-workload estimates including Docker image, quantized weights, and dataset. Note that dlrm-v3 dominates the total at roughly 1.2 TB on its own, and also requires about 3 TB of host system memory.

## Llama 2 70B

**Llama 2 70B** is Meta's 70-billion-parameter open-weight language model fine-tuned for dialogue and summarization tasks. It serves as one of the core LLM benchmarks in MLPerf Inference, evaluated on the CNN/DailyMail summarization dataset. AMD submission uses **MXFP4** (weight-matrix MX floating-point 4-bit) quantization, which aggressively reduces memory bandwidth pressure on the MI355X's high-bandwidth memory while maintaining accuracy above the 99% of FP32 baseline target. The standard scenario tests end-to-end summarization throughput and latency; the interactive variant adds per-token latency constraints representative of a live chat application.

### Step 1: Prepare the Docker Container - Llama 2 70B

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_llama2_70b_inference_6.1
```

Start the Docker container:

```bash
docker run -it --name llama2_test \
  --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --entrypoint bash \
  rocm/amd-mlperf:mi355x_llama2_70b_inference_6.1
```

### Step 2: Download the Reference Model and Dataset - Llama 2 70B

From within the Docker container, download the quantized model using this command:

```bash
git clone https://huggingface.co/amd/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-6.0MLPerf-GPTQ  /model/llama2-70b-chat-hf/fp4_quantized_gptq
```

Download the OpenOrca dataset following the instructions in the [MLCommons inference repository](https://github.com/mlcommons/inference/tree/master/language/llama2-70b).

```{note}
The llama2-70b model on Hugging Face is gated. You must accept the model license at `https://huggingface.co/meta-llama/Llama-2-70b-hf` before the download will succeed with your token.
```

### Step 3: Run the Benchmark Harness - Llama 2 70B

Note: The commands below are for MI355X. To execute on a different GPU, change the config file as appropriate.

#### Offline Scenario Performance Benchmark - Llama 2 70B

Run the offline scenario performance test to obtain performance of your system:

```bash
python /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/llama2-70b-99/ \
  --config-name offline_mi355x test_mode=performance \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Offline/performance/run_1
```

Run the offline scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/llama2-70b-99 \
  config_name=offline_mi355x \
  test_mode=accuracy \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Offline/accuracy
```

The `mlperf_log_accuracy.json` file is processed to verify the accuracy of the offline scenario:

```bash
bash /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
  /lab-mlperf-inference/results/llama2-70b/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server Scenario Performance Benchmark - Llama 2 70B

Run the server scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/llama2-70b-99/ \
  --config-name server_mi355x test_mode=performance \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/performance/run_1
```

Run the server scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/llama2-70b-99 \
  config_name=server_mi355x \
  test_mode=accuracy \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/accuracy
```

The `mlperf_log_accuracy.json` is processed to verify the accuracy of the server scenario:

```bash
bash /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
  /lab-mlperf-inference/results/llama2-70b/Server/accuracy/mlperf_log_accuracy.json
```

#### Interactive Scenario Performance Benchmark - Llama 2 70B

Run the interactive scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/llama2-70b-99/ \
  --config-name interactive_mi355x test_mode=performance \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/performance/run_1
```

Run the interactive scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/llama2-70b-99 \
  config_name=interactive_mi355x \
  test_mode=accuracy \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/accuracy
```

The `mlperf_log_accuracy.json` is processed to verify the accuracy of the interactive scenario:

```bash
bash /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
  /lab-mlperf-inference/results/llama2-70b/Interactive/accuracy/mlperf_log_accuracy.json
```

The accuracy run must produce a ROUGE score at or above the threshold value specified in the [MLCommons inference repository](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#accuracy-target).

## GPT-OSS 120B

**GPT-OSS 120B** is an open-weight 120-billion-parameter Mixture-of-Experts (MoE) language model. MoE architectures activate only a subset of parameters per token, making them throughput-efficient for large-scale inference while maintaining high model capacity.

### Step 1: Prepare the Docker Container - GPT-OSS 120B

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.1
```

Start the Docker container:

```bash
docker run -it --name gptoss_test \
  --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --entrypoint bash \
  rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.1
```

### Step 2: Download the Reference Model and Dataset - GPT-OSS 120B

From within the Docker container, download the quantized model using this command:

```bash
git clone https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8-Mlperf  /model/gpt-oss-120b/fp4_quantized
```

From within the Docker container, download and process the dataset:

```bash
cd /lab-mlperf-inference
bash setup/gpt-oss-120b/dataset_and_model/prepare_dataset.sh
```

### Step 3: Run the Benchmark Harness - GPT-OSS 120B

Note: The commands below are for MI355X. To execute on a different GPU, change the config file as appropriate.

#### Offline Scenario Performance Benchmark - GPT-OSS-120B

Run the offline scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
  --config-name offline_mi355x test_mode=performance \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1
```

Run the offline scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/gpt-oss-120b \
  config_name=offline_mi355x \
  test_mode=accuracy \
  harness_config.dataset_path=/data/gpt-oss-120b/perf_eval_ref.parquet \
  harness_config.accuracy_dataset_path=/data/gpt-oss-120b/acc_eval_ref.parquet \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy
```

The `mlperf_log_accuracy.json` file is processed to verify the accuracy of the offline scenario:

```bash
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
  /lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server Scenario Performance Benchmark - GPT-OSS-120B

Run the server scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
  --config-path /lab-mlperf-inference/code/gpt-oss-120b/ \
  --config-name server_mi355x test_mode=performance \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1
```

Run the server scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/gpt-oss-120b \
  config_name=server_mi355x \
  test_mode=accuracy \
  harness_config.dataset_path=/data/gpt-oss-120b/perf_eval_ref.parquet \
  harness_config.accuracy_dataset_path=/data/gpt-oss-120b/acc_eval_ref.parquet \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy
```

The `mlperf_log_accuracy.json` file is processed to verify the accuracy of the server scenario:

```bash
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
  /lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy/mlperf_log_accuracy.json
```

The accuracy run must meet the 99% threshold relative to the reference score. The exact reference score is documented in the [MLCommons inference repository](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#accuracy-target) under the gpt-oss model directory.

### Multi-Node Submission

AMD submitted multi-node results for **GPT-OSS 120B** at cluster scale using AMD Instinct MI355X GPUs. The topology used was as follows:

- **Cluster:** 9 nodes × 8 AMD Instinct MI355X GPUs = **72 total GPUs**
- **Interconnect:** [PLACEHOLDER: InfiniBand / Ethernet spec]
- **MPI / launcher:** [PLACEHOLDER: mpirun / torchrun / custom launcher]

#### Step 1: Prepare the Docker Container - GPT-OSS-120B Multi-Node

Pull the Docker image containing the required code and scripts. For example, for GPT-OSS 120B:

```bash
docker pull rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.1
```

Start the Docker container:

```bash
docker run -it --name gptoss_test \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.1
```

#### Step 2: Download the Reference Model and Dataset - GPT-OSS-120B Multi-Node

From within the Docker container, download the quantized model using this command:

```bash
git clone https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8-Mlperf  /model/gpt-oss-120b/fp4_quantized
```

From within the Docker container, download and process the dataset:

```bash
bash /lab-mlperf-inference/setup/download_gptoss_120b.sh
```

To enable a distributed SUT with ZMQ, choose one node in the cluster as the Head node where the SUT client will run. Get the IP address of the head node using:

```bash
hostname -I
```

#### Step 3: Run the Benchmark Harness - GPT-OSS-120B Multi-Node

##### Running the Benchmark Under the Server Scenario

To run the benchmark under the server scenario, use the run_harness.sh script to start the SUT client. Make sure `device_count` is set to the number of all the healthy GPUs across all the nodes in the cluster.

```bash
bash run_harness.sh --config-path gpt-oss-120b/ --config-name server_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/gpt-oss-120b_server_performance_zmq port=12345 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps x 0.9>
```

The following flags can be appended to the command for the purpose of debugging:

```bash
harness_config.target_qps=300 harness_config.duration_sec=30 harness_config.debug_record_sample_latencies=True harness_config.debug_print_finished=True harness_config.debug_dump_model_output=True
```

Use the `distributed_async_server.py` script to start a worker on each node, including the Head node. Use the IP of the Head node for the Head node IP:

```bash
python harness_llm/backends/vllm/zmq/distributed_async_server.py --config-path gpt-oss-120b/ --config-name server_mi355x node_id=`hostname` headnode_address=<Head node IP>:12345
```

The steps for running the benchmark for the offline scenario are similar to the steps for running the benchmark for the server scenario.

The `run_harness.sh` script with `config-name` set to `offline_mi355x` is used to start the SUT client:

```bash
bash run_harness.sh --config-path gpt-oss-120b/ --config-name offline_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/gptoss_offline_performance_zmq port=12345 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps x 0.9>
```

Start the worker across all nodes with the following command:

```bash
python harness_llm/backends/vllm/zmq/distributed_sync_offline.py --config-path gpt-oss-120b/ --config-name offline_mi355x node_id=`hostname` headnode_address=<IP>:12345
```

## DeepSeek-R1

**DeepSeek-R1** is a reasoning-focused large language model. AMD submission runs the quantized `S3_sq_a05_v2` checkpoint on MI355X with the **SGLang** inference backend. Scenarios: **Offline** and **Server**.

### Step 1: Prepare the Docker Container - DeepSeek-R1

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_deepseek_r1_inference_6.1
```

Start the Docker container:

```bash
docker run -it --name deepseek-r1_test \
  --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $(pwd)/model:/model -v $(pwd)/data:/data \
  rocm/amd-mlperf:mi355x_deepseek_r1_inference_6.1
```

You start in `/lab-mlperf-inference/code`, where `main.py` lives. The `-v $(pwd)/model:/model` and `-v $(pwd)/data:/data` mounts keep the downloaded model and dataset on the host so they persist across containers.

### Step 2: Download the Reference Model and Dataset - DeepSeek-R1

From within the Docker container, download the quantized model:

```bash
HUGGINGFACE_ACCESS_TOKEN="<your-token>"
hf download amd/Deepseek-S3_sq_a05_v2_mlperf6_1 \
  --token "${HUGGINGFACE_ACCESS_TOKEN}" \
  --local-dir /model/S3_sq_a05_v2
```

```{note}
The DeepSeek-R1 model on Hugging Face is gated. You must have a valid Hugging Face access token. Generate one at `https://huggingface.co/settings/tokens`.
```

Download the DeepSeek-R1 evaluation dataset:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data/deepseek-r1 https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
```

The configs read `/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`, so make sure the eval file ends up there (move or symlink it if the downloader nests it under a subfolder).

### Step 3: Run the Benchmark Harness - DeepSeek-R1

Run from `/lab-mlperf-inference/code`. Every run must pass `--backend sglang`.

Before your first accuracy score, install the eval dependencies once (clones the PRM800K and LiveCodeBench graders):

```bash
bash scripts/setup_deepseek_accuracy_env.sh
```

```{note}
`setup_deepseek_accuracy_env.sh` is a **one-time** step. It installs the PRM800K (math) and LiveCodeBench graders the scorer needs. Run it before your first `check_deepseek_accuracy_scores.sh`, otherwise scoring aborts with a missing-module error.
```

#### Offline Scenario - DeepSeek-R1

```bash
# Performance
python3 main.py --config-path deepseek-r1 --config-name offline_mi355x \
  --backend sglang test_mode=performance \
  harness_config.output_log_dir=results/deepseek-r1/Offline/performance/run_1

# Accuracy (generates mlperf_log_accuracy.json)
python3 main.py --config-path deepseek-r1 --config-name offline_mi355x \
  --backend sglang test_mode=accuracy \
  harness_config.output_log_dir=results/deepseek-r1/Offline/accuracy

# Score accuracy (writes accuracy.txt next to the input)
bash scripts/check_deepseek_accuracy_scores.sh \
  results/deepseek-r1/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server Scenario - DeepSeek-R1

```bash
# Performance
python3 main.py --config-path deepseek-r1 --config-name server_mi355x \
  --backend sglang test_mode=performance \
  harness_config.output_log_dir=results/deepseek-r1/Server/performance/run_1

# Accuracy
python3 main.py --config-path deepseek-r1 --config-name server_mi355x \
  --backend sglang test_mode=accuracy \
  harness_config.output_log_dir=results/deepseek-r1/Server/accuracy

# Score accuracy
bash scripts/check_deepseek_accuracy_scores.sh \
  results/deepseek-r1/Server/accuracy/mlperf_log_accuracy.json
```

## DLRM-v3

**DLRM-v3** (Deep Learning Recommendation Model, version 3) is the recommendation workload in MLPerf Inference, and v6.1 is AMD first submission of it. Unlike earlier DLRM generations, DLRM-v3 is built around a **Hierarchical Sequential Transduction Unit (HSTU)**: it treats each user's interaction history as a token sequence, runs stacked causal attention layers over it, and scores 2,048 candidate items per request. It therefore stresses two subsystems at once — a **~1 TB sparse embedding table** that no single GPU can hold and must be row-sharded across all eight GPUs, and a long-sequence attention stack that dominates GPU compute time. AMD submission runs the dense HSTU path in **FP8 (E4M3) with FP32 accumulation** and keeps the embedding table at the reference precision, reading it directly over the AMD Infinity Fabric™ (xGMI) mesh. It uses full-causal attention with no sliding window, at batch 64 on MI355X and batch 32 on MI350X.

**Storage:** budget roughly **1.2 TB** of free disk for this workload — a ~964 GB checkpoint, a ~140 GB preprocessed dataset, and the container image. The host also needs about **3 TB of system memory** to stage the checkpoint.

```{note}
DLRM-v3's sparse embedding engine maps peer GPU memory over AMD Infinity Fabric, which depends on the host stack as well as the container. Use Ubuntu 24.04 with Linux kernel 6.8 or newer, `amdgpu` 6.16.6 or newer, and host ROCm 7.2 or newer. On older host stacks the identical container fails during embedding-engine initialization.
```

Unlike the other workloads in this blog, DLRM-v3 is driven by a standalone runner repository rather than a single pre-built image. The runner provisions the container, builds the two components that must be compiled for the `gfx950` target, stages the model and dataset, and runs the performance, accuracy, and compliance tests. This is the procedure AMD used to produce the submission results.

### Step 1: Prepare the Docker Container - DLRM-v3

Pull the Docker image:

```bash
docker pull rocm/amd-mlperf-inference:mi355x_dlrm_inference_6.1
```

Extract the host wrapper and harness launcher scripts from the image (one-time setup):

```bash
mkdir -p dlrmv3-host-runner/scripts/image dlrmv3-host-runner/scripts/run
docker run --rm --entrypoint /bin/bash \
  -v "$PWD/dlrmv3-host-runner:/out" \
  rocm/amd-mlperf-inference:mi355x_dlrm_inference_6.1 -lc '
    cp -a /opt/dlrmv3/host-runner/scripts/image/. /out/scripts/image/
    cp -a /opt/dlrmv3/runner/scripts/run/run_gold.sh \
          /opt/dlrmv3/runner/scripts/run/run_accuracy.sh \
          /opt/dlrmv3/runner/scripts/run/score_accuracy.py \
          /opt/dlrmv3/runner/scripts/run/_test08_chain.sh \
          /out/scripts/run/
  '
export RUNNER=$PWD/dlrmv3-host-runner
```

### Step 2: Download the Reference Model and Dataset - DLRM-v3

DLRM-v3 uses a trained checkpoint and a preprocessed synthetic streaming dataset, both published on MLCommons storage. The dataset and checkpoint are **not** included in the image.

- **Checkpoint:** download from [https://inference.mlcommons-storage.org/metadata/dlrm-v3-checkpoint.uri](https://inference.mlcommons-storage.org/metadata/dlrm-v3-checkpoint.uri) (~964 GB, sharded)
- **Dataset:** `dlrmv3_preprocessed_full` (~140 GB)

Set the environment variables pointing to your local copies:

```bash
export IMAGE_TAG=rocm/amd-mlperf-inference:mi355x_dlrm_inference_6.1
export DATASET=/path/to/dlrmv3_preprocessed_full
export CHECKPOINT=/path/to/dlrm-v3-checkpoint
export RESULTS_ROOT=/path/to/dlrmv3-results
export TRITON_CACHE=/path/to/dlrmv3-triton-cache
mkdir -p "$RESULTS_ROOT" "$TRITON_CACHE"
```

```{note}
**Host requirements** (qualified MI355X 8-GPU box — `run_gold.sh` checks most of these before launch):
- ≥ 220 GB free VRAM per GPU (export `MIN_FREE_GB=220`; `run_gold` default is 210)
- SMT enabled and expected CPUs online (≥ 256 online CPUs)
- CPU governor set to `performance` on all online CPUs (or export `SKIP_CPU_CHECK=1`)
- Supported KMD: `amdgpu` driver version ≥ 6.16.13
```

### Step 3: Run the Benchmark Harness - DLRM-v3

All runs go through the `run_image.sh` launcher, which provisions the container with GPU device access and the capabilities the sparse embedding engine needs for peer memory over the fabric. The harness shards the embedding table across all eight GPUs on its own, so no GPU-count flag is required.

```{note}
Every scenario needs a warm Triton autotuning cache. The first run after a fresh container start compiles kernels on the fly and can miss the latency target near the throughput knee; re-run once the cache is populated.
```

#### Server Scenario Performance Benchmark - DLRM-v3

```bash
bash "$RUNNER/scripts/image/run_image.sh" server
```

This produces output similar to the following:

```text
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second : 12198.83
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
================================================
Additional Stats
================================================
Scheduled samples per second : 12200.90
50.00 percentile latency (ns) : 50550699
99.00 percentile latency (ns) : 60426940
99.90 percentile latency (ns) : 67225573
```

The 99th-percentile latency of about 60.43 ms is well under the 80 ms Server bound, so the result is `VALID`.

#### Offline Scenario Performance Benchmark - DLRM-v3

```bash
BATCH=64 \
CONF=user_mi355x8_nve_b64_qps12200_OFFLINE10min.conf \
SCENARIO=Offline \
bash "$RUNNER/scripts/image/run_image.sh" server
```

#### Accuracy Benchmark - DLRM-v3

DLRM-v3 is scored by **grouped AUC (GAUC)**, and a valid result requires GAUC at or above **99.9% of the reference**. Run the accuracy test for each scenario; MLPerf requires it to use the same FP8 configuration as the performance runs, which the launcher does by default. The script generates `mlperf_log_accuracy.json` and then scores GAUC against the reference:

**Offline accuracy:**

```bash
CONF=user_mi355x8_nve_b64_qps12200_OFFLINE10min.conf \
SCENARIO=Offline \
bash "$RUNNER/scripts/image/run_image.sh" accuracy
```

**Server accuracy:**

```bash
CONF=user_mi355x8_nve_b64_qps12200_PROD10min.conf \
SCENARIO=Server \
bash "$RUNNER/scripts/image/run_image.sh" accuracy
```

AMD submitted lifetime GAUC on MI355X is **0.78629** — a PASS with margin.

#### Compliance Test (TEST08) - DLRM-v3

A valid submission must also clear the TEST08 compliance audit, which verifies that sampled inference outputs match the reference within tolerance:

```bash
bash "$RUNNER/scripts/image/run_image.sh" test08
```

AMD submission reports `TEST PASS` with zero unmatched entries on both systems.

#### Read the Verdicts - DLRM-v3

```bash
cat $RESULTS_ROOT/artifacts/gold_server_run_*/mlperf_log_summary.txt   # performance: "Result is : VALID"
cat $RESULTS_ROOT/artifacts/gold_acc_*/accuracy_metrics.txt            # accuracy: lifetime GAUC
```

## Wan 2.2-t2v

**Wan 2.2-t2v** (Wan2.2-T2V-A14B) is a 14-billion-parameter text-to-video generative model. The model uses a Mixture-of-Experts architecture with two experts activated sequentially during the denoising process: a High Noise Expert active in the early denoising stages and a Low Noise Expert that completes the process.

### Step 1: Prepare the Docker Container - Wan 2.2-t2v

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_wan2_2_inference_6.1
```

Start the Docker container:

```bash
mkdir -p ./mlperf_outputs
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --network=host --privileged \
  --shm-size 128G \
  --name ${USER}-mlperf-inference-wan22 \
  -v ./mlperf_outputs:/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/runs \
  -w /app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/ \
  rocm/amd-mlperf:mi355x_wan2_2_inference_6.1 \
  /bin/bash
```

### Step 2: Download the Reference Model and Dataset - Wan 2.2-t2v

Download the Wan 2.2 model from within the container:

```bash
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

Fetch the prompts and fixed latents:

```bash
python3 -m tools.fetch_data --with-calibration --with-samples-list
```

### Step 3: Run the Benchmark Harness - Wan 2.2-t2v

Wan 2.2-t2v is evaluated under two scenarios: **Offline** (maximize throughput) and **Single Stream** (one request at a time, measuring end-to-end latency).

Switch to the `/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/` folder inside the Docker container. Use `run_scenarios.sh` to run the accuracy, performance, compliance, and, optionally, VBench benchmarks for both Single Stream and Offline scenarios:

```bash
# Run both SingleStream and Offline (default)
./run_scenarios.sh
```

You can also run specific scenarios or skip specific benchmarks:

```bash
# Run only one scenario
./run_scenarios.sh SingleStream
./run_scenarios.sh Offline

# Skip VBench or compliance
./run_scenarios.sh --skip-vbench
./run_scenarios.sh --skip-compliance

# Preview commands without executing
./run_scenarios.sh --dry-run
./run_scenarios.sh --help
```

Results are written under `runs/`. If you used the bind-mount above, that tree appears on the host as `./mlperf_outputs/`.

## Llama 3.1 8B

**Llama 3.1 8B** is Meta's 8-billion-parameter instruction-tuned language model, evaluated on the CNN/DailyMail summarization dataset. AMD submission uses **MXFP4** quantization with the **vLLM** inference backend. Scenarios: **Offline**, **Server**, and **Interactive**.

### Step 1: Prepare the Docker Container - Llama 3.1 8B

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_llama3_1_8b_inference_6.1
```

Start the Docker container:

```bash
docker run -it --name llama3_1-8b_test \
  --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $(pwd)/model:/model -v $(pwd)/data:/data \
  rocm/amd-mlperf:mi355x_llama3_1_8b_inference_6.1
```

You start in `/lab-mlperf-inference/code`, where `main.py` lives. The `-v $(pwd)/model:/model` and `-v $(pwd)/data:/data` mounts keep the downloaded model and dataset on the host so they persist across containers.

### Step 2: Download the Reference Model and Dataset - Llama 3.1 8B

From within the Docker container, download the MXFP4-quantized model:

```bash
hf download amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ --local-dir /model
```

Download the CNN/DailyMail evaluation dataset:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /data https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
```

The configs read `/data/cnn_eval.json`, so make sure the eval file ends up there (move or symlink it if the downloader nests it under a subfolder).

### Step 3: Run the Benchmark Harness - Llama 3.1 8B

Run from `/lab-mlperf-inference/code`. Every run must pass `--backend vllm`.

Note: The commands below are for MI355X. To execute on a different GPU, change the config file as appropriate.

#### Offline Scenario - Llama 3.1 8B

```bash
# Performance
python3 main.py --config-path llama3.1-8b --config-name offline_mi355x \
  --backend vllm test_mode=performance \
  harness_config.output_log_dir=results/llama3_1-8b/Offline/performance/run_1

# Accuracy (generates mlperf_log_accuracy.json)
python3 main.py --config-path llama3.1-8b --config-name offline_mi355x \
  --backend vllm test_mode=accuracy \
  harness_config.output_log_dir=results/llama3_1-8b/Offline/accuracy

# Score accuracy (ROUGE; writes accuracy.txt next to the input)
bash scripts/check_llama3_1_8b_accuracy_scores.sh \
  results/llama3_1-8b/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server Scenario - Llama 3.1 8B

```bash
# Performance
python3 main.py --config-path llama3.1-8b --config-name server_mi355x \
  --backend vllm test_mode=performance \
  harness_config.output_log_dir=results/llama3_1-8b/Server/performance/run_1

# Accuracy
python3 main.py --config-path llama3.1-8b --config-name server_mi355x \
  --backend vllm test_mode=accuracy \
  harness_config.output_log_dir=results/llama3_1-8b/Server/accuracy

# Score accuracy
bash scripts/check_llama3_1_8b_accuracy_scores.sh \
  results/llama3_1-8b/Server/accuracy/mlperf_log_accuracy.json
```

#### Interactive Scenario - Llama 3.1 8B

```bash
# Performance
python3 main.py --config-path llama3.1-8b --config-name interactive_mi355x \
  --backend vllm test_mode=performance \
  harness_config.output_log_dir=results/llama3_1-8b/Interactive/performance/run_1

# Accuracy
python3 main.py --config-path llama3.1-8b --config-name interactive_mi355x \
  --backend vllm test_mode=accuracy \
  harness_config.output_log_dir=results/llama3_1-8b/Interactive/accuracy

# Score accuracy
bash scripts/check_llama3_1_8b_accuracy_scores.sh \
  results/llama3_1-8b/Interactive/accuracy/mlperf_log_accuracy.json
```

The `check_llama3_1_8b_accuracy_scores.sh` step reads the dataset from `/data/cnn_eval.json` and the model from `/model/` by default, and writes `accuracy.txt` (ROUGE scores) next to each `mlperf_log_accuracy.json`.

## Expected Results

| Model | Platform | Scenario | Metric | Expected Result |
| --- | --- | --- | --- | --- |
| dlrm-v3 | MI355X | Offline | queries/s | ~13,000 |
| dlrm-v3 | MI355X | Server | queries/s | ~12,000 |
| dlrm-v3 | MI350X | Offline | queries/s | ~10,500 |
| dlrm-v3 | MI350X | Server | queries/s | ~9,200 |
| dlrm-v3 | MI350P | Offline | queries/s | ~4,500 |
| dlrm-v3 | MI350P | Server | queries/s | ~4,200 |
| gpt-oss-120b | MI355X | Offline | tokens/s | ~120,000 |
| gpt-oss-120b | MI355X | Server | tokens/s | ~111,000 |
| gpt-oss-120b | MI350P | Offline | tokens/s | ~48,000 |
| gpt-oss-120b | MI350P | Server | tokens/s | ~40,000 |
| gpt-oss-120b (72 GPUs) | MI355X | Offline | tokens/s | ~1,000,900 |
| gpt-oss-120b (72 GPUs) | MI355X | Server | tokens/s | ~950,468 |
| llama2-70b | MI355X | Interactive | tokens/s | ~72,000 |
| llama2-70b | MI355X | Offline | tokens/s | ~105,000 |
| llama2-70b | MI355X | Server | tokens/s | ~101,000 |
| llama2-70b | MI350P | Interactive | tokens/s | ~21,000 |
| llama2-70b | MI350P | Offline | tokens/s | ~41,000 |
| llama2-70b | MI350P | Server | tokens/s | ~40,000 |
| llama3.1-8b | MI355X | Offline | tokens/s | ~165,000 |
| llama3.1-8b | MI355X | Server | tokens/s | ~155,000 |
| llama3.1-8b | MI355X | Interactive | tokens/s | ~140,000 |
| deepseek-r1 | MI355X | Offline | tokens/s | ~51,000 |
| deepseek-r1 | MI355X | Server | tokens/s | ~41,000 |
| wan2.2-t2v | MI355X | Offline | samples/s | ~0.07 |
| wan2.2-t2v | MI355X | Single Stream | latency (s) | ~16.0 |
| wan2.2-t2v | MI350X | Offline | samples/s | ~0.05 |
| wan2.2-t2v | MI350X | Single Stream | latency (s) | ~20.000 |
| wan2.2-t2v | MI350P | Offline | samples/s | ~0.035 |
| wan2.2-t2v | MI350P | Single Stream | latency (s) | ~41.000 |

Expect a certain degree of variability in your runs, generally within 3%. If you see larger deviations, they may be due to:

- ECC not enabled (required by MLCommons rules and affects HBM read/write performance)
- Wrong Docker image tag (using a development image instead of the submission image)
- Incorrect dataset preprocessing (data format mismatch changes token counts and query counts)
- Thermal throttling (ensure adequate system cooling for sustained workloads)

## Summary

AMD MLPerf Inference v6.1 submission demonstrates the breadth and maturity of the ROCm software stack for production AI inference workloads. Across recommendation systems (dlrm-v3), large dense language models (llama2-70b), MoE language models (gpt-oss-120b), and text-to-video generation (Wan 2.2-t2v), every workload follows the same three-step recipe: prepare the container, download the weights and dataset, and run the benchmark harness.

This is AMD fifth consecutive MLPerf Inference submission, each one deepening toolchain integration, expanding the workload coverage, and improving performance through ROCm-native optimizations. The Docker images package all dependencies — ROCm runtime, quantization kernels, inference server, and harness configuration — so any MI355X, MI350X, or MI350P system with ECC enabled and sufficient storage can reproduce the results directly.

Run the benchmarks on your own system and compare your numbers against the table in the Expected Results section above. If you encounter issues or have questions, engage with the community:

- **MLCommons inference GitHub:** [https://github.com/mlcommons/inference](https://github.com/mlcommons/inference)
- **ROCm GitHub Discussions:** [https://github.com/ROCm/ROCm/discussions](https://github.com/ROCm/ROCm/discussions)
- **AMD ROCm blogs:** [https://rocm.blogs.amd.com](https://rocm.blogs.amd.com)

## Related Resources

- [AMD MLPerf Inference v6.1 Technical Details Blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-v6.1/README.html) — optimization details
- [AMD MLPerf Inference v6.0 Reproduction Blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf_v6.0-repro/README.html) — the v6.0 version of this guide
- [AMD MLPerf Inference v6.0 Highlights Blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html) — v6.0 results analysis
- [MLCommons Inference GitHub](https://github.com/mlcommons/inference) — harness source code and per-model documentation
- [MLCommons Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) — closed division rules, accuracy constraints, ECC requirements
- [MLCommons Results Visualizer](https://mlcommons.org/visualizer) — interactive comparison of all v6.1 results across vendors
- [ROCm Quick Start Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) — ROCm installation on Ubuntu
- [AMD Instinct MI355X Product Page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi355x.html) — hardware specifications

## Disclaimers

AMD Cautionary Statement - https://www.amd.com/en/legal/copyright.html

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
