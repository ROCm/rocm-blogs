---
blogpost: true
blog_title: "Reproducing the AMD Instinct™ GPUs MLPerf Inference v5.1 Submission"
date: 09 Sep 2025
author: 'Meena Arunachalam, Miro Hodak, Poovaiah Palangappa, Wei-Ting Liao, Uma Kannikanti, Fulu Li, Karan Verma, Neha Mathews, Yamini Kamisetty, Chelsea Iluno, Ean Garvey, Kumar Deepak, Yixing Xu, Zhe Li, Guanchen Li, Xuanwu Yin, Dong Li, Clint Greene, Eliot Li'
thumbnail: 'mlperf_inf_v5.1_repro_thumbnail.png'
tags: AI/ML, GenAI, Performance, Optimization, MLPerf Inference, MLPerf
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Provide instructions to potential customers and partners to verify our MLPerf Inference v5.1 submission result.
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Poovaiah Palangappa, Wei-Ting Liao, Uma Kannikanti, Fulu Li, Karan Verma, Chelsea Iluno, Ean Garvey, Kumar Deepak, Yixing Xu, Zhe Li, Guanchen Li, Xuanwu Yin, Dong Li, Eliot Li, Clint Greene"
        "description lang=en": "In this blog, we will provide step by step instruction on how to reproduce AMD's MLPerf Inference v5.1 Submission"
        "keywords": "MLPerf, Inference, Optimization, Llama 2, SDXL, Mixtral"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Clint Greene, Eliot Li"
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

# Reproducing the AMD Instinct™ GPUs MLPerf Inference v5.1 Submission

MLPerf Inference v5.1 marks AMD’s third round of submissions and the most ambitious yet. This round features submissions on AMD Instinct MI325X and MI355X systems, including multi-node inference and models in MXFP4 datatype. Building upon the success in [MLPerf Inference v5.0](https://rocm.blogs.amd.com/artificial-intelligence/reproducing-amd-mlperf-inference-submission/README.html), AMD has submitted improved results for Llama 2 70B and SDXL on the MI325X platform in this round using new optimization techniques. For a deeper look at these optimizations, see our [Technical Dive into AMD's MLPerf Inference v5.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html). Additionally, explore how we optimized Llama 3.1 405B through pruning and fine-tuning in [Slim Down Your Llama: Pruning & Fine-Tuning for Maximum Performance](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-llama-pruning/README.html). In addition, AMD has made submissions for the following workloads:

- **Mixtral 8x7B** – A mixture-of-experts (MoE) model.
- **Llama 2 70B** interactive - The original Llama 2 70B workload was first introduced in MLPerf inference v4.0.  This interactive workload with the same model demands tighter latency targets of 450 ms TTFT and 40 ms TPOT (25 tokens/s per user).
- **Llama 3.1 405B** - This workload was introduced in MLPerf inference v5.0, and has a server latency requirement of TTFT: 6000 ms and TPOT: 175 ms.

MLPerf is designed to drive innovation across both software and hardware by allowing participants to reimplement reference workloads. It features two divisions tailored to different objectives: the Closed division, which ensures apples-to-apples comparisons by using the same reference model, and the Open division, which encourages exploration of novel approaches. For the first time, AMD is competing in the Open division, leveraging advanced optimization techniques such as model pruning and fine-tuning to push performance boundaries. Combined with our Closed division submissions, this broader scope highlights how AMD Instinct platforms accelerate a diverse array of cutting-edge inference workloads.

Momentum across the ecosystem remains strong, with eight AMD partners submitting results on Instinct platforms. Every submission falls into the “available” category, meaning these systems are ready for rental or purchase today. In this blog, we’ll guide you through reproducing our results on systems from your preferred vendors and share key insights from this latest round of MLPerf Inference.

## MLPerf Inference v5.1 Submission Overview

To help you navigate AMD's submissions, here's a summary table listing the models, datatypes, platforms, and divisions involved:

| Model     | Datatype | Instinct Platform | Division |
| ---- | ---|:----------: | :----------: |
| [Llama 2 70B](llama-2-70b-submissions) | FP8 | MI325X | Closed  |
| [Llama 2 70B interactive](llama-2-70b-submissions) | FP8 | MI325X | Closed  |
| [Mixtral 8x7B](#mixtral-8x7b-submission) | FP8 | MI325X | Closed  |
| [SD-XL](#sd-xl-submission) | FP8 | MI325X | Closed  |
| [Llama 3.1 405B Pruned](#llama-3-1-405b-submissions) | MXFP4 | MI355X | Open  |
| [Llama 3.1 405B Pruned + Finetuned](#llama-3-1-405b-submissions) | MXFP4 | MI355X | Open  |
| [Llama 2 70B](llama-2-70b-open-division-submissions) | MXFP4 | MI355X | Open  |
| [Llama 2 70B](llama-2-70b-open-division-submissions) | MXFP4 | 4xMI355X | Open  |
| [Llama 2 70B](llama-2-70b-open-division-submissions) | MXFP4 | 8xMI355X | Open  |

## System Requirements

To follow along with this blog, you will need the following:

- [AMD Instinct MI355X Platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html) (for multi-node submissions, 8 systems are needed)
- [AMD Instinct MI325X Platform](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x/platform.html)
- ROCm 6.4.0 or newer
- [Supported Linux distributions for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)

Refer to the [ROCm Quick Start Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) for installation steps.

To reproduce each submission, you'll typically follow these four steps:

1. Download the reference model and dataset.
2. Prepare the Docker container.
3. Quantize the model to the specified format.
4. Run the benchmarking scripts.

The next sections walk through this process for each submission, grouped by division.

## Closed Division Submissions

The Closed division evaluates submissions using standardized models and datasets, enabling direct comparison across hardware vendors. AMD submitted results for Llama 2 70B, Mixtral 8x7B, and SDXL in this category for the MI325X. Below are detailed instructions to reproduce each.

### Llama 2 70B Submissions

Llama 2 70B was originally introduced in MLPerf Inference v4.0. In this round, both offline, server and interactive scenarios were evaluated on the MI325X platform using FP8 quantization.

https://docs.mlcommons.org/inference/benchmarks/language/llama2-70b/

First, pull the Docker image containing the required scripts and code, and start the container for the benchmark.

```bash
docker pull rocm/amd-mlperf:mi325x_llama2_70b_inference_5.1
```

Run the command below to start the Docker container:

```bash
docker run -it \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi325x_llama2_70b_inference_5.1
```

#### Prepare the Model and Dataset

Inside the Docker container, download the quantized model using this command:

```bash
git lfs install
git clone https://huggingface.co/amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2 /model/llama2-70b-chat-hf/fp8_quantized/
```

Inside the Docker container, download and process the dataset:

```bash
bash /lab-mlperf-inference/setup/download_dataset.sh
```

Runtime tunables

(Optional) To boost the machine’s performance, execute the following script before any performance test (should be run once after a reboot):

```bash
bash /lab-mlperf-inference/setup/runtime_tunables.sh
```

#### Offline Scenario Performance Benchmark

Run the offline scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name offline_mi325x \
   test_mode=performance \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Offline/performance/run_1
```

Run the offline scenario accuracy test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name offline_mi325x \
   test_mode=accuracy \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Offline/accuracy
```

The above step will generate the mlperf_log_accuracy.json, which can then be processed to verify the offline scenario accuracy:

```bash
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server Scenario Performance Benchmark

Run the server scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name server_mi325x \
   test_mode=performance \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/performance/run_1
```

Run the server scenario accuracy test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name server_mi325x \
   test_mode=accuracy \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/accuracy
```

The above step will generate a file `mlperf_log_accuracy.json`, which can then be processed to verify the server scenario accuracy:

```bash
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b/Server/accuracy/mlperf_log_accuracy.json
```

#### Interactive Mode

Run the Interactive scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name interactive_mi325x \
   test_mode=performance \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/performance/run_1
```

Run the Interactive scenario accuracy test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/llama2-70b/ \
   --config-name interactive_mi325x \
   test_mode=accuracy \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/accuracy
```

The above step will generate a file `mlperf_log_accuracy.json`, which can then be processed to verify the interactive scenario accuracy:

```bash
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b/Interactive/accuracy/mlperf_log_accuracy.json
```

### Mixtral 8x7B Submission

Mixtral 8x7B is a mixture-of-experts model making its MLPerf debut. Like Llama 2, it was submitted in the Closed division with FP8 on MI325X.
https://docs.mlcommons.org/inference/benchmarks/language/mixtral-8x7b/

- First time submission with this model.
- MOE model

#### Prepare Docker Container

First, pull the Docker image containing the required scripts and codes.

```bash
docker pull rocm/amd-mlperf:mi325x_mixtral_8x7b_inference_5.1
```

Run the docker container using the following command:

```bash
docker run -it \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi325x_mixtral_8x7b_inference_5.1
```

#### Retrieve the Model and Dataset

Inside the Docker container, download the quantized model and dataset:

```bash
huggingface-cli download amd/Mixtral-8x7B-Instruct-v0.1_FP8_MLPerf_V3 --local-dir /model/mixtral-8x7b/fp8_quantized/
```

```bash
mkdir -p /data/mixtral-8x7b
wget https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl -O /data/mixtral-8x7b/mlperf_mixtral8x7b_dataset_15k.pkl
```

Runtime tunables

(Optional) To boost the machine’s performance, execute the following script before any performance test (should be run once after a reboot):

```bash
bash /lab-mlperf-inference/setup/runtime_tunables.sh
```

#### Offline Scenario

Run the offline scenario performance test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/mixtral-8x7b/ \
   --config-name offline_mi325x \
   test_mode=performance \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Offline/performance/run_1
```

Run the offline scenario accuracy test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/mixtral-8x7b/ \
   --config-name offline_mi325x \
   test_mode=accuracy \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Offline/accuracy
```

The above step will generate the mlperf_log_accuracy.json, which can then be processed to verify the offline scenario accuracy:

##### Evaluate accuracy

```bash
bash /lab-mlperf-inference/code/scripts/check_mixtral_accuracy_scores.sh \
   /lab-mlperf-inference/results/mixtral-8x7b/Offline/accuracy/mlperf_log_accuracy.json
```

#### Server scenario

Run the server scenario performance test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/mixtral-8x7b/ \
   --config-name server_mi325x \
   test_mode=performance \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Server/performance/run_1
```

Run the server scenario accuracy test:

```bash
python /lab-mlperf-inference/code/main.py \
   --config-path /lab-mlperf-inference/code/harness_llm/models/mixtral-8x7b/ \
   --config-name server_mi325x \
   test_mode=accuracy \
   harness_config.user_conf_path=/lab-mlperf-inference/code/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Server/accuracy
```

The above step will generate the mlperf_log_accuracy.json, which can then be processed to verify the server scenario accuracy:

```bash
bash /lab-mlperf-inference/code/scripts/check_mixtral_accuracy_scores.sh \
   /lab-mlperf-inference/results/mixtral-8x7b/Server/accuracy/mlperf_log_accuracy.json
```

### SDXL submission

The software used in this submission is fully open source, leveraging IREE (an MLIR-based compiler and runtime) and the shark-ai shortfin serving platform and SHARK Tank model implementation.

For mlperf inference v5.1, GIL-free harness execution via python3.13t (free-threaded) support was leveraged to boost inference throughput, reducing CPU bottlenecks (interpreter locks) in the asynchronous python code ([shortfin SDXL service](https://github.com/nod-ai/shark-ai/blob/shared/mlperf-v5.1-sdxl-nogil-spin/shortfin/python/shortfin_apps/sd/components/service.py)/[mlperf harness](https://github.com/nod-ai/SHARK-MLPERF/tree/v5.1/code/stable-diffusion-xl/SDXL_inference)) used to execute SDXL inference. This submission's offline scenario result for MI325x also benefitted indirectly from compiler improvements to matrix multiplication code generation.

#### Setup procedure for SDXL benchmark

There are several steps in the process of setting up a workspace for reproducing the MLPerf SDXL results. We use Docker to set up a suitable environment for ease of reproducibility. Detailed instructions for running each step of the process are provided below.

##### System preparation

The SDXL MLPerf submission we will reproduce uses the following non-default compute and memory partitioning available for Instinct GPUs:

- [CPX compute partitioning](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html#compute-partitioning-modes): Each MI325X GPU has its compute power divided into 8 partitions, meaning we have 64 CPX partitions (devices) to work with.

- [NPS1 memory partitioning](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html#memory-partitioning-modes).

To achieve this partitioning, first verify the current system configuration by running `rocm-smi`, a utility provided with your ROCm installation. You should see output containing information about the system memory and compute partitioning:

```{image} images/rocm-smi-output.png
:width: 975px
:height: 290px
```

If rocm-smi does not work, see the official [ROCm documentation for troubleshooting install issues](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/reference/install-faq.html).

To set the partitions correctly, ensure you set memory partitioning first (if the machine is not already in NPS1 mode), using the following command:

```bash
sudo rocm-smi --setmemorypartition NPS1
```

Next, we will set perf determinism in SPX mode before switching to CPX:

```bash
sudo rocm-smi --setcomputepartition SPX
sudo rocm-smi --setperfdeterminism 2100
sudo rocm-smi --setcomputepartition CPX
```

After configuration is complete, these partitions should be reflected in the rocm-smi output, showing 64 available devices in NPS1 mode.

An important last step for machine configuration is to tweak some CPU, power, and numa settings on your machine -- run the following exactly:

```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches && \
sudo cpupower idle-set -d 2 && \
sudo cpupower frequency-set -g performance && \
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog && \
echo 0 | sudo tee /proc/sys/kernel/numa_balancing && \
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space && \
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled && \
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```

If you do not have `cpupower` on your machine, `sudo apt install linux-tools-common` before running the above commands.

##### Running the SDXL benchmark

To run the submission scenarios, we need the model weights, validation dataset, and source files for both the mlcommons/inference repository and the AMD SHARK team’s harness. The harness is a server which interfaces the shortfin serving platform SDXL API with the MLPerf loadgen library.

First, clone the repository containing the Dockerfiles and submission-related source files:

```bash
git clone https://github.com/nod-ai/SHARK-MLPERF -b v5.1
cd SHARK-MLPERF/code/stable-diffusion-xl
```

#### Setup the inference Docker container

Note: By default, the docker run command that follows will link your machine’s local filesystem directories – /data/mlperf_sdxl/data and /data/mlperf_sdxl/models/ to volumes in your docker container. You may change these links if desired, or remove them if you are not running quantization from scratch and understand that the files will not persist outside of your docker container.

From code/stable-diffusion-xl/:

##### Build the containers

```bash
./build_docker.sh
./build_docker_nogil.sh
```

#### Run precompilation

The model artifacts for SDXL inference need to be exported from the model checkpoint and compiled for the hardware target through IREE.
The first container is where the precompilation steps and server scenario will be executed.

```bash
./run_docker.sh
```

Inside the container:

Download data and base weights

```bash
./download_data.sh
./download_model.sh
```

Execute precompilation with shark-ai / IREE tooling

```bash
# Compile the SHARK engines (Offline)
IREE_BUILD_DIR=/iree/build-offline/ IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325_bs32.mlir --model_json sdxl_config_fp8_sched_unet_bs32.json

# Compile the SHARK engines (Server)
IREE_BUILD_DIR=/iree/build-server/ IREE_BUILD_MP_CONTEXT="fork" ./precompile_model_shortfin.sh --td_spec attention_and_matmul_spec_gfx942_MI325.mlir --model_json sdxl_config_fp8_sched_unet_bs2.json
```

Preprocess the coco dataset and prepare for run execution

```bash
python3.11 preprocess_data.py
```

#### Run scenarios and reproduce results

##### Server Mode

Now that you have successfully compiled artifacts for running SDXL, you can run the Server scenario in the current docker container:

```bash
./run_scenario_server_MI325x_cpx.sh
```

This will run:

- Performance Run
- Accuracy Run
- Accuracy Validation
- Compliance Test 01 and 04

for the server scenario, likewise for the offline script which we will run in a separate docker container as follows:

##### Offline Mode (Using GIL-free python)

```bash
exit
./run_docker_nogil.sh
```

This will pick up the previous container's compiled artifacts automatically.

Once you are in this container, run:

``` bash
PYTHON_GIL=0 ./run_scenario_offline_MI325x_cpx.sh
```

By default, the scripts will save your submission results to `code/stable-diffusion-xl/SDXL_inference/Submission/...` including accuracy and compliance test results.

The offline scenario uses a large sample count and can take up to 3 hours to complete all scenario tests. Server mode is shorter and should take under 45 minutes.

#### Expected results

While running the scenario runner scripts, you will see loadgen output after each run in the scenario. The first run is the performance run which gives an accurate throughput performance calculation. The other runs are for testing accuracy and compliance with submission constraints.

To verify that the results are within acceptable accuracy, you may view the accuracy.txt generated by the accuracy validation test. This file is saved by default in the harness working directory under the following directories:

Offline mode - `Submission/closed/AMD/results/8xMI325x_2xEPYC-9655/stable-diffusion-xl/Offline/accuracy/accuracy.txt`.

Server mode - `Submission/closed/AMD/results/8xMI325x_2xEPYC-9655/stable-diffusion-xl/Server/accuracy/accuracy.txt` .

In the accuracy.txt file, you will see accuracy scores, e.g. `Accuracy Results: {'FID_SCORE': '23.813210898858188', 'CLIP_SCORE': '31.749402277171612'}`. These scores must be within the submission constraints: `FID_SCORE ∈ (23.0108, 23.9501)`, `CLIP_SCORE ∈ (31.686, 31.813)` for the results to be considered valid.

The performance run output for Offline mode should look like this:

```bash
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 18.5856
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 107574744790
Max latency (ns)                : 5778682867542
Mean latency (ns)               : 2954453799716
50.00 percentile latency (ns)   : 2965965347259
90.00 percentile latency (ns)   : 5224408938958
95.00 percentile latency (ns)   : 5510491274623
97.00 percentile latency (ns)   : 5628167067300
99.00 percentile latency (ns)   : 5734881585613
99.90 percentile latency (ns)   : 5769285663572

================================================
Test Parameters Used
================================================
samples_per_query : 107400
target_qps : 17
target_latency (ns): 0
max_async_queries : 1
min_duration (ms): 600000
max_duration (ms): 0
min_query_count : 1
max_query_count : 107400
qsl_rng_seed : 1780908523862526354
sample_index_rng_seed : 14771362308971278857
schedule_rng_seed : 18209322760996052031
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 5000

No warnings encountered during test.

No errors encountered during test.
```

The server mode output should resemble the following:

```bash
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 16.20
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
Early Stopping Result:
 * Run successful.

================================================
Additional Stats
================================================
Scheduled samples per second : 16.66
Min latency (ns)                : 6861142555
Max latency (ns)                : 18656990527
Mean latency (ns)               : 10165896851
50.00 percentile latency (ns)   : 8919662797
90.00 percentile latency (ns)   : 15313217419
95.00 percentile latency (ns)   : 17300266456
97.00 percentile latency (ns)   : 17580579392
99.00 percentile latency (ns)   : 18010059342
99.90 percentile latency (ns)   : 18533803943

================================================
Test Parameters Used
================================================
samples_per_query : 1
target_qps : 16.5
target_latency (ns): 20000000000
max_async_queries : 0
min_duration (ms): 600000
max_duration (ms): 0
min_query_count : 100
max_query_count : 0
qsl_rng_seed : 1780908523862526354
sample_index_rng_seed : 14771362308971278857
schedule_rng_seed : 18209322760996052031
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 5000

No warnings encountered during test.

No errors encountered during test.
```

##### Tweaking and troubleshooting the SDXL benchmark

Here are a few tips if you run into problems running the benchmark or having difficulty getting an acceptable result:

- Latency requirements fail in the SDXL Server Scenario

If you see the following lines in your performance results, it indicates that your system is unable to process the number of prompts fast enough to satisfy the latency constraints.

```bash
Result is : INVALID
  Performance constraints satisfied : No
  Min duration satisfied : Yes
  Min queries satisfied : Yes
```

Try lowering the QPS specified in SHARK-MLPERF/code/stable-diffusion-xl/SDXL_inference/run_scenario_server_MI325_cpx.sh to 16.2, just above the expected throughput. Rerun the benchmark. The result should meet the latency constraint with this change.

#### Troubleshooting / FAQ

If you don’t see 64 devices when you run rocm-smi, it is either due to a ROCm driver version mismatch or you might need to run the last blacklisting step in system setup and then reboot the system.

For any questions or issues with these instructions or contents of the SHARK-MLPERF repository, please [file an issue on the SHARK-MLPERF github](https://github.com/nod-ai/SHARK-MLPERF/issues/new).

If quantization takes more than 2 hours, ensure you are in SPX mode or skip quantization and use the public Hugging Face pre-quantized weights (the precompile script will handle this for you).

## Open Division Submissions

The Open division allows for model modifications, enabling participants to explore optimizations such as pruning and fine-tuning. AMD has made several LLM submissions on the MI355X platform using the MXFP4 datatype.

### Llama 3.1 405B Submissions

AMD’s most complex submission in this round leverages the MI355X’s support for the MXFP4 data type. To optimize inference, we explore two distinct approaches for running Llama 3.1 405B. In the first approach, the model is pruned, fine-tuned, and then quantized to MXFP4 before inference. In the second approach, the model is quantized to MXFP4, pruned, and then used for inference measurements. Detailed steps for both approaches are provided below.

<!-- - https://docs.mlcommons.org/inference/benchmarks/language/llama3_1-405b/
- Open category - allow us to unleash the power of model pruning techniques to optimize performance.
- Only submit with MI355X with FP4 -->

### Llama 3.1 405B: Pruning and finetuning

In this approach we start with the original BF16 model: which is pruned with the layers that we intend to drop and then fine tuned with LoRA adapter to bring back the accuracy and then finally, used in inference.

#### Model and Data Preparation

Pull the Docker image containing the required scripts and code, and start the container for the benchmark.

```bash
docker pull rocm/amd-mlperf:mi355x_llama3_1_405b_inference_5.1
```

Start the Docker container for the setup and move to the root folder:

```bash
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_llama3_1_405b_inference_5.1
cd /
```

Access your HuggingFace token and run the following command:

```bash
# Set env variable for your Hugging Face token. Generate an access token on HuggingFace if you don't have one.
export HF_TOKEN="<hf_your_access_token>"
```

Download the `Llama-3.1-405B-Instruct` model and dataset by running the commands below. The model will be saved at /model/Llama-3.1-405B-Instruct and the dataset will be saved at /data/.

```bash
bash script/download_model.sh

sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl /data/ -P
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl /data/ -P

```

##### Pruning the Llama 3.1 405B Model

Use the following script to prune the `[62, 103]` layers of the `Llama-3.1-405B-Instruct` model:

```bash
python script/prune_llama.py
```

The pruned model is saved at `/model/MLPerf-Pruned-Llama-3.1-405B-Instruct`.

###### Training and Validation Dataset Preparation

The pruned model needs to be fine tuned to ensure its accuracy is not degraded significantly. In order to prepare the training and validation datasets, first install the required packages:

```bash
cd /RULER/docker
pip install -r requirements.txt
```

Next, download the data by running the following scripts:

```bash
cd /RULER/scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
```

Generate the first part of the data as follows:

```bash
cd /RULER/scripts
bash run.sh llama3.1-405b-mxfp4 synthetic
```

After this run, navigate to `/RULER/scripts/synthetic.yaml` and modify  `num_needle_v` in `niah_multivalue` from `8` to `16`. Change `RESULTS_DIR` in `run.sh` from `RESULTS_DIR="/data/original_8/${MAX_SEQ_LENGTH}"` to `RESULTS_DIR="/data/original_16/${MAX_SEQ_LENGTH}"`.

```bash
bash run.sh llama3.1-405b-mxfp4 synthetic
```

After this run, navigate to `/RULER/scripts/synthetic.yaml` and modify `num_needle_v` in `niah_multivalue` to `32`. Change `RESULTS_DIR` in `run.sh` from `RESULTS_DIR="/data/original_16/${MAX_SEQ_LENGTH}"` to `RESULTS_DIR="/data/original_32/${MAX_SEQ_LENGTH}"`.

```bash
bash run.sh llama3.1-405b-mxfp4 synthetic
```

Finally, after generating all three parts of the data, run the following script to combine a total of 9 .jsonl files and split the combined file into `train.json` and `valid.json` files for model training in LLaMA-Factory:

```bash
python combine_data.py
```

The two .json files can be found in the `/data` folder.

##### LoRA Fine Tuning

The next step is to fine tune the pruned Llama 3.1 405B model using LoRA. First, build the docker image for LoRA fine tuning by running the script below:

Environment setup inside the docker can be done as follows:

```bash
cd /LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install liger-kernel
pip install deepspeed==0.16.9
```

Finally, finish the LoRA fine-tuning process and get the LoRA adapter by running the command below:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
```

After the training is done, merge the LoRA adapter into the original model and get the fine-tuned model by running the command below:

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

You can modify the `adapter_name_or_path` in `examples/merge_lora/llama3_lora_sft.yaml` with the path of any checkpoint for merging. The fine-tuned model is saved at `/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft`.

Specifically, we use the checkpoint at step 140 for inference, which means that we change the `adapter_name_or_path` into `saves/llama3-405b_prune_finetune/lora/sft/checkpoint-140`.

##### Quantization of the Combined BF16 LoRA adapter and Pruned BF16 model of Llama 3.1 405B

This section describes how to quantize the combined BF16 LoRA adapter and Pruned BF16 version of Llama 3.1 405B into FP4 datatype.

Install the MLCommons MLC Automation framework and download the MLPerf Calibration dataset:

```bash
pip install mlc-scripts
mlcr get,dataset,mlperf,inference,llama3,_calibration --outdirname=./ -j
```

Finish the quantization process by running the script:

```bash
python /quantization/mxfp4_quantization.py
```

Copy the tokenizer by running the command below:

```bash
cp /model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft/tokenizer* /model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize/
```

The quantized model is saved at `/model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize`.

### Llama 3.1 405B Submission: Pruning only

- Instructions for downloading model and dataset
Pull the Docker image containing the required scripts and codes, and start the container for the benchmark.

```bash
docker pull rocm/amd-mlperf:mi355x_llama3_1_405b_inference_5.1
```

Start the Docker container for the setup and move to the root folder:

```bash
docker run -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_llama3_1_405b_inference_5.1
cd /
```

Access your HuggingFace token and run the following command:

```bash
# Set env variable for your Hugging Face token. Generate an access token on HuggingFace if you don't have one.
export HF_TOKEN="<hf_your_access_token>"
```

Download the `Llama-3.1-405B-Instruct` model and dataset by running the commands below. The model will be saved at /model/Llama-3.1-405B-Instruct and the dataset will be saved at /data/.

```bash
bash script/download_model.sh

mv /model/Llama-3.1-405B-Instruct /model/llama3.1-405b/orig/

sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl /data/llama3.1-405b/ -P
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3.1_405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl /data/llama3.1-405b/ -P

```

-Instructions for quantizing the model

```bash
cd /lab-mlperf-inference/setup/
bash quantize_llama3_1_405b.sh
```

The quantized model is saved at /model/llama3.1-405b/fp4_quantized/

-Instructions for pruning the model

Set the environment variables `GIT_ROOT,MODEL_PATH` as follows:

```bash
export $GIT_ROOT=/lab-mlperf-inference/setup/
export $MODEL_PATH=/model/llama3.1-405b/fp4_quantized/
python3 $GIT_ROOT/drop_layers.py --model $MODEL_PATH --initial 59 --final 84
```

The pruned model is saved at /model/llama3.1-405b/fp4_quantized/pruned_59_84/

### Llama 3.1 405B Inference Setup

#### Generate Submission Result

This section describes how to conduct performance measurement and accuracy validation according to MLPerf rules and implementation guidelines.

##### Running Performance Measurement for Offline Scenario

The MLPerf offline scenario evaluation is conducted using the mlperf harness on VLLM serving engine. We need to emphasize that it is a good idea to check that the model path and the dataset path in /lab-mlperf-inference/code/harness_llm/models/llama3-405b/offline_mi355x.yaml is accurate compared with the actual model path and the dataset path in the container. In the current offline_mi355x.yaml file, the model path is pointed to /model/llama3.1-405b/fp4_quantized/pruned_59_84/ for the pruning only case. To conduct inference experiments for the pruning and finetuning case, you need to change the model path in offline_mi355x.yaml to /model/MLPerf-Pruned-Llama-3.1-405B-Instruct-lora-sft-quantize/ to make it run correctly. Please use the following commands to launch the offline scenario performance evaluation:

```bash
cd /lab-mlperf-inference/code/
export VLLM_USE_V1=0
python3 main.py --config-path harness_llm/models/llama3-405b --config-name offline_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_offline_performance
```

The results are saved in `mlperf_log_summary.txt` in the following folder:

```bash
code/results/llama3_offline_performance
```

###### Checking Accuracy of the Inference Results

In this section, we describe how to check accuracy of the inference results based on MLPerf implementation guidelines.

```bash
export VLLM_USE_V1=0
python3 main.py --config-path harness_llm/models/llama3-405b --config-name offline_mi355x --backend vllm test_mode=accuracy harness_config.output_log_dir=results/llama3_offline_performance
```

The results are saved in `mlperf_log_accuracy.json` in the folder `code/results/llama3_offline_performance`.

In the `code` folder, run the following script with input `mlperf_log_accuracy.json` to get the accuracy scores:

```bash
cd /lab-mlperf-inference/code
cp results/llama3_offline_performance/mlperf_log_accuracy.json ./
bash check_llama3_accuracy_scores.sh "mlperf_log_accuracy.json"
```

The results are saved in the file `accuracy.txt`, which looks like the following:

```text
Here are the results for the pruning only case:

Results
{'rougeL': 21.587649286570944, 'exact_match': 86.5952150893448, 'gen_len': 26809005, 'gen_num': 8313, 'gen_tok_len': 6318740, 'tokens_per_sample': 760.1}
```

For Llama 3.1 405B workloads, reference score according to MLPerf, Running the GPU implementation in FP16 precision resulted in the following FP16 accuracy targets:

```json
{
        'rougeL': 21.6666,
        'exact_match': 90.1335,
        'tokens_per_sample': 684.68,
}
```

The accuracy target is 99% for rougeL and exact_match, and 90% for tokens_per_sample.

The accuracy results can be similar to what is shown as follows:

```text
Here are the results for the pruning and finetuning case:

Results

{'rougeL': 21.170078928684507, 'exact_match': 87.22471872931833, 'gen_len': 20197937, 'gen_num': 8313, 'gen_tok_len': 5008413, 'tokens_per_sample': 602.5}

```

The performance results can be similar to what is shown as follows:

```console
Here are the results for the pruning only case:

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 2.49831
Tokens per second: 1942.23
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

Here are the results for the pruning and finetuning case:

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 3.34884
Tokens per second: 2019.04
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================

```

### Llama 2 70B Open Division Submissions

In this variant, Llama 2 70B is quantized to MXFP4 and evaluated on MI355X in 1-, 4-, and 8-node configurations.

#### Inference Docker Container Setup

```bash
docker pull rocm/amd-mlperf:mi355x_llama2_70b_inference_5.1
```

#### Single Node Submission

#### Download the Model and Dataset

Inside the Docker container, download the quantized model using this command:

```bash
git lfs install
git clone https://huggingface.co/amd/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-MLPerf-GPTQ
```

Inside the Docker container, download and process the dataset:

```bash
bash /lab-mlperf-inference/setup/download_dataset.sh
```

##### Performance Measurement

The MLPerf offline scenario is evaluated using the MLPerf harness on the vLLM serving engine. Please use the following commands to launch the offline scenario performance evaluation:

```bash
python3 main.py --config-path harness_llm/models/llama2-70b --config-name offline_mi355x --backend vllm test_mode=performance harness_config.output_log_dir=results/llama2_offline_performance
```

The results are saved in `mlperf_log_summary.txt` under the folder `code/results/llama2_offline_performance`. The output should resemble the following:

```bash
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 321.002
Tokens per second: 92081
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 7946678275629
Max latency (ns)                : 8038821082037
Mean latency (ns)               : 844406901550
50.00 percentile latency (ns)   : 8001897404803
90.00 percentile latency (ns)   : 8037419092960
95.00 percentile latency (ns)   : 8038116444342
97.00 percentile latency (ns)   : 8038398226199
99.00 percentile latency (ns)   : 8038679207142
99.90 percentile latency (ns)   : 8038806534288


================================================
Test Parameters Used
================================================
samples_per_query : 2577300
target_qps : 330
ttft_latency (ns): 2000000000
tpot_latency (ns): 200000000
max_async_queries : 1
min_duration (ms): 7100000
max_duration (ms): 0
min_query_count : 1
max_query_count : 0
qsl_rng_seed : 6023615788873153749
sample_index_rng_seed : 15036839855038426416
schedule_rng_seed : 9933818062894767841
accuracy_log_rng_seed : 0
accuracy_log_probability : 0
accuracy_log_sampling_target : 0
print_timestamps : 0
performance_issue_unique : 0
performance_issue_same : 0
performance_issue_same_index : 0
performance_sample_count : 24576
WARNING: sample_concatenate_permutation was set to true. 
Generated samples per query might differ from the configured value.
Check the generated_samples_per_query line in the detailed log for the real
samples_per_query value

2 warnings encountered. See detailed log.

No errors encountered during test.
```

##### Accuracy Measurement

The MLPerf offline accuracy is evaluated using the same MLPerf harness on VLLM serving engine. Please use the following commands to launch the offline scenario performance evaluation:

```bash
python3 main.py --config-path harness_llm/models/llama2-70b --config-name offline_mi355x --backend vllm test_mode=accuracy harness_config.output_log_dir=results/llama2_offline_accuracy
```

The above step will generate the mlperf_log_accuracy.json, which can then be processed to verify the offline scenario accuracy:

```bash
bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2_offline_accuracy/mlperf_log_accuracy.json
```

The output of accuracy evaluation should resemble the following:

```bash
{'rouge1': 44.4931, 'rouge2': 22.1416, 'rougeL': 28.9048, 'rougeLsum': 42.0484, 'gen_len': 28125378, 'gen_num': 24576, 'gen_tok_len': 7155345, 'tokens_per_sample': 291.2}
```

## Summary

MLPerf Inference v5.1 showcases AMD’s growing capabilities in accelerating state-of-the-art inference workloads across both standardized (Closed) and flexible (Open) benchmarking tracks. Key takeaways from this round include:

- **Broader Model Coverage:** Submissions span Llama 2 70B, Llama 3.1 405B, Mixtral, and SDXL across different latency and throughput profiles.
- **Advanced Optimization:** AMD utilized techniques such as model pruning, quantization to MXFP4, and LoRA fine-tuning to improve accuracy and performance.
- **Scalability:** Multi-node results demonstrate the Instinct platform’s readiness for enterprise-scale deployment.
- **Partner Ecosystem:** Eight partners submitted results using AMD hardware, reinforcing the strength of the surrounding ecosystem.

Through this blog, developers can reproduce AMD’s results and gain hands-on experience with MLPerf-compliant benchmarking on Instinct GPUs. For specific insights into how we optimized Llama 3.1 405B, check out [Slim Down Your Llama: Pruning & Fine-Tuning for Maximum Performance](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-llama-pruning/README.html). To understand the technical details behind our submission, read our [Technical Dive into AMD's MLPerf Inference v5.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html). Whether you're optimizing LLMs or deploying AI at scale, AMD provides a robust and open platform to meet your needs.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
