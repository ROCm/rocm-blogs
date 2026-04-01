---
blogpost: true
blog_title: "Reproducing the AMD MLPerf Inference v6.0 Submission Result"
date: 1 Apr 2026
author: 'Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Li, Rajesh Poornachandran, Karan Verma, Jesus Carabano Bravo, Nico Holmberg, Mikko Lauri, Eliot Li'
thumbnail: 'mlperf_inf_v6.0_repro_thumbnail.png'
tags: MLPerf, MLPerf Inference, GenAI
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Provide instructions to potential customers and partners to verify our MLPerf Inference v6.0 submission result.
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Li, Rajesh Poornachandran, Karan Verma, Jesus Carabano Bravo, Nico Holmberg, Mikko Lauri, Eliot Li"
        "description lang=en": "Provide instructions to potential customers and partners to verify our MLPerf Inference v6.0 submission result."
        "keywords": "MLPerf, MLPerf Inference, Optimization, Llama 3.1,"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Eliot Li"
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

# Reproducing the AMD MLPerf Inference v6.0 Submission Result

MLPerf Inference v6.0 marked AMD’s fourth round of submissions to MLPerf Inference. This blog provides a step-by-step guide to reproducing AMD's results on different vendor systems.

AMD's results in MLPerf Inference v6.0 build upon AMD's success in [MLPerf Inference v5.1](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference5.1-repro/README.html), and featured submissions on the AMD Instinct MI355X system, including running models on multi-node systems with the MXFP4 datatype.

Nine AMD partners submitted results on Instinct platforms in the "Available" category. These systems are available for rent or purchase immediately.

The following is a summary of the models, datatypes, platforms, and cluster sizes used in AMD's submission:

| Model     | Datatype | Instinct Platform | Cluster Size (GPUs) |
| ---- |:----------: | :----------: | ---- |
| [Llama 2 70B](#llama-2-70b-submission) | WMXFP4 | MI355X | 8 |
| [Llama 2 70B interactive](#llama-2-70b-submission) | WMXFP4 | MI355X | 8 |
| [gpt-oss-120b](#gpt-oss-120b-submission) | WMXFP4 | MI355X | 8 |
| [gpt-oss-120b interactive](#gpt-oss-120b-submission) | WMXFP4 | MI355X | 8 |
| [Wan-2.2-T2V-A14B](#wan22-t2v-a14b-submission) | BF16 | MI355X | 8 |
| [Llama 2 70B](#multi-node-submissions) | WMXFP4 | MI355X | 87 |
| [Llama 2 70B interactive](#multi-node-submission) | WMXFP4 | MI355X | 87 |
| [gpt-oss-120b](#multi-node-submission) | WMXFP4 | MI355X | 94 |

Each [MI355X platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html) comes with 8 MI355X GPUs. In the clusters where the multi-node benchmarks were run, some GPUs were not in a healthy state. Therefore, the submissions did not use all GPUs in the cluster.

See [AMD Instinct™ GPUs MLPerf Inference v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html) for  details about AMD's submission to MLPerf Inference v6.0.

## System Requirements

You will need the following to reproduce the submissions:

- An [AMD Instinct MI355X Platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html). For multi-node submissions, 11 or 12 systems are required.
- [ROCm 7.1.0](https://rocm.docs.amd.com/en/docs-7.1.0/) or [later](https://rocm.docs.amd.com/en/latest/index.html).
- [Supported Linux distributions for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).

Refer to the [ROCm Quick Start Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) for installation steps.

The following three steps are used to reproduce each submission:

1. Prepare the Docker container.
2. Download the reference model and dataset.
3. Run the benchmarking scripts.

## Single Node Submissions

The scripts for [Llama 2 70B](https://github.com/mlcommons/inference/tree/master/language/llama2-70b), [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), and [Wan2.2-T2V-A14B](https://github.com/mlcommons/inference/tree/master/text_to_video/wan-2.2-t2v-a14b) are run on single node systems.

### Llama 2 70B Submission

AMD's submission for [Llama 2 70B](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) included offline, server, and interactive scenarios, and were evaluated on the MI355X platform using FP4 quantization.

#### Benchmark Llama 2 70B

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_llama2_70b_inference_6.0
```

Start the Docker container:

```bash
docker run -it --name llama2_test \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_llama2_70b_inference_6.0
```

From within the Docker container, download the quantized model using this command:

```bash
git clone https://huggingface.co/amd/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-6.0MLPerf-GPTQ  /model/llama2-70b-chat-hf/fp4_quantized_gptq
```
  
From within the Docker container, download and process the dataset:

```bash
bash /lab-mlperf-inference/setup/download_llama2_dataset.sh
```

##### Offline Scenario Performance Benchmark - Llama 2 70B

Run the offline scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
--config-path /lab-mlperf-inference/code/llama2-70b-99/  \
--config-name offline_mi355x test_mode=performance \
harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Offline/performance/run_1
```

It will produce output similar to the following:

```text

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 365.738
Tokens per second: 103480
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
 
================================================
Additional Stats
================================================
Min latency (ns)                : 3977557787592
Max latency (ns)                : 4031743344146
Mean latency (ns)               : 4004257081878
50.00 percentile latency (ns)   : 4007366560006
90.00 percentile latency (ns)   : 4030752663028
95.00 percentile latency (ns)   : 4031248429133
97.00 percentile latency (ns)   : 4031445761195
99.00 percentile latency (ns)   : 4031644062306
99.90 percentile latency (ns)   : 4031733347753

 
...
```

> **Note:**
>
> Results depend on the specific system where the benchmarking script was run.

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
bash  /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
/lab-mlperf-inference/results/llama2-70b/Offline/accuracy/mlperf_log_accuracy.json
```

##### Server Scenario Performance Benchmark - Llama 2 70B

Run the server scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py  --config-path /lab-mlperf-inference/code/llama2-70b-99/  
--config-name server_mi355x test_mode=performance 
harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf 
harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/performance/run_1
```

It will produce output similar to the following:

```text

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 354.41
Completed tokens per second: 100282.36
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
TTFT Early Stopping Result:
* Run successful.
TPOT Early Stopping Result:
* Run successful.
 
================================================
Additional Stats
================================================
Scheduled samples per second : 264.96
Min latency (ns)                : 83955567
Max latency (ns)                : 146131556252
Mean latency (ns)               : 17499356575
50.00 percentile latency (ns)   : 14710161989
90.00 percentile latency (ns)   : 32087812398
95.00 percentile latency (ns)   : 40555938229
97.00 percentile latency (ns)   : 47456533681
99.00 percentile latency (ns)   : 62057076953
99.90 percentile latency (ns)   : 103924598064

Completed tokens per second                 : 74638.54
Min First Token latency (ns)                : 18033503
Max First Token latency (ns)                : 1492253501
Mean First Token latency (ns)               : 200849949
50.00 percentile first token latency (ns)   : 187392125
90.00 percentile first token latency (ns)   : 256051079
95.00 percentile first token latency (ns)   : 298376130
97.00 percentile first token latency (ns)   : 419503563
99.00 percentile first token latency (ns)   : 674674289
99.90 percentile first token latency (ns)   : 930948482

Min Time per Output Token (ns)                : 58799
Max Time per Output Token (ns)                : 258647608
Mean Time per Output Token (ns)               : 61403731
50.00 percentile time to output token (ns)   : 58683300
90.00 percentile time to output token (ns)   : 69719511
95.00 percentile time to output token (ns)   : 71851225
97.00 percentile time to output token (ns)   : 74031682
99.00 percentile time to output token (ns)   : 191656553
99.90 percentile time to output token (ns)   : 215074330

Scheduled samples per second : 359.71
Min latency (ns)                : 137662641
Max latency (ns)                : 164982849642
Mean latency (ns)               : 41319947501
50.00 percentile latency (ns)   : 35850130747
90.00 percentile latency (ns)   : 75485984191
95.00 percentile latency (ns)   : 92453536522
97.00 percentile latency (ns)   : 105844207799
99.00 percentile latency (ns)   : 136596635604
99.90 percentile latency (ns)   : 160745270332
 
Completed tokens per second                 : 100282.36
Min First Token latency (ns)                : 20429511
Max First Token latency (ns)                : 1609568898
Mean First Token latency (ns)               : 616415078
50.00 percentile first token latency (ns)   : 637051878
90.00 percentile first token latency (ns)   : 850114882
95.00 percentile first token latency (ns)   : 897936566
97.00 percentile first token latency (ns)   : 930504055
99.00 percentile first token latency (ns)   : 1019241339
99.90 percentile first token latency (ns)   : 1324136675
 
Min Time per Output Token (ns)                : 221245
Max Time per Output Token (ns)                : 628524568
Mean Time per Output Token (ns)               : 144435776
50.00 percentile time to output token (ns)   : 153201297
90.00 percentile time to output token (ns)   : 157543008
95.00 percentile time to output token (ns)   : 158645028
97.00 percentile time to output token (ns)   : 159432349
99.00 percentile time to output token (ns)   : 161051159
99.90 percentile time to output token (ns)   : 165253430
...
```

> **Note:**
>
> Results depend on the specific system where the benchmarking script was run.

Run the server scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/llama2-70b-99 \
  config_name=server_mi355x \
  test_mode=accuracy \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/accuracy
```

The `mlperf_log_accuracy.json` is processed to verify the accuracy of the offline scenario:

```bash
bash  /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
/lab-mlperf-inference/results/llama2-70b/Server/accuracy/mlperf_log_accuracy.json
```

##### Interactive Scenario Performance Benchmark - Llama 2 70B

Run the interactive scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py  --config-path /lab-mlperf-inference/code/llama2-70b-99/  
--config-name interactive_mi355x test_mode=performance 
harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf 
harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/performance/run_1
```

It will produce output similar to the following:

```text
...

MLPerf Results Summary
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 260.10
Completed tokens per second: 73608.25
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
TTFT Early Stopping Result:
* Run successful.
TPOT Early Stopping Result:
* Run successful.
 
Additional Stats
Scheduled samples per second : 263.64
Min latency (ns)                : 130998483
Max latency (ns)                : 39622077035
Mean latency (ns)               : 10121546184
50.00 percentile latency (ns)   : 8825377806
90.00 percentile latency (ns)   : 18211506137
95.00 percentile latency (ns)   : 22236840085
97.00 percentile latency (ns)   : 25499348470
99.00 percentile latency (ns)   : 32974558540
99.90 percentile latency (ns)   : 38478075878
 
Completed tokens per second                 : 73608.25
Min First Token latency (ns)                : 23433435
Max First Token latency (ns)                : 908172693
Mean First Token latency (ns)               : 241975280
50.00 percentile first token latency (ns)   : 227905995
90.00 percentile first token latency (ns)   : 363500186
95.00 percentile first token latency (ns)   : 392345040
97.00 percentile first token latency (ns)   : 410349165
99.00 percentile first token latency (ns)   : 447556635
99.90 percentile first token latency (ns)   : 654750233
 
Min Time per Output Token (ns)                : 104083
Max Time per Output Token (ns)                : 56060003
Mean Time per Output Token (ns)               : 34999429
50.00 percentile time to output token (ns)   : 35971914
90.00 percentile time to output token (ns)   : 37702735
95.00 percentile time to output token (ns)   : 38112643
97.00 percentile time to output token (ns)   : 38340518
99.00 percentile time to output token (ns)   : 38889954
99.90 percentile time to output token (ns)   : 40407775

...
```

> **Note:**
>
> Results depend on the specific system where the benchmarking script was run.

Run the interactive scenario accuracy test to generate the `mlperf_log_accuracy.json` file:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/llama2-70b-99 \
  config_name=interactive_mi355x \
  test_mode=accuracy \
  harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Interactive/accuracy
```  

The `mlperf_log_accuracy.json` file is processed to verify the interactive scenario accuracy:

```bash
bash  /lab-mlperf-inference/code/scripts/setup_llama2_accuracy_env.sh

bash /lab-mlperf-inference/code/scripts/check_llama2_accuracy_scores.sh \
/lab-mlperf-inference/results/llama2-70b/Interactive/accuracy/mlperf_log_accuracy.json
```

### GPT-OSS-120B Submission

AMD's submission for [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) included offline and server scenarios.

#### Run the GPT-OSS-120B Benchmark

Pull the Docker image containing the required code and scripts:

```bash
docker pull rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.0
```

Start the Docker container:

```bash
docker run -it --name gptoss_test \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_gptoss_120b_inference_6.0
```

From within the Docker container, download the quantized model using this command:

```bash
git clone https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8-Mlperf  /model/gpt-oss-120b/fp4_quantized
```
  
From within the Docker container, download and process the dataset:

```bash
bash /lab-mlperf-inference/setup/download_gptoss_120b.sh
```

##### Offline Scenario Performance Benchmark - GPT-OSS-120B

Run the offline scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
--config-path /lab-mlperf-inference/code/gpt-oss-120b/  \
--config-name offline_mi355x test_mode=performance \
harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/performance/run_1
```

It will produce output similar to the following:

```text
...

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 69.1714
Tokens per second: 95004
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
 
================================================
Additional Stats
================================================
Min latency (ns)                : 2663965464007
Max latency (ns)                : 2681513066718
Mean latency (ns)               : 2674855205662
50.00 percentile latency (ns)   : 2675725005988
90.00 percentile latency (ns)   : 2681058093143
95.00 percentile latency (ns)   : 2681286265507
97.00 percentile latency (ns)   : 2681376911068
99.00 percentile latency (ns)   : 2681466716310
99.90 percentile latency (ns)   : 2681508604794

...
```

> **Note:**
>
> Results depend on the specific system where the benchmarking script was run.

Run the offline scenario accuracy test to generate the `mlperf_log_accuracy.json`:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/gpt-oss-120b \
  config_name=offline_mi355x \
  test_mode=accuracy \
  harness_config.dataset_path=/data/gptoss-120b/perf/perf_eval_ref.parquet \
  harness_config.accuracy_dataset_path=/data/gptoss-120b/acc/acc_eval_ref.parquet \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy
```

The `mlperf_log_accuracy.json` is processed to verify the accuracy of the offline scenario:

```bash
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
/lab-mlperf-inference/results/gpt-oss-120b/Offline/accuracy/mlperf_log_accuracy.json
```

##### Server Scenario Performance Benchmark - GPT-OSS-120B

Run the server scenario performance benchmark:

```bash
python /lab-mlperf-inference/code/main.py \
--config-path /lab-mlperf-inference/code/gpt-oss-120b/  \
--config-name server_mi355x test_mode=performance \
harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/performance/run_1
```

It will produce output similar to the following:

```text
...

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 59.86
Completed tokens per second: 82136.08
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
TTFT Early Stopping Result:
* Run successful.
TPOT Early Stopping Result:
* Run successful.
 
================================================
Additional Stats
================================================
Scheduled samples per second : 61.29
Min latency (ns)                : 4729250425
Max latency (ns)                : 249451351723
Mean latency (ns)               : 93118913728
50.00 percentile latency (ns)   : 92090855721
90.00 percentile latency (ns)   : 120200988231
95.00 percentile latency (ns)   : 129539808521
97.00 percentile latency (ns)   : 136326992569
99.00 percentile latency (ns)   : 150838236030
99.90 percentile latency (ns)   : 180265327699
 
Completed tokens per second                 : 82136.08
Min First Token latency (ns)                : 33029822
Max First Token latency (ns)                : 1909164219
Mean First Token latency (ns)               : 687235954
50.00 percentile first token latency (ns)   : 649078453
90.00 percentile first token latency (ns)   : 1070548973
95.00 percentile first token latency (ns)   : 1169717780
97.00 percentile first token latency (ns)   : 1230144725
99.00 percentile first token latency (ns)   : 1347438897
99.90 percentile first token latency (ns)   : 1607218398
 
Min Time per Output Token (ns)                : 18332542
Max Time per Output Token (ns)                : 75446232
Mean Time per Output Token (ns)               : 67443317
50.00 percentile time to output token (ns)   : 69554179
90.00 percentile time to output token (ns)   : 71481632
95.00 percentile time to output token (ns)   : 71935465
97.00 percentile time to output token (ns)   : 72171402
99.00 percentile time to output token (ns)   : 72613294
99.90 percentile time to output token (ns)   : 73229337

...
```

> **Note:**
>
> Results depend on the specific system where the benchmarking script was run.

Run the interactive scenario accuracy test to generate the `mlperf_log_accuracy.json`:

```bash
python /lab-mlperf-inference/code/main.py \
  config_path=/lab-mlperf-inference/code/gpt-oss-120b \
  config_name=server_mi355x \
  test_mode=accuracy \
  harness_config.dataset_path=/data/gptoss-120b/perf/perf_eval_ref.parquet \
  harness_config.accuracy_dataset_path=/data/gptoss-120b/acc/acc_eval_ref.parquet \
  harness_config.user_conf_path=/lab-mlperf-inference/code/gpt-oss-120b/user_mi355x.conf \
  harness_config.output_log_dir=/lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy
```

The `mlperf_log_accuracy.json` is processed to verify the accuracy of the offline scenario:

```bash
bash /lab-mlperf-inference/code/scripts/check_gptoss_accuracy_scores.sh \
/lab-mlperf-inference/results/gpt-oss-120b/Server/accuracy/mlperf_log_accuracy.json
```

### Wan2.2-T2V-A14B Submission

To run AMD's submission for [Wan2.2-T2V-A14B](https://github.com/mlcommons/inference/tree/master/text_to_video/wan-2.2-t2v-a14b), pull the Docker image `rocm/mlperf-inference:mi355x_wan2_2_inference_6.0`:

```bash
docker pull rocm/amd-mlperf:mi355x_wan2_2_inference_6.0
```

Launch the Docker container:

```bash
docker run \
    -it \
    --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --network host \
    --privileged \
    --shm-size 128G \
    --name ${USER}-mlperf-inference-xdit \
    -v ./mlperf_outputs:/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/outputs \
    -w /app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/ \
    rocm/amd-mlperf:mi355x_wan2_2_inference_6.0 \
    /bin/bash
```

Download the Wan 2.2 model from within the container:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

Switch to the `/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/` folder inside the Docker container.

Use `run_scenarios.sh` to run the accuracy, performance, compliance, and, optionally, VBench benchmarks for both single stream and offline scenarios:

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

## Multi-Node Submissions

The multi-node benchmarks for Llama 2 70B and GPT-OSS 120B involve these steps:

- Start the Docker container
- Download the model and dataset onto all the nodes
- For each scenario
  - Start the System Under Test (SUT) client
  - Start workers across all the nodes

The procedure for running the Llama 2 70B benchmark is described below. The procedure for GPT-OSS 120B is similar.

### Common Setup Across All Nodes

Pull the Docker image containing the required code and scripts. For example, for Llama 2 70B:

```bash
docker pull rocm/amd-mlperf:mi355x_llama2_70b_inference_6.0
```

Start the Docker container. For example, for Llama 2 70B:

```bash
docker run -it --name llama2_test \
--ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN \
--device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
rocm/amd-mlperf:mi355x_llama2_70b_inference_6.0
```

From within the Docker container, download the quantized model. For example, for Llama 2 70B:

```bash
git clone https://huggingface.co/amd/Llama-2-70b-chat-hf-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-6.0MLPerf-GPTQ  /model/llama2-70b-chat-hf/fp4_quantized_gptq
```

From within the Docker container, download and process the dataset. For example, for Llama 2 70B, use the `download_llama2_dataset.sh` script:

```bash
bash /lab-mlperf-inference/setup/download_llama2_dataset.sh
```

### Distributed SUT with ZMQ

Choose one node in the cluster as the Head node where the SUT client will run. Get the IP address of the head node using:

```bash
hostname -I
```

#### Running the Benchmark Under the Server Scenario

To run the benchmark under the server scenario, use the `run_harness.sh` script to start the SUT client, make sure `device_count` is the number of all the healthy GPUs across all the nodes in the cluster.

```bash
bash run_harness.sh --config-path llama2-70b-99/ --config-name server_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/llama2_server_performance_zmq port=12345 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps x 0.9>
```

```{note}
In the command above with argument port=12345, both port 12345 and port 12346 will be used when running the benchmark.  Make sure both ports are available when choosing the port in your run. For details, see the blog: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html#multi-node-inference that describes the technical details of this submission.
```

The following flags can be appended to the command for the purpose of debugging:

```bash
harness_config.target_qps=300 harness_config.duration_sec=30 harness_config.debug_record_sample_latencies=True harness_config.debug_print_finished=True harness_config.debug_dump_model_output=True
```

Use the `distributed_async_server.py` script to start a worker on each node, including the Head node. Use the IP of the Head node for the `Head node IP`:

```bash
python harness_llm/backends/vllm/zmq/distributed_async_server.py --config-path llama2-70b-99/ --config-name server_mi355x node_id=`hostname` headnode_address=<Head node IP>:12345
```

The output should resemble the following. Note that the result depends on your specific cluster and this is provided for reference only.

```text
...

MLPerf Results Summary
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Completed samples per second    : 3591.83
Completed tokens per second: 1016375.00
Result is : VALID
  Performance constraints satisfied : Yes
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
TTFT Early Stopping Result:
* Run successful.
TPOT Early Stopping Result:
* Run successful.
 
Additional Stats
Scheduled samples per second : 3647.85
Min latency (ns)                : 75001597
Max latency (ns)                : 108813997627
Mean latency (ns)               : 27280367810
50.00 percentile latency (ns)   : 23694235007
90.00 percentile latency (ns)   : 49629946900
95.00 percentile latency (ns)   : 60809901215
97.00 percentile latency (ns)   : 69560515738
99.00 percentile latency (ns)   : 89413303887
99.90 percentile latency (ns)   : 105243224037
 
Completed tokens per second                 : 1016375.00
Min First Token latency (ns)                : 18200453
Max First Token latency (ns)                : 1183960508
Mean First Token latency (ns)               : 311773426
50.00 percentile first token latency (ns)   : 302434196
90.00 percentile first token latency (ns)   : 401168352
95.00 percentile first token latency (ns)   : 463184768
97.00 percentile first token latency (ns)   : 512165926
99.00 percentile first token latency (ns)   : 640102305
99.90 percentile first token latency (ns)   : 862499278
 
Min Time per Output Token (ns)                : 22754
Max Time per Output Token (ns)                : 626231995
Mean Time per Output Token (ns)               : 95756208
50.00 percentile time to output token (ns)   : 100361604
90.00 percentile time to output token (ns)   : 103481706
95.00 percentile time to output token (ns)   : 104341682
97.00 percentile time to output token (ns)   : 104930792
99.00 percentile time to output token (ns)   : 106190403
99.90 percentile time to output token (ns)   : 110248923

...
```

The steps for running the benchmark for the offline scenario are similar to the steps for running the benchmark for the server scenario.

The `run_harness.sh` script with `config-name` set  to `offline_mi355x` is used to start the SUT client:

```bash
bash run_harness.sh --config-path llama2-70b-99/ --config-name offline_mi355x --backend zmq test_mode=performance harness_config.output_log_dir=results/llama2_offline_performance_zmq port=12345 harness_config.device_count=<SUM-OF-ALL-GPUS> harness_config.target_qps=<Node-count x single-node-qps x 0.9>
```

Start the worker across all nodes:

```bash
python harness_llm/backends/vllm/zmq/distributed_sync_offline.py --config-path llama2-70b-99/ --config-name offline_mi355x node_id=`hostname` headnode_address=<IP>:12345
```

The output should resemble the following. Note that the result depends on your specific cluster and this is provided for reference only.

```text
...

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 3629.98
Tokens per second: 1.04211e+06
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
 
================================================
Additional Stats
================================================
Min latency (ns)                : 1329550068207
Max latency (ns)                : 1387907267424
Mean latency (ns)               : 1359512982501
50.00 percentile latency (ns)   : 1359534252556
90.00 percentile latency (ns)   : 1382367854076
95.00 percentile latency (ns)   : 1385117397134
97.00 percentile latency (ns)   : 1386338710412
99.00 percentile latency (ns)   : 1387581425152
99.90 percentile latency (ns)   : 1387874730034

...
```

## Summary

AMD's MLPerf Inference v6.0 submissions highlight the company's expanding capabilities in accelerating complex inference workloads across both closed and open divisions. Using the information in this blog, developers can reproduce AMD’s results and gain hands-on experience with MLPerf-compliant benchmarking on Instinct GPUs. For technical details behind AMD's submission, see [AMD Instinct™ GPUs MLPerf Inference v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html).

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
