---
blogpost: true
blog_title: "Reproducing AMD MLPerf Training v6.1 Submission Result"
date: "16 Nov 2026"
author: "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Tamar Al-Tahan, Eliot Li"
thumbnail: 'mlperf-trainingv61-repro-thumbnail.png'
tags: "AI/ML, Fine-Tuning, GenAI, Performance, MLPerf, MLPerf Training"
category: "Applications & models"
target_audience: "AI developers, AI practitioners"
key_value_propositions: "Provide instructions to readers on how to reproduce AMD's MLPerf Training v6.1 submission result."
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Tamar Al-Tahan, Eliot Li"
        "description lang=en": "Learn how to reproduce AMD's MLPerf Training v6.1 submission result."
        "keywords": "MLPerf, Training, Optimization, Llama3.1-8B, Llama2-70B-LoRA, DLRM v4, Flux.1, GPT-OSS-20B"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Tamar Al-Tahan, Eliot Li"
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

# Reproducing AMD MLPerf Training v6.1 Submission Result

This blog provides a step-by-step guide for reproducing AMD's MLPerf Training v6.1 submission results. AMD submitted results on five benchmarks this round — an expanded portfolio compared to the three benchmarks submitted in v6.0:

- **Llama3.1-8B pretraining** — full pretraining from random weights on the C4 dataset
- **Llama2-70B LoRA fine-tuning** — parameter-efficient fine-tuning of Llama 2 70B using the GovReport dataset
- **DLRMv4** — recommendation model training on the Yambda dataset
- **Flux.1** — text-to-image diffusion model training, with two recipes: a6w4 + fp4 and fp6 + fp4
- **GPT-OSS-20B** — pretraining of a 20-billion-parameter open-weight large language model

All AMD first-party results were submitted on AMD Instinct **MI355X** and **MI350X** GPUs. Primus, AMD's unified training framework introduced in v6.0, powers the language-model and Flux.1 benchmarks in this submission. DLRMv4 uses the HSTU stack.
Readers looking for context on the performance results and competitive analysis should refer to the companion blog post: [Technical Dive into AMD's MLPerf Training v6.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.1/README.html).

The table below provides a summary of AMD submissions covered in this blog:

| Submission ID | Model | Dataset | Convergence Target | GPUs in This Guide |
| --- | --- | --- | --- | --- |
| 6.1-0062 | Llama3.1-8B Pretraining | C4 | Log perplexity ≤ 3.3 | 8 |
| 6.1-0062 | Llama2-70B LoRA Fine-Tuning | SCROLLS GovReport | Cross-entropy loss ≤ 0.925 | 8 |
| 6.1-0061 | DLRMv4 | Yambda | AUC ≥ 0.75 | 8 |
| 6.1-0064 | Flux.1, a6w4 + fp4 recipe | CC12M | Validation loss ≤ 0.586 | 32 |
| 6.1-0001 | Flux.1, fp6 + fp4 recipe | CC12M | Validation loss ≤ 0.586 | 32 |
| 6.1-0062 | GPT-OSS-20B | C4 | Log perplexity ≤ 3.34 | 8 |

## Prerequisites

To follow this guide you need a Linux machine with AMD MI355X GPUs, the Docker image tag for each benchmark (see **Docker Images** below), and this page. For Flux.1, which runs on four nodes under Slurm, check the Slurm login node and every compute node.

| Requirement | How to check | If it is missing |
| --- | --- | --- |
| AMD Instinct MI355X GPUs, 8 per node (4 nodes for Flux.1) | `amd-smi list` lists 8 GPUs | If fewer than 8 appear, something is wrong with the node. Reboot it or power cycle it first |
| Ubuntu 22.04 or later | `lsb_release -d` | Install Ubuntu 22.04 or later |
| amdgpu driver (versions below) | `cat /sys/module/amdgpu/version` | Install it with the [ROCm installation guide](https://rocm.docs.amd.com) |
| ROCm tools on the host | `rocm_agent_enumerator` prints the GPU architectures | Install ROCm as above; the launch scripts call this tool |
| Idle GPUs | `amd-smi monitor` shows near-zero GFX use and `docker ps` lists no containers | Stop the other workloads |
| Docker without `sudo` | `docker ps` runs | `sudo usermod -aG docker $USER`, then log out and back in |
| Passwordless `sudo` | `sudo -n true && echo OK` prints `OK` | Configure it; the launch scripts use `sudo` to tune the host |
| `cpupower` | `cpupower --version` | `sudo apt-get install linux-tools-common linux-tools-$(uname -r)` |
| Python 3 with pip | `python3 -m pip --version` | `sudo apt-get install python3-pip` |
| A writable `/data` folder | `test -w /data && echo OK` | `sudo mkdir -p /data && sudo chown $USER /data`, or use another path in every command |
| Internet access | `curl -sI https://github.com` | Allow Docker Hub, `training.mlcommons-storage.org`, `raw.githubusercontent.com`, and `github.com` |

Two benchmarks have extra requirements:

- **Llama2-70B LoRA:** To download the model from HuggingFace, sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT for [`meta-llama/Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf) and create a HuggingFace token in your [token settings](https://huggingface.co/settings/tokens). Steps 2 and 4 use the token as `HF_TOKEN`.
- **Flux.1:** Slurm, shared storage that every node can reach, and an RDMA network between the nodes (`ls /dev/infiniband` lists the RDMA devices on each node). The launcher stops if the container cannot open an RDMA device, because the job would otherwise run over TCP, too slowly to finish.

AMD's submitted runs used the ROCm versions specified in the table below. The ROCm version is the one inside each Docker image, so the ROCm installed on your host does not need to match it.

| Model | amdgpu driver (host) | ROCm (in the image) |
| --- | --- | --- |
| Llama3.1-8B | 6.19.4 | 7.15.0 |
| Llama2-70B LoRA | 6.19.4 | 7.15.0 |
| DLRMv4 | 6.19.4 | 7.2.1 |
| Flux.1, a6w4 + fp4 recipe | 6.16.6 | 7.15.0 |
| Flux.1, fp6 + fp4 recipe | 6.14.14 | 10.0.0 |
| GPT-OSS-20B | 6.19.4 | 7.15.0 |

```{note}
Before every run, the launch scripts run `runtime_tunables.sh` on the host with `sudo`. It drops the filesystem caches, loads the `amdgpu` driver, disables CPU idle state 2, sets the CPU governor to `performance`, disables the NMI watchdog, NUMA balancing, and address-space randomization, and enables transparent huge pages. These settings last until the machine reboots, so use a machine that you are allowed to tune.
```

### Disk Space, Time, and Where Things Live

Before downloading anything, check free space with `df -h` on `/data` and on the Docker root folder (`docker info --format '{{.DockerRootDir}}'`) to ensure you have enough space to run the benchmark.

| Model | Image | Data and model | Total |
| --- | --- | --- | --- |
| Llama3.1-8B | 46 GB | 117 GB (85 GB dataset + 32 GB model) | about 163 GB |
| Llama2-70B LoRA | 47 GB | about 270 GB | about 320 GB |
| DLRMv4 | 86 GB | 245 GB (227 GB dataset + 18 GB index) | about 330 GB |
| GPT-OSS-20B | 43 GB | 117 GB (85 GB dataset + 32 GB model) | about 160 GB |
| Flux.1, a6w4 + fp4 recipe | 45 GB per node | about 2.5 TB shared, plus about 2.5 TB of local disk per node | about 2.5 TB shared, plus about 2.55 TB on each node |
| Flux.1, fp6 + fp4 recipe | 63 GB per node | about 2.4 TB shared | about 2.4 TB shared, plus about 63 GB on each node |

Llama3.1-8B and GPT-OSS-20B use the same dataset and tokenizer, so one download can serve both by pointing `DATADIR` and `MODELDIR` at the same location for both models.

Trial times from AMD's submitted runs are included below for reference. A trial takes longer than its time to train because startup (model construction, data setup, and warm-up) happens before the timed region.

| Model | One trial, launch to finish | Of which startup | A full 10-trial set |
| --- | --- | --- | --- |
| Llama3.1-8B | about 77 min | about 7 min | about 13 hours |
| Llama2-70B LoRA | about 18 min | about 11 min | about 3 hours |
| DLRMv4 | about 25 min | about 4 min | about 4 hours |
| GPT-OSS-20B | about 61 min | about 7 min | about 10 hours |
| Flux.1, a6w4 + fp4 recipe | about 54 min | about 6 min | about 9.5 hours |
| Flux.1, fp6 + fp4 recipe | about 52 min | about 4 min | about 9 hours |

### Get the Implementation Code

The benchmark code and launch scripts are inside each Docker image, in `/workspace/code`. Step 1 of each benchmark copies that folder to the host (`~/mlperf/<benchmark>_code`, or `/shared/mlperf_flux1/flux1_code` for the Flux.1 a6w4 + fp4 recipe and `/shared/mlperf_flux1_fp6/flux1_fp6_code` for the Flux.1 fp6 + fp4 recipe), and the launch scripts run that host copy.

### Docker Images

Each benchmark has its own docker image that is available on [docker hub](https://hub.docker.com/r/rocm/amd-mlperf/tags):

| Model | Image | Image size |
| --- | --- | --- |
| Llama2-70B LoRA | `rocm/amd-mlperf:llama2_70b_training_6.1` | 47.4 GB |
| Llama3.1-8B | `rocm/amd-mlperf:llama31_8b_training_6.1` | 46.2 GB |
| DLRMv4 | `rocm/amd-mlperf:dlrmv4_training_6.1` | 85.7 GB |
| Flux.1, a6w4 + fp4 recipe | `rocm/amd-mlperf:mlperf_flux1_6.1` | 45.6 GB |
| Flux.1, fp6 + fp4 recipe | Built with Dockerfile (`mlperf_flux1_fp6:6.1`) | 63 GB |
| GPT-OSS-20B | `rocm/amd-mlperf:gpt_oss_20b_training_6.1` | 42.9 GB |

If you prefer to build the docker images yourself, please follow the instruction in the [Appendix](#appendix-build-the-docker-images-instead-of-pulling).

## How Results Are Checked

All five benchmarks are scored the same way. This section explains the method once, and the final step of each benchmark shows how to apply it to that benchmark's logs.

### What the Logs Record

Each trial writes one log, `$LOGDIR/<DATESTAMP>_<N>.log`, where `<DATESTAMP>` is the launch time as a string of digits and `<N>` is the trial number, which starts at 1 (Flux.1 starts at 0). The commands in each benchmark's Step 5 use a `*` in place of the timestamp, so `"$LOGDIR"/*_1.log` selects trial 1 (`*_0.log` for Flux.1); change the number to read another trial. They also use `$LOGDIR` from Step 3, so set it again if you opened a new terminal. Scoring uses only the MLPerf log lines, which start with `:::MLLOG` followed by a JSON record. The following three keys in the record are important to determine the result:

- `run_start` marks the start of the timed region, after initialization (model construction, data setup, and warm-up) has finished.
- `eval_accuracy` holds the quality metric from one evaluation. Its `metadata.samples_count` is the number of training samples consumed up to that evaluation.
- `run_stop` marks the end of the timed region. Its `metadata.status` is `success` when an evaluation met the quality target, and `aborted` when training reached its step limit first.

For example, these are the last two records of a submitted Llama3.1-8B trial, which reached the target at its 15th evaluation:

```text
:::MLLOG {"namespace": "", "time_ms": 1792039530706, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 3.299457550048828, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pretrain_trainer.py", "lineno": 336, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1792039530840, "event_type": "INTERVAL_END", "key": "run_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pretrain_trainer.py", "lineno": 216, "samples_count": 184320, "status": "success"}}
```

### Quality, Time to Train, and Score

To understand how MLPerf Training submissions are evaluated, it is important to understand the following metrics:

- **Quality.** Training pauses at a fixed interval of training samples and evaluates the model on a fixed validation set. Each benchmark's final step gives its metric, interval, and validation set. Training stops at the first evaluation that meets the target and logs `run_stop` with `status: success`.
- **Time to train.** A trial's time to train is the `run_stop` timestamp minus the `run_start` timestamp. It includes the time spent in evaluations, but not initialization.
- **Score.** A submission is a set of 10 trials. The fastest and slowest trials are dropped, and the score is the mean time to train of the remaining eight. All 10 trials should converge.

### Convergence and Log Checks

To simplify the scoring process, MLCommons provides two checkers, in the `mlperf_logging` package:

- **RCP checker.** Reference convergence points (RCPs) record how many samples the reference implementation needs to converge at a given global batch size. The RCP checker compares the mean samples-to-converge of your 10-trial set against them and ends with `RCP test PASSED` or `RCP test FAILED`. If a set converges in fewer samples than the reference mean but is still within the allowed range, it passes, and the checker reports a normalization factor that scales the score up.
- **Compliance checker.** This checks that a trial log contains every required key with allowed values, for example the hyperparameters fixed by the rules. The Llama3.1-8B, DLRMv4, Flux.1, and GPT-OSS-20B launchers run it after every trial and append the result to `$LOGDIR/compliance_<DATESTAMP>.out`, where each trial's section ends with `INFO - SUCCESS` or `ERROR - FAILED`. The Llama2-70B LoRA launcher skips it, so that benchmark's final step shows how to run it on the host.

Install the checkers on the host once:

```bash
python3 -m pip install "git+https://github.com/mlcommons/logging.git@6.1.0-rc2"
```

The RCP checker reads only files named `result_<N>.txt`, so each benchmark's final step copies the trial logs to those names before running it. It needs a full set of 10 trials: with fewer logs it prints `nan` and `RCP test FAILED`, so skip it after a single test trial. Use a fresh `$LOGDIR` for each 10-trial set, so that the glob `"$LOGDIR"/*_[0-9]*.log` matches only that set.

## Benchmark 1: Llama3.1-8B Pretraining

### Background - Llama3.1-8B

The Llama3.1-8B pretraining benchmark measures how quickly a model can be trained from **random weights** to reach a target validation loss on the C4 (Colossal Cleaned Common Crawl) dataset. AMD led the development of this benchmark in MLPerf Training. No checkpoint conversion is required. **Disk space required:** ~85 GB dataset + ~32 GB model directory.

### Step 1: Pull the Docker Image and Copy the Code - Llama3.1-8B

```bash
mkdir -p ~/mlperf && cd ~/mlperf

docker pull rocm/amd-mlperf:llama31_8b_training_6.1

container_id=$(docker create rocm/amd-mlperf:llama31_8b_training_6.1) && \
docker cp $container_id:/workspace/code ./llama31_8b_code && \
docker rm $container_id
```

To build the image instead, see the **Build the Images Instead of Pulling** section at the end of this guide.

### Step 2: Prepare Dataset and Tokenizer - Llama3.1-8B

```bash
mkdir -p /data/mlperf_llama31_8b
cd /data/mlperf_llama31_8b

# Download training data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After the download is complete, the data directory contains files ending in both `.idx` and `.bin` with the following naming conventions:

- Training partitions: `c4-train.en_6_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

### Step 3: Set Up Environment - Llama3.1-8B

Enter the `llama31_8b_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd ~/mlperf/llama31_8b_code

mkdir -p /data/mlperf_llama31_8b/results

export DATADIR=/data/mlperf_llama31_8b/data
export MODELDIR=/data/mlperf_llama31_8b/model/
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama31_8b_training_6.1

sudo chmod -R 777 $LOGDIR
```

Source the MI355X configuration file:

```bash
source config_MI355X_1x8x1.sh
```

The `source` command loads the settings into your current shell. Run Step 4 in this same shell, or repeat this step in a new terminal.

### Step 4: Run Training - Llama3.1-8B

Run one trial first to confirm that everything works. It takes about 77 minutes, including about 7 minutes of startup. The console can stay quiet while training. The trial is finished when the output shows `ENDING TIMING RUN` followed by `RESULT,LLAMA3.1_8B,,<seconds>,AMD,<start time>`, the total run time in seconds including startup.

```bash
export NEXP=1
bash run_with_docker.sh
```

For a compliant 10-run submission (about 13 hours):

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, the trial logs are in `$LOGDIR`.

### Step 5: Check Quality and Validate Results - Llama3.1-8B

Quality is validation log perplexity on held-out C4 text (lower is better), evaluated every 12,288 training sequences. A trial succeeds at the first evaluation at or below **3.3**.

Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/*_1.log
```

The trial converged if the last `eval_accuracy` value is 3.3 or lower and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. Across the 10 submitted trials, the target was met at the 15th or 16th evaluation (184,320–196,608 samples), with times to train of 69.9–74.8 minutes.

The last throughput record gives the average training throughput; expect approximately **44 samples/second** on MI355X:

```bash
grep overall_throughput "$LOGDIR"/*_1.log | tail -1
```

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight; the AMD submission scored **72.94 minutes**. Then run the RCP checker, after copying each trial log to the `result_<N>.txt` name it reads:

```bash
i=0; for f in "$LOGDIR"/*_[0-9]*.log; do cp "$f" "$LOGDIR/result_$((i++)).txt"; done
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

Finally, confirm that every trial's section of `$LOGDIR/compliance_*.out` ends with `INFO - SUCCESS`.

## Benchmark 2: Llama2-70B LoRA Fine-Tuning

### Background - Llama2-70B LoRA

The Llama2-70B LoRA fine-tuning benchmark measures how quickly the Llama 2 70B model can be fine-tuned to reach the target cross-entropy loss on the SCROLLS GovReport summarization dataset (pre-tokenized, packed to sequence length 8,192), using Low-Rank Adaptation (LoRA). **Disk space required:** ~270 GB for model download and conversion.

Downloading the model needs a HuggingFace token (`HF_TOKEN`) with access to `meta-llama/Llama-2-70b-hf`; see **Prerequisites**.

### Step 1: Pull the Docker Image and Copy the Code - Llama2-70B LoRA

```bash
mkdir -p ~/mlperf && cd ~/mlperf

docker pull rocm/amd-mlperf:llama2_70b_training_6.1

container_id=$(docker create rocm/amd-mlperf:llama2_70b_training_6.1) && \
docker cp $container_id:/workspace/code ./llama2_70b_code && \
docker rm $container_id
```

To build the image instead, see the **Build the Images Instead of Pulling** section at the end of this guide.

### Step 2: Download and Preprocess Data - Llama2-70B LoRA

Start a container with the host download directory mounted at `/data`. The preparation script is the copy in the image. The commands below are run in the shell you start in, so the folder you are in does not matter:

```bash
mkdir -p /data/mlperf_llama2/data

docker run --rm -it -v /data/mlperf_llama2/data:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_6.1
```

From inside the container, run the preparation script `prepare_data_and_model.sh`:

```bash
export HF_TOKEN=<your_huggingface_token>
bash ./scripts/prepare_data_and_model.sh
```

Exit the container. The data directory on the host should look like this:

```text
/data/mlperf_llama2/data
├── train.npy                          # Packed training data (seq_length=8192)
├── validation.npy                     # Packed validation data
├── packed_metadata.jsonl              # Sequence metadata for Megatron-Bridge
└── megatron_checkpoints/
    └── Llama-2-70b-hf/
        ├── latest_checkpointed_iteration.txt
        ├── latest_train_state.pt
        └── iter_0000000/
            ├── __0_0.distcp           # Sharded weights (~64 GB)
            ├── __0_1.distcp           # Sharded weights (~64 GB)
            ├── common.pt
            ├── .metadata
            ├── metadata.json
            ├── run_config.yaml
            ├── modelopt_run_config.yaml
            ├── train_state.pt
            └── tokenizer/
                ├── tokenizer.model
                ├── tokenizer_config.json
                └── special_tokens_map.json
```

### Step 3: Set Up Environment - Llama2-70B LoRA

Enter the `llama2_70b_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd ~/mlperf/llama2_70b_code

mkdir -p /data/mlperf_llama2/results

export DATADIR=/data/mlperf_llama2
export MODELDIR=/data/mlperf_llama2/data
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_6.1

sudo chmod -R 777 $LOGDIR
```

Source the MI355X configuration:

```bash
source config_MI355X_1x8x1.sh
```

The `source` command loads the settings into your current shell. Run Step 4 in this same shell, or repeat this step in a new terminal.

### Step 4: Run Training - Llama2-70B LoRA

Run one trial first to confirm that everything works. It takes about 18 minutes, including about 11 minutes of startup. The console can stay quiet while training. The trial is finished when the output shows `ENDING TIMING RUN` followed by `RESULT,LLAMA2_70B_LORA_SFT,,<seconds>,AMD,<start time>`, the total run time in seconds including startup.

```bash
export HF_TOKEN=<your_huggingface_token>
export NEXP=1
bash run_with_docker.sh
```

For a compliant 10-run submission (about 3 hours):

```bash
export HF_TOKEN=<your_huggingface_token>
export NEXP=10
bash run_with_docker.sh
```

After completion, the trial logs are in `$LOGDIR`.

### Step 5: Check Quality and Validate Results - Llama2-70B LoRA

Quality is the cross-entropy loss on the GovReport validation set (lower is better), evaluated every 384 training samples after the first three evaluations are skipped. A trial succeeds at the first evaluation at or below **0.925**.

Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/*_1.log
```

The trial converged if the last `eval_accuracy` value is 0.925 or lower and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. All 10 submitted trials met the target at the fifth evaluation (3,072 samples), with a final loss of 0.907–0.913 and times to train of 6.84–6.86 minutes.

The last throughput record gives the average training throughput; expect approximately **7.5 samples/second** on MI355X:

```bash
grep '"key": "throughput"' "$LOGDIR"/*_1.log | tail -1
```

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight; the AMD submission scored **6.98 minutes**. Then run the RCP checker, after copying each trial log to the `result_<N>.txt` name it reads:

```bash
i=0; for f in "$LOGDIR"/*_[0-9]*.log; do cp "$f" "$LOGDIR/result_$((i++)).txt"; done
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

The launcher does not run the compliance checker, so run it on the host for each trial log. Every trial should print `INFO - SUCCESS`:

```bash
for f in "$LOGDIR"/*_[0-9]*.log; do
    python3 -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 "$f" 2>&1 | grep -E "SUCCESS|FAILED"
done
```

## Benchmark 3: DLRMv4

### Background - DLRMv4

DLRMv4 is the latest generation of the Deep Learning Recommendation Model benchmark in MLPerf Training. It trains an HSTU ranking model on the Yambda-5b music-recommendation dataset and is one of the computationally heaviest benchmarks in the suite due to its large embedding tables and high memory requirements. **Disk space required:** ~227 GB for the prepared dataset, plus an ~18 GB time-window index built before the first run.

> **Note:** DLRMv4 in MLPerf Training v6.1 uses the **Yambda** dataset — a change from the Criteo Terabyte dataset used by prior DLRM benchmarks. The prepared dataset is published by MLCommons at [`dlrmv4_dataset.uri`](https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri).

### Step 1: Pull the Docker Image and Copy the Code - DLRMv4

```bash
mkdir -p ~/mlperf && cd ~/mlperf

docker pull rocm/amd-mlperf:dlrmv4_training_6.1

container_id=$(docker create rocm/amd-mlperf:dlrmv4_training_6.1) && \
docker cp $container_id:/workspace/code ./dlrmv4_code && \
docker rm $container_id
```

To build the image instead, see the **Build the Images Instead of Pulling** section at the end of this guide.

### Step 2: Prepare Dataset - DLRMv4

```bash
mkdir -p /data/mlperf_dlrmv4
cd /data/mlperf_dlrmv4

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri
```

After the download is complete, the data directory should contain the following folders:

- Training cache: `processed_5b/hstu_cache_L4086/`
- Item popularity: `processed_5b/item_popularity.npy`
- Item mappings: `shared_metadata/album_item_mapping.parquet` and `shared_metadata/artist_item_mapping.parquet`

Verify the checksums and build the time-window index once before the first run. The first command runs from the `dlrmv4_code` folder that you copied in Step 1:

```bash
cd ~/mlperf/dlrmv4_code

DLRM_DATA_PATH=/data/mlperf_dlrmv4/data bash scripts/verify_dataset.sh

docker run --rm -v /data/mlperf_dlrmv4/data:/data \
    rocm/amd-mlperf:dlrmv4_training_6.1 \
    python3 -m generative_recommenders.dlrm_v4.datasets.offline_ts_indices \
    --cache-dir /data/processed_5b/hstu_cache_L4086 \
    --history-length 4086 --min-history 4086 --window-seconds 86400
```

### Step 3: Set Up Environment - DLRMv4

Enter the `dlrmv4_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd ~/mlperf/dlrmv4_code

mkdir -p /data/mlperf_dlrmv4/results

export DATADIR=/data/mlperf_dlrmv4/data
export LOGDIR=/data/mlperf_dlrmv4/results
export CONT=rocm/amd-mlperf:dlrmv4_training_6.1

sudo chmod -R 777 $LOGDIR
```

Source the MI355X configuration:

```bash
source config_MI355X_1x8x1.sh
```

The `source` command loads the settings into your current shell. Run Step 4 in this same shell, or repeat this step in a new terminal.

### Step 4: Run Training - DLRMv4

Run one trial first to confirm that everything works. It takes about 25 minutes, including about 4 minutes of startup. The console can stay quiet while training. The trial is finished when the output shows `ENDING TIMING RUN` followed by `RESULT,DLRMV4,,<seconds>,AMD,<start time>`, the total run time in seconds including startup.

```bash
export NEXP=1
bash run_with_docker.sh
```

For a compliant 10-run submission (about 4 hours):

```bash
export NEXP=10
bash run_with_docker.sh
```

To run the same single-node configuration through Slurm instead, edit `job.sh` to set `--partition`, `CONT`, `DATADIR`, and `LOGDIR`, then submit it. `job.sh` is set for 1 node and `DGXSYSTEM=MI355X_1x8x1`, and runs 10 trials by default (`NEXP=10`). `CONT` must be an image every node can pull. Run the following Slurm command to submit the job:

```bash
sbatch job.sh
```

### Step 5: Check Quality and Validate Results - DLRMv4

Quality is the AUC of the model's `listen_plus` prediction on a held-out time window (higher is better), evaluated every 2,293,760 training samples after the first 57,344,000. A trial succeeds at the first evaluation at or above **0.75**.

Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/*_1.log
```

The trial converged if the last `eval_accuracy` value is 0.75 or higher and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. Convergence varies more between trials than in the other benchmarks: across the 10 submitted trials, the target was met after 4 to 15 evaluations (64,225,280–89,456,640 samples), with times to train of 16.5–25.5 minutes.

The last throughput record gives the average training throughput; expect approximately **62,400 samples/second** on MI355X:

```bash
grep overall_throughput "$LOGDIR"/*_1.log | tail -1
```

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight; the AMD submission scored **18.95 minutes**. Then run the RCP checker, after copying each trial log to the `result_<N>.txt` name it reads:

```bash
i=0; for f in "$LOGDIR"/*_[0-9]*.log; do cp "$f" "$LOGDIR/result_$((i++)).txt"; done
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

Finally, confirm that every trial's section of `$LOGDIR/compliance_*.out` ends with `INFO - SUCCESS`.

## Benchmark 4: Flux.1

### Background - Flux.1

Flux.1 is the text-to-image diffusion model benchmark, new to AMD's MLPerf Training submissions in the v6.1 round. It measures how quickly the 12-billion-parameter Flux.1 Schnell model can be trained to reach the target validation loss (≤ 0.586) on the CC12M dataset, using 32 GPUs (4 nodes × 8 MI355X GPUs). The model has 19 double-stream (MMDiT) layers and 38 single-stream layers, with a hidden size of 3,072 and 24 attention heads.

This is one benchmark with two training recipes, which differ in the precision of the forward GEMMs of the transformer's linear layers. Both recipes run all backward GEMMs in MXFP4:

- **a6w4 + fp4 recipe:** the forward GEMMs use the a6w4 format. It runs from the published image `rocm/amd-mlperf:mlperf_flux1_6.1`.
- **fp6 + fp4 recipe:** runs the forward GEMMs in MXFP6 on Primus's Megatron-LM backend, while attention, normalization and communication run in BF16. It has no published image, so Step 1 builds it.

Pick one recipe and follow its steps; they are independent, and each has its own code folder, data, and results. The a6w4 + fp4 recipe needs about 2.5 TB on shared storage, plus about 2.5 TB of local disk per node. The fp6 + fp4 recipe needs about 2.4 TB on shared storage.

### Recipe A: a6w4 + fp4

#### Step 1: Pull the Docker Image and Copy the Code - Flux.1 (a6w4 + fp4)

```bash
mkdir -p /shared/mlperf_flux1 && cd /shared/mlperf_flux1

docker pull rocm/amd-mlperf:mlperf_flux1_6.1

container_id=$(docker create rocm/amd-mlperf:mlperf_flux1_6.1) && \
docker cp $container_id:/workspace/code ./flux1_code && \
docker rm $container_id
```

The job runs its scripts from `flux1_code` on every node, so `/shared/mlperf_flux1` must be shared storage that every node can reach; replace it with your own shared path in every Flux.1 command. Every node also needs the image. Pull it on each node, for example with `srun --nodes=4 --ntasks-per-node=1 docker pull rocm/amd-mlperf:mlperf_flux1_6.1` (add your usual `--partition` option). To build the image instead, see the **Build the Images Instead of Pulling** section at the end of this guide.

#### Step 2: Prepare Dataset - Flux.1 (a6w4 + fp4)

Download the preprocessed data to shared storage, running the script from the `flux1_code` folder that you copied in Step 1:

```bash
cd /shared/mlperf_flux1/flux1_code

PREPROCESSED_PATH=/shared/mlperf_flux1/data ./scripts/download_ref_data.sh
```

The data directory must contain the three downloaded directories, which training reads as-is:

```text
/shared/mlperf_flux1/data/
├── cc12m_preprocessed
├── coco_preprocessed
└── empty_encodings
```

The script `job.sh` copies the data to `/data/mlperf_flux1` on each node (override with `FLUX_LOCAL_DATADIR`), so `DATADIR` must be set to a different path from `/data/mlperf_flux1`.

#### Step 3: Set Up Environment - Flux.1 (a6w4 + fp4)

Enter the `flux1_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd /shared/mlperf_flux1/flux1_code

mkdir -p /shared/mlperf_flux1/results

export DATADIR=/shared/mlperf_flux1/data
export LOGDIR=/shared/mlperf_flux1/results
export CONT=rocm/amd-mlperf:mlperf_flux1_6.1

sudo chmod -R 777 $LOGDIR
```

`LOGDIR` must also be on shared storage. `sudo chmod -R 777 $LOGDIR` lets the container write its logs there.

Source the 4-node (32-GPU) MI355X configuration:

```bash
source config_MI355X_4x8x32.sh
```

The `source` command loads the settings, including `NNODES`, into your current shell. Run Step 4 in this same shell, or repeat this step in a new terminal.

#### Step 4: Run Training - Flux.1 (a6w4 + fp4)

Edit the `#SBATCH` header in `job.sh` to set your partition (`sinfo` lists the partitions you can use; replace `<your_partition>` with one of them), and uncomment `--nodelist` if you need specific nodes:

```bash
cd /shared/mlperf_flux1/flux1_code
sed -i 's/--partition=YOUR_PARTITION/--partition=<your_partition>/' job.sh
```

Submit from the `flux1_code` folder, in the same shell where you sourced the configuration, so that the job inherits `NNODES` and the other settings. Run one trial first to confirm that everything works:

```bash
export NEXP=1
sbatch --nodes=$NNODES job.sh
```

`sbatch` prints `Submitted batch job <jobid>` and returns at once; the job keeps running if you log out. Check its state with `squeue -u $USER`.

For a compliant 10-trial submission:

```bash
export NEXP=10
sbatch --nodes=$NNODES job.sh
```

The job runs in three stages:

1. **Stage the dataset.** Each node copies `cc12m_preprocessed`, `coco_preprocessed`, and `empty_encodings` from `DATADIR` to its local disk at `/data/mlperf_flux1` (override with `FLUX_LOCAL_DATADIR`), then writes a marker that records the source. Files already present with the same size are skipped, so only the first job copies the full ~2.5 TB. Training refuses to start on a node without a complete copy, so `FLUX_STAGE_LOCAL=0`, which skips this stage, only works when every node is already staged.
2. **Train.** Each node starts one container, runs `runtime_tunables.sh` to tune the host, checks that an RDMA device is usable, and runs the `NEXP` trials back to back in that container.
3. **Record.** After each trial, the first node writes that trial's MLPerf records to `$LOGDIR/<DATESTAMP>_<N>.log`, numbered from 0, and runs the compliance checker.

Slurm writes the job's console output to `flux1_primus_<jobid>.out` and `flux1_primus_<jobid>.err` in the submission directory. The script `job.sh` requests a 60-hour time limit, which covers staging plus 10 trials of about 54 minutes each.

#### Step 5: Check Quality and Validate Results - Flux.1 (a6w4 + fp4)

Quality is the diffusion validation loss on 29,696 COCO-2014 samples (lower is better), evaluated every 262,144 training samples. A trial succeeds at the first evaluation at or below **0.586**.

Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/*_0.log
```

The trial converged if the last `eval_accuracy` value is 0.586 or lower and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. In AMD's 4-node runs, converged trials met the target after 33 to 35 evaluations (8,650,752–9,175,040 samples), taking 47 to 50 minutes.

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight. Then run the RCP checker, after copying each trial log to the `result_<N>.txt` name it reads:

```bash
i=0; for f in "$LOGDIR"/*_[0-9]*.log; do cp "$f" "$LOGDIR/result_$((i++)).txt"; done
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

Finally, confirm that every trial's section of `$LOGDIR/compliance_*.out` ends with `INFO - SUCCESS`.

### Recipe B: fp6 + fp4

The network settings are written for the AMD Pensando™ Pollara 400 AI NICs (RoCE v2, one NIC per GPU) that AMD's runs used; Step 3 shows where to change them for other RDMA NICs.

#### Step 1: Get the Code and Build the Docker Image - Flux.1 (fp6 + fp4)

This recipe has no published image, so build it from the submission code in the MLCommons results repository. Copy the implementation folder to shared storage:

```bash
mkdir -p /shared/mlperf_flux1_fp6 && cd /shared/mlperf_flux1_fp6

git clone https://github.com/mlcommons/training_results_v6.1.git
cp -r training_results_v6.1/AMD/benchmarks/flux1/implementations/4xMI355X_EPYC_9575F_primus_megatron ./flux1_fp6_code
```

The job runs its scripts from `flux1_fp6_code` on every node, so `/shared/mlperf_flux1_fp6` must be on shared storage that every node can reach; replace it with your own shared path in every command below.

Every node needs the image. Build it on all four nodes at once (add your usual `--partition` option):

```bash
srun --nodes=4 --ntasks-per-node=1 \
    bash -c 'cd /shared/mlperf_flux1_fp6/flux1_fp6_code && docker build -t mlperf_flux1_fp6:6.1 .'
```

The `Dockerfile` starts from the public `rocm/primus:v26.7` image and installs AITER, Primus-Turbo and Primus at pinned commits. The build does not need a GPU. It also compiles the AITER kernels that training uses into the image, so nothing is compiled during a trial. The image records the commits it was built from in `/opt/mlperf/stack.txt`.

#### Step 2: Prepare Dataset - Flux.1 (fp6 + fp4)

Run the following commands to download the dataset, convert it to the format training reads, and verify it:

```bash
mkdir -p /shared/mlperf_flux1_fp6/data

docker run --rm -v /shared/mlperf_flux1_fp6/data:/data mlperf_flux1_fp6:6.1 \
    bash /workspace/code/scripts/prepare_data.sh

docker run --rm -v /shared/mlperf_flux1_fp6/data:/data mlperf_flux1_fp6:6.1 \
    bash /workspace/code/scripts/verify_data.sh /data/mlperf_flux1
```

The script `prepare_data.sh` downloads the MLCommons-preprocessed CC12M training set, the COCO-2014 evaluation subset and the empty encodings, and converts them to [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) WebDataset shards in one streaming pass. It downloads four files at a time, converts each into one shard and deletes it, and keeps at most six downloaded files (about 1.2 GB) on disk at once. So the peak disk use is the final dataset size plus about 1.2 GB, not a raw download plus a converted copy. If the script stops partway, run it again: it skips the shards that already exist. After it completes, the data directory should contain the following:

- Training: `mlperf_flux1/train/` (4,762 shards, 1,099,776 samples)
- Validation: `mlperf_flux1/val/` (129 shards, 29,696 samples)
- Empty encodings: `mlperf_flux1/empty_encodings/`
- Energon metadata: `mlperf_flux1/.nv-meta/`

The script `verify_data.sh` checks these counts, that every validation sample has its evaluation timestep, and the checksum of every shard against the manifest shipped with the code. It ends with `[verify_data] PASS`. If a shard fails its checksum, for example one cut short when an earlier run of `prepare_data.sh` stopped, delete that shard and run `prepare_data.sh` again.

Every node reads the data from shared storage at a rate of about 1.5 GB/s per node. If your shared storage is slower, copy `mlperf_flux1` to the same local path on every node and set `DATADIR` there in Step 3.

#### Step 3: Set Up Environment - Flux.1 (fp6 + fp4)

The script `job.sh` sets the image and the paths itself, so set them in that file rather than in your shell. Enter the `flux1_fp6_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd /shared/mlperf_flux1_fp6/flux1_fp6_code

mkdir -p /shared/mlperf_flux1_fp6/results

export LOGDIR=/shared/mlperf_flux1_fp6/results

sed -i \
    -e 's|^export CONT=.*|export CONT=mlperf_flux1_fp6:6.1|' \
    -e 's|^export DATADIR=.*|export DATADIR=/shared/mlperf_flux1_fp6/data|' \
    -e "s|^export LOGDIR=.*|export LOGDIR=$LOGDIR|" \
    -e 's|^export CACHEDIR=.*|export CACHEDIR=/tmp/mlperf_flux1_fp6_cache|' \
    -e 's|--partition=YOUR_PARTITION|--partition=<your_partition>|' \
    -e '/^export CONT=/a export CONT_PULL=0' \
    job.sh

sudo chmod -R 777 $LOGDIR
```

The environment variable `LOGDIR` must be on shared storage, and `CACHEDIR` holds the compile caches on each node's local disk. `CONT_PULL=0` makes the launcher use the image you built instead of pulling it. `sinfo` lists the partitions you can use. `job.sh` uses the 4-node (32-GPU) configuration `config_MI355X_04x08x32_mxfp6heal_hb128.sh`, so there is no configuration to source.

The network settings are in `config_common.sh`. If your nodes do not have eight Pollara AI NICs named `ionic_0` through `ionic_7`, set `NCCL_IB_HCA` to your RDMA devices (`ls /sys/class/infiniband` lists them) and `NCCL_IB_GID_INDEX` to their RoCE v2 GID index. If your site provides an RCCL topology file for the nodes, set `RCCL_TOPO_FILE` in `job.sh` to its path on the host.

#### Step 4: Run Training - Flux.1 (fp6 + fp4)

Run one trial first to confirm that everything works. It takes about 52 minutes, including about 4 minutes of startup before the timed run begins:

```bash
cd /shared/mlperf_flux1_fp6/flux1_fp6_code
NEXP=1 sbatch job.sh
```

For a compliant 10-trial submission (about 9 hours), submit the job with:

```bash
sbatch job.sh
```

`sbatch` prints `Submitted batch job <jobid>` and returns at once; check the job's state with `squeue -u $USER`. Slurm writes the job's console output to `flux1_<jobid>.out` and `flux1_<jobid>.err` in the submission folder, and each node writes its console output for trial `<N>` to `$LOGDIR/MI355X_04x08x32_mxfp6heal_hb128_run<N>_node<rank>.log`. A trial is finished when that output shows `ENDING TIMING RUN` followed by a line of the form `RESULT,FLUX1,<seed>,<seconds>,AMD,<start time>`, the total run time including startup.

```{note}
Before the first trial, the launcher runs `runtime_tunables.sh`, which logs the host settings and warns where they differ from the ones AMD's runs used; it does not change them. Before every trial, it drops the page cache, with passwordless `sudo` or through the privileged container, and does not start the trial if it cannot.
```

#### Step 5: Check Quality and Validate Results - Flux.1 (fp6 + fp4)

Quality is the diffusion validation loss on 29,696 COCO-2014 samples (lower is better), evaluated every 262,144 training samples. A trial succeeds at the first evaluation at or below **0.586**.

This submission writes each trial's MLPerf log to `$LOGDIR/result_<N>.txt`, numbered from 0. Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/result_0.txt
```

The trial converged if the last `eval_accuracy` value is 0.586 or lower and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. In AMD's 10 submitted trials, the target was met after 32 to 36 evaluations (8,388,608–9,437,184 samples), taking 45.0 to 50.3 minutes.

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight; the AMD submission scored **47.11 minutes**. The logs already have the names the RCP checker reads, so run it on `$LOGDIR` directly:

```bash
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

Finally, confirm that every trial's compliance file ends with `INFO - SUCCESS`:

```bash
tail -n 1 "$LOGDIR"/raw/*_node0/compliance_*.out
```

## Benchmark 5: GPT-OSS-20B

### Background - GPT-OSS-20B

GPT-OSS-20B is a 20-billion-parameter Mixture-of-Experts language model pretraining benchmark new to AMD's MLPerf Training portfolio in v6.1, trained on the C4 dataset to a target log perplexity of ≤ 3.34. **Disk space required:** ~85 GB dataset + ~32 GB model directory.

### Step 1: Pull the Docker Image and Copy the Code - GPT-OSS-20B

```bash
mkdir -p ~/mlperf && cd ~/mlperf

docker pull rocm/amd-mlperf:gpt_oss_20b_training_6.1

container_id=$(docker create rocm/amd-mlperf:gpt_oss_20b_training_6.1) && \
docker cp $container_id:/workspace/code ./gpt_oss_20b_code && \
docker rm $container_id
```

To build the image instead, see the **Build the Images Instead of Pulling** section at the end of this guide.

### Step 2: Prepare Dataset - GPT-OSS-20B

Run the following commands to download the dataset and tokenizer:

```bash
mkdir -p /data/gpt_oss_20b
cd /data/gpt_oss_20b

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After the download is complete, the data directory should contain the following:

- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

### Step 3: Set Up Environment - GPT-OSS-20B

Enter the `gpt_oss_20b_code` folder that you copied in Step 1, then set the paths and the image:

```bash
cd ~/mlperf/gpt_oss_20b_code

mkdir -p /data/gpt_oss_20b/results

export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=rocm/amd-mlperf:gpt_oss_20b_training_6.1

sudo chmod -R 777 $LOGDIR
```

Source the MI355X configuration:

```bash
source config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh
```

The `source` command loads the settings into your current shell. Run Step 4 in this same shell, or repeat this step in a new terminal.

### Step 4: Run Training - GPT-OSS-20B

Run one trial first to confirm that everything works. It takes about 61 minutes, including about 7 minutes of startup. The console can stay quiet while training. The trial is finished when the output shows `ENDING TIMING RUN` followed by `RESULT,GPT_OSS_20B,,<seconds>,AMD,<start time>`, the total run time in seconds including startup.

```bash
export NEXP=1
bash run_with_docker.sh
```

For a compliant 10-run submission (about 10 hours):

```bash
export NEXP=10
bash run_with_docker.sh
```

Logs default to MLPerf records and run summaries. Set `MLPERF_VERBOSE_LOGS=1` for debugging.

### Step 5: Check Quality and Validate Results - GPT-OSS-20B

Quality is validation log perplexity on held-out C4 text (lower is better), evaluated every 12,288 training sequences. A trial succeeds at the first evaluation at or below **3.34**.

Print the trial's start, evaluation, and stop records:

```bash
grep -E '"key": "(run_start|eval_accuracy|run_stop)"' "$LOGDIR"/*_1.log
```

The trial converged if the last `eval_accuracy` value is 3.34 or lower and `run_stop` shows `"status": "success"`. Its time to train is the `run_stop` `time_ms` minus the `run_start` `time_ms`, in milliseconds. Across the 10 submitted trials, the target was met at the 18th or 19th evaluation (221,184–233,472 samples), with times to train of 54.0–57.0 minutes.

The last throughput record gives the average training throughput; expect approximately **68 samples/second** on MI355X:

```bash
grep overall_throughput "$LOGDIR"/*_1.log | tail -1
```

To score a 10-trial set, drop the fastest and slowest trials and average the time to train of the other eight; the AMD submission scored **56.18 minutes**. Then run the RCP checker, after copying each trial log to the `result_<N>.txt` name it reads:

```bash
i=0; for f in "$LOGDIR"/*_[0-9]*.log; do cp "$f" "$LOGDIR/result_$((i++)).txt"; done
python3 -m mlperf_logging.rcp_checker "$LOGDIR" --rcp_version 6.1.0
```

Finally, confirm that every trial's section of `$LOGDIR/compliance_*.out` ends with `INFO - SUCCESS`.

## Troubleshooting

Here is a list of common issues and their resolutions:

- **`permission denied` when running `docker`:** Your user is not in the `docker` group. Run `sudo usermod -aG docker $USER`, then log out and back in.
- **`sudo: a password is required` in the console output:** The launch scripts cannot type a password. Configure passwordless `sudo` for your user, then start the run again.
- **`cpupower: command not found`:** Install it with `sudo apt-get install linux-tools-common linux-tools-$(uname -r)`.
- **`docker pull` fails with `unauthorized` or `pull access denied`:** Check that the image tag is spelled exactly as in the **Docker Images** table, and run `docker login` if your account needs it.
- **A download stops partway:** Run the same download command again from the same folder. It resumes partial files and verifies checksums when it finishes.
- **`No space left on device`:** Compare the sizes in **Disk Space, Time, and Where Things Live** with `df -h` on the Docker root folder and on `/data`. Free space, then start again.
- **Llama2-70B LoRA download fails with 401 or 403:** The HuggingFace token is wrong, or the account has not accepted the license for `meta-llama/Llama-2-70b-hf`.
- **The run ends within a minute or two:** Another job may be using the GPUs. Check `amd-smi monitor` and `docker ps`, and read the console output for the first error.
- **The console is quiet for minutes:** This is normal. Startup takes several minutes, and training prints only when it evaluates the model. Check that the container is still running with `docker ps`.
- **Starting over, or changing a script or configuration:** The launcher runs the copy in the code folder (`~/mlperf/<folder>`, `/shared/mlperf_flux1/flux1_code` for the Flux.1 a6w4 + fp4 recipe, or `/shared/mlperf_flux1_fp6/flux1_fp6_code` for the Flux.1 fp6 + fp4 recipe), not the one inside the image, so edit that copy. To start from a clean copy, delete the folder with `sudo rm -rf` (the container writes files as root), then repeat Step 1. Repeating Step 1 into a folder that already exists nests a second copy inside it and keeps the old files.
- **Stopping a run:** Press Ctrl+C, then find the container with `docker ps` and remove it with `docker rm -f <container name>`. For a Flux.1 job, run `scancel <jobid>`.
- **Flux.1 stops at startup with an RDMA message:** The container could not open an RDMA device. Check `ls /dev/infiniband` on every node and that the nodes are connected to the RDMA network.

## Expected Results

The time-to-train values for the MLPerf Training v6.1 submissions by AMD are included below for reference. You should achieve similar results by following the procedures described above. Keep in mind that slight variations in hardware setups and conditions may lead to minor discrepancies from AMD’s results.

| Submission ID | Model | MI355X TTT (min) |
| --- | --- | ---: |
| 6.1-0062 | Llama3.1-8B Pretraining | 72.94 |
| 6.1-0062 | Llama2-70B LoRA Fine-Tuning | 6.98 |
| 6.1-0061 | DLRMv4 | 18.95 |
| 6.1-0062 | GPT-OSS-20B | 56.18 |
| 6.1-0064 | Flux.1, a6w4 + fp4 recipe (4 nodes, 32 GPUs) | 47.18 |
| 6.1-0001 | Flux.1, fp6 + fp4 recipe (4 nodes, 32 GPUs) | 47.11 |

## Summary

This guide covers the full reproduction workflow for AMD's MLPerf Training v6.1 submission across five benchmarks — Llama3.1-8B pretraining, Llama2-70B LoRA fine-tuning, DLRMv4, Flux.1 (two recipes, a6w4 + fp4 and fp6 + fp4), and GPT-OSS-20B — on AMD Instinct MI355X GPUs. All benchmarks follow the same three-phase pattern:

1. **Pull** the Docker image and copy the code out of it (the Flux.1 fp6 + fp4 recipe builds its image from the MLCommons code instead).
2. **Prepare** the dataset and source the GPU-specific configuration file.
3. **Run and check** each trial's quality and time to train from its MLPerf log, then score the 10-trial set and run the MLCommons RCP and compliance checkers.

This round marks a significant expansion of AMD's MLPerf Training portfolio — from three benchmarks in v6.0 to five in v6.1 — covering language model pretraining and fine-tuning, recommendation systems, and generative image modeling. Dive into the [MLPerf Training v6.1 blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.1/README.html) to understand the optimizations behind these submissions, then put them into practice on AMD Instinct™ GPUs with ROCm.

## Related Resources

- [Technical Dive into AMD's MLPerf Training v6.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.1/README.html)
- [Reproducing AMD MLPerf Training v6.0 Submission Results](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training6.0-repro/README.html)
- [Technical Dive into AMD's MLPerf Training v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html)
- [MLPerf Training v6.1 Official Results — MLCommons](https://mlcommons.org/benchmarks/training/)
- [ROCm Documentation](https://rocm.docs.amd.com)
- [AMD Instinct GPU Product Page](https://www.amd.com/en/products/accelerators/instinct.html)
- [Primus — AMD's Large-Scale Training Framework](https://github.com/AMD-AGI/Primus)
- [ROCm GitHub Repository](https://github.com/ROCm/ROCm)

## Appendix: Build the Docker Images Instead of Pulling

AMD's submission code is also published in the MLCommons results repository. Each implementation folder there contains its Dockerfile and every file that the Dockerfile copies, so you can build an image from it instead of pulling one:

```bash
git clone https://github.com/mlcommons/training_results_v6.1.git
cd training_results_v6.1/AMD
```

Run the build from that benchmark's implementation folder, and tag it with the image name from the **Docker Images** table so that `CONT` and the other commands in this guide work unchanged. If you build an image, skip the `docker pull` in that benchmark's Step 1 and copy the code from your image with the same `docker create` and `docker cp` commands, or run the benchmark from the implementation folder itself.

| Model | Implementation folder | Build command |
| --- | --- | --- |
| Llama3.1-8B | `benchmarks/llama31_8b/implementations/MI355X_EPYC_9575F_primus` | `docker build -t rocm/amd-mlperf:llama31_8b_training_6.1 -f Dockerfile .` |
| Llama2-70B LoRA | `benchmarks/llama2_70b_lora/implementations/MI355X_EPYC_9575F_primus` | `docker buildx build --network=host --load -t rocm/amd-mlperf:llama2_70b_training_6.1 .` |
| DLRMv4 | `benchmarks/dlrmv4/implementations/MI355X_EPYC_9575F_hstu` | `docker build --network=host -t rocm/amd-mlperf:dlrmv4_training_6.1 -f Dockerfile .` |
| Flux.1, a6w4 + fp4 recipe | `benchmarks/flux1/implementations/4xMI355X_EPYC_9575F_primus` | `DOCKER_BUILDKIT=1 docker build -t rocm/amd-mlperf:mlperf_flux1_6.1 -f Dockerfile .` |
| Flux.1, fp6 + fp4 recipe | `benchmarks/flux1/implementations/4xMI355X_EPYC_9575F_primus_megatron` | `docker build -t mlperf_flux1_fp6:6.1 .` (Step 1 already builds it) |
| GPT-OSS-20B | `benchmarks/gpt_oss_20b/implementations/MI355X_EPYC_9575F_primus` | `docker build -t rocm/amd-mlperf:gpt_oss_20b_training_6.1 .` |

The Dockerfiles pin every dependency to a fixed commit, and some of them are compiled from source:

- **Llama2-70B LoRA** compiles Transformer Engine and Primus-Turbo, which takes a while. Run `sudo chmod -R 777 .` in the implementation folder before building so that all scripts are writable.
- **DLRMv4** compiles FBGEMM-GPU, because there is no public gfx950 build. This takes 30–60 minutes; `--build-arg MAX_JOBS=<n>` sets the compile parallelism.
- **Flux.1** runs on every node in the Slurm job, so each node needs the image. Either build it on every node, or push it to a registry that all nodes can reach and set `CONT` to that name.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, EPYC, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See www.mlcommons.org for more information. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
