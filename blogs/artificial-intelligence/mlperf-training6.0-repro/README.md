---

blogpost: true
blog_title: "Reproducing AMD MLPerf Training v6.0 Submission Result"
date: 16 Jun 2026
author: 'Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sarthak Arora, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Karan Verma, Eliot Li'
thumbnail: 'mlperf-training6.0-repro-thumbnail.png'
tags: AI/ML, Fine-Tuning, GenAI, Optimization, Performance, MLPerf, MLPerf Training
category: Applications & models
target_audience: AI developers, AI practitioners
key_value_propositions: Provide instructions to readers on how to reproduce AMD's MLPerf Training v6.0 submission result.
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sarthak Arora, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Karan Verma, Eliot Li"
        "description lang=en": "Learn how to reproduce AMD's MLPerf Training v6.0 submission result."
        "keywords": "MLPerf, Training, Optimization, Llama 2, Llama 3.1, LoRA, Flux.1"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
    "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Ravi Dwivedula, Sarthak Arora, Sathish Sanjeevi, Su Ann Chong, Sudhir Kylasa, Karan Verma, Eliot Li"

---

# Reproducing AMD MLPerf Training v6.0 Submission Result

This blog provides a step-by-step guide for reproducing AMD's MLPerf Training 6.0 submission results. AMD submitted results on three benchmarks this round:

- **Llama 2 70B LoRA fine-tuning** — parameter-efficient fine-tuning of Llama 2 70B using GovReport data
- **Llama 3.1 8B pretraining** — full pretraining from random weights on a subset of the C4 dataset
- **Flux.1-schnell** — text-to-image training benchmark on MI325X (8-node)

Results were submitted on AMD Instinct MI325X, MI350X and MI355X GPUs. For the first time, AMD's [Primus](https://github.com/AMD-AGI/Primus) training framework was used in all LLM submissions (Llama 2 70B LoRA fine-tuning and Llama 3.1 8B pretraining). This guide covers the reproduction procedure for each benchmark, including environment setup, dataset preparation, training configuration, execution, and result validation.

Readers looking for context on the performance results and competitive analysis should refer to the companion blog post: [Technical Dive into AMD's MLPerf Training v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html).

## Prerequisites

To follow along with this blog, you will need:

- AMD Instinct MI325X, MI350X, or MI355X platform
- ROCm 7.2.2 or later
- Any Linux distribution supported by the selected ROCm version
- Docker
- Slurm (for the multinode Flux.1-schnell training submission)
- At least 6 TB of disk space (for Flux.1-schnell dataset preparation)

See the [ROCm Quick start installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) for information on how to install ROCm.

In the following sections, you will first find instructions to set up the environment and preprocess the dataset for training. Then, you will go through the steps to run the model training job and collect benchmark results. Finally, you will be presented with the benchmark results submitted to the MLPerf Training v6.0 round by AMD.

## Primus

The LLM benchmarks submitted by AMD in this round are powered by **[Primus](https://github.com/AMD-AGI/Primus)**, AMD's unified, modular training framework for large-scale foundation model training on AMD Instinct GPUs. Primus abstracts over multiple training backends — Megatron-LM and TorchTitan — through a single CLI and configuration system, enabling pretraining, fine-tuning, and RLHF workflows without managing each backend separately.

The Primus ecosystem used in these submissions consists of three components:

- **Primus (Primus-LM):** The unified training framework. Provides the `primus-cli`, configuration templates, workflow orchestration, and ROCm-native integrations with Megatron-LM and TorchTitan.
- **Primus-Turbo:** A high-performance operator library delivering optimized FlashAttention, GEMM, and communication kernels for Instinct GPUs via AITER, Composable Kernel, and ROCm Triton — activated as a non-intrusive drop-in over the training framework.
- **Primus-SaFE:** A cluster management and fault tolerance layer for enterprise-grade training resilience at scale.

All three are bundled in the Docker images for all benchmarks. No additional installation or configuration is required — Primus is activated automatically by the benchmark configuration scripts.

Users interested in using Primus for their own workloads can get started at [github.com/AMD-AGI/Primus](https://github.com/AMD-AGI/Primus). Pre-built Docker images are available from the `rocm/primus` registry on Docker Hub.

## Llama 2 70B LoRA Fine-tuning

The Llama 2 70B LoRA fine-tuning benchmark measures how quickly a model can be fine-tuned to reach a target ROUGE-2 score on the GovReport summarization dataset, using Low-Rank Adaptation (LoRA) with FP8 precision.

- **Disk space required:** ~270 GB for model download and conversion  
- **Dataset:** [GovReport](https://gov-report-data.github.io/) (pre-tokenized, packed to sequence length 8,192)  
- **Model:** Llama 2 70B with fused QKV  
- **Convergence target:** ROUGE-2 score of 0.925
- **Docker image:** `rocm/amd-mlperf:llama2_70b_training_6.0`

### Set up Docker Image - Llama 2 70B

Pull the docker image from the registry:

```bash
docker pull rocm/amd-mlperf:llama2_70b_training_6.0
```

### Prepare Dataset - Llama 2 70B

GovReport is a dataset for long document summarization that consists of reports written by U.S. government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has a length of 8,192.

The model used in this submission is the Llama 2 70B with fused QKV. You will need 270 GB of disk space to download and convert the model.

#### Download and Preprocess Data & Model - Llama 2 70B

To download the model from Huggingface, you will need to sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT as well as obtain a Huggingface Token (`HF_TOKEN`).

Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/data/mlperf_llama2` as the host download directory:

```bash
container_id=$(docker create rocm/amd-mlperf:llama2_70b_training_6.0) && \
docker cp $container_id:/workspace/code ./code && \
docker rm $container_id

docker run -it -v /data/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_6.0
```

Start the script for downloading and preprocessing data from within the container (`HF_TOKEN` is needed to download the model weights):

```bash
export HF_TOKEN=<your-huggingface-token>
bash ./scripts/prepare_data_and_model.sh
```

Exit the container by running:

```bash
exit
```

#### Verify Data

After completion of the above step, your data directory, on the host, should look like this:

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

### Run Training - Llama 2 70B

#### Setup Environment - Llama 2 70B

Set the directory for the data, model and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`. In this example we use `/data/mlperf_llama2/results` as the results directory, so please make sure to create this directory

```bash
export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_6.0
```

To set appropriate configuration and system-specific hyperparameters for each AMD Instinct platform, use the appropriate configuration file listed in the table below:

| Instinct Platform | configuration file​    |
| ----------------- | ---------------------- |
| MI350X            | config_MI350X_1x8x1.sh |
| MI355X            | config_MI355X_1x8x1.sh |

For example, to set the configuration for the MI355X platform on a single node, use the config file `config_MI355X_1x8x1.sh`:

```bash
source config_MI355X_1x8x1.sh
```

##### Launch a Single Training Run - Llama 2 70B

To perform a single run for Llama 2 70B LoRA fine-tuning, set `NEXP` to `1`, and then run the `run_with_docker.sh` script:

```bash
export NEXP=1
bash run_with_docker.sh
```

##### Launch 10 Training Runs - Llama 2 70B

To prepare for a 10 run result, simply set the environment variable `NEXP` to `10` before running the `run_with_docker.sh` script:

```bash
export NEXP=10
bash run_with_docker.sh
```

After the benchmarking is completed, the logs will be available under the directory `$LOGDIR`, where you can also find the benchmark results.

```{note}
Note: To optimize the machine's performance, the training script will also execute runtime_tunables.sh script before any training run.
```

## Llama 3.1 8B Pretraining

AMD led the development of the Llama 3.1 8B pretraining benchmark in MLPerf Training. Unlike Llama 2 70B LoRA, this benchmark starts from **random weights** — no checkpoint conversion is required. It trains on a subset of the C4 (Colossal Cleaned Common Crawl) dataset and targets a validation loss perplexity of **3.3**.

- **Disk space required:** ~80 GB dataset + ~30 GB tokenizer  
- **Dataset:** C4 (last 256 of 1,024 training shards, randomly shuffled)  
- **Convergence target:** Validation loss perplexity ≤ 3.3  
- **Docker image:** `rocm/amd-mlperf:llama31_8b_training_6.0`

### Setup Docker Image - Llama 3.1 8B

Pull the docker image from the registry:

```bash
docker pull rocm/amd-mlperf:llama31_8b_training_6.0

container_id=$(docker create rocm/amd-mlperf:llama31_8b_training_6.0) && \
docker cp $container_id:/workspace/code ./code && \
docker rm $container_id
```

### Prepare Dataset and Model - Llama 3.1 8B

The dataset used for this benchmark according to MLCommons is the [c4/en/3.0.1 dataset](https://huggingface.co/datasets/allenai/c4) from AllenAI.

#### Download Preprocessed Data

The pre-tokenized dataset and the tokenizer are available for download. You can navigate to your desired download directory and run the following commands to download the dataset and tokenizer. In this example, the folder `/data/mlperf_llama31_8b` is used as the download directory on the host.

```bash
# desired download directory
mkdir -p /data/mlperf_llama31_8b
cd /data/mlperf_llama31_8b

# download training data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

mv llama3_1_8b_tokenizer model
```

After the download is completed, you should see files with the following naming conventions under the data directory, ending with both `.idx` and `.bin` extensions:

- Training partitions: `c4-train.en_6_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

The size of the data directory is ~80 GB, and ~30 GB for the model directory.

### Run Pre-training Benchmark - Llama 3.1 8B

#### Setup Environment - Llama 3.1 8B

Set the directory for the data, model and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`, In this example the folder `/data/mlperf_llama31_8b/results` is used as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_llama31_8b/data
export MODEL=/data/mlperf_llama31_8b/model/
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama31_8b_training_6.0
```

#### Set Configurations - Llama 3.1 8B

Similar to the Llama 2 70B LoRA fine-tuning case, there is a configuration file for each Instinct platform submission for the Llama 3.1 8B benchmark:

| Instinct platform | Configuration file ​      |
| ----------------- | ------------------------- |
| MI350X            | config_MI350X_1x8x1.sh    |
| MI355X            | config_MI355X_1x8x1.sh    |

For example, to configure the environment for a MI355X platform, use:

```bash
source config_MI355X_1x8x1.sh
```

#### Launch a Single Training Run - Llama 3.1 8B

If you want to perform a single run of the pretraining benchmark, use:

```bash
export NEXP=1
bash run_with_docker.sh
```

#### Launch 10 Training Runs - Llama 3.1 8B

If you would like to prepare for a 10-run submission, use:

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, the logs will be available under the directory `$LOGDIR`.

```{note}
To optimize the machine's performance, the training script will also execute the `runtime_tunables.sh` script before any training run.
```

## Flux.1-schnell Text-to-Image Training

This benchmark trains the Flux.1-schnell model using preprocessed CC12M training data and COCO validation data. See the [MLCommons benchmark specification](https://github.com/mlcommons/training/blob/master/text_to_image/README.md) for quality rules and dataset details. AMD's v6.0 submission for this benchmark targets the MI325X platform in an 8-node × 8-GPU configuration (`config_MI325X_08x08x32.sh`) coordinated via Slurm.

- At least 6TB disk space is required.
- GPUs are not required for dataset preparation.

### Set up Docker Image - Flux.1

Pull the docker image from the registry and copy the `/workspace/code` folder (benchmark code and launch scripts) from the container to the host:

```bash
docker pull rocm/amd-mlperf:flux1_training_6.0
```

Make sure the image is accessible on every node of the run environment.

### Prepare Dataset - Flux.1

The dataset download and preprocessing scripts are included in the container. GPUs are not required for dataset preparation. In this example, `/data/mlperf_flux1/data` is used as the host download directory. The dataset path must be on a shared filesystem reachable from every node in the Slurm job.

```bash
mkdir -p /data/mlperf_flux1/data

docker run -it --rm --network=host --ipc=host \
    --volume /data/mlperf_flux1/data:/dataset \
    rocm/amd-mlperf:flux1_training_6.0
```

From inside the container:

```bash
pip install datasets

cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

mkdir energon
python /workspace/code/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed --output_path /dataset/energon/train --num_workers 8
python /workspace/code/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed --output_path /dataset/energon/val --num_workers 8

cd energon
energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./
# Select y for duplicate keys
# Select y for creating interactively
# Select class 11

cp -r ../empty_encodings .
```

Optionally, reclaim disk space by deleting `/dataset/cc12m_preprocessed` and `/dataset/coco_preprocessed` after conversion.

Exit the container:

```bash
exit
```

After completion, your data directory on the host should look like this:

```text
/data/mlperf_flux1/data
└── energon
    ├── train
    ├── val
    └── empty_encodings
```

### Run Training (Slurm, 8-node MI325X)

All training commands below should be run from the `./code` directory on the host. Edit `job.sh` in `./code` to set Slurm options, partition, and the variables below. Set these inside `job.sh` (do not rely on exporting them in your shell before `sbatch`).

| Variable     | Example value for MI325X submission      |
| ------------ | -------------------------------------- |
| `CONT`       | `rocm/amd-mlperf:flux1_training_6.0`   |
| `DATADIR`    | `/data/mlperf_flux1/data`              |
| `LOGDIR`     | `/data/mlperf_flux1/results`           |
| `DGXSYSTEM`  | `MI325X_08x08x32`                      |
| `NEXP`       | `10` (use `1` for a smoke run)         |

Example `job.sh` excerpt:

```bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --job-name=flux1_mlperf
#SBATCH --output=flux1_%j.out
#SBATCH --error=flux1_%j.err
#SBATCH --partition=<YOUR_PARTITION>

export CONT=rocm/amd-mlperf:flux1_training_6.0
export DATADIR=/data/mlperf_flux1/data
export LOGDIR=/data/mlperf_flux1/results
export DGXSYSTEM=MI325X_08x08x32
export NEXP=10

export ROOT_SEED=${ROOT_SEED:-$RANDOM}

srun bash run_with_docker_slurm.sh
```

Ensure `$LOGDIR` exists and is writable on the shared filesystem (`mkdir -p "$LOGDIR" && sudo chmod -R 777 "$LOGDIR"`).

Submit the training job from the head node:

```bash
cd ./code
sbatch job.sh
```

The script `run_with_docker_slurm.sh` launches one container per node, sources `config_MI325X_08x08x32.sh` (selected by `DGXSYSTEM`), sets up RCCL/torch distributed across the cluster, and runs the training loop. Per-run logs land under `$LOGDIR` on the shared filesystem.

```{note}
To optimize each node's performance, the training script also executes `runtime_tunables.sh` on every node before each training run.
```

### Evaluation of Training Result

The evaluation of the training run is summarized in the following table:

| Parameter               | Value                            |
| ----------------------- | -------------------------------- |
| Quality target          | 0.586                            |
| Evaluation frequency    | Every 262,144 training samples   |
| Evaluation thoroughness | 29,696 samples                   |

### Run Training

Use the same `run_start` / `run_stop` timestamps for time-to-train. Convergence is validation loss at or below **0.586** (see Evaluation above). For a submission score, average the 8 of 10 runs after dropping the fastest and slowest.

## Interpreting Results

Below is the log from one of the training runs on the AMD MI355X platform for the Llama 3.1 8B pretraining benchmark:

```text
...

:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "submission_benchmark", "value": "llama31_8b", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "submission_division", "value": "closed", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "submission_status", "value": "", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "submission_org", "value": "AMD", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "submission_platform", "value": "MI355X", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "tensor_parallelism", "value": 1, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "pipeline_parallelism", "value": 1, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "context_parallelism", "value": 1, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "expert_parallelism", "value": 1, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "micro_batch_size", "value": 2, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "config_filename", "value": "config_MI355X_1x8x1.sh", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "POINT_IN_TIME", "key": "lowest_numerical_precision_linear", "value": "mxfp4", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 66}}
:::MLLOG {"namespace": "", "time_ms": 1778651800896, "event_type": "INTERVAL_END", "key": "init_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 68}}
:::MLLOG {"namespace": "", "time_ms": 1778651870949, "event_type": "INTERVAL_START", "key": "run_start", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 193}}

...
:::MLLOG {"namespace": "", "time_ms": 1778657039449, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 139, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657039450, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 141, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657049081, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 3.2914416790008545, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 156, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657049082, "event_type": "INTERVAL_END", "key": "eval_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 170, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657049088, "event_type": "INTERVAL_END", "key": "epoch_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 220, "samples_count": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657049089, "event_type": "INTERVAL_END", "key": "run_stop", "value": null, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 224, "samples_count": 184320, "status": "success"}}
:::MLLOG {"namespace": "", "time_ms": 1778657057151, "event_type": "POINT_IN_TIME", "key": "run_duration", "value": "5178.13s -> 86.3 minutes", "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 87, "samples": 184320}}
:::MLLOG {"namespace": "", "time_ms": 1778657057152, "event_type": "POINT_IN_TIME", "key": "overall_throughput", "value": 35.6, "metadata": {"file": "/opt/venv/lib/python3.12/site-packages/primus_mllog/mlperf_pre_training.py", "lineno": 92, "samples": 184320}}

```

From the logs, the `run_start` and `run_stop` events happened at timestamp `1778651870949` and `1778657049089` in milliseconds respectively. The difference between these two timestamps is the time taken for the training to finish, and is equal to `86.3 min`. In most cases, this should serve as a reliable estimate of the MLPerf Training score, but obtaining an MLPerf compliant score is more involved, as described below.

Time-to-train is determined by two components: throughput and number of processed samples. Throughput is a measure of hardware performance and from the log above, it is about 30.7 samples per second for the Llama 3.1 8B pretraining benchmark running on MI355X. You can find throughput printed in your output file by searching for the last occurrence of `throughput` in the file. If your throughput is significantly lower, this indicates a misconfiguration or hardware issue that needs to be addressed.

Number of processed samples, given by `samples_count` in the output file, is determined by how fast the training converges to the defined accuracy target. In MLPerf Training, convergence needs to match the [Reference Convergence Checkpoints](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#13-reference-convergence-points-rcps) (RCPs). For hyperparameters used in the AMD submission for an individual run of the Llama 3.1 8B pretraining benchmark on MI355X, a value of `184,320` as in the log shown above is most common.

To obtain an MLPerf compliant Time-to-Train score, you will need 10 consecutive runs. Calculate timings for each run using the `run_start` and `run_stop` timestamps as described above.  Disregard the slowest and fastest runs and average the remaining 8 runtimes. Next, check convergence of runs by running [RCP Checker Script](https://github.com/mlcommons/logging/tree/44b4810e65e8c0a7d9e4e207c60e51d9458a3fb8/mlperf_logging/rcp_checker) on the directory containing the set of 10 runs. The output should look like this:

```text
INFO - ------------------------------
INFO -  Running RCP Checker, pass: pruned_rcps
INFO - ------------------------------
INFO -  RCP Record: {'Benchmark': 'llama31_8b', 'BS': 32, 'Hyperparams': {'opt_base_learning_rate': 0.0008, 'opt_learning_rate_warmup_samples': 4096, 'gradient_accumulation_steps': 1}, 'Epochs to converge': [172032, 172032, 172032, 172032, 172032, 172032, 172032, 172032, 172032, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 184320, 196608], 'RCP Mean': np.float64(178858.66666666666), 'RCP Stdev': np.float64(7165.0736883859045), 'Max Speedup': np.float64(1.0299965952657717), 'Min Epochs': np.float64(173649.76494948068)}
INFO -  Submission mean epochs: 185856.0000
INFO - RCP found, RCP test PASSED
** Logging output also at rcp_checker.log
```

Make sure there is no error in the output. If RCP checker prints a normalization factor, the average of 8 runs should be multiplied by the factor to arrive at the final score.

## Expected Results

### Llama 2 70B LoRA Fine-tuning Results

For MI355X, the score should be about 8.5 mins. The AMD MLPerf Training v6.0 submission score is `8.27` mins (Submission ID 6.0-0036).

For MI350X, the score should be about 10 mins. The AMD MLPerf Training v6.0 submission score is `10.25` mins  (Submission ID 6.0-0035).

### Llama 3.1 8B Pretraining Results

For MI355X, the score should be about 87 mins. The AMD MLPerf Training v6.0 submission score is `86.84` mins (Submission ID 6.0-0036).

For MI350X, the score should be about 110 mins. The AMD MLPerf Training v6.0 submission score is `109.76` mins  (Submission ID 6.0-0035).

### Flux.1-schnell Text-to-Image Training Results

For MI325X (8 nodes × 8 GPUs), the score should be about 92.5 mins. The AMD MLPerf Training v6.0 submission score is `92.36` mins (Submission ID 6.0-0037).

## Summary

This guide covered the full reproduction workflow for AMD's MLPerf Training v6.0 submission across three benchmarks — Llama 2 70B LoRA fine-tuning, Llama 3.1 8B pretraining, and Flux.1-schnell text-to-image training — on AMD Instinct MI325X, MI350X and MI355X GPUs.  You can view the official MLPerf Training v6.0 results on [MLCommons](https://mlcommons.org/benchmarks/training/). Keep in mind that slight variations in hardware setups and conditions may lead to minor discrepancies from AMD’s results. Primus, AMD's unified training framework for large-scale foundation model training, powers all benchmarks and is bundled in the respective Docker images with no additional setup required. For more insights into the techniques used to optimize training workloads in these submissions, check out this [blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v6.0/README.html). Ready to go further? Build on this foundation to scale your AI workloads with AMD Instinct™ GPUs and ROCm.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
