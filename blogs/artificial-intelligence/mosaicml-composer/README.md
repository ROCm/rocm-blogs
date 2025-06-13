---
blogpost: true
blog_title: 'Distributed fine-tuning of MPT-30B using Composer on AMD GPUs'
date: 28 Jan 2025
author: Vara Lakshmi Bayanagari
thumbnail: 'thumbnail-390.jpg'
tags: LLM, PyTorch, AI/ML, Fine-Tuning
category: Applications & models
language: English
target_audience: 'machine learning engineers'
key_value_propositions: 'This blog is an introduction to a new distributed training framework, Composer, by MosaicML. It shows how seamlessly Composer can be launched on AMD GPUs using SLURM on multinode setup and using docker for single node experiments'
myst:
    html_meta:
        "author": "Vara Lakshmi Bayanagari"
        "description lang=en": "This blog uses Composer, a distributed framework, on AMD GPUs to fine-tune MPT-30B in single node as well as multinode"
        "keywords": "mosaicml, LLM, mpt30b, AMD GPU, MI250, MI300, Natural Language Processing"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
---

# Distributed fine-tuning of MPT-30B using Composer on AMD GPUs

[Composer](https://github.com/mosaicml/composer), developed by MosaicML, is an open-source deep learning training library built on top of PyTorch, designed to simplify and optimize distributed training workflows. It supports scalable training on multiple nodes and efficiently handles datasets of various sizes. Composer integrates advanced techniques such as PyTorch Fully Sharded Data Parallelism (FSDP), elastic sharded checkpointing, training callbacks, and speed-up algorithms to enhance training performance and flexibility. It closely resembles PyTorch’s torchrun and has demonstrated [exceptional efficiency](https://www.databricks.com/blog/stable-diffusion-2) when scaling to hundreds of GPUs.

In this blog, we'll discuss how to fine-tune the MPT-30B model for an instruction tuning task using MosaicML's distributed Composer framework. MPT-30B is a 30 billion parameter decoder-style Transformer model that is considered an [open source alternative](https://github.com/vbayanag/llm-foundry/) to GPT models by the ML community. It performs on par with other prominent LLMs such as Llama-30B and Falcon-40B and even outperforms GPT3 (for the 175B model) on many evaluation tasks.

All the scripts required for finetuning MPT-30B model are packaged into a docker image. We shall use this image to launch these scripts in two different settings - distributed environment as well as a single node environment. Through these two experiments, we can observe how seamlessly Composer scales from a single node to two nodes.

## Requirements

AMD's [Infinity Hub](https://account.amd.com/en/member/infinity_hub_containers.html) serves as a comprehensive repository, offering an extensive array of Docker images for various large language models (LLMs). This platform significantly simplifies access to these advanced models, eliminating the complexities typically associated with software configuration and setup. Users can effortlessly deploy and train these models using our Docker containers that have been optimized for performance on AMD GPUs. Sign up on [Infinity Hub](https://account.amd.com/en/member/infinity_hub_containers.html) to gain swift access to these models.

For this blog, we’ll be utilizing the [MPT-30B](https://account.amd.com/content/dam/account/en/member/infinity_hub_containers/MPT-30B_Training.md) Docker image from AMD’s Infinity Hub. This image provides a robust platform for deploying and fine-tuning MPT-30B, ensuring a seamless and efficient workflow. This docker image requires [AMD Instinct Accelerator](https://www.amd.com/en/products/accelerators/instinct.html) and comes with ROCm software installed. Refer to [ROCm](https://rocm.docs.amd.com/en/docs-6.1.2/) documentation for more details on the GPU software stack.

This blog requires SLURM setup across different nodes in a cluster of GPUs. If you don't have access to multiple nodes, you can skip to ##single-node-training-without-SLURM section to train MPT-30B using Composer on a single node.

## Multinode Composer training using SLURM

Multinode training using SLURM requires two scripts: a SLURM script and a launch script. In our SLURM file `train.sbatch`, we define our hardware requirements for 2 nodes with 8 GPUs each as follows:

```bash
#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00                                  #specify time for the job
##SBATCH --nodelist=                                     #specify specific nodes, if you want those specific nodes
##SBATCH --partition=                                    #specify your partition if required

srun bash train.sh
```

The launch script `train.sh` file shown below defines environment variables, creates the workspace, and launches the Docker image. Update the `DOCKER_USER` and `DOCKER_PASS` fields below with your credentials to seamlessly pull the Docker image from Infinity Hub.

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=7501
#echo "MASTER_ADDR="$MASTER_ADDR
#echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
#8GPUS per node
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:=8}

export DOCKER_USER=<INSERT USERID>
export DOCKER_PASS=<INSERT PASSWORD>

#RCCL info
export NCCL_DEBUG="${NCCL_DEBUG:=WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:=WARN}"
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:=1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
export NCCL_CHECKS_DISABLE=1
export NCCL_ALGO=Ring

docker login packages.xilinx.com -u "$DOCKER_USER" --password "$DOCKER_PASS"
docker pull packages.xilinx.com/instinct/dev-benchmark-300x:mpt-30b_rocm-v6.1.3_llm-foundry-v0.7.0_train

docker  run  --rm   --ipc=host --cap-add=SYS_PTRACE --network=host --device=/dev/kfd --device=/dev/dri \
                --security-opt seccomp=unconfined --group-add video --privileged   \
                        -v /data/data:/data \
                        --env SLURM_NNODES=$SLURM_NNODES \
                        -e MASTER_ADDR=$MASTER_ADDR \
                        -e MASTER_PORT=$MASTER_PORT \
                        -e WORLD_SIZE=$WORLD_SIZE \
                        -e LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE \
                        -e NODE_RANK=$SLURM_NODEID \
                        -e LOCAL_RANK=$SLURM_LOCALID \
                        -e NCCL_DEBUG=$NCCL_DEBUG \
                        -e NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS \
                        -e PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER \
                        -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
                        -e NCCL_CHECKS_DISABLE=$NCCL_CHECKS_DISABLE \
                        -e NCCL_ALGO=$NCCL_ALGO \
                        packages.xilinx.com/instinct/dev-benchmark-300x:mpt-30b_rocm-v6.1.3_llm-foundry-v0.7.0_train \
                        /bin/bash -c "bash run_mpt30b.sh"
```

The Docker image contains the training script `run_mpt30b.sh`, as shown below. It launches training runs through Composer, which seamlessly facilitates multi node training.

```bash
#!/bin/bash

cd /app/llm-foundry/scripts/train
HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2 USE_ROCMLINEAR=1 composer train.py yamls/finetune/mpt-30b-instruct.yaml global_train_batch_size=1024 device_train_microbatch_size=8 max_seq_len=8192 precision=amp_fp16 max_duration=1ep eval_interval=1ep
```

In our training run, we fine-tune the MPT-30B model with the `mosaicml/instruct-v3` dataset hosted on Hugging Face for a batch size of 1024 and micro batch size of 8 for one epoch. Composer automatically adjusts the number of gradient accumulation steps based on the batch size and device batch size. Composer is device-count agnostic when you update the argument in the above command to `device_train_microbatch_size: auto`. This automatically sets the micro batch size without OOM errors. The YAML config `llm-foundry/scripts/train/yamls/finetune/mpt-30b-instruct.yaml` also uses FSDP optimizations to decrease the memory footprint, allowing for larger batch sizes and increased throughput. When employing FSDP and elastic checkpointing, keep in mind that [it might require about 12 times as much memory (in GB) as the number of model parameters, for example, 30B parameters requires ~360GB of total shared VRAM](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#faq-how-many-gpus-do-i-need-to-train-a-llm-).

Launch the training using the command `sbatch train.sbatch` from the login node of the cluster. The (truncated) training output on 16 GPUs (2 nodes) is as follows:

```text
.
.
.
2024-09-30 02:05:36,662: rank0[73][MainThread]: INFO: __main__: Starting training...
2024-09-30 02:05:36,662: rank0[73][MainThread]: INFO: composer.trainer.trainer: Using precision Precision.AMP_FP16
******************************
Config:
composer_commit_hash: None
composer_version: 0.21.3
enabled_algorithms/GradientClipping: true
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 8
num_nodes: 2
rank_zero_seed: 17
time/remaining_estimate_unit: hours
******************************
.
.
.
[epoch=1][batch=5/5]:
         Train time/batch: 4
         Train time/sample: 4096
         Train time/batch_in_epoch: 4
         Train time/sample_in_epoch: 4096
         Train time/token: 29254495
         Train time/token_in_epoch: 29254495
         Train trainer/device_train_microbatch_size: 8
         Train loss/train/total: 8.4213
         Train metrics/train/LanguageCrossEntropy: 8.4620
         Train metrics/train/LanguagePerplexity: 4731.5786
         Train metrics/train/TokenAccuracy: 0.0445
         Train throughput/batches_per_sec: 0.0027
         Train throughput/samples_per_sec: 2.7330
         Train throughput/device/batches_per_sec: 0.0002
         Train throughput/device/samples_per_sec: 0.1708
         Train throughput/tokens_per_sec: 20082.8418
         Train throughput/device/tokens_per_sec: 1255.1776
         Train time/train: 0.5249
         Train time/val: 0.0000
         Train time/total: 0.5249
         Train time/remaining_estimate: 0.0000
         Train lr-DecoupledAdamW/group0: 0.0000
```

You can customize the YAML file within the Docker container to suit your specific requirements — adding additional callbacks, employing a different dataset, utilizing memory monitors, or even fine-tuning an alternative LLM. This section offered a comprehensive guide to designing a SLURM-based multinode workload using MosaicML’s Composer to fine-tune the MPT-30B model. To leverage the Kubernetes Engine for a multinode workload on AMD GPUs, you can refer to [this](../sdxl-multinode-oke/README.md) blog.

## Single node training without SLURM

Single node training is a preferable choice for developers looking to get going with Composer on AMD GPUs without any additional hassle of framework setup. Composer scripts can simply be run on a single node using the docker image obtained from AMD's [Infinity Hub](https://account.amd.com/en/member/infinity_hub_containers.html) as shown below. As explained in the #requirements section, this docker image comes with all dependencies installed and requires only a single command launch. First, run the docker command below on compute node terminal.

```bash
docker run --cap-add=SYS_PTRACE --device=/dev/dri --device=/dev/kfd --group-add video --ipc=host \
  --network=host --privileged --rm --security-opt seccomp=unconfined -it -w /app \
  packages.xilinx.com/instinct/dev-benchmark-300x:mpt-30b_rocm-v6.1.3_llm-foundry-v0.7.0_train
```

Inside the docker container, you can launch the training script with the same hyperparameters as described in the previous section.

```bash run_mpt30b.sh```

On a single node of 8 MI300X GPUs, the output consists of the following, highlighting the number of devices and nodes.

```text
.
.
.
******************************
Config:
composer_commit_hash: None
composer_version: 0.21.3
enabled_algorithms/GradientClipping: true
node_name: unknown because NODENAME environment variable not set
num_gpus_per_node: 8
num_nodes: 1
rank_zero_seed: 17
time/remaining_estimate_unit: hours

******************************
.
.
.
[epoch=1][batch=3/5]:
         Train time/batch: 2
         Train time/sample: 2048
         Train time/batch_in_epoch: 2
         Train time/sample_in_epoch: 2048
         Train time/token: 14750088
         Train time/token_in_epoch: 14750088
         Train trainer/device_train_microbatch_size: 8
         Train loss/train/total: 8.3177
         Train metrics/train/LanguageCrossEntropy: 8.4361
         Train metrics/train/LanguagePerplexity: 4610.4463
         Train metrics/train/TokenAccuracy: 0.0441
         Train throughput/batches_per_sec: 0.0014
         Train throughput/samples_per_sec: 1.4307
         Train throughput/device/batches_per_sec: 0.0002
         Train throughput/device/samples_per_sec: 0.1788
         Train throughput/tokens_per_sec: 10347.9543
         Train throughput/device/tokens_per_sec: 1293.4943
         Train time/train: 0.5992
         Train time/val: 0.0000
         Train time/total: 0.5992
         Train time/remaining_estimate: 0.2630
         Train lr-DecoupledAdamW/group0: 0.0000
```

We can see that Composer scaled across two nodes as expected as seen in the output metrics such as `Train throughput/batches_per_sec: 0.0014` on single node vs `Train throughput/batches_per_sec: 0.0027` on two nodes.

## Summary

In this blog, we showed you how to fine-tune the MPT-30B model for an instruction tuning task using MosaicML's distributed Composer framework. More specifically, we discussed two approaches in which Composer can perform and scale - distributed framework using SLURM, and single node framework. To observe Composer's behavior on two nodes framework we used SLURM scripts for resource allocations across nodes. Where as for a hassle free Composer experiment without any additional framework setup we demonstrated that a single node section can work perfectly. This blog also highlighted the ease of use of Composer with the docker image from AMD's [Infinity Hub](https://account.amd.com/en/member/infinity_hub_containers.html) and a simple training script launch inside the container.
If you want to know more about AI development on AMD GPUs, visit the [AI developer hub](https://www.amd.com/rocm-ai-developer).

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
