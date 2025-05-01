---
blogpost: true
blog_title: "Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration"
date: 24 Apr 2025
author: 'Yusheng Su, Vicky Tsang, Yao Liu, Zicheng Liu'
thumbnail: 'verl.jpg'
tags: Fine-Tuning, Reinforcement Learning, AI/ML
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: verl, an efficient RLHF framework designed for scalable and high-performance training.
language: English
myst:
    html_meta:
        "author": "Yusheng Su, Vicky Tsang, Yao Liu, Zicheng Liu"
        "description lang=en": "Deploy verl on AMD GPUs for fast, scalable RLHF training with ROCm optimization, Docker scripts, and impressive throughput-convergence results"
        "keywords": "verl, Training, AMD GPUs."
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_hardware_deployment": "Instinct GPUs"
        "amd_deployment_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_deployment_tools": "ROCm Software"
        "amd_applications": "AI Training"
        "amd_blog_category_topic": "Enterprise & Data Center Trends"
        "amd_blog_authors": Yusheng Su, Vicky Tsang, Yao Liu, Zicheng Liu"
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
# Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration

In this blog post, we provide an overview of Volcano Engine Reinforcement Learning for LLMs (verl) and discuss its benefits in large-scale reinforcement learning from human feedback (RLHF). We also detail the modifications made to the codebase to optimize verl’s performance on AMD Instinct GPUs. Next, we walk through the process of building the Docker image using a Dockerfile on the user side, along with training scripts tailored for both single-node and multi-node setups. Lastly, we present verl’s performance results, focusing on throughput and convergence accuracy achieved on AMD Instinct™ MI300X GPUs.
Follow this guide to get started with verl on AMD Instinct GPUs and accelerate your RLHF training with ROCm-optimized performance.

## Key Takeaways

1. verl framework and its advantages
2. AMD ROCm software support and docker image for the [v0.3.0.post0](https://github.com/volcengine/verl/releases/tag/v0.3.0.post0) version verl
3. Single-node and multi-node training scripts for verl
4. Throughput and convergence accuracy
  
## Introducing verl: A Scalable RLHF Training Framework

To develop intelligent large-scale foundation models, post-training is just as important as pre-training. Among post-training paradigms, reinforcement learning from human feedback (RLHF) has emerged as a critical technique, though its full potential has not been thoroughly explored until now. Since the release of ChatGPT at the end of 2023, the effectiveness of RLHF in enhancing large-scale pre-trained language models (LLMs) has become increasingly evident. More recently, the
release of several O1/R1-series models trained with RLHF has once again highlighted its power, particularly in improving reasoning capabilities. Despite the growing recognition of its importance, there remains a lack of mature, open-source RLHF frameworks ---particularly those capable of
supporting training with both high efficiency and scalability.

Building on this foundation, an open-source community ([Volcengine](https://github.com/volcengine/verl)) Introduce verl, an efficient RLHF framework designed for scalable and high-performance training. It integrates state-of-the-art, high-throughput LLM training engines--- FSDP and Megatron---with advanced inference engines --- vLLM and SGLang. It also uses Ray as part of a hybrid orchestration engine to schedule and coordinate training and inference tasks in parallel, enabling optimized resource utilization and potential overlap between these phases. This dynamic resource allocation strategy significantly improves overall system efficiency.

## Enabling verl on AMD Instinct™ with ROCm and Docker

To ensure verl functions effectively on AMD Instinct GPUs, we contributed some key ROCm enhancements. First, we updated the verl codebase to ensure compatibility with the ROCm kernel, enabling stable and efficient execution on AMD GPUs (see PR: [\[Hardware\] Support AMD (ROCMm Kernel) #360](https://github.com/volcengine/verl/pull/360)). Additionally, we addressed issues in the third-party library **Ray** on AMD Instinct, allowing it to handle dynamic resource allocation reliably---an essential step toward achieving high overall training efficiency (see PR: [Replace AMD device env var with HIP_VISIBLE_DEVICES #51104](https://github.com/ray-project/ray/pull/51104)).Additionally, to better support AMD Instinct GPUs, we provide [Dockerfiles](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.rocm) to simplify the setup of the verl training environment. The aforementioned supports have been integrated into the official verl upstream repository and are included in the [v0.3.0.post0](https://github.com/volcengine/verl/releases/tag/v0.3.0.post0) version.

## Running verl on AMD GPUs: Single-Node and Multi-Node Setups

Let's get started with running verl on AMD Instinct:

### Single-node Training

The first step is to launch the verl docker image. You can build by your own with:

```shell
FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

# Set working directory
WORKDIR $PWD/app

# Set environment variables
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# Install vllm
RUN pip uninstall -y vllm && \
    rm -rf vllm && \
    git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && \
    rm -rf vllm

# Copy the entire project directory
COPY . .

# Install dependencies
RUN pip install "tensordict<0.6" --no-deps && \
    pip install accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    numpy \
    pandas \
    peft \
    "pyarrow>=15.0.0" \
    pylatexenc \
    "ray[data,train,tune,serve]" \
    torchdata \
    transformers \
    wandb \
    orjson \
    pybind11 && \
    pip install -e . --no-deps
```

After building the Docker image (verl-rocm) using the provided Dockerfile with the command `docker build -t verl-rocm .`, you can proceed to the next steps.

#### Docker Launch

```shell
docker run --rm -it \
	--device /dev/dri \
	--device /dev/kfd \
	-p 8265:8265 \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
	-v $HOME/.ssh:/root/.ssh \
	-v $HOME:$HOME \
	--shm-size 128G \
	-w $PWD \
	verl-rocm 
```

#### Data Prepare

```shell
python3 examples/data_preprocess/gsm8k.py --local_dir ../data/gsm8k
```
  
#### Model Loading

```shell  
python3 -c "import transformers;transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
python3 -c "import transformers;transformers.pipeline('text-generation', model='deepseek-ai/deepseek-llm-7b-chat')"
```

#### Configuration Setting

 ```shell
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
train_files="../data/gsm8k/train.parquet"
test_files="../data/gsm8k/test.parquet"
```
  
You can choose any model supported by verl and assign it to the `$MODEL_PATH` variable. In our case, we use`Qwen/Qwen2-7B-Instruct` and `deepseek-ai/deepseek-llm-7b-chat`. As for the dataset, you are free to use any dataset of your choice---just ensure it is converted into the required format. In this example, we use `gsm8k`, as verl already provides preprocessing code to format it appropriately.

#### Environment Variable Setup

```shell
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
GPUS_PER_NODE=8
```

You must assign `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.
This is the most crucial part---and the only difference compared to running on CUDA version torch.  

Then, you can run the RLHF algorithms provided by verl, including PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, and others. In this example, we use PPO and GRPO to illustrate the workflow.

PPO:

```shell
MODEL_PATH="Qwen/Qwen2-7B-Instruct" # You can use: deepseek-ai/deepseek-llm-7b-chat
TP_VALUE=2 #If deepseek, set TP_VALUE=4
INFERENCE_BATCH_SIZE=32 #If deepseek, set INFERENCE_BATCH_SIZE=32
GPU_MEMORY_UTILIZATION=0.4 #If deepseek, set GPU_MEMORY_UTILIZATION=0.4

python3 -m verl.trainer.main_ppo  \
	data.train_files=$train_files  \
	data.val_files=$test_files  \
	data.train_batch_size=1024 \
	data.max_prompt_length=1024 \
	data.max_response_length=512 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
	actor_rollout_ref.rollout.name=vllm  \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	critic.optim.lr=1e-5 \
	critic.model.use_remove_padding=True \
	critic.model.path=$MODEL_PATH \
	critic.model.enable_gradient_checkpointing=True \
	critic.ppo_micro_batch_size_per_gpu=32 \
	critic.model.fsdp_config.param_offload=False \
	critic.model.fsdp_config.optimizer_offload=False \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console','wandb'] \
	trainer.project_name='ppo_qwen_llm' \
	trainer.experiment_name='ppo_trainer/run_qwen2-7b.sh_default' \
	trainer.n_gpus_per_node=8 \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=10 \
	trainer.total_epochs=50

```

GRPO:

```shell
MODEL_PATH="Qwen/Qwen2-7B-Instruct" #  You can use: deepseek-ai/deepseek-llm-7b-chat
TP_VALUE=2 #If deepseek, set TP_VALUE=2
INFERENCE_BATCH_SIZE=40 #If deepseek, set INFERENCE_BATCH_SIZE=110
GPU_MEMORY_UTILIZATION=0.6 #If deepseek, set GPU_MEMORY_UTILIZATION=0.6

python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.train_batch_size=1024 \
	data.max_prompt_length=512 \
	data.max_response_length=1024 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=256 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console','wandb'] \
	trainer.project_name='grpo_qwen_llm' \
	trainer.experiment_name='grpo_trainer/run_qwen2-7b.sh_default' \
	trainer.n_gpus_per_node=8 \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=10 \
	trainer.total_epochs=50 
```

### Multi-node Training
  
Once you have successfully run single-node training and wish to scale up to multi-node training on AMD clusters using the Slurm cluster management system, you can refer to the provided [multi-node training script.](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst#slurm_scriptsh)

Our guide walks you through each step of the process---from Slurm configuration, environment setup, and Docker/Podman container setup, to Ray cluster initialization, data preprocessing, model setup, and finally, training launch---to help you smoothly launch multi-node training.

#### Multi-node training

```shell
sbatch slurm_script.sh
```

For more detailed guidance, comprehensive single-node and multi-node training tutorials are available in the official verl [GitHub introduction](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) and [documentation](https://verl.readthedocs.io/en/latest/index.html).

## Performance Benchmarks: Throughput and Convergence on MI300X vs. H100

In this section, we run our modified verl (v0.3.0.post0) and present the throughput and convergence accuracy results on H100 and MI300, respectively, using the same hyperparameter settings.  

PPO:
|Platform   |Model   |TP_VALUE |INFERENCE_BATCH_SIZE |GPU_MEMORY_UTILIZATION |Throughput (Token/GPU/Sec)|Convergence(Acc.)
|--|--|--|--|--|--|--|
|H100| Qwen2-7B-Instruct| 2| 32| 0.4| 907.24| 87.6|
|MI300^\[1\]^| Qwen2-7B-Instruct| 2| 32| 0.4| 921.24| 87.5|
|--|--|--|--|--|--|--|
|H100| deepseek-llm-7b-chat| 4| 32| 0.4| 623.52| 70.3|
|MI300^\[1\]^| deepseek-llm-7b-chat| 4| 32| 0.4| 767.47| 70.3|

GRPO:
|Platform| Model| TP_VALUE| INFERENCE_BATCH_SIZE| GPU_MEMORY_UTILIZATION| Throughput(Token/GPU/Sec)| Convergence (Acc.)|
|--|--|--|--|--|--|--|
|H100| Qwen2-7B-Instruct| 2| 40| 0.6| 1544.30|90.0|
|MI300^\[1\]^| Qwen2-7B-Instruct| 2 |40 |0.6| 1747.94| 89.7|
|--|--|--|--|--|--|--|
|H100| deepseek-llm-7b-chat| 2| 110| 0.4| 1624.42| 71.2|
|MI300^\[1\]^| deepseek-llm-7b-chat| 2| 110| 0.4| 1899.04| 70.9|

Note: For throughput (measured in Tokens/GPU/Second), we run 350 training steps for each setting. To ensure stability, we exclude any step where step % 10 == 0, then compute the average throughput over the remaining steps.
Note: For convergence (measured by accuracy), we run 350 training steps and report the highest accuracy achieved during this period, as the training typically converges within these steps.

## Summary

As RLHF becomes a cornerstone in fine-tuning LLMs, verl offers a scalable, open-source solution optimized for AMD Instinct GPUs with full ROCm support. This blog walks you through setting up verl using Docker, configuring training scripts for single- and multi-node clusters, and evaluating performance (throughput and accuracy) across leading models on both MI300 and H100 platforms. We hope this work enables more AMD Instinct users to adopt verl for RLHF training, ultimately contributing to the development of more powerful foundation models.

## Contributors

Core contributors: Yusheng Su, Vicky Tsang, Yao Liu, Zicheng Liu

Contributors: Xiaodong Yu, Gowtham Ramesh, Jiang Liu, Zhenyu Gu, Vish Vadlamani, Emad Barsoum

Thanks to the IT AI Sys team for providing cluster support and system configuration: Kobawala, Arhat

## SYSTEM CONFIGURATION

AMD Instinct ™ MI300X platform\
System Model: ORACLE SERVER X10-2c\
CPU: Intel(R) Xeon(R) Platinum 8480+
NUMA: 2 NUMA node per socket. NUMA auto-balancing disabled / Memory:
2048 GiB, (32x 64GiB Samsung M321R8GA0BB0-CQKZJ DDR6 4400 MT/s)\
Disk: 30.72 TB (8x INTEL SSDPF2KX038T1S 3.84 TB)\
GPU: 8x AMD Instinct MI300X 192GB HBM3 750W\
Host OS: Ubuntu 22.04.4 LTS\
System Bios Vendor: American Megatrends International, LLC.\
Host GPU Driver: (amdgpu version): ROCm 6.3.1

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
