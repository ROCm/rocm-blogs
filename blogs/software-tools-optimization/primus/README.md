---
blogpost: true
blog_title: "Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs"
date: 22 Aug 2025
author: 'Yao Fu, Wen Xie, Peng Wen, Chen Xiaoming, Li Xiaobo, Liz Li, Vidushi Goyal, Anshul Gupta'
thumbnail: 'primus.jpg'
tags: AI/ML
category: Software tools & optimizations
target_audience: AI ENTHUSIAST AND DEVELOPERS
key_value_propositions: Primus simplifies LLM training with a unified, backend-agnostic framework—fast launches, reliable scaling, and easier debugging on AMD GPUs.
language: English
myst:
    html_meta:
        "author": "Yao Fu"
        "description lang=en": "Primus streamlines LLM training on AMD GPUs with unified configs, multi-backend support, preflight validation, and structured logging."
        "keywords": "PRIMUS, Training, LLM, AMD GPUs"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools, ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Adaptive & Embedded Computing"
        "amd_blog_authors": "Yao Fu"
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

# Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs

Training large language models (LLMs) at scale is inherently complex. Different frameworks expose inconsistent interfaces, multi-GPU and distributed setups require brittle scripting, and backend-specific quirks introduce overhead that slows down training iterations. Primus tackles these challenges with a streamlined, backend-agnostic training framework that helps developers launch, customize, and scale training jobs faster on AMD GPUs.

In this blog, we’ll explore how [Primus](https://github.com/AMD-AIG-AIMA/Primus) works, how to get started, and how it simplifies model training at scale—across ROCm compatible clusters and development environments.

## What is PRIMUS?

[Primus](https://github.com/AMD-AIG-AIMA/Primus)  is a unified and modular training framework that supports both Megatron-LM and TorchTitan backends, making it easy to configure, extend, and optimize large model training on AMD hardware. It is designed to unify the training interface across engines like Megatron-LM and TorchTitan. Primus provides a YAML-driven configuration system, modular backend support, built-in preflight validation, and structured logging utilities.

Key Highlights:

- Unified CLI with YAML-Based Configuration
- Modular Multi-Backend Design
- Built-in Preflight Validation for Reliable Launch
- Structured Logging for Debuggability

## Unified CLI with YAML-Based Configuration

Primus lets you train large models with a single command by abstracting backend-specific arguments into a structured, composable YAML format. This approach eliminates repetitiveness, enhances reproducibility, and simplifies collaboration across teams and environments. At its core, Primus adopts a declarative design: experiments are defined in YAML rather than managed through extensive CLI flags or ad-hoc scripts, enabling a cleaner and more maintainable training workflow.

Below is a YAML file that defines a Primus training experiment in a clean, reproducible contract for your training experiment.

```bash
work_group: AMD
user_name: root
exp_name: llama3.1_8b-pretrain
workspace: ./output

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml

    # model to run
    model: llama3.1_8B.yaml
    overrides:
      train_iters: 50
      micro_batch_size: 2
      global_batch_size: 128
    
      # parallel
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
    
      # data
      mock_date: true
      train_data_path: null
      valid_data_path: null
      test_data_path: null
```

Launching an experiment becomes a single, reproducible command:

```bash
export EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml
bash examples/run_pretrain.sh
```

## Megatron-LM Integration

Primus integrates seamlessly with Megatron-LM while hiding backend complexity from users.

- Structured configuration mapping automatically translates YAML fields into Megatron’s expected : argparse.Namespace.
- Automatic patching: adjusts tokenizer types, attention backends, and optimizer overrides without manual intervention.

This design lets you run directly on the upstream Megatron codebase with minimal modification, while benefiting from Primus’s streamlined configuration and launch workflow.

## Modular Multi-Backend Design

Primus provides out-of-the-box support for Megatron-LM, including advanced parallelism strategies such as tensor parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP). Its modular backend architecture makes it easy to extend to additional training engines—such as the planned TorchTitan, all accessible through the same unified interface. Each backend implements a common launcher, which encapsulates environment setup, argument translation, and execution logic. Instead of forcing one configuration to work across multiple engines, Primus cleanly isolates backend-specific behaviors.

This allows developers to:

- Switch backends with minimal changes
- Avoid manual environment reconfiguration
- Launch training consistently across machines and clusters

Our future with TorchTitan will involve integration to introduce FP8 precision and fused attention support on AMD GPUs—while preserving the same streamlined CLI and YAML-driven workflow.

## Built-in Preflight Validation for Reliable Launch

Primus features a robust preflight system that verifies environment variables, checks parallelism configuration, estimates model size and FLOPs--catching potential issues before training begins. For HPC workflows, optional Slurm job script generation is also supported, enabling smooth integration into distributed computing environments. Before launching training, Primus performs a comprehensive preflight check to help developers validate whether their cluster environment is ready for large-scale distributed training. Unlike traditional static config validation, Primus preflight actively probes the runtime environment and generates a structured diagnostic report.

This allows developers to have:

- **Cluster health check** verifies cluster connectivity and RDMA/RCCL communication availability
- **GPU diagnostics** checks GPU availability, topology, and memory bandwidth
- **Network performance benchmarking** measures effective inter-node bandwidth to identify potential bottlenecks

The results are automatically compiled into a readable PDF report, allowing users to review system readiness, share diagnostics with others, and quickly identify performance bottlenecks or misconfigured nodes—before any training job is launched. Primus is an ideal choice for robust, production-ready deployment across clusters and CI pipelines.

## Structured Logging for Debuggability

Debugging distributed training can be very difficult. Primus introduces structured logging that organizes output by experiment and rank, making it easier to isolate issues and trace behavior in large-scale runs. Distributed training often produces scattered and overwhelming logs across multiple ranks and nodes—making debugging difficult and time-consuming. Primus introduces a structured, hierarchical logging system that organizes logs by experiment, module, rank, and severity level to streamline analysis and issue tracking.

The directory layout follows a clean convention, as shown below:

```bash
{workspace}/
└── {work_group}/{user_name}/{exp_name}/logs/pre_trainer/
    ├── rank-0/
    │   ├── debug.log
    │   ├── error.log
    │   ├── info.log
    │   └── warning.log
    ├── rank-1/
    │   └── ...
    ...
    ├── rank-7/
```

This allows developers to have:

- **Log root follows workspace hierarchy**

The entire log output is stored under the path:

```bash
{workspace}/{work_group}/{user_name}/{exp_name}/logs/
```

Note: This ensures logs are cleanly separated across teams, users, and experiments.

- **Module-specific isolation**
  - Each module (e.g., pre_trainer) logs independently to avoid file conflicts and improve modularity.
- **Rank-separated logs**
  - Each rank writes its own set of logs (rank-0, rank-1, etc.), enabling fine-grained debugging in distributed runs.
- **Log-level segregation**
  - Each rank produces logs segmented by severity (debug.log, info.log, warning.log, error.log), making it easy to search or aggregate critical issues.

This structure makes it easy to:

- Trace errors down to specific ranks or modules
- Monitor and compare behaviors across workers
- Organize logs for archiving, auditing, or sharing across teams

By aligning logs with user-defined metadata (work_group, user_name, exp_name), Primus ensures distributed training is not only reproducible—but also observable and diagnosable at scale.

## How to Get Started with Primus?

You can try Primus today using two workflows: Docker Interactive Mode or Quick Start Mode.

### Setup Docker

Use the official ROCm Megatron Docker image for a consistent runtime:

```bash
docker pull docker.io/rocm/megatron-lm-training-private:20250813
# IMP NOTE change docker image to docker.io/rocm/megatron-lm:v25.7_py310 before release
```

### Interactive Mode

You will enter the container and execute training inside.

```bash
# Launch the container
docker run -it --device /dev/dri --device /dev/kfd --device /dev/infiniband --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 128G --name primus_training_env rocm/megatron-lm-training-private:20250813
# IMP NOTE change docker image to docker.io/rocm/megatron-lm:v25.7_py310 before release

# Install dependencies
pip install -r requirements.txt

# Export your HF_TOKEN in the workspace
export HF_TOKEN=<your_hftoken>

# Launch training(e.g llama3.1_8B)
EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml bash ./examples/run_pretrain.sh
```

________________________________________

### Quick Start Mode

You do not need to enter the Docker container. Clone the repository locally and just set the config and run.

#### 1. Clone the Repository

Clone the repository and install dependencies:

```bash
# Clone with submodules
git clone -b v0.1.0-rc1 --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus

# Or initialize submodules if already cloned
git submodule update --init --recursive

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Launch Training

Use the run_local_pretrain.sh script to start training.

```bash
# Export DOCKER IMAGE
export DOCKER_IMAGE=rocm/megatron-lm-training-private:20250813
# IMP NOTE change docker image to docker.io/rocm/megatron-lm:v25.7_py310 before release

# Export your HF_TOKEN in the workspace
export HF_TOKEN=<your_hftoken>

# Example for megatron llama3.1_8B
export EXP=examples/megatron/configs/llama3.1_8B-pretrain.yaml 
bash ./examples/run_local_pretrain.sh
```

## Summary

In this blog we demonstrate how to use Primus to simplify large-model training on AMD ROCm. [Primus](https://github.com/AMD-AIG-AIMA/Primus) streamlines large-model training on AMD ROCm—making experiments easier to configure, safer to launch, and faster to debug. From fine-tuning to massive pretraining on AMD Instinct™ MI300X GPUs, Primus gives you a consistent, reliable workflow. Whether you're fine-tuning a 7B model or scaling pretraining on AMD Instinct™ MI300X GPUs, Primus helps you iterate faster and with greater confidence.

## Additional Resources

[Primus GitHub Repository](https://github.com/AMD-AIG-AIMA/Primus)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
