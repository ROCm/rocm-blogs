---
blogpost: true
blog_title: 'Getting started with AMD ROCm containers: from base images to custom solutions'
thumbnail: './images/2025-01-09-containers.png'
description: 'This blog post provides an overview of the ROCm container ecosystem along with examples of extending these containers for custom use cases.'
date: 16 Jan 2025
author: Matt Elliott
tags: AI/ML
category: Software tools & optimizations
language: English
myst:
  html_meta:
    "author": "Matt Elliott"
    "description lang=en": "Getting started with AMD ROCm containers: from base images to custom solutions"
    "keywords": "AI/ML"
    "property=og:locale": "en_US"
---
# Getting started with AMD ROCm containers: from base images to custom solutions

Having worked in technology for over two decades, I've witnessed firsthand how containerization has transformed the way we develop and deploy applications. Containers package applications with their dependencies into standardized units, making software portable and consistent across different environments. When we combine this containerization power with AMD Instinct™ Accelerators, we get a powerful solution for quickly deploying AI and machine learning workloads. In this blog, the first in a series exploring ROCm containerization, I want to share my insights about [AMD ROCm™](https://rocm.docs.amd.com/en/latest/ "https://rocm.docs.amd.com/en/latest/") containers and show you how to build and customize your own GPU-accelerated workloads. You'll learn how to select appropriate base images, modify containers for your specific needs, and implement best practices for GPU-enabled containerization - all with hands-on examples you can try yourself.

## Why use containers for AI workloads?

Recently I encountered a scenario that many of you might recognize: a complex ML application that worked perfectly in development but failed in production. The root cause? Subtle differences in framework versions, system libraries, and Python dependencies across environments. These key advantages highlight just how essential containers have become in today’s development environments:

1. **Consistency**: Your development environment matches your production environment exactly
2. **Portability**: Applications run consistently across different systems and ROCm versions
3. **Isolation**: Multiple applications can use different ROCm versions on the same system

## The ROCm container ecosystem

AMD provides several types of official containers through their [Docker Hub repositories](https://hub.docker.com/u/rocm "https://hub.docker.com/u/rocm"). The list below highlights some of the most popular containers.

### Development containers

Each of these containers include ROCm tools and libraries.

- [rocm/dev-ubuntu-22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04 "https://hub.docker.com/r/rocm/dev-ubuntu-22.04"): Development image built on Ubuntu 22.04  
- [rocm/dev-ubuntu-24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04 "https://hub.docker.com/r/rocm/dev-ubuntu-24.04"): Development image built on Ubuntu 24.04  
- [rocm/dev-centos-7](https://hub.docker.com/r/rocm/dev-centos-7 "https://hub.docker.com/r/rocm/dev-centos-7"): Development image built on CentOS 7
  
### Framework-specific containers

- [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch "https://hub.docker.com/r/rocm/pytorch"): Pre-built PyTorch environment
- [rocm/pytorch-nightly](https://hub.docker.com/r/rocm/pytorch-nightly "https://hub.docker.com/r/rocm/pytorch-nightly"): Nightly builds for PyTorch
- [rocm/tensorflow](https://hub.docker.com/r/rocm/tensorflow "https://hub.docker.com/r/rocm/tensorflow"): Pre-built TensorFlow environment
- [rocm/tensorflow-build:](https://hub.docker.com/r/rocm/tensorflow-build "https://hub.docker.com/r/rocm/tensorflow-build") Nightly builds for Tensorflow
- [rocm/jax](https://hub.docker.com/r/rocm/jax "https://hub.docker.com/r/rocm/jax"): Pre-built JAX environment
- [rocm/jax-build](https://hub.docker.com/r/rocm/jax-build "https://hub.docker.com/r/rocm/jax-build"): Nightly builds for Jax
- [rocm/vllm](https://hub.docker.com/r/rocm/vllm "https://hub.docker.com/r/rocm/vllm"): ROCm optimized vLLM build for LLM inference and benchmarking
- [rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev "https://hub.docker.com/r/rocm/vllm-dev"): Development build of vLLM

### An important note about tags

Starting with ROCm 6.2.1, we made an important change to our tagging strategy:

- **ROCm 6.2.1 and later**: `rocm/pytorch:latest` points to the latest stable PyTorch release  
- **Pre-ROCm 6.2.1**: `rocm/pytorch:latest` pointed to the development version  

For example:

```bash
# Latest stable PyTorch environment
docker pull rocm/pytorch:latest

# Development version (if needed)
docker pull rocm/pytorch-nightly:latest
```

## Container requirements and setup

To grant container access to GPUs, you need specific Docker options. Here's the command structure with explanations:

```bash
docker run \
  --device=/dev/kfd \     # Main compute interface shared by all GPUs
  --device=/dev/dri \     # Direct Rendering Interface for each GPU
  --security-opt seccomp=unconfined \  # Enables memory mapping for HPC environments
  --group-add video \
  --ipc=host \
  -e HIP_VISIBLE_DEVICES=0,1 \  # Optional: restrict to specific GPUs
  <image>
```

Check out the [ROCm installation prerequisites](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html#configuring-permissions-for-gpu-access "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html#configuring-permissions-for-gpu-access") page for more information on configuring user and group permissions. If you want to limit a container to using a specific GPU, you have two options:

You can restrict GPU access in two ways:

1. Using the `HIP_VISIBLE_DEVICES` environment variable (recommended)
2. Explicitly mapping specific GPU devices (e.g., `/dev/dri/renderD128` for the first GPU)

## Practical examples

Let's explore two common container configurations that showcase two of the available ROCm containers. We'll start with a basic development container that's perfect for GPU-accelerated application development, then move to a specialized container for machine learning workloads.

### Basic development container

This Dockerfile creates what I like to call a "ROCm developer's workshop" - it's a minimal but complete environment for GPU development. It's designed for teams building GPU-accelerated applications, providing essential ROCm tools while remaining easy to customize.

```dockerfile
FROM rocm/dev-ubuntu-22.04:latest

# Install basic development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Add your project files
COPY . .
```

I’ll break down each section:

```dockerfile
FROM rocm/dev-ubuntu-22.04:latest
```

We're starting with AMD's official ROCm container image, which comes with all the essential ROCm tools and libraries pre-installed. It's like getting a workbench with all your specialized GPU tools already mounted on the wall.

```dockerfile
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
```

Here's where we add our everyday development tools. Think of it as adding your general-purpose toolset:

- `build-essential`: Your basic compiler and build tools
- `git`: For version control
- `cmake`: For project building and configuration
- `python3-pip`: For Python package management

The `rm -rf /var/lib/apt/lists/*` at the end is a best practice I always include - it cleans up the package lists after installation to keep our image slim.

```dockerfile
WORKDIR /workspace
COPY . .
```

This sets up our active workspace and copies in our project files. I've found `/workspace` to be a clean, logical place for development work.

#### Use case example

This container is perfect for a team developing a GPU-accelerated image processing library. The ROCm base image provides the frameworks needed, build tools to compile C++ code, and Python support to create convenient bindings. Build the container with this command:

```bash
docker buildx build -t rocm-dev-envionment .
```

Source code can be mounted as a volume when launching the container:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host -v /path/to/code:/workspace \
  rocm-dev-envionment
```

This setup creates an ideal development loop - you can edit code on your host machine and compile/test inside the container with all the right dependencies in place.

I'd recommend this as a starting point for any ROCm development project - it's minimal enough to understand easily but complete enough to be immediately useful. You can always add more tools as your project's needs grow.

### ML training container

Here's a more sophisticated example for ML training:

```dockerfile
# Start from latest stable PyTorch ROCm container
FROM rocm/pytorch:latest

# Install additional Python packages
RUN pip3 install \
    pandas \
    scikit-learn \
    matplotlib \
    wandb

# Set up training directory structure
WORKDIR /opt/training
RUN mkdir -p /opt/training/data /opt/training/models

# Copy training code
COPY train.py .
COPY requirements.txt .

# Default command
CMD ["python3", "train.py"]
```

This Dockerfile creates a ready-to-run machine learning training environment built on AMD's official PyTorch container. I particularly like using this setup because it combines PyTorch's GPU acceleration with essential data science tools like `pandas` and `scikit-learn`, while also integrating `wandb` for experiment tracking. The container is structured with dedicated directories for data and model artifacts (`/opt/training/data` and `/opt/training/models`), which you can easily mount as volumes to persist your work. When you launch this container, it automatically runs your training script (`train.py`), making it perfect for both interactive development and automated training pipelines.

A practical tip I've learned from using this setup: while the container comes ready to train, you'll want to mount your dataset directory and model output directory as volumes when you run it. Something like `docker run -v $(pwd)/data:/opt/training/data -v $(pwd)/models:/opt/training/models your-image-name` will do the trick. This keeps your valuable data and trained models safely stored on your host system while giving you all the benefits of a containerized training environment. A [Docker Compose](https://docs.docker.com/compose/ "https://docs.docker.com/compose/") file can be used to simplify the process of starting and stopping the container with the desired folder mounts and other options. The process to build and run this container is similar to the previous example:

```bash
docker buildx build -t rocm-training .
docker run -it --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host \
  -v /home/user/models:/opt/training/models \
  -v /home/user/data:/opt/training/data \
  rocm-training
```

## Summary

Containers have transformed how we develop and deploy GPU-accelerated applications. Through this blog, we've explored how to build custom ROCm containers from base images, set up development environments for GPU-accelerated applications, and create specialized containers for ML training workloads. You've learned practical approaches to mounting volumes, managing dependencies, and implementing GPU support in your containers. These examples demonstate how the ROCm container ecosystem provides a robust foundation for both development and production deployments, making it easier than ever to work with AMD GPUs. In the next blog post, I'll walk through an advanced example using [vLLM](https://github.com/vllm-project/vllm "https://github.com/vllm-project/vllm") for both inferencing and benchmarking.
