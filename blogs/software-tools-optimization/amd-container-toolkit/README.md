---
blogpost: true
blog_title: "Unlocking GPU-Accelerated Containers with the AMD Container Toolkit"
date: 03 July 2025
author: 'Abhishek Patil'
thumbnail: '2025-05-21-container-toolkit.jpg'
tags: Optimization, Performance, AI/ML, Kubernetes
category: Ecosystems and Partners
target_audience: GPU developers, DevOps engineers, ML/AI practitioners, and software engineers working with containerized workflows on AMD Instinct platforms.
key_value_propositions: The blog highlights how the AMD Container Toolkit simplifies and streamlines the deployment of ROCm-based GPU workloads in containerized environments. It demonstrates how the toolkit provides a consistent and optimized developer experience across Docker platforms, automates container image building and runtime configuration, and enables better integration with AMD Instinct GPUs.
language: English
myst:
    html_meta:
        "author": "Abhishek Patil"
        "description lang=en": "Simplify GPU acceleration in containers with the AMD Container Toolkit—streamlined setup, runtime hooks, and full ROCm integration."
        "keywords": "AMD Container Toolkit, ROCm, Docker GPU, GPU container runtime, MI300, MI250, AMD GPUs, ROCm stack, AI/ML workloads, HPC, GPU containerization, AMD Instinct"
        "vertical": "AI, Developers, HPC, Data Science, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Abhishek Patil"
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

# Unlocking GPU-Accelerated Containers with the AMD Container Toolkit

In the rapidly evolving fields of high-performance computing (HPC), artificial intelligence (AI), and machine learning (ML), containerization has become a cornerstone of modern application deployment. Containers provide a lightweight, portable, and scalable way to package applications and their dependencies. The integration of GPUs into these environments has become imperative. However, leveraging GPU acceleration within containers has historically been a complex and error-prone process, particularly when ensuring seamless access to GPU hardware resources.

The  [AMD Container Toolkit Documentation](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/) is a game-changing solution that addresses these challenges by providing a robust and streamlined framework for developers and organizations—simplifying integration and fully utilizing AMD Instinct™ accelerators in containerized environments. This toolkit provides platform engineers, DevOps teams, and HPC architects with the tools they need to deploy GPU-accelerated workloads efficiently. Whether you're running AI/ML training pipelines, HPC simulations, or inference workloads, the AMD Container Toolkit ensures seamless access to AMD Instinct™ accelerators in containerized environments.

This blog post explores the AMD Container Toolkit in depth, covering its architecture, installation, configuration, and real-world applications. By the end, you'll have a clear understanding of how this toolkit can empower your GPU-accelerated workloads.

---

## What is the AMD Container Toolkit?

The [AMD Container Toolkit Documentation](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/) is an extensible suite of runtime tools and configurations that streamlines deployment and enables seamless access to AMD GPUs within Linux containers. It is designed to work with Instinct GPUs, providing the necessary runtime hooks, libraries, and device configurations to ensure GPU workloads run efficiently in containerized environments without compromising security or performance. It represents a significant step forward in simplifying GPU access in containerized environments.

---

## Why AMD Container Toolkit?

The AMD Container Toolkit addresses several critical challenges in GPU-accelerated containerization:

- **Simplified GPU Access:** Provides a streamlined and consistent way to access AMD GPUs within containers, eliminating the need for complex manual configurations.
- **Production-Ready Deployments:** Ensures enterprise-grade stability and performance, making it suitable for mission-critical workloads.
- **Security and Performance:** Exposes GPU resources to containers while maintaining strong isolation and optimal performance.

---

### Key Features

- **Runtime Hooks:** Dynamically configure containers to include environment variables and device files.
- **Device Discovery:** Provides tools to enumerate and manage GPU devices within containers.
- **Framework Support:** Compatible with popular ML frameworks like PyTorch and TensorFlow.

---

## Architecture Overview

The AMD Container Toolkit Runtime is designed to integrate with standard container runtimes—such as Docker, without requiring any changes to existing container images. Its primary role is to ensure that AMD GPUs are accessible inside containers while maintaining a clean separation between application and hardware configuration.

At the core of the toolkit is a lightweight wrapper around the low-level container runtime, `runc`. This wrapper is responsible for preparing the container environment so that AMD GPUs can be accessed during runtime. It configures the container with the necessary device files and runtime settings to make GPU resources available transparently.

Specifically, the wrapper performs the following tasks:

- **Device Access Configuration:** Grants access to the appropriate GPU device nodes inside the container by mounting the relevant `/dev` entries.
- **Environment Setup:** Propagates key environment variables required for ROCm to operate effectively within the containerized environment.

This approach enables GPU support to be dynamically injected at runtime, ensuring that containers remain lightweight, portable, and agnostic of the underlying hardware configuration. There is no need to embed GPU-specific logic or dependencies directly into container images.

---

## System Requirements

Before installing the AMD Container Toolkit, ensure your system meets the following prerequisites:

- **Operating System:**  
  Ubuntu 22.04 LTS (Jammy Jellyfish) or Ubuntu 24.04 LTS (Noble Numbat)

- **Docker Version:**  
  Docker version **25 or above** is required. The Container Device Interface (CDI) format—used by modern container runtimes to abstract and expose GPUs—is not supported in older Docker versions.

> **Note:**  
> Please refer to the [compatibility matrix](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/requirements.html#compatibility-matrix) before proceeding.

---

## Installation and Setup

### Step-by-Step Installation

To get started with the AMD Container Toolkit, follow these steps:

#### 1. Install System Prerequisites

```bash
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
```

#### 2. Install the AMDGPU Driver

Refer to the latest ROCm documentation for driver installation in the [ROCm Install Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

Download the AMDGPU driver installer package from the [Radeon Repository](https://repo.radeon.com/amdgpu-install).

#### 3. Configure Repositories

```bash
sudo apt update
sudo apt install wget gpg
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
```

Add the appropriate repository to your sources list based on your Ubuntu version:

```bash
# For Ubuntu 22.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amd-container-toolkit/apt/ jammy main" | sudo tee /etc/apt/sources.list.d/amd-container-toolkit.list

# For Ubuntu 24.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amd-container-toolkit/apt/ noble main" | sudo tee /etc/apt/sources.list.d/amd-container-toolkit.list
```

#### 4. Install the AMD Container Toolkit

Update the package list and install the container toolkit:

```bash
sudo apt update
sudo apt install amd-container-toolkit
```

#### 5. Install Docker (if not already installed)

```bash
sudo apt install docker.io
```

> **Important:**  
> Docker version **25 or above** is required. The Container Device Interface (CDI) format—used by modern container runtimes to abstract and expose GPUs—is not supported in older Docker versions.
>
> For a detailed walkthrough, refer to the [Quick Start Guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html).

---

## Verifying AMD Container Runtime Hook Functionality

After installing and configuring the AMD Container Runtime, it's important to verify that the runtime hook is functioning correctly and that the ROCm stack is accessible inside containers. You can do this by running the `rocm-smi` utility within a container.

### Example: Running `rocm-smi` in a Container

```bash
amd-ctk runtime configure --runtime=docker

docker run --rm --runtime=amd -e AMD_VISIBLE_DEVICES=all rocm/rocm-terminal rocm-smi
```

### Expected Result

If the runtime is set up correctly, the command should display the output of `rocm-smi` inside the container. This confirms that the container can access the host's AMD GPUs and that the runtime hook is working as intended.

---

## Core Concepts

The AMD Container Toolkit operates by intercepting and modifying the Open Container Initiative (OCI) specifications generated by the container daemon. It injects the necessary GPU devices into the OCI spec, enabling containers to access AMD GPUs seamlessly. This process involves:

- Using the `AMD_VISIBLE_DEVICES` environment variable to specify GPU access.
- Employing the `amd-ctk` CLI to configure the container runtime and manage device visibility.

## Docker Runtime Integration

The toolkit integrates with Docker to provide GPU access in containers. Key functionalities include:

- **Configuring Docker to use the AMD container runtime:**

    ```bash
    amd-ctk runtime configure --runtime=docker
    ```

- **Specifying required GPUs (choose one of the following methods):**

  - Using the `AMD_VISIBLE_DEVICES` environment variable:

      ```bash
      docker run --rm --runtime=amd -e AMD_VISIBLE_DEVICES=all rocm/rocm-terminal rocm-smi
      ```

  - Using the Container Device Interface (CDI):

      ```bash
      sudo amd-ctk cdi generate --output=/etc/cdi/amd.json
      docker run --rm --device amd.com/gpu=all rocm/rocm-terminal rocm-smi
      ```

- **Listing available GPUs:**

    ```bash
    amd-ctk cdi list
    ```

## Device Discovery and Enumeration

Using the `amd-ctk` CLI, developers can:

- Enumerate available GPU devices.
- Generate CDI specifications for device access.
- Manage device visibility within containers.

## Framework Integration

The AMD Container Toolkit supports integration with popular machine learning frameworks, enabling developers to leverage GPU acceleration in their applications. For example, integrating with PyTorch or TensorFlow involves:

- Building container images with the necessary framework dependencies.
- Configuring the container runtime to provide GPU access.
- Running training or inference workloads within the containerized environment.

For detailed examples and guidance, refer to the [Framework Integration](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/framework-integration.html) section of the documentation.

## Troubleshooting

Common issues and their resolutions include:

- **Docker daemon not recognizing the AMD container runtime:**  
    Ensure that the runtime is correctly configured and that the Docker daemon has been restarted.

- **Containers unable to access GPUs:**  
    Verify that the `AMD_VISIBLE_DEVICES` environment variable is set appropriately and that the necessary devices are visible within the container.

For a comprehensive list of troubleshooting steps, consult the [Troubleshooting guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/troubleshooting.html).

## Migration Guide: NVIDIA to AMD Container Toolkit

Transitioning from NVIDIA's container toolkit to AMD's involves:

1. Uninstalling the NVIDIA container runtime.
2. Installing the AMD Container Toolkit as outlined in the [Quick Start Guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html).
3. Updating container configurations to use the AMD runtime and environment variables.

Detailed migration steps are available in the [Migration Guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/migration-guide.html).

## Real-World Use Cases

### 1. AI and Machine Learning

- Supports frameworks like PyTorch and TensorFlow
- Efficient model training and inference
- Consistent ML workflow environments

### 2. Development Environments

- Reproducible development environments
- Consistency between development and production
- Simplified dependency management

### 3. Production Deployments

- Scalable container orchestration
- Efficient resource utilization
- Robust monitoring and management

---

## Best Practices and Troubleshooting

### Best Practices

- **Resource Management:** Define resource limits, monitor GPU utilization, implement scaling strategies.
- **Security:** Apply regular updates, enforce access controls, follow container isolation best practices.
- **Performance Optimization:** Use suitable base images, optimize GPU memory usage, implement efficient data pipelines.

### Common Troubleshooting Scenarios

- **Container Start-up Issues:** Check GPU driver installation, runtime configuration, and device permissions.
- **Performance Problems:** Monitor resources, check for memory leaks, verify GPU accessibility.

---

## Summary

The AMD Container Toolkit is a vital component for organizations leveraging AMD GPUs for HPC and AI workloads. The toolkit stands as a production-ready solution for simplifying GPU access, facilitating integration with popular frameworks, and enabling ROCm-powered GPU acceleration in modern containerized environments. By following the installation and configuration steps above, users can effectively deploy and manage GPU-accelerated containers.

In this blog, we presented the AMD Container Toolkit’s architecture, installation process, runtime configuration, and practical use cases for GPU-accelerated containers. Whether building scalable AI pipelines or prototyping machine learning models in Docker, the AMD Container Toolkit streamlines integration, boosts performance, and ensures portability.

As container adoption continues to surge, AMD’s commitment to open standards and robust developer tooling ensures that GPU acceleration is accessible, scalable, and primed for the workloads of tomorrow.

For more details and updates, visit the [AMD Container Toolkit Documentation](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/).

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
