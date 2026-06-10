---
blogpost: true
blog_title: "Accelerating Autonomous Driving Model Training on AMD ROCm™ Software"
date: 08 Dec 2025
author: 'Fuwei Yang, Mingjie Lu, Bin Ding, Fan Wang, Treemann Zheng, Zhaodong Bing, Dong Li, Emad Barsoum'
thumbnail: 'autodrive_rocm.jpg'
tags: AI/ML
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: Provides out-of-the-box training examples for the most widely used autonomous driving models in the industry.
language: English
myst:
    html_meta:
        "author": "Fuwei Yang, Mingjie Lu, Bin Ding, Fan Wang, Treemann Zheng, Zhaodong Bing, Dong Li, Emad Barsoum"
        "description lang=en": "Learn how to deploy AMD GPUs for high-performance autonomous driving related model training with ROCm optimization."
        "keywords": "Autonomous Driving, Training"
        "vertical": "AI, Robotics"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Fuwei Yang, Mingjie Lu, Bin Ding, Fan Wang, Treemann Zheng, Zhaodong Bing, Dong Li, Emad Barsoum"
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

# Accelerating Autonomous Driving Model Training on AMD ROCm™ Software

The autonomous driving industry is undergoing rapid growth, driven by advances in AI and the increasing demand for safer, smarter transportation. At the core of this transformation are complex deep learning models that enable vehicles to perceive, reason, and navigate dynamic environments. Efficiently training these models at scale has become a key challenge for companies working to bring autonomous systems to the road.

While NVIDIA GPUs have long been the default platform for model training, this reliance brings drawbacks such as vendor lock-in, supply limitations, and reduced flexibility in cost optimization. AMD Instinct™ GPUs offer a strong alternative with competitive performance and improved availability. However, ecosystem readiness has lagged—particularly for domain-specific models used in autonomous driving—due to software compatibility gaps and the lack of off-the-shelf training pipelines.

To help close this gap, we're excited to introduce [awesome-rocm-autodrive](https://github.com/AMD-AGI/awesome-rocm-autodrive/tree/main), an open-source collection of popular autonomous driving models running on AMD GPUs using the ROCm™ stack. By providing a unified Docker image, ready-to-run examples for popular autonomous-driving models, and clear tuning guides, the project helps narrow the ecosystem gap and accelerates developer onboarding.

In this blog, you will learn the practical steps required to train state-of-the-art autonomous driving models on AMD GPUs—from setting up a unified ROCm training environment to applying performance-critical optimizations. You will also see concrete examples and benchmark data that help you replicate and scale your workloads efficiently.

## Introducing awesome-rocm-autodrive

[awesome-rocm-autodrive](https://github.com/AMD-AGI/awesome-rocm-autodrive/tree/main) is a comprehensive repository that provides out-of-the-box training examples for the most widely used autonomous driving models in the industry. Our growing collection covers essential areas of autonomous driving AI, including 3D perception, bird's-eye-view understanding, HD mapping and end-to-end driving models. We continuously expand our model zoo to incorporate the latest advances in the field, ensuring developers have access to cutting-edge architectures optimized for AMD GPUs.

### Key Components

- **ROCm-optimized MMCV**: We've ported and optimized the core MMCV library for AMD ROCm, ensuring seamless compatibility with existing PyTorch workflows
- **Pre-configured Docker Images**: Ready-to-use containers with all dependencies properly configured
- **Optimized Training Examples**: Diverse training examples across major autonomous driving tasks, with ROCm-specific optimizations included in each example.
- **Comprehensive Documentation**: Step-by-step guides for setup and deployment

### Getting Started

Using our repository is straightforward:

1\. Clone the repo:

```shell
git clone <https://github.com/AMD-AGI/awesome-rocm-autodrive.git>
cd awesome-rocm-autodrive

```

2\. Build the Docker Image

```shell
cd docker
docker build -t rocm-autodrive .

```

Or directly pull the docker image provided by AMD:

```shell
docker pull amdagi/autodrive_training_rocm6.4:v1

```

If you want to build mmcv separately, just

```shell
MMCV_WITH_OPS=1 pip install .

```

3\. Launch Docker Container

```shell
docker run --rm -it --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $PWD:/workspace \
  amdagi/autodrive_training_rocm6.4:v1

```

4\. Run an Example

git clone the repo link provided in README or examples, repare the dataset, and run it as described in the corresponding README.

## ROCm-Based Performance Optimizations

### NHWC layout for convolutions

One of our key performance optimizations involves leveraging the NHWC (Number-Height-Width-Channels) memory layout for convolutional operations on ROCm. While PyTorch defaults to Number-Channels-Height-Width (NCHW) layout, we've found that NHWC can deliver significant performance improvements on AMD GPUs, particularly for convolution-heavy models common in autonomous driving applications.

The NHWC layout stores pixel data with channels as the innermost dimension, which aligns better with how modern GPUs access memory during convolution operations. This layout enables more efficient memory coalescing and reduces the number of memory transactions required. On AMD ROCm, the MIOpen library provides highly optimized kernels for NHWC convolutions that can leverage the full memory bandwidth of AMD GPUs.

To enable the NHWC layout:

```shell
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM=1

```

It is also necessary to revise the model definition by adding the following code:

```python
model = model.to (memory_format=torch.channels_last)

```

### GEMM Optimization for Coordination Transformation

In BEVFormer-like models, point cloud coordinate transformations are a critical part of projecting multi-view features into the bird’s-eye view (BEV) space. This process involves several transformation matrices—such as the lidar-to-camera projection, image intrinsic parameters, and camera-to-BEV transforms. Each of these matrices has specific shapes depending on the batch size and the number of cameras. For example, the lidar-to-camera projection:

```python

## lidar2cam: shape (B, N_cam, 1, 1 ,1, 3, 3)

## points: shape (B, N_cam, X, Y, Z, 3, 1)

points = lidar2cam.matmul(points)

```

However, the default behavior of batchmatmul broadcasts the two input tensors to the same shape and then flattens all leading dimensions into the batch dimension. This effectively results in an operation like:

```python
torch.bmm(lidar2cam.view(B * N_cam * X * Y * Z, 3, 3), points.view(B * N_cam * X * Y * Z, 3, 1))

```

Such flattening can lead to an excessively large batch dimension, which introduces significant inefficiencies in both memory usage and computation. To address this, we can restructure the tensor layout by moving the shared spatial dimensions (e.g., X, Y, Z) to the innermost position (out_dim), and placing the non-shared dimensions (e.g., B, N_cam) at the outermost (batch_dim). This layout optimization substantially reduces the effective tensor size and alleviates performance degradation caused by large batch dimensions. The optimized bmm operation can be expressed as:

```python
torch.bmm(lidar2cam.view(B * N_cam, 3, 3), points.view(B * N_cam, 3, X * Y * Z)).permute(0, 2, 1).view(B, N_cam, X, Y, Z, 3).unsqueeze(-1)

```

We provided an [example to showcase the difference](https://github.com/AMD-AGI/awesome-rocm-autodrive/blob/main/tests/compare_bmm.py). Run the example and you will find that optimized bmm reduces the execution time from 7.2 ms to 0.09 ms in AMD Instinct™ MI325X While maintaining mathematical equivalence.

### MIOpen tuning

For users seeking the best performance on AMD GPUs, especially when training large or compute-intensive models, we recommend optional but effective tuning step: MIOpen tuning. This step is not required to get started but can improve training speed and efficiency with limited extra efforts.

MIOpen tuning would be effective for convolution-heavy models like ResNet, EfficientNet, a detailed guidance of how to conduct MIOpen tuning could be found here: [miopen tuning guidance](https://github.com/ROCm/MIOpen/blob/develop/docs/conceptual/tuningdb.rst) . In some of the [examples](https://github.com/binding7012/Sparse4D/tree/main/tune_result/sparse4d_miopen_tuning), we have put the tuned MIOpen config into the corresponding github repo, to allow users to deploy them without manual tuning.

### Custom HIP kernel optimization

We observed that some CUDA kernels in MMCV or the original model repositories do not fully leverage AMD GPU performance under AMD ROCm. This is particularly common in the autonomous driving domain, where many models rely on custom CUDA kernels such as deformable convolution, voxelization, NMS and ROI align.

While AMD ROCm HIPIFY tooling can automatically convert most of these kernels for functional use on AMD GPUs, performance is not always optimal out of the box. In practice, some kernels require further tuning or redesign to fully utilize the GPU’s compute and memory bandwidth. Two typical examples are the deformable aggregation kernel and the voxelization kernel. We applied a series of HIP optimizations to improve their performance significantly on AMD GPUs.

#### Deformable Aggregation Kernel Optimization

- **Hierarchical Reduction Basics**: A major bottleneck in GPU reduction is the overuse of atomic operations, which serialize parallel workloads when multiple threads write to the same memory location. This contention becomes more severe as thread count increases, making naive atomic-add approaches inefficient for large-scale computation.
- **Multi-level Reduction Strategy**: To address this, we apply a hierarchical reduction approach. Within each thread block, we use tree-based reductions in shared memory, with `__syncthreads()` for synchronization. For larger groups, warp-level reductions with `__shfl_down` further improve efficiency.
- **Dynamic Memory and Atomic Minimization**: We dynamically allocate shared memory based on runtime parameters to adapt to varying reduction group sizes. Using a "reduce-then-atomize" pattern, we first aggregate locally within a block or warp, and only then apply a minimized number of atomic operations by representative threads. This significantly reduces contention and improves memory bandwidth utilization.

By combining hierarchical reduction strategies with dynamic memory management and atomic operation minimization, we achieved notable performance improvements in deformable aggregation. This optimization serves not only as a performance fix for deformable aggregation but also as a general design pattern for developers working with reduction-heavy or atomics-heavy kernels on ROCm.

#### Voxelization Kernel Optimization

The original voxelization kernel has multiple memory and control flow inefficiencies. Our enhancements include:

- **Shared Memory Caching**: Frequently accessed data is cached in fast local memory, reducing pressure on HBM.
- **Vectorized Loads and Loop Unrolling**: Leveraging ROCm’s wide VGPR resources, we apply vectorized memory access and aggressive loop unrolling to reduce instruction overhead.
- **Read-Only Cache Paths**: We take advantage of hardware read-only data cache paths for invariant inputs by telling the compiler explicitly to speed up input data loading.
- **Branch divergence minimization through logic refactoring**: To avoid wavefront divergence caused by complex control flow inside main loops, conditional logic is carefully rewritten to either enable predictation or reduce divergence points.
- **Preprocessing with rocPRIM**: We pre-sort the point cloud using rocPRIM, enabling the kernel to operate in a fully parallel fashion rather than sequential processing.

By combining these techniques, we achieved substantial performance gains for both kernels on AMD GPUs. These improvements are part of our broader effort to make autonomous driving models run efficiently on ROCm. These improvements are part of our broader effort to make autonomous driving models run efficiently on ROCm and you can try them out yourself by checking our repo.

## Summary

This blog walked through the unified ROCm Docker, the ready-to-run training examples, and the tuning best practices that help you train popular autonomous-driving models efficiently on AMD GPUs. With these tools, you can quickly reproduce results, optimize performance, and accelerate your development workflow.

The release of awesome-rocm-autodrive marks an important step toward democratizing autonomous-driving development. We will continue expanding this ecosystem by adding new models—such as end-to-end autonomous driving architectures and multi-sensor fusion networks—while also optimizing widely used MMCV customer kernels, improving training performance, and providing updated benchmarks and multi-node training guides.

By making it easy to leverage AMD GPUs for autonomous-driving development, we help teams focus on what matters most: building safer, more capable vehicles. Providing developers with a choice in hardware accelerators also drives innovation and reduces costs across the industry.

Visit the [GitHub repository](https://github.com/AMD-AGI/awesome-rocm-autodrive) to get started today. Together, we can accelerate the future of autonomous driving on AMD GPUs.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
