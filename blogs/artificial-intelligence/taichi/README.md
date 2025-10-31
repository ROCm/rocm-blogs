---
blogpost: true
blog_title: "Accelerating Parallel Programming in Python with Taichi Lang on AMD GPUs"
date: 31 Jul 2025
author: 'Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani'
thumbnail: 'taichi_blog_thumbnail.png'
tags: AI/ML, Scientific Computing
category: Applications & models
target_audience: The target audience is AI developers interested in AMD GPU kernel offload
key_value_propositions: This blog enables customers to develop Python applications with AMD GPU kernel acceleration using Taichi Lang.
language: English
myst:
    html_meta:
        "author": "Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "This blog provides a how-to guide on installing and programming with Taichi Lang on AMD Instinct GPUs."
        "keywords": "Taichi Lang, Python, AI, machine learning, domain specific language, GPU programming, GPU acceleration"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Design, Simulation & Modeling"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# Accelerating Parallel Programming in Python with Taichi Lang on AMD GPUs

[Taichi Lang](https://www.taichi-lang.org/) is an open-source, imperative, parallel programming language for high-performance numerical computation.
It is embedded in Python and uses just-in-time (JIT) compiler frameworks (e.g. LLVM) to offload the compute-intensive
Python code to the native GPU or CPU instructions. The language has broad applications spanning real-time physical simulation,
numerical computation, augmented reality, artificial intelligence, vision and robotics, visual effects in films and games,
general-purpose computing, and much more [^1].

Taichi's imperative programming paradigm distinguishes it from frameworks like PyTorch and allows for flexible programming with
no particular programming patterns for writing numerical computations (e.g. element-wise operations).
This flexible, imperative programming paradigm allows the user to write
large amounts of code in a single kernel. Taichi also has its own intermediate representation which enables optimizations that are not dependent
on the accelerator and compiler backends like common subexpression elimination, dead code elimination,
control flow graph analysis, etc. Taichi also goes beyond a Python JIT compiler.
Taichi decouples the computation from data structures with a set of generic data containers, called SNodes [^2].
SNodes can be used to compose hierarchical, dense or sparse, multi-dimensional fields conveniently [^3].

This blog is a guide for programming with Taichi Lang on AMD Instinct GPUs. In this blog, we provide an overview of Taichi,
instructions for installing Taichi and several examples for running a Taichi program on AMD Instinct GPUs.

[^1]: https://github.com/taichi-dev/taichi
[^2]: https://docs.taichi-lang.org/docs/master/internal#data-structure-organization
[^3]: https://docs.taichi-lang.org/docs/overview

## ROCm Taichi

* [Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/taichi-install.html)

* [Docker image](https://hub.docker.com/r/rocm/taichi/tags)

* [GitHub](https://github.com/ROCm/taichi)

## Getting Started with Taichi

When writing compute-intensive tasks, users can make use of the two decorators `@ti.func` and `@ti.kernel`.
Functions decorated with `@ti.kernel` are kernels that serve as the entry points where Taichi's runtime takes over the tasks,
and they must be directly invoked by Python code. Functions decorated with `@ti.func` are building blocks of kernels and can only be
invoked by another Taichi function or a kernel. These decorators instruct Taichi to take over the computation tasks and
compile the decorated functions to machine code using JIT compiler.
As a result, calls to these functions are executed on multi-core CPUs or GPUs.

Below we show how simple it is to use the Taichi `@ti.func` and `@ti.kernel` decorators to accelerate a python code. First we show the python code without using Taichi. In this example, we have the function `inv_square` which acts as a building block function for the kernel `partial_sum`.

Example without Taichi:

```python
def inv_square(x):  # A function
    return 1.0 / (x * x)

def partial_sum(n: int) -> float:  # A kernel
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total

partial_sum(1000)
```

To write this code as a Taichi code we simply import and initialize Taichi for acceleration:

```python
import taichi as ti
ti.init(arch=ti.gpu)
```

and decorate the building block function and kernel with the `@ti.func` and `@ti.kernel` decorators, respectively:

```python
@ti.func
def inv_square(x):  # A Taichi function
    return 1.0 / (x * x)

@ti.kernel
def partial_sum(n: int) -> float:  # A kernel
    total = 0.0
    for i in range(1, n + 1):
        total += inv_square(n)
    return total
```

## Docker Environments for Taichi Lang on AMD GPUs

### Use a Prebuilt Docker Image with Taichi Pre-Installed

To simplify running Taichi programs on AMD GPUs, we recommend using the pre-built docker image.
To do this, pull the docker image:

```bash
docker pull rocm/taichi:taichi-1.8.0b1_rocm6.3.2_ubuntu22.04_py3.10.12
```

and launch the docker container:

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --name taichi_lang rocm/taichi:taichi-1.8.0b1_rocm6.3.2_ubuntu22.04_py3.10.12 bash
```

You will also need to install `git` in the Docker container to clone the Taichi Examples repo in later instructions:

```bash
sudo apt-get update && apt-get install -y git
```

### Build Your Own Docker Image

You can also install taichi in an existing docker image with ROCm 6.3.2. To do this, copy the instructions below into a Dockerfile:

```text
FROM rocm/dev-ubuntu-22.04:6.3.2

ARG LLVM_VERSION=15
ARG GPU_TARGETS=gfx90a

ENV DEBIAN_FRONTEND=noninteractive
ENV TAICHI_SRC=/app/taichi
ENV LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}
ENV PATH=${LLVM_DIR}/bin:$PATH
ENV TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN=OFF -DTI_WITH_OPENGL=OFF -DTI_BUILD_TESTS=ON -DTI_BUILD_EXAMPLES=OFF -DCMAKE_PREFIX_PATH=${LLVM_DIR}/lib/cmake -DCMAKE_CXX_COMPILER=${LLVM_DIR}/bin/clang++ -DTI_WITH_AMDGPU=ON -DTI_WITH_CUDA=OFF -DTI_AMDGPU_ARCHS=${GPU_TARGETS}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget \
    freeglut3-dev libglfw3-dev libglm-dev libglu1-mesa-dev \
    libjpeg-dev liblz4-dev libpng-dev libssl-dev \
    libwayland-dev libx11-xcb-dev libxcb-dri3-dev libxcb-ewmh-dev \
    libxcb-keysyms1-dev libxcb-randr0-dev libxcursor-dev libxi-dev \
    libxinerama-dev libxrandr-dev libzstd-dev \
    python3-pip cmake pybind11-dev ca-certificates \
    llvm-${LLVM_VERSION} clang-${LLVM_VERSION} lld-${LLVM_VERSION} \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --recursive -b amd-integration https://github.com/ROCm/taichi.git \
    && cd taichi \
    && ./build.py \
    && python3 -m pip install /app/taichi/dist/taichi*.whl 
```

Build the docker container with the following command:

```bash
docker build  -t taichi-lang-dev .
```

Launch the docker container using the following docker run command:

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --name taichi_lang taichi-lang-dev bash
```

## Prepared Taichi Lang Examples

AMD has a collection of select examples to demonstrate the use of Taichi Lang on AMD Instinct GPUs. One such example is the "count primes" program. In this example we have the function `is_prime` which will be used in the kernel `count_primes`. In the code below, we write this example as a Taichi program by decorating `is_prime` with the Taichi decorator `@ti.func` and decorating `count_primes` with the Taichi decorator `@ti.kernel`. To run this example, copy the code below to a file named count_primes.py:

```python
import taichi as ti
ti.init(arch=ti.gpu)

@ti.func
def is_prime(n: int):
    result = True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result

@ti.kernel
def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1

    return count

print(count_primes(1000000))
```

Once this file has been created, execute the code in your docker container with the following command:

```bash
python3 count_primes.py
```

The output should be similar to the output below:

```text
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
78498
```

Another example is a longest common subsequence kernel. In this example we do not need a helper function, so the only decorator we use is `@ti.kernel` to accelerate the kernel function `compute_lcs`. To run this example, copy the code below into a file named lcs.py:

```python
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

benchmark = True

N = 15000

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))

if benchmark:
    a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
    b_numpy = np.random.randint(0, 100, N, dtype=np.int32)
else:
    a_numpy = np.array([0, 1, 0, 2, 4, 3, 1, 2, 1], dtype=np.int32)
    b_numpy = np.array([4, 0, 1, 4, 5, 3, 1, 2], dtype=np.int32)

@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True) # Disable auto-parallelism in Taichi
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
               f[i, j] = ti.max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                          ti.max(f[i - 1, j], f[i, j - 1]))

    return f[len_a, len_b]


print(compute_lcs(a_numpy, b_numpy))
```

Once this file has been created, execute the code in your docker container with the following command:

```bash
python3 lcs.py
```

The output should be similar to the output below:

```text
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
2706
```

The remainder of these examples have been collected in the taichi_examples github repo. To clone this github repo:

```bash
git clone https://github.com/ROCm/taichi_examples.git
```

You may go to the `taichi_example/examples` directory to examine the examples to see how Taichi Lang is implemented in the code.

To run the available examples, install the dependencies and use the scripts in the repo to run the examples in batches:

```bash
pip3 install pillow
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3/
cd taichi_examples
./run_demos.sh
./run_algorithm_graph_examples.sh
```

The output should be similar to the output below.

run_demos.sh output:

```text
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
2710
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
78498
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
Elapsed 0.07401013374328613 seconds
Elapsed 0.07366204261779785 seconds
Elapsed 0.07427334785461426 seconds
Elapsed 0.07503175735473633 seconds
Elapsed 0.0747842788696289 seconds
Elapsed 0.0749671459197998 seconds
Elapsed 0.07503128051757812 seconds
Elapsed 0.07497262954711914 seconds
=========================================================================
Kernel Profiler(count, default) @ AMDGPU on AMD Instinct MI210
=========================================================================
[      %     total   count |      min       avg       max   ] Kernel name
-------------------------------------------------------------------------
[100.00%   0.002 s      8x |    0.256     0.262     0.271 ms] ti_pad_c80_0_kernel_0_range_for
-------------------------------------------------------------------------
[100.00%] Total execution time:   0.002 s   number of results: 1
=========================================================================
```

run_algorithm_graph_examples.sh output:

```text
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
running in graph mode
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
0.0
4.0
0.0
0.0
4.0
0.0
0.0
4.0
0.0
0.0
[Taichi] version 1.8.0b1, llvm 15.0.0, commit eeb3354a, linux, python 3.10.12
[Taichi] Starting on arch=amdgpu
```

## Summary

In this blog, we have provided an overview of Taichi Lang and the distinguishing features of its imperative programming paradigm.
The provided step-by-step guide should enable users to install Taichi Lang in a ROCm 6.3.2 docker environment.
With this installation users can run Taichi Lang programs with features enabled for AMD Instinct GPUs.

AMD continues to enhance Taichi support through ongoing development on its latest ROCm software and AMD Instinct GPU products. Keep an eye out for updates and new blogs as we share our progress.

## Acknowledgements

The author wishes to acknowledge the AMD teams that supported this work, whose contributions were
instrumental in enabling Taichi Lang:
*Tiffany Mintz, Debasis Mandal, Yao Liu, Phani Vaddadi, Vish Vadlamani, Ritesh Hiremath,
Bhavesh Lad, Radha Srimanthula, Anisha Sankar, Amit Kumar, Ram Seenivasan, Kiran Thumma,
Aakash Sudhanwa, Aditya Bhattacharji*

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
