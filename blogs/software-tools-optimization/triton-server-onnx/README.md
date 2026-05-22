---
blogpost: true
blog_title: "From Build to Benchmark: ONNX Model Serving with Triton Inference Server on AMD GPUs"
date: 22 May 2026
author: 'Fabricio Flores, Ted Themistokleous, Lin Sun, Jorge Parada'
thumbnail: 'tritonserver-onnx-amd-blog.png'
tags: Serving, GenAI, LLM, Performance
category: Software tools & optimizations
target_audience: AI engineers, SW
key_value_propositions: Showcase triton inference server with onnx backend for onnx model serving
language: English
myst:
    html_meta:
        "author": "Fabricio Flores, Ted Themistokleous, Lin Sun, Jorge Parada"
        "description lang=en": "Step-by-step guide to building, deploying, and benchmarking ONNX models with Triton Inference Server and MIGraphX on AMD GPUs"
        "keywords": "ONNX, triton inference server, triton, LLM, migraphx, onnxruntime"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Fabricio Flores, Ted Themistokleous, Lin Sun, Jorge Parada"
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

# From Build to Benchmark: ONNX Model Serving with Triton Inference Server on AMD GPUs

Triton Inference Server is an open-source platform designed to streamline AI inferencing. It supports the deployment, scaling, and inference of trained models from multiple frameworks, including ONNX Runtime, TensorFlow, PyTorch, and others. It runs across cloud, data center, and edge environments, making it adaptable for diverse AI workloads.

Some of the Triton Inference Server capabilities include:

* **Framework flexibility**: You can deploy models from different frameworks (see the [Triton Inference Server Backend documentation](https://github.com/triton-inference-server/backend)) regardless of underlying infrastructure. This flexibility allows you to run multiple models or multiple instances of the same model on the same hardware, improving resource utilization.
* **Hardware and deployment versatility**: Triton is optimized for both GPU and CPU-based environments, allowing deployment on a variety of hardware. You can use Triton Inference Server in the cloud, in data centers, or on edge devices.
* **Performance optimization**: Triton Inference Server enhances inference performance through dynamic batching, which aggregates multiple inference requests into batches to optimize processing and enable concurrent model execution. This capacity allows multiple models to run simultaneously, which is important for real-time applications that require minimal latency.

This blog is part of a series on deploying AI models with Triton Inference Server on AMD GPUs. In a previous blog, [Triton Inference Server with vLLM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton_server_vllm/README.html), you can find a step-by-step guide for serving large language models using the vLLM backend. More recently, [Serving CTR Recommendation Models with Triton Inference Server using the ONNX Runtime Backend](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html) introduced the ONNX Runtime and Python backends along with performance benchmarks comparing AMD Instinct MI355X and NVIDIA B200 GPUs.

In this blog, you will learn how to set up a Triton Inference Server with the [ONNX Runtime](https://onnxruntime.ai/) backend on AMD GPUs, using ROCm and [MIGraphX](https://github.com/ROCm/AMDMIGraphX/) as the graph optimization accelerator. You will walk through the process of building the server from source, configuring the model repository, running inference, and benchmarking performance with the ResNet50-v2 image classification model from the [ONNX Model Zoo](https://github.com/onnx/models).

## Requirements

* **AMD GPU**: See the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems. This blog was tested on a machine with AMD Instinct MI300X GPUs.
* **ROCm 7.2+**: See the [ROCm installation guide for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) for installation instructions.
* **Docker**: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) for installation instructions.
* **ONNX models**: This blog uses ResNet50-v2 from the [ONNX Model Zoo](https://github.com/onnx/models). Instructions for downloading the model are provided in the [Setting up the model repository](#setting-up-the-model-repository) section.

## Triton Inference Server: ONNX Runtime backend

A Triton Inference Server backend is the component responsible for executing an AI model during inference. Each backend wraps a specific machine learning framework such as [ONNX Runtime](https://onnxruntime.ai/), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), or others. Each backend is implemented as a shared library and models are configured to use a specific backend through the model configuration file (`config.pbtxt`). For a list of all supported backends, see [Where can I find all the backends that are available for Triton?](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton). This blog focuses on the ONNX Runtime backend.

When you use ONNX Runtime as the backend on AMD GPUs, the inference pipeline relies on an execution provider (also called an execution accelerator) to bridge the gap between the framework and the hardware. An execution provider is a specialized software interface that translates a model's computational graph into optimized instructions for a specific hardware architecture. On AMD Instinct GPUs, this role is filled by [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/), AMD's graph inference engine, which compiles and optimizes model graphs for high-performance execution. MIGraphX acts as a compiler that takes your generic ONNX model and produces a highly optimized set of instructions tailored for AMD hardware, similar to how a C compiler optimizes source code for a particular CPU architecture.

The inference pipeline works as follows:

1. **Triton Inference Server** receives the inference request and routes it to the appropriate model.
2. **ONNX Runtime backend** loads and manages the ONNX model.
3. **MIGraphX execution provider** compiles and optimizes the model graph for AMD GPU hardware.
4. **AMD GPU** executes the optimized computation.

Here are some key features of the ONNX Runtime backend with MIGraphX on AMD GPUs:

* **MIGraphX integration**: When you enable the MIGraphX execution accelerator in the model configuration `config.pbtxt`, ONNX Runtime offloads graph compilation and execution to MIGraphX. This provides optimized kernel selection and memory management tailored for AMD Instinct GPUs.
* **Model caching**: MIGraphX compiles model graphs at load time, which can take several minutes for large models. You can cache the compiled models to disk using the `ORT_MIGRAPHX_MODEL_CACHE_PATH` environment variable. With model caching, subsequent server restarts will be much faster as no model compilation is needed.
* **Dynamic batching**: Triton supports dynamic batching, which groups incoming inference requests into batches of optimal sizes to improve GPU utilization and throughput. You can configure this through the `dynamic_batching` section in `config.pbtxt`.

```{note}
The `config.pbtxt` file is the model configuration file for Triton Inference Server, and you need one of these files for every model you deploy. It is written in [Protocol Buffers text format](https://protobuf.dev/reference/protobuf/textformat-spec/) and lives alongside the model file in the model repository. It defines the backend, input/output tensors, batching strategy, and optimization settings for each model. You will learn how to set up your model repository and write these configuration files in the sections below.
```

The ROCm-enabled Triton Inference Server and its components are maintained across several repositories. All components listed below are based on [**Triton Inference Server r25.12**](https://github.com/ROCm/triton-inference-server-server#rocm-enabled-repository-branches):

| Component | Repository | Branch |
|-----------|------------|--------|
| Server | [ROCm/triton-inference-server-server](https://github.com/ROCm/triton-inference-server-server) | `rocm7.2_r25.12` |
| Core | [ROCm/triton-inference-server-core](https://github.com/ROCm/triton-inference-server-core) | `rocm7.2_r25.12` |
| Backend | [ROCm/triton-inference-server-backend](https://github.com/ROCm/triton-inference-server-backend) | `rocm7.2_r25.12` |
| Third Party | [ROCm/triton-inference-server-third_party](https://github.com/ROCm/triton-inference-server-third_party) | `rocm7.2_r25.12` |
| ONNX Runtime Backend | [ROCm/triton-inference-server-onnxruntime_backend](https://github.com/ROCm/triton-inference-server-onnxruntime_backend) | `rocm7.2_r25.12` |
| ONNX Runtime | [ROCm/onnxruntime](https://github.com/ROCm/onnxruntime) | See build defaults |
| MIGraphX | [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | `develop` |

## Building Triton Inference Server with ONNX Runtime backend

To deploy ONNX models on AMD GPUs, you need to build a Triton Inference Server Docker image with ROCm support and the ONNX Runtime backend. This section provides build instructions for Ubuntu 24.04 and Debian 12. The build process has two steps: building a base Docker image with the ROCm stack, and then building the Triton Inference Server image on top of it.

### On Ubuntu 24.04

#### Prerequisites (Ubuntu 24.04)

* Docker installed and running
* AMD GPU with ROCm support
* ROCm 7.2 or compatible version installed on the host

**Step 1**: Build the base Docker image with Ubuntu 24.04 and ROCm 7.2.

Clone the ROCm Triton server repository and run the base image build script:

```bash
git clone -b rocm7.2_r25.12 https://github.com/ROCm/triton-inference-server-server.git
cd triton-inference-server-server
bash scripts/build_ubuntu24.04_rocm_72_base.sh
```

**Step 2**: Build the Triton server Docker image.

From the same directory, run the build script:

```bash
python3 build.py \
  --no-container-pull \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-rocm \
  --enable-metrics \
  --verbose \
  --endpoint=grpc \
  --endpoint=http \
  --backend=onnxruntime \
  --linux-distro=ubuntu
```

### On Debian 12

#### Prerequisites (Debian 12)

* Docker installed and running
* AMD GPU with ROCm support
* ROCm 7.2 or compatible version installed on the host

**Step 1**: Build the base Docker image with Debian 12 and ROCm 7.2.

Clone the ROCm Triton server repository and run the base image build script:

```bash
git clone -b rocm7.2_r25.12 https://github.com/ROCm/triton-inference-server-server.git
cd triton-inference-server-server
bash scripts/build_debian12_rocm_72_base.sh
```

**Step 2**: Build the Triton server Docker image.

From the same directory, run the build script:

```bash
python3 build.py \
  --no-container-pull \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-rocm \
  --enable-metrics \
  --verbose \
  --endpoint=grpc \
  --endpoint=http \
  --backend=onnxruntime \
  --linux-distro=debian
```

### Build options explained

The following table summarizes the key build flags:

| Flag | Description |
|------|-------------|
| `--enable-rocm` | Enable ROCm support for AMD GPUs |
| `--endpoint=grpc --endpoint=http` | Enable both HTTP and gRPC inference protocols |
| `--backend=onnxruntime` | Include the ONNX Runtime backend with MIGraphX |
| `--linux-distro` | Target Linux distribution (`ubuntu` or `debian`) |

When you enable ROCm, the build automatically includes MIGraphX as the execution accelerator for the ONNX Runtime backend. The build compiles ONNX Runtime from the [ROCm/onnxruntime](https://github.com/ROCm/onnxruntime) fork and MIGraphX from [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX).

To verify that the image was built successfully, run:

```bash
docker images | grep tritonserver
```

You should see output similar to:

```bash
REPOSITORY     TAG       IMAGE ID       CREATED         SIZE
tritonserver   latest    fffefb8a8258   2 hours ago     XX.XGB
```

## Setting up the model repository

A model repository is a directory structure that contains the models Triton Inference Server will serve. Each model is organized in a specific layout that Triton Inference Server scans and loads at startup. Similar to a library's shelving system: each model has its own shelf (directory), with labeled versions and a catalog card (configuration file) describing its properties.

The structure of the model repository is as follows:

```bash
model_repository/
    ├── resnet50_onnx/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.onnx
```

Each model directory contains:

* A **version directory** (`1/`) that holds the actual model file (`model.onnx`). Triton supports multiple versions, so you could have `1/`, `2/`, etc.
* A **configuration file** (`config.pbtxt`) that defines the backend, input/output tensor names, shapes, data types, batching settings, and GPU optimization parameters.

### Downloading the model

The following commands create the model repository directory structure and download the ResNet50-v2 ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models):

```bash
mkdir -p model_repository/resnet50_onnx/1

# Download ResNet50-v2
wget -O model_repository/resnet50_onnx/1/model.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
```

```{note}
The ONNX Model Zoo repository structure may change over time. If the download URL above does not work, visit the [ONNX Model Zoo](https://github.com/onnx/models) directly and navigate to the ResNet50 model page.
```

### Model configuration

Each model needs a `config.pbtxt` file that tells Triton Inference Server how to load and serve it. When using the ONNX Runtime backend on AMD GPUs, the configuration must specify `onnxruntime` as the backend and include the MIGraphX execution accelerator in the `optimization` block.

Create the configuration file for ResNet50-v2 at `model_repository/resnet50_onnx/config.pbtxt`:

```text
name: "resnet50_onnx"
backend: "onnxruntime"
max_batch_size: 8

dynamic_batching {
    max_queue_delay_microseconds: 100
}

input [
    {
        name: "data"
        data_type: TYPE_FP32
        dims: [3, 224, 224]
    }
]

output [
    {
        name: "resnetv24_dense0_fwd"
        data_type: TYPE_FP32
        dims: [1000]
    }
]

instance_group [
    {
        kind: KIND_GPU
        count: 1
        gpus: [0]
    }
]

optimization {
    execution_accelerators {
        gpu_execution_accelerator {
            name: "migraphx"
            parameters: {
                key: "migraphx_max_dynamic_batch"
                value: "8"
            }
        }
    }
}
```

### Understanding the key configuration parameters

The following table explains the key parameters used in the configuration files:

| Parameter | Description |
|-----------|-------------|
| `backend: "onnxruntime"` | Tells Triton to use the ONNX Runtime backend for this model |
| `max_batch_size` | Maximum batch size that Triton can combine through dynamic batching |
| `dynamic_batching` | Enables Triton to group incoming requests into batches for better GPU utilization |
| `instance_group` | Defines where to run the model (`KIND_GPU` or `KIND_CPU`) and on which GPU(s) |
| `execution_accelerators` | Specifies MIGraphX as the GPU execution accelerator for graph optimization |
| `migraphx_max_dynamic_batch` | Maximum batch size MIGraphX should pre-compile for dynamic batching |

## Running the Triton Inference Server

With the model repository set up, you can start the Triton Inference Server container.  Start the container with the `docker run` command below, which exposes the AMD GPU devices, sets MIGraphX optimization environment variables, and mounts the model repository.

```bash
docker run -it --rm \
  --name tritonserver_container \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --net=host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/cache/ \
  -e ORT_MIGRAPHX_CACHE_PATH=/cache/ \
  -e ORT_MIGRAPHX_COMPILE_BATCHES="1,2,4,8" \
  -v $(pwd)/model_repository:/models \
  -v $(pwd)/migraphx_cache:/cache \
  tritonserver \
  tritonserver --model-repository=/models --exit-on-error=false
```

### Important parameters

The following table explains the key Docker and environment variable parameters used in the `docker run` command:

| Parameter | Description |
|-----------|-------------|
| `--device=/dev/kfd --device=/dev/dri` | Grant access to AMD GPU devices inside the container. |
| `--ipc=host --net=host` | Share host IPC namespace and network stack for GPU inter-process communication and direct port access. |
| `-p 8000:8000 -p 8001:8001 -p 8002:8002` | Expose HTTP (8000), gRPC (8001), and metrics (8002) ports. |
| `-v $(pwd)/model_repository:/models` | Mount your model repository inside the container. |
| `-v $(pwd)/migraphx_cache:/cache` | Mount a directory for caching compiled MIGraphX models. This path must be outside the model repository (`/models`) so Triton does not try to load the cache as a model. |
| `ORT_MIGRAPHX_MODEL_CACHE_PATH` | Directory for caching compiled MIGraphX models. On subsequent restarts, the server loads from cache instead of recompiling. |
| `ORT_MIGRAPHX_CACHE_PATH` | Alternative cache path variable used by some ONNX Runtime versions. Set to the same value as `ORT_MIGRAPHX_MODEL_CACHE_PATH`. |
| `ORT_MIGRAPHX_COMPILE_BATCHES` | Comma-separated list of batch sizes to pre-compile (e.g., `"1,2,4,8"`). Pre-compiling avoids recompilation when different batch sizes arrive at runtime. |
| `--model-repository=/models` | Tells Triton where to find the model directories inside the container. |
| `--exit-on-error=false` | Keeps the server running even if some models fail to load, allowing healthy models to serve requests. |

```{note}
The first time you start the server, MIGraphX will compile the model graphs for each batch size listed in `ORT_MIGRAPHX_COMPILE_BATCHES`. This can take several minutes depending on the model size and the number of batch sizes. Subsequent starts will load from the cache directory, making startup much faster.
```

When the server is ready, you will see output similar to:

```console
I0403 12:00:00.000000 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I0403 12:00:00.000001 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I0403 12:00:00.000002 1 http_server.cc:315] Started Metrics Service at 0.0.0.0:8002
```

## Performing inference with ResNet50-v2

With the Triton Inference Server running, you can now send inference requests to the ResNet50-v2 model. This section shows you how to set up a Python environment, install the Triton client library, and run a sample inference script.

### Setting up the Python environment

Create a Python virtual environment and install the Triton client library with its dependencies:

```bash
python3 -m venv triton-client-env
source triton-client-env/bin/activate
pip install tritonclient[http] numpy Pillow
```

### Inference with ResNet50-v2

The following Python script sends an inference request to the ResNet50-v2 model. For demonstration purposes, this example uses random input data. In a real application, you would preprocess an actual image (resize to 224x224, normalize with ImageNet mean and standard deviation, and convert to CHW (Channel-Height-Width tensor format)).

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

inputs = [httpclient.InferInput("data", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("resnetv24_dense0_fwd")]

result = client.infer("resnet50_onnx", inputs, outputs=outputs)

output_data = result.as_numpy("resnetv24_dense0_fwd")
print(f"Model: resnet50_onnx")
print(f"Output shape: {output_data.shape}")
print(f"Top-5 class indices: {np.argsort(output_data.flatten())[-5:][::-1]}")
```

Running this script produces output similar to:

```console
Model: resnet50_onnx
Output shape: (1, 1000)
Top-5 class indices: [490 904 794 556 599]
```

```{note}
The class indices shown above correspond to ImageNet class labels (e.g., 504 = coffee mug, 968 = cup).  Since the input image is random, the model output does not correspond to any meaningful classification result and is provided for illustration purposes only.
```

## Measuring performance with Performance Analyzer

Triton Inference Server includes a tool called [Performance Analyzer](https://github.com/triton-inference-server/perf_analyzer) (`perf_analyzer`) that measures inference throughput and latency. It works by sending a sustained stream of requests to the server and collecting timing statistics over a measurement window.

`perf_analyzer` is available in the official Triton SDK container image (`nvcr.io/nvidia/tritonserver:25.12-py3-sdk`). Since it only sends requests over HTTP or gRPC, it does not require GPU access. It just needs network connectivity to the running Triton Inference Server.

Run the following command to benchmark the ResNet50-v2 model with a stream of requests using a batch size of 1 and a concurrency level of 1 (one request at a time):

```bash
docker run --rm --network=host \
  nvcr.io/nvidia/tritonserver:25.12-py3-sdk \
  perf_analyzer \
  -u localhost:8000 \
  -m resnet50_onnx \
  -i http \
  --input-data random \
  --measurement-interval=30000 \
  --concurrency-range 1 \
  -b 1
```

The key flags are:

| Flag | Description |
|------|-------------|
| `-u localhost:8000` | Triton server URL. |
| `-m resnet50_onnx` | Model name to benchmark. |
| `-i http` | Protocol to use (`http` or `grpc`). |
| `--input-data random` | Generate random input tensors matching the model's expected shape. |
| `--measurement-interval=30000` | Collect measurements over a 30-second window for stable results. |
| `--concurrency-range 1` | Number of concurrent client requests in flight. |
| `-b 1` | Batch size per request. |

After the measurement window completes, `perf_analyzer` prints a report similar in format to the example below. The exact numbers depend on the hardware, driver and ROCm versions, container build, model build, client/server placement, and overall system load. Treat the values shown only as a guide to interpreting the report layout, not as performance claims.

```console
*** Measurement Settings ***
  Batch size: 1
  Service Kind: TRITON
  Using "time_windows" mode for stabilization
  Stabilizing using average latency and throughput
  Measurement window: 30000 msec
  Using synchronous calls for inference

Request concurrency: 1
  Client:
    Request count: 40063
    Throughput: 336.724 infer/sec
    Avg latency: 2614 usec (standard deviation 536 usec)
    p50 latency: 2628 usec
    p90 latency: 3089 usec
    p95 latency: 3131 usec
    p99 latency: 3270 usec
    Avg HTTP time: 2610 usec (send/recv 189 usec + response wait 2421 usec)
  Server:
    Inference count: 40064
    Execution count: 40064
    Successful request count: 40064
    Avg request latency: 1718 usec (overhead 19 usec + queue 184 usec + compute input 72 usec + compute infer 1431 usec + compute output 10 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 336.724 infer/sec, latency 2614 usec
```

```{note}
The throughput and latency values shown above are illustrative output from a single, untuned `perf_analyzer` run and are included solely to explain the structure of the report. They are not official benchmark results, are not intended for hardware or software comparisons, and should not be cited as performance claims. Your own results will vary with hardware, ROCm and driver versions, container and model build, configuration, and system conditions.
```

The output is split into two sections:

* **Client**: Metrics as seen from the client side. **Throughput** is the number of inferences per second the server sustained. The **latency percentiles** (p50, p90, p95, p99) show how long individual requests took end-to-end, including network overhead.
* **Server**: Metrics reported by the Triton server itself. The **Avg request latency** breakdown shows where time is spent: queue wait, input preparation, actual model computation (`compute infer`), and output formatting. This breakdown is useful for identifying bottlenecks. For example, a large `queue` time suggests the server is saturated and could benefit from higher concurrency or more model instances.

You can experiment with different batch sizes (`-b 4`, `-b 8`) and concurrency levels (`--concurrency-range 1:4`) to find the configuration that maximizes throughput for your workload.

## Summary

In this blog, you walked through the deployment and serving of the ResNet50-v2 ONNX model using Triton Inference Server with the ONNX Runtime backend, powered by AMD Instinct MI300X GPUs and the ROCm software platform. You learned how to:

* Build a Triton Inference Server Docker image with ROCm and MIGraphX support on both Ubuntu 24.04 and Debian 12.
* Set up a model repository with ONNX models and configure them with MIGraphX as the execution accelerator.
* Start the Triton Inference Server with AMD GPU device access and MIGraphX optimization environment variables.
* Send inference requests using the Python `tritonclient` library.
* Use `perf_analyzer` to measure model throughput and latency across different batch sizes and concurrency levels.

By using MIGraphX as the graph optimization engine within the ONNX Runtime backend, you can take full advantage of AMD Instinct GPU hardware for high-throughput, low-latency inference of ONNX models.

To explore other Triton Inference Server backends on AMD GPUs, see these related posts:

* [Triton Inference Server with vLLM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton_server_vllm/README.html): A step-by-step guide to serving large language models (Phi-2, Mistral, Llama-3) using the vLLM backend on AMD Instinct GPUs.
* [Serving CTR Recommendation Models with Triton Inference Server using the ONNX Runtime Backend](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html): An overview of the ONNX Runtime and Python backend support, the version 25.12 upgrade, and performance benchmarks on AMD Instinct MI355X.

## Acknowledgements

Thank you to the broader AMD team whose contributions made this work possible: Eliot Li, Ted Maeurer, David Rooney, Yao Liu, Christopher Austen, Khalique Ahmed, Titir Santra, Vish Vadlamani.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
