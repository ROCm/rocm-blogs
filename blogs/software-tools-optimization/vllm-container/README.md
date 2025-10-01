---
blogpost: true
blog_title: 'How to Build a vLLM Container for Inference and Benchmarking'
thumbnail: '2025-02-20-vllm-container.png'
target_audience: 'Data center administrators'
key_value_propositions: 'This post, the second in a series, provides a walkthrough for building a vLLM container that can be used for both inference and benchmarking.'
date: 21 Feb 2025
author: "Matt Elliott"
tags: LLM, AI/ML, GenAI
category: Software tools & optimizations
language: English
myst:
    html_meta:
        "author": "Matt Elliott"
        "description lang=en": "This post, the second in a series, provides a walkthrough for building a vLLM container that can be used for both inference and benchmarking."
        "keywords": "AI/ML"
        "property=og:locale": "en_US"
        "amd_category": 'Developer Resources'
        "amd_asset_type": 'Blogs'
        "amd_blog_type": 'Technical Articles & Blogs'
        "amd_technical_blog_type": 'Benchmarks and Testing'
        "amd_developer_type": 'ML/AI Developer'
        "amd_deployment": 'Servers'
        "amd_blog_hardware_platforms": 'Instinct GPUs'
        "amd_blog_development_tools": 'ROCm Software'
        "amd_blog_applications": 'AI Inference, AI Training'
        "amd_blog_topic_categories": 'Software & Ecosystem'
        "amd_blog_releasedate": Fri Feb 21, 12:00:00 PST 2025
---

# How to Build a vLLM Container for Inference and Benchmarking

Welcome back! If you've been following along with this [series](https://rocm.blogs.amd.com/software-tools-optimization/rocm-containers/README.html), you've already learned about the basics of ROCm containers. Today, we'll build on that foundation by creating a container for large language model inference with [vLLM](https://docs.vllm.ai/).

In my [last post](https://rocm.blogs.amd.com/software-tools-optimization/rocm-containers/README.html), I introduced the AMD ROCm™ container ecosystem, and demonstrated how to extend AMD-provided containers for custom use cases. In this post, I’ll provide a walkthrough for building a vLLM container that can be used for both inference and benchmarking. This enables a straightforward process to ensure you can validate performance before serving models in production.

Before getting started, make sure your system meets these minimum requirements:

- **ROCm**: Version 6.3.0 or later
- **GPU**: AMD Instinct™ MI300X accelerator or [other ROCm-supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- **Docker**: Engine 20.10 or later with buildx support
- **Python**: Version 3.8 or later

```{note}
While these are the minimum requirements, we recommend using the latest stable versions of ROCm and Docker for optimal performance and compatibility.
```

## Building the container

When it comes to deploying large language models for inference, I've found vLLM to be a solid choice for high-performance inference on AMD GPUs. What I love about this next example is its versatility - you can use it for both benchmarking and inference. Here's an example Dockerfile to start with:

```dockerfile
FROM rocm/vllm-dev:main

# Install basic development tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    safetensors

# Create non-root user for security
RUN useradd -m -u 2000 vllm
WORKDIR /app
RUN chown vllm:vllm /app

# Create directories for models and benchmarks
RUN mkdir -p /data/benchmarks && \
   chmod 777 /data/benchmarks

# Switch to non-root user
USER vllm

# Make our entrypoint script executable
COPY --chown=vllm:vllm entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
```

### Container entrypoint

A container entrypoint is a script or command that runs when the container starts. Think of it as the container's main function - it's the first thing that executes and controls how your container behaves. In our case, we want the container to be flexible enough to handle both inference and benchmarking tasks, which is why we've created this entrypoint script. Save this as `entrypoint.sh` in the same folder as the `dockerfile`:

```bash
#!/bin/bash

# Base configuration with defaults
MODE=${MODE:-"serve"}
MODEL=${MODEL:-"amd/Llama-3.2-1B-FP8-KV"}
PORT=${PORT:-3000}

# Benchmark configuration with defaults
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-256}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
NUM_ROUNDS=${NUM_ROUNDS:-3}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-8192}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}

# Additional args passed directly to vLLM
EXTRA_ARGS=${EXTRA_ARGS:-""}

case $MODE in
  "serve")
    echo "Starting vLLM server on port $PORT with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"
    python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL \
      --port $PORT \
      $EXTRA_ARGS
    ;;
    
  "benchmark")
    echo "Running vLLM benchmarks with model: $MODEL"
    echo "Additional arguments: $EXTRA_ARGS"
    
    # Create timestamped directory for this benchmark run
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BENCHMARK_DIR="/data/benchmarks/$TIMESTAMP"
    mkdir -p "$BENCHMARK_DIR"
    
    # Throughput benchmark
    echo "Running throughput benchmark..."
    python3 /app/vllm/benchmarks/benchmark_throughput.py \
      --model $MODEL \
      --input-len $INPUT_LEN \
      --output-len $OUTPUT_LEN \
      --num-prompts $NUM_PROMPTS \
      --max-num-batched-tokens $MAX_BATCH_TOKENS \
      --output-json "$BENCHMARK_DIR/throughput.json" \
      $EXTRA_ARGS
    echo "Throughput benchmark complete - results saved in $BENCHMARK_DIR/throughput.json"
    
    # Latency benchmark
    echo "Running latency benchmark..."    
    python3 /app/vllm/benchmarks/benchmark_latency.py \
      --model $MODEL \
      --input-len $INPUT_LEN \
      --output-len $OUTPUT_LEN \
      --output-json "$BENCHMARK_DIR/latency.json" \
      $EXTRA_ARGS
    echo "Latency benchmark complete - results saved in $BENCHMARK_DIR/latency.json"

    echo "All results have been saved to $BENCHMARK_DIR"
    ;;
    
  *)
    echo "Unknown mode: $MODE"
    echo "Please use 'serve' or 'benchmark'"
    exit 1
    ;;
esac
```

```{note}
The examples in this post use [amd/Llama-3.2-1B-FP8-KV](https://huggingface.co/amd/Llama-3.2-1B-FP8-KV), a quantized version of Meta's Llama 3.2 1B model. You can find thousands of additional models to experiment with at [HuggingFace](https://huggingface.co/).
```

This script makes our container flexible by using environment variables to control its behavior. The most important is `MODE`, which lets you switch between two main functions:

- `MODE=serve`: Starts an OpenAI-compatible API server for model inference
  - Set `MODEL` to specify which model to load
  - Use `PORT` to control which port the API listens on
  - Pass additional vLLM arguments through `EXTRA_ARGS`

- `MODE=benchmark`: Runs performance tests and saves detailed metrics
  - Configure test parameters like `INPUT_LEN` and `OUTPUT_LEN`
  - Results are automatically saved with timestamps
  - Both throughput and latency tests are included

For example, to run inference:

```bash
docker run -e MODE=serve -e MODEL="your-model" your-image
```

Or to benchmark:

```bash
docker run -e MODE=benchmark -e INPUT_LEN=1024 your-image
```

Now we're ready to build the container:

```bash
# Build the container
docker buildx build -t vllm-toolkit .

# Optionally tag with version
docker tag vllm-toolkit vllm-toolkit:main
```

For teams working together, I recommend pushing the image to a local container registry. This ensures everyone uses the same container version and reduces build time across the team:

```bash
# Tag for your registry
docker tag vllm-toolkit localhost:5000/vllm-toolkit:main

# Push to local registry
docker push localhost:5000/vllm-toolkit:main
```

```{note}
You can find more information about deploying a local container registry here: [https://hub.docker.com/_/registry](https://hub.docker.com/_/registry)
```

Here's a screen recording of the container build process:

<div id="vllm-container-1"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/vllm-container-1.cast', document.getElementById('vllm-container-1'));
</script>

### Available vLLM Containers

AMD provides two main vLLM container options:

- [rocm/vllm](https://hub.docker.com/r/rocm/vllm): Production-ready container
  - Pinned to a specific version, for example: `rocm/vllm-dev:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6`
  - Designed for stability
  - Optimized for deployment

- [rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev): Development container with the latest vLLM features
  - `nightly`, `main` and other specialized builds available:
    - `nightly` tags are built daily from the latest code, but may contain bugs
    - `main` tags are more stable builds, updated after testing
  - Includes development tools
  - Best for testing new features or custom modifications

When building your own container, you can choose either as a base image depending on your needs. For example, to use the `rocm/vllm` container:

```dockerfile
FROM rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6
// ...rest of Dockerfile remains the same...
```

## Container usage

With our container built and ready to go, let's explore how to put it to work. We'll look at two main use cases: running inference with your models and benchmarking their performance. Whether you're serving models in production or testing different configurations, these examples will show you how to get the most out of your vLLM container.

### Creating a docker run alias

Let's start with a helpful time-saving tip. Running ROCm containers requires a few command-line arguments to enable GPU access and proper permissions. We can create a simple alias to streamline this process:

```bash
# Add to your ~/.bashrc or ~/.zshrc
alias rdr='docker run -it --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=$(getent group video | cut -d: -f3) \
    --group-add=$(getent group render | cut -d: -f3) \
    --ipc=host \
    --security-opt seccomp=unconfined'

# Reload your shell config to enable the alias
source ~/.bashrc  # or source ~/.zshrc
```

This alias, `rdr` (ROCm Docker Run), encapsulates all the standard arguments needed for GPU-enabled containers. Now instead of typing out the full command, you can simply use:

```bash
rdr -v /path/to/your/models:/home/vllm/.cache/huggingface \
    -e MODE="serve" \
    -e MODEL="your-model" \
    -p 8000:8000 \
    your-vllm-image
```

The alias handles all the essential container configurations:

- GPU device access (`--device=/dev/kfd --device=/dev/dri`)
- Group membership (dynamically fetched)
- Shared memory settings (`--ipc=host`)
- Security configurations (`--security-opt seccomp=unconfined`)
- Common docker flags (`-it --rm`)

```{note}
If you prefer a more permanent solution, you could create a shell script in `/usr/local/bin/rdr` with these commands. This would make the functionality available system-wide without requiring alias setup.
```

### Container security and GPU access

You may be wondering about the `--group-add` commands in the `rdr` alias we defined. These arguments allow the container to run as a non-root user. It's a container security best practice to run as a non-root user - that's why our Dockerfile creates a dedicated `vllm` user:

```dockerfile
# Create non-root user for security
RUN useradd -m -u 2000 vllm
WORKDIR /app
RUN chown vllm:vllm /app
```

However, running as a non-root user creates a challenge: how do we ensure our container can still access the GPU? On Linux systems, GPU access is managed through group permissions. Specifically, for AMD GPUs, we need our user to be in both the `video` and `render` groups. We could hardcode the group IDs, but that's not very portable - these IDs can vary across different systems. Instead, I prefer to detect them dynmically by using the `getent` command in the custom `rdr` alias. This neat little trick looks up the system's group IDs when we launch our container, granting the necessary group permissions for GPU access. This approach provides advantages:

1. **Better Security**: We're following the principle of least privilege - our container only gets the permissions it actually needs
2. **Easy Portability**: The container will work across different systems, regardless of their specific group IDs
3. **Clear Intent**: Anyone reading our deployment scripts can easily understand what system access we require

By combining proper group permissions with these flags, we've created a setup that's both secure and functional - our non-root container can access the GPU while following security best practices.

## Inference

The `vllm-toolkit` container exposes an OpenAI-compatible API endpoint, making it easy to integrate with existing tools and workflows.

```bash
rdr -v /data/hf_home:/home/vllm/.cache/huggingface \
    -e MODE="serve" \
    -e MODEL="amd/Llama-3.2-1B-FP8-KV" \
    -p 8000:8000 \
    your-vllm-image
```

This command mounts the model directory, exposes port 8000 for API access, and runs the container in serve mode. When running in serve mode, you can interact with the model using the OpenAI client library, which provides a familiar interface for developers:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="example")

# Example chat completion
completion = client.chat.completions.create(
    model="amd/Llama-3.2-1B-FP8-KV",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement in simple terms"}
    ]
)
print(completion.choices[0].message.content)
```

The client code demonstrates one of vLLM's key features - a consistent OpenAI-compatible API interface regardless of which model you're using. Whether you're running Llama, Mistral, or any other supported model, your application code remains exactly the same. Here's what this looks like in action:

<div id="vllm-container-2"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/vllm-container-2.cast', document.getElementById('vllm-container-2'));
</script>

## Benchmarking

Testing your model's performance helps you understand its resource requirements and performance characteristics. The benchmark mode runs a series of standard tests that measure:

- Tokens per second
- Memory usage patterns
- Response latency distribution

Here's how to run the benchmark:

```bash
rdr -v /path/to/your/models:/data/models \
    -v /path/to/save/results:/data/benchmarks \
    -e MODE=benchmark \
    your-vllm-image
```

The benchmark results are saved to a timestamped directory in the folder mapped to `/data/benchmarks`. I find it helpful to run these tests with different batch sizes and model configurations to find the optimal settings for your specific use case.

### Customizing Benchmark Parameters

Passing in environment variables allows you to customize the benchmarking process. For example:

```bash
rdr -e MODE=benchmark \
    -e INPUT_LEN=1024 \
    -e OUTPUT_LEN=512 \
    -e NUM_PROMPTS=500 \
    -e EXTRA_ARGS="--dtype bfloat16" \
    vllm-toolkit:main
```

 The entrypoint script accepts variables that let you adjust these parameters:

- `INPUT_LEN`: How long should each test prompt be? (default: 512 tokens)
- `OUTPUT_LEN`: How much text should the model generate? (default: 256 tokens)
- `NUM_PROMPTS`: How many test cases to run (default: 1000)
- `EXTRA_ARGS`: Any additional vLLM parameters you want to experiment with

I've chosen these defaults to give you meaningful results in a reasonable amount of time. But don't hesitate to adjust them for your specific needs! For instance, you might want to use longer `INPUT_LEN` and `OUTPUT_LEN` if you're working with document processing. You can also pass additional vLLM arguments via `EXTRA_ARGS`. For example:

| Parameter                | Description                                  | Example                   |
|--------------------------|----------------------------------------------|---------------------------|
| `--dtype`                | Set data type (fp16, bf16, fp8)              | `--dtype bf16`            |
| `--max-batch-tokens`     | Max tokens in a single batch                 | `--max-batch-tokens 4096` |
| `--tensor-parallel-size` | Set tensor parallelism level                 | `-tp 4`                   |
| `--gpu-memory-util`      | Target GPU memory utilization (0.0 - 1.0)    | `--gpu-memory-util 0.9`   |

This short screen capture shows a full benchmark run:

<div id="vllm-container-3"></div>
<script>
  AsciinemaPlayer.create('../../_static/asciinema/vllm-container-3.cast', document.getElementById('vllm-container-3'));
</script>

#### Choosing the right benchmarking framework

While our vLLM container provides a straightforward way to get started with inference and basic benchmarking, it's worth understanding how it fits into the broader AMD ML ecosystem. The vLLM container we've built is an excellent starting point for getting hands-on experience with model deployment and performance testing. It's perfect for development and initial benchmarking, providing a simple way to validate your setup and run basic performance tests.

For more comprehensive benchmarking, AMD's [MAD (Model Automation and Dashboarding)](https://github.com/ROCm/MAD) framework offers a much more in-depth solution. MAD includes extensive benchmarking capabilities for vLLM, supporting a wide range of models (from Llama 3.1 to Mixtral, Mistral, and more) and providing detailed performance metrics. It's particularly valuable when you need:

- Standardized benchmark reporting
- Support for multiple model architectures
- Automated testing across different configurations
- Detailed throughput and latency measurements
- FP16 and FP8 performance comparisons

For users with training needs, the [rocm/pytorch-training](https://hub.docker.com/r/rocm/pytorch-training) container provides your foundational ML development environment, equipped with PyTorch and ROCm tools for general ML work.

In practice, you might start with our simple vLLM container for initial testing and development, then move to MAD for thorough performance validation before production deployment. Each tool serves a specific purpose in your ML workflow, with MAD being the go-to solution for serious performance benchmarking and validation.

## Summary

Over these two blog posts, we've explored the full spectrum of ROCm container development - from basic development environments to specialized AI inference solutions. We [started](https://rocm.blogs.amd.com/software-tools-optimization/rocm-containers/README.html) with the fundamentals of ROCm containers, learning how to build custom development environments and ML training setups. Now, we've advanced to creating a production-ready vLLM container that handles both inference and benchmarking.

This two-part series demonstrates the flexibility of the ROCm container ecosystem. Whether you're just starting with GPU development or deploying large language models in production, these containerized solutions provide a solid foundation. They combine the portability and reproducibility of containers with the performance of AMD GPUs, giving you the tools you need to build and deploy GPU-accelerated applications efficiently. The patterns we've explored - from basic development containers to sophisticated inference solutions - can be adapted for your specific needs.

Ready to take your LLM inference to the next level? I recommend checking out our comprehensive guide on [Best practices for competitive inference optimization on AMD Instinct™ MI300X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html). For a deep dive into vLLM's capabilities, our four-part series on [vLLM x AMD: Efficient LLM Inference](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus-part1.html) explores advanced optimization techniques.

### Additional Resources

For more information about working with vLLM and AMD GPUs:

- [Triton Inference Server with vLLM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton_server_vllm/README.html)
- [Using AMD ROCm™ vLLM Docker Image with AMD Instinct™ MI300X](https://www.amd.com/en/developer/resources/technical-articles/how-to-use-prebuilt-amd-rocm-vllm-docker-image-with-amd-instinct-mi300x-accelerators.html)
- vLLM x AMD Series:
  - [Part 1: Prefix Caching](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus-part1.html)
  - [Part 2: Chunked Prefill](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus-part2.html)
  - [Part 3: Speculative Decoding](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus-part3.html)
  - [Part 4: vLLM Deployment Walkthrough](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus-part4.html)

<p style="font-size: 14px;">
Disclaimers<br>
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
</p>
