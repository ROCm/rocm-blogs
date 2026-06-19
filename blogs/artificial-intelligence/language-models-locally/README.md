---
blogpost: true
blog_title: "A Practical Guide to Running LLMs on AMD Radeon™ GPUs"
date: 19 Jun 2026
author: 'Hisham Chowdhury, Owen Zhang'
thumbnail: 'Running-LLMs-on-AMD-Radeon-GPUs-Thumbnail.png'
tags: Partner Applications, AI/ML, GenAI, Installation, LLM
category: Applications & models
target_audience: AI developers and enthusiasts
key_value_propositions: This blog shows you the different paths to get LLM on your system as an end-user using AMD Radeon GPUs
language: English
myst:
    html_meta:
        "author": "Hisham Chowdhury, Owen Zhang"
        "description lang=en": "This guide describes how to run LLMs on AMD Radeon™ GPUs using a range of partner frameworks, tools, and runtimes, with step-by-step setup instructions and performance optimization tips."
        "keywords": "LLM, Lemonade, LM Studio, Ollama, Llama.cpp, Setup, Optimization, Developer, End-user, AI"
        "vertical": "AI, Developers, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Hisham Chowdhury, Owen Zhang"
---

<!--
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
-->

<!-- markdownlint-disable MD036 -->

# A Practical Guide to Running LLMs on AMD Radeon™ GPUs

Running large language models on AMD Radeon™ GPUs has never been more accessible or more exciting. Thanks to rapid advancements in open‑source tooling and GPU acceleration, both Radeon™ integrated GPUs (iGPU) and discrete GPUs (dGPU) have become powerful, cost‑effective platforms for local AI. Whether you prefer a polished desktop application, a lightweight command‑line workflow, or a fully customizable runtime, a rich ecosystem of tools now makes it easy to deploy cutting‑edge models on your system. With today’s software stack, you can run state‑of‑the‑art language models directly on your Radeon™‑powered PC, whether you’re using integrated graphics or a high‑performance discrete card.

This guide describes how to run LLMs on AMD Radeon™ GPUs using a range of partner frameworks, tools, and runtimes. We’ll walk through step‑by‑step setup instructions for several popular LLM environments and show you how to configure them for optimal performance on Radeon™ hardware.

## Getting Things Ready

**LLM toolkit:**

- [Lemonade](https://lemonade-server.ai/) — a user‑friendly launcher built for seamless AMD acceleration. Supports GGUF and ONNX model formats, as well as CPU, GPU, and NPU execution.

- [LM Studio](https://lmstudio.ai/) — a desktop environment for downloading, serving, and interacting with models

- [Ollama](https://ollama.com/) — a simple command‑line and graphical user interface (GUI) tool with an extensive model library

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the low‑level, highly optimized foundation powering many local‑AI solutions. It also serves as the foundation for many higher-level LLM frameworks.

Each of the three approaches above offers unique strengths. From plug‑and‑play convenience to deep configurability, together they form a flexible toolkit for anyone looking to run fast, private, offline AI workloads on AMD GPUs.

Let’s dive in and explore what’s possible.

**Prerequisites**

- [Git](https://git-scm.com/install/)
- [Git LFS](https://git-lfs.com/)
- [Conda/Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Python 3.10 or later](https://www.python.org/downloads/)
- [CMake](https://cmake.org/download/)
- [Ninja](https://ninja-build.org/)
- [Clang](https://clang.llvm.org/)

**Hardware requirements**

- [AMD Radeon GPU](https://www.amd.com/en/products/graphics/desktops/radeon.html)

## Converting the Model

If you already have a **GGUF** model, proceed to the **Run LLMs on AMD Radeon™ GPUs Locally** section.

This section explains how to convert a **PyTorch** checkpoint into the **GGUF** format so it can be used by the frameworks covered in this guide.

Lemonade, LM Studio, Ollama, and llama.cpp all support GGUF models, making it a unified format for running LLMs efficiently across different tools.

**Note:**

If you plan to use standard or built‑in model downloads provided by these frameworks, this conversion step can be skipped.

1. **Set up the environment.**

```bash
conda create -n llm_convert python=3.12.4
conda activate llm_convert

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

2. **Download the models from Hugging Face using Git LFS or the Hugging Face CLI.**

**Git LFS**

```bash
git lfs install
git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct
```

**Hugging Face CLI**

```bash
pip install huggingface-hub
huggingface-cli download microsoft/Phi-3.5-mini-instruct --local-dir ./Phi-3.5-mini-instruct
```

3. **Format conversion to GGUF.**

```bash
python convert_hf_to_gguf.py ./Phi-3.5-mini-instruct --outfile phi-3.5-mini-instruct-fp16.gguf
```

**Note:** The conversion script automatically detects the model architecture. For most modern models (including Phi-3.5, Llama, Mistral, etc.), no special tokenizer handling is required.

## Run LLMs on AMD Radeon™ GPUs Locally

This section walks through running LLMs on AMD Radeon™ GPUs using a range of partner frameworks, tools, and runtimes, with step‑by‑step setup instructions and configuration tips for optimal performance on Radeon hardware.

- **Lemonade** — a user‑friendly launcher built for seamless AMD acceleration. Supports GGUF and ONNX model formats, as well as CPU, GPU, and NPU execution.

- **LM Studio** — a desktop environment for downloading, serving, and interacting with models

- **Ollama** — a simple command‑line and graphical user interface (GUI) tool with an extensive model library

- **llama.cpp** — the low‑level, highly optimized foundation powering many local‑AI solutions. This is also a foundation for many higher level LLM frameworks

### Lemonade

Lemonade's server application provides users with the ability to run GGUF and ONNX models under the same UI. It's built on key technologies including AMD ROCm™ software, Llama.cpp, and ONNXGenAI frameworks. It also provides access to both the GPU and NPU for running LLMs.

Follow the instructions at [https://lemonade-server.ai/](https://lemonade-server.ai/) to install the Lemonade server on your local system and run LLMs.

**Key features:**

- Text generation for natural language interfaces, agents, and tool driven workflows.
- Image generation for content creation and visual feedback loops inside applications.
- Speech-to-text and speech recognition for accessibility scenarios and hands-free interaction.

### LM Studio

Download LM Studio for your platform [here](https://lmstudio.ai/).

The standard model can be downloaded using LM Studio’s model manager.

To use a custom generated GGUF model with LM Studio:

1. Open the LM Studio app and go to the `model` folder.
2. Go to models > My Models.
3. Create the `lmstudio-community` folder if it doesn't already exist.
4. Create the `phi-3.5-mini` folder: `lmstudio-community/phi-3.5-mini/`.
5. Copy your `phi-3.5-mini-instruct-Q4_K_M.gguf` model into this folder.
6. Go back to My Models and refresh to ensure it appears.
7. Load the model in chat and start chatting.

### llama.cpp Command Line Tools

If the **Convert the model** section has been completed and llama.cpp is set up, navigate to your llama.cpp directory.

**Windows**

Clone the repository.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

**Linux**

Clone the repository.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Build with ROCm Backend (Recommended for Best Performance)

**Windows**

This step can be skipped by downloading prebuilt llama.cpp with ROCm backend binaries from [Llama.cpp+ROCm prebuilt binaries](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/windows/llm/llamacpp.html).

```bash
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**Note:** Replace `gfx1100` with your AMD GPU architecture:

- gfx1100: RDNA<sup>TM</sup> 3 GPU Architecture
- gfx115x: RDNA<sup>TM</sup> 3.5 GPU Architecture
- gfx12xx: RDNA<sup>TM</sup> 4 GPU Architecture

**Troubleshooting:**
If you encounter OpenMP-related linking issues, add `-DGGML_OPENMP=OFF` to cmake flags before building.

**Prerequisites:**

- ROCm software installation (download from [AMD ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/windows/llm/llamacpp.html))
- Ninja build system
- Clang compiler

⚠️ **CRITICAL: Set Environment Variable Before Running (Multi-GPU Systems)**

When your platform has multiple AMD GPUs, you must set `HIP_VISIBLE_DEVICES` to specify which AMD GPU to use:

**Windows PowerShell:**

```powershell
$env:HIP_VISIBLE_DEVICES="0"  # Use first AMD adapter (GPU 0)
$env:HIP_VISIBLE_DEVICES="1"  # Use second AMD adapter (GPU 1)
```

**Windows CMD:**

```cmd
set HIP_VISIBLE_DEVICES=0     :: Use first AMD adapter (GPU 0)
set HIP_VISIBLE_DEVICES=1     :: Use second AMD adapter (GPU 1)
```

**Linux:**

```bash
export HIP_VISIBLE_DEVICES=0  # Use first AMD adapter (GPU 0)
export HIP_VISIBLE_DEVICES=1  # Use second AMD adapter (GPU 1)
```

**Device Selection:**

- **Device 0:** First AMD adapter (primary GPU) - **Used by default if not set**
- **Device 1:** Second AMD adapter (if you have multiple GPUs)
- **Device 2+:** Third or later adapter (for multi-GPU systems)

**Note:** By default, HIP uses device 0 (first AMD GPU). The device number corresponds to the order AMD GPUs are detected by the system. If you have only one AMD GPU or want to use the first GPU, you can omit setting `HIP_VISIBLE_DEVICES`.

The compiled binaries will be in: `build\bin\Release\` on Windows or `build/bin/` on Linux.

**Linux**

This step can be skipped by downloading prebuilt llama.cpp with ROCm backend binaries from [Llama.cpp+ROCm prebuilt binaries](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/linux/llm/llamacpp.html).

```bash
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

**Note:** Replace `gfx1100` with your AMD GPU architecture:

- **gfx1100:** RDNA<sup>TM</sup> 3 GPU Architecture
- **gfx115x:** RDNA<sup>TM</sup> 3.5 GPU Architecture
- **gfx12xx:** RDNA<sup>TM</sup> 4 GPU Architecture

**Troubleshooting:**

If you encounter OpenMP-related linking issues, add `-DGGML_OPENMP=OFF` to cmake flags before building.

**Prerequisites:**

- ROCm installation (see the installation guide for [AMD ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html))
- Ninja build system
- Clang compiler

The compiled binaries will be in: `build/bin/` on Linux.

### Build with Vulkan Backend

**Prerequisites:** Install the [Vulkan SDK](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan).

**Windows**

Build `llama.cpp` with Vulkan backend.

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

The compiled binaries will be in: `build\bin\Release\` on Windows.

**Linux**

Build `llama.cpp` with Vulkan backend.

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

The compiled binaries will be in: `build/bin/` on Linux.

### Quantize the Model

If you need to reduce your model’s memory footprint or disk size, you can quantize the weights to a lower bit‑width. Quantization converts the original FP16 weights into more compact representations—such as INT4—resulting in significantly lower RAM usage and faster loading times, with minimal impact on model quality for many use cases.

**Windows**

The following example demonstrates how to convert an FP16 model to a 4‑bit Q4_K_M format using llama.cpp:

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
build\bin\Release\llama-quantize.exe phi-3.5-mini-instruct-fp16.gguf phi-3.5-mini-instruct-Q4_K_M.gguf Q4_K_M
```

**Quantization Options:**

- **Q4_K_M:** 4-bit quantization, medium quality (recommended balance of size and quality)
- **Q4_K_S:** 4-bit quantization, small size (more compression, slight quality loss)
- **Q5_K_M:** 5-bit quantization, medium quality (better quality, larger file)
- **Q8_0:** 8-bit quantization (highest quality, larger file)

After quantization, the new .gguf file will be much smaller and can be used directly at runtime with llama.cpp or any inference runtime that supports GGUF formats.

**Linux**

The following example demonstrates how to convert an FP16 model to a 4‑bit Q4_K_M format using llama.cpp:

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
build/bin/llama-quantize phi-3.5-mini-instruct-fp16.gguf phi-3.5-mini-instruct-Q4_K_M.gguf Q4_K_M
```

**Quantization Options:**

- **Q4_K_M:** 4-bit quantization, medium quality (recommended balance of size and quality)
- **Q4_K_S:** 4-bit quantization, small size (more compression, slight quality loss)
- **Q5_K_M:** 5-bit quantization, medium quality (better quality, larger file)
- **Q8_0:** 8-bit quantization (highest quality, larger file)

After quantization, the new .gguf file will be much smaller and can be used directly at runtime with llama.cpp or any inference runtime that supports GGUF formats.

### Run llama-cli with Your Model

**Windows**

Basic Usage:

```bash
build\bin\Release\llama-cli.exe -m phi-3.5-mini-instruct-Q4_K_M.gguf -p "What is 2+2? Answer:" -ngl 33 -c 4096 -n 100
```

**Interactive Chat Mode:**

```bash
build\bin\Release\llama-cli.exe -m phi-3.5-mini-instruct-Q4_K_M.gguf -ngl 33 -c 4096 --interactive
```

**With Phi-3.5 Chat Format:**

```bash
build\bin\Release\llama-cli.exe -m phi-3.5-mini-instruct-Q4_K_M.gguf -p "<|system|>You are a helpful AI assistant.<|end|><|user|>What is the capital of France?<|end|><|assistant|>" -ngl 33 -c 4096 -n 100
```

**Key Parameters:**

- `-m`: Path to your GGUF model file
- `-p`: Your prompt text
- `-c`: **REQUIRED** - Context window size (use 4096 or 8192, NOT the full 128K)
- `-ngl`: Number of GPU layers to offload (33 for Phi-3.5-mini = all layers, or use -1)
- `-n`: Maximum tokens to generate (default: 128, increase for longer responses)
- `--interactive`: Enable interactive chat mode

**Troubleshooting:**

- **Error "ErrorOutOfDeviceMemory":** Reduce `-c` value (try 2048 or lower)
- **Slow performance:** Ensure `-ngl 33` is set to use GPU
- **Model not found:** Use absolute path or ensure you're in the correct directory
- **Context too large:** Use `-c 4096` instead of default (which tries to use full 131K)

**Performance Tips:**

- `-c 4096`: Good balance (uses ~1.5GB VRAM for the KV cache)
- `-c 8192`: Better for longer conversations (uses ~3GB VRAM for KV cache)
- `-c 2048`: Use if GPU memory is limited
- `-ngl 33`: Offloads all 33 layers to GPU (recommended)
- Model requires ~2GB VRAM + additional memory for the context cache (varies with `-c` value)

**Linux**

Basic Usage:

```bash
build/bin/llama-cli -m phi-3.5-mini-instruct-Q4_K_M.gguf -p "What is 2+2? Answer:" -ngl 33 -c 4096 -n 100
```

**Interactive Chat Mode:**

```bash
build/bin/llama-cli -m phi-3.5-mini-instruct-Q4_K_M.gguf -ngl 33 -c 4096 --interactive
```

**With Phi-3.5 Chat Format:**

```bash
build/bin/llama-cli -m phi-3.5-mini-instruct-Q4_K_M.gguf -p "<|system|>You are a helpful AI assistant.<|end|><|user|>What is the capital of France?<|end|><|assistant|>" -ngl 33 -c 4096 -n 100
```

**Key Parameters:**

- `-m`: Path to your GGUF model file
- `-p`: Your prompt text
- `-c`: **REQUIRED** - Context window size (use 4096 or 8192, NOT the full 128K)
- `-ngl`: Number of GPU layers to offload (33 for Phi-3.5-mini = all layers, or use -1)
- `-n`: Maximum tokens to generate (default: 128, increase for longer responses)
- `--interactive`: Enable interactive chat mode

**Troubleshooting:**

- Error "ErrorOutOfDeviceMemory": Reduce `-c` value (try 2048 or lower)
- Slow performance: Ensure `-ngl 33` is set to use GPU
- Model not found: Use absolute path or ensure you're in the correct directory
- Context too large: Use `-c 4096` instead of default (which tries to use full 131K)

**Performance Tips:**

- `-c 4096`: Good balance (uses ~1.5GB VRAM for KV cache)
- `-c 8192`: Better for longer conversations (uses ~3GB VRAM for KV cache)
- `-c 2048`: Use if GPU memory is limited
- `-ngl 33`: Offloads all 33 layers to GPU (recommended)
- Model requires ~2GB VRAM + plus additional memory for the context cache (varies with `-c` value)

### Ollama

Ollama provides a simple way to run large language models locally with ROCm support for AMD GPUs.

#### Installation

**Prerequisites for AMD GPU Support:**

- ROCm must be installed on your system (see the [ROCm installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/))
- Ollama will automatically detect and use AMD GPUs with ROCm

**Windows**

1. Download the Windows installer:
   - Visit [https://ollama.com/download](https://ollama.com/download)
   - Click "Download for Windows" to get `OllamaSetup.exe`

2. Run the installer:

   ```powershell
   # Double-click OllamaSetup.exe or run from PowerShell:
   Start-Process "$env:USERPROFILE\Downloads\OllamaSetup.exe"
   ```

3. Follow the installation wizard:
   - Accept the license agreement
   - Choose installation location (default: `C:\Program Files\Ollama`)
   - Complete the installation

4. Verify installation:

   ```bash
   ollama --version
   ```

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Running with ROCm Backend

**Windows**

Ollama automatically detects and uses AMD GPUs with ROCm support.

*Option A: Use a Model from Ollama Library*

Pull and run a pre-configured model:

```bash
ollama run phi3.5
```

*Option B: Use Your Custom GGUF Model (from the "Converting the Model" section")

Create a Modelfile to import your converted model:

1. Create a `Modelfile`:

   ```text
   # Create Modelfile in the same directory as your GGUF model
   FROM ./phi-3.5-mini-instruct-Q4_K_M.gguf

   TEMPLATE """<|system|>
   {{ .System }}<|end|>
   <|user|>
   {{ .Prompt }}<|end|>
   <|assistant|>
   """

   PARAMETER stop "<|end|>"
   PARAMETER stop "<|endoftext|>"
   ```

2. Create the model in Ollama:

   ```bash
   ollama create phi3.5-custom -f Modelfile
   ```

3. Run your custom model:

   ```bash
   ollama run phi3.5-custom
   ```

*Example Output:*

```text
>>> What is 2+2?
The sum of 2 + 2 is 4. This arithmetic operation follows the basic principles
of addition, where combining two sets or instances of a number results in their
total when added together.
```

*Useful Commands:*

```bash
# List all models
ollama list

# Delete a model
ollama rm phi3.5-custom

# Show model information
ollama show phi3.5-custom
```

*Multi-GPU Systems:*

If you have multiple AMD GPUs, set the `HIP_VISIBLE_DEVICES` environment variable before running Ollama:

```bash
# Windows PowerShell
$env:HIP_VISIBLE_DEVICES="1"; ollama run phi3.5

# Linux
export HIP_VISIBLE_DEVICES=1
ollama run phi3.5
```

*Available Models:*

Visit [https://ollama.com/library](https://ollama.com/library) to browse hundreds of available models, including:

- phi3.5, phi3, phi3:medium
- llama3.1, llama3.2, llama2
- mistral, mixtral
- codellama, deepseek-coder
- And many more

**Linux**

Ollama automatically detects and uses AMD GPUs with ROCm support.

*Option A: Use a Model from Ollama Library*

Pull and run a pre-configured model:

```bash
ollama run phi3.5
```

*Option B: Use Your Custom GGUF Model from Section 1*

Create a Modelfile to import your converted model:

1. Create a `Modelfile`:

   ```text
   # Create Modelfile in the same directory as your GGUF model
   FROM ./phi-3.5-mini-instruct-Q4_K_M.gguf

   TEMPLATE """<|system|>
   {{ .System }}<|end|>
   <|user|>
   {{ .Prompt }}<|end|>
   <|assistant|>
   """

   PARAMETER stop "<|end|>"
   PARAMETER stop "<|endoftext|>"
   ```

2. Create the model in Ollama:

   ```bash
   ollama create phi3.5-custom -f Modelfile
   ```

3. Run your custom model:

   ```bash
   ollama run phi3.5-custom
   ```

*Example Output:*

```text
>>> What is 2+2?
The sum of 2 + 2 is 4. This arithmetic operation follows the basic principles
of addition, where combining two sets or instances of a number results in their
total when added together.
```

*Useful Commands:*

```bash
# List all models
ollama list

# Delete a model
ollama rm phi3.5-custom

# Show model information
ollama show phi3.5-custom
```

*Multi-GPU Systems:*

If you have multiple AMD GPUs, set the `HIP_VISIBLE_DEVICES` environment variable before running Ollama:

```bash
# Windows PowerShell
$env:HIP_VISIBLE_DEVICES="1"; ollama run phi3.5

# Linux
export HIP_VISIBLE_DEVICES=1
ollama run phi3.5
```

*Available Models:*

Visit [https://ollama.com/library](https://ollama.com/library) to browse hundreds of available models including:

- phi3.5, phi3, phi3:medium
- llama3.1, llama3.2, llama2
- mistral, mixtral
- codellama, deepseek-coder
- And many more

### Python Bindings

#### Install llama-cpp-python with Vulkan Support

**Windows**

```bash
set CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python
```

**Linux**

```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python
```

#### Using Phi-3.5 chat format

The examples below demonstrate the basic usage of the Phi-3.5 model on Windows and Linux using llama.cpp with Python.

##### Windows

```python
from llama_cpp import Llama

llm = Llama(
    model_path="phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    chat_format="chatml"  # Phi-3.5 uses ChatML format
)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=256,
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
```

**Key Parameters:**

- `model_path`: Path to your GGUF model
- `n_gpu_layers`: Number of layers to offload to GPU (-1 = all layers)
- `n_ctx`: Context window size (max 128000 for Phi-3.5, but 4096-8192 recommended)
- `chat_format`: Set to "chatml" for Phi-3.5 models
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative)
- `max_tokens`: Maximum tokens to generate

##### Linux

```python
from llama_cpp import Llama

llm = Llama(
    model_path="phi-3.5-mini-instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    chat_format="chatml"  # Phi-3.5 uses ChatML format
)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=256,
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
```

**Key Parameters:**

- `model_path`: Path to your GGUF model
- `n_gpu_layers`: Number of layers to offload to GPU (-1 = all layers)
- `n_ctx`: Context window size (max 128000 for Phi-3.5, but 4096-8192 recommended)
- `chat_format`: Set to "chatml" for Phi-3.5 models
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative)
- `max_tokens`: Maximum tokens to generate

## Summary

In this blog you explored how running large language models locally on AMD Radeon GPUs has moved from an experimental niche to a practical, flexible option for developers and enthusiasts. You learned what the ROCm-based software stack and companion tools like Lemonade, LM Studio, Ollama, and llama.cpp each bring to the table, how they map to different workflows (from lightweight on-device inference to multi-GPU training and fine-tuning), and why that choice matters for performance, privacy, and cost.

As both hardware and software continue to evolve, the AMD AI ecosystem is poised to make local AI even more capable and widely accessible. Now is the perfect time to explore what’s possible with Radeon-powered AI.

## Additional Resources

- [Lemonade](https://lemonade-server.ai/)
- [LM Studio](https://lmstudio.ai/)
- [Ollama](https://ollama.com/)
- [LLM on AMD GPUs with ONNX GenAI](https://community.amd.com/t5/ai/llm-on-amd-gpu-memory-footprint-and-performance-improvements-on/ba-p/686157)
- [HuggingFace: Optimized onnx genai models for AMD GPUs](https://huggingface.co/collections/amd/amdgpu-onnxgenai)
- [https://www.amd.com/en/developer/resources/technical-articles/2026/lemonade-for-local-ai.html](https://www.amd.com/en/developer/resources/technical-articles/2026/lemonade-for-local-ai.html)
- [https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- [https://www.amd.com/en/developer/resources/technical-articles/2025/amd-ryzen-ai-max-395--a-leap-forward-in-generative-ai-performanc.html](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-ryzen-ai-max-395--a-leap-forward-in-generative-ai-performanc.html)
- [https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#high-level-api](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#high-level-api)
- [https://www.amd.com/en/developer/resources/technical-articles/unlocking-a-wave-of-llm-apps-on-ryzen-ai-through-lemonade-server.html](https://www.amd.com/en/developer/resources/technical-articles/unlocking-a-wave-of-llm-apps-on-ryzen-ai-through-lemonade-server.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
