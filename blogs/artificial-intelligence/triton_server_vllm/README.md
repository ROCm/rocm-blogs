---
blogpost: true
blog_title: 'Triton Inference Server with vLLM on AMD GPUs'
thumbnail: '232285500_AMD_EPYC_Siena_beautyshot_02.png'
date: 8 January 2025
author: Fabricio Flores, Tiffany Mintz, Eliot Li, Yao Liu, Ted Themistokleous, Brian Pickrell, Vish Vadlamani
tags: LLM, AI/ML
category: Applications & models
key_value_propositions: ""
target_audience: ""
language: English
myst:
    html_meta:
        "description lang=en": "This blog provides a how-to guide on setting up a Triton Inference Server with vLLM backend powered by AMD GPUs, showcasing robust performance with several LLMs"
        "author": "Fabricio Flores, Tiffany Mintz, Eliot Li, Yao Liu, Ted Themistokleous, Brian Pickrell, Vish Vadlamani"
        "keywords": "Triton Inference Server, vLLM, ROCm, AMD, GPU, MI300, MI250, MI210"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
---

# Triton Inference Server with vLLM on AMD GPUs

Triton Inference Server is an open-source platform designed to streamline AI inferencing. It supports the deployment, scaling, and inference of trained AI models from various machine learning and deep learning frameworks including Tensorflow, PyTorch, and vLLM, making it adaptable for diverse AI workloads. It is designed to work across multiple environments, including cloud, data centers and edge devices.

Some of the Triton Inference Server capabilities include:

* **Framework flexibility**: Allows for the deployment of models from different frameworks (see [Triton Inference Server Backend](https://github.com/triton-inference-server/backend)) regardless of underlying infrastructure. This flexibility allows for running multiple models or multiple instances of the same model on the same hardware, improving resource utilization.

* **Hardware and deployment versatility**: It is optimized for both GPU and CPU-based environments, which allows for its deployment on a variety of hardware. Triton Inference Server can be used in the cloud, data centers or on edge, making it highly versatile.

* **Performance optimization**: It enhances inference performance through dynamic batching, which aggregates smaller inference requests to optimize processing and enable concurrent model execution. This capability allows multiple models to run simultaneously, making it crucial for real-time applications that require minimal latency.

In this blog we will show you, step-by-step, how to set up a Triton Inference Server with [vLLM](https://docs.vllm.ai/en/latest/) backend on AMD GPUs, using ROCm. We start by briefly introducing some of the key aspects of using [vLLM](https://docs.vllm.ai/en/latest/) as a backend for the Triton Inference Server. We then present a detailed how-to guide showing you how to set up the Triton Inference Server with vLLM backend, with inference testing performed on 3 LLMs: `microsoft/phi-2`, `mistral-7b-instruct` and `meta-llama/Meta-Llama-3-8B-Instruct.`

## Requirements

* AMD GPU: See the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported hardware and operating systems. This blog was tested on a machine with 8 AMD Instinct MI210 GPUs.

* ROCm 6.1+: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) for installation instructions.

* Docker: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) for installation instructions.

* Hugging Face Access Token: This blog requires a [Hugging Face](https://huggingface.co/) account with a newly generated [User Access Token](https://huggingface.co/docs/hub/security-tokens).

* Access to the `mistral` and `Llama-3` models on Hugging Face. These are [gated models](https://huggingface.co/docs/hub/models-gated) on Hugging Face, to request access see [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

You can find the files related to this blog post in this
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/triton_server_vllm).

## Triton Inference Server: vLLM Backend

A Triton Inference Server backend refers to the component responsible for executing the AI model during inference. A backend is a wrapper around a specific machine learning framework such as [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [vLLM](https://docs.vllm.ai/en/latest/), or others. Each backend is implemented as a shared library and models can be configured to use specific backends. For example, if a model is using PyTorch, the backend will be configured to interact with PyTorch libraries.

The Triton Inference Server project provides a set of supported backends that are tested and updated with each release. For a list of the supported backends, see [Where can I find all the backends that are available for Triton?](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton). This blog focuses on the [vLLM](https://docs.vllm.ai/en/latest/) backend.

Using vLLM as a backend enables serving of large language models (LLMs) with features designed for high throughput and low latency. vLLM is a specialized engine optimized for handling LLM inference, particularly in scenarios where continuous batching and memory efficiency are critical.

These are some of the key aspects of vLLM with Triton Inference Server:

* **vLLM integration**: vLLM is integrated into Triton Inference Server starting from the 23.10 release. It can be used through pre-built containers that include the vLLM backend or by building a custom container. This integration allows serving models like Facebook’s OPT series, LLaMA models, and others through Triton Inference Server’s flexible and scalable architecture.

* **Configuration and deployment**: When setting up vLLM as a backend, it is necessary to configure the model repository. This repository includes the `model.json` and `config.pbtxt` files. These configurations define model parameters such as memory utilization, batch sizes, and model-specific settings.

* **Performance features**: vLLM’s backend in Triton Inference Server supports asynchronous inference, which is crucial for tasks like large-scale text generation and processing. Features such as tensor parallelism and paged attention enhance multi-GPU performance, making vLLM suitable for handling large models across distributed systems.

* **Deployment options**: vLLM-backend models can be deployed on various platforms, including cloud environments. The containerized deployment ensures that models can be scaled horizontally based on performance demands, with support for Kubernetes and other orchestration systems.

Using vLLM as the backend for Triton Inference Server provides access to a highly optimized serving engine tailored to the specific demands of LLMs, while also leveraging Triton Inference server's robust infrastructure for scalable inference.

## Setting up Triton Inference Server with vLLM backend

To perform inference with large language models using the Triton Inference Server and vLLM backend, follow these steps:

* **Set up Triton Inference Server with vLLM backend**: We are configuring a `docker compose` file that includes a Triton Inference Server container with the vLLM backend. The `docker compose` refers to a Docker image with Triton Inference Server pre-installed (the Docker image can be built from source or pulled from a registry), defines GPU access, sets the repository path, and exposes the necessary ports.

* **Prepare the model repository**: A model repository is a directory or set of directories that contain the models that will be served for inference. Each model is organized in a specific structure within the repository. This structure is scanned and loaded every time Triton Inference Server starts.

    The structure of the model repository is as follows:

    ```text
    model_repository/
        ├── <model_name_1>/
        │   ├── config.pbtxt  # Configuration file that describes the model
        │   ├── 1/  # Version directory (Triton Inference Server supports versioning)
        │   │   └── model.onnx  # The actual model file (e.g., ONNX, PyTorch, vLLM)
        │   └── 2/
        │       └── model.onnx
        ├── <model_name_2>/
        │   ├── config.pbtxt
        │   ├── 1/
        │   │   └── model.json
        │   └── 2/
        │       └── model.json
    ```

    The `model_repository` is the root directory that contains one or more subdirectories, each representing a model. Each model is organized into a **Model directory** (`<model_name>`), where the directory name corresponds to the model's name. Within the model directory, there are **Version directories** (`1/`, `2/`) allowing for multiple versions of the same model to exist. Each version directory contains the actual model files. These files enable Triton Inference Server to identify and serve the correct version. The **Model file** (`model.onnx`, `model.json`, and so on) stores the model architecture and inference parameters. Finally, **Configuration file** (`config.pbtxt`) defines the input and output tensor names, shapes, data types, and other configurations.

* **Define the model configuration and model files**: When using the vLLM backend, the model configuration file must specify the backend type in addition to the data types and shapes. A simplified version of a `config.pbtxt` would look like:

    ```json
    backend: "vllm"

    input [
    {
        name: "text_input"
        data_type: TYPE_STRING
        dims: [ 1 ]
    }
    ]

    output [
    {
        name: "text_output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
    ]
    ```

    While the model file `model.json`, which specifies the parameters for model initialization and inference, would look like:

    ```json
    {
        "model":"meta-llama/Meta-Llama-3-8B-Instruct",
        "gpu_memory_utilization": 0.8,
        "tensor_parallel_size": 2,
        "trust_remote_code": true,
        "disable_log_requests": true,
        "enforce_eager": true,
        "max_model_len": 2048
    }
    ```

    Among these parameters, `model` specify the model's name, `gpu_memory_utilization` restricts the model to only use a certain percent of the GPU memory, `tensor_parallel_size` defines the number of GPUs the model should use for parallel processing. For more information on the additional parameters and configuration files, see the Triton Inference Server-vLLM documentation: [Start Triton Inference Server](https://github.com/AI-General/triton-tutorials/blob/main/Quick_Deploy/vLLM/README.md#step-2-start-triton-inference-server).

We have created a Docker compose configuration that automates the entire setup of Triton Inference Server with vLLM backend. This setup includes building a Docker image, configuring AMD GPU access through a `docker-compose.yaml` file, and setting up the model repository (`./triton_server_vllm/src/model_repository`) with 3 different LLMs to test. Using this setup, run the `docker compose build` and `docker compose up` commands to launch the Triton Inference Server without the need to manually complete the previous steps.

Let's begin by building a Triton Inference Server Docker image from source. Clone the repository for the Triton Inference Server, specifically the AMD ROCm version:

```bash
git clone https://github.com/ROCm/tritoninferenceserver-vllm.git
```

Next, navigate to the `tritoninferenceserver-vllm` directory and run the `build-vllm-docker.py` Python script that builds the Docker image:

```bash
cd tritoninferenceserver-vllm

python3 build-vllm-docker.py --no-container-pull --enable-logging --enable-stats \
  --enable-tracing --enable-rocm  --endpoint=grpc \
  --image gpu-base,rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2 \
  --endpoint=http --backend=python --backend=vllm
```

The newly built Docker image is named `tritonserver`. To verify its existence, use the following command:

```bash
docker images | grep tritonserver
```

Where the output will be similar to:

```bash
REPOSITORY                TAG            IMAGE ID       CREATED         SIZE
tritonserver              latest         fffefb8a8258   22 hours ago    62.8GB
```

With the `tritonserver` Docker image built, let's return to the original directory and clone this blog's repository:

```bash
cd ..
git clone https://github.com/ROCm/rocm-blogs.git
cd rocm-blogs/blogs/artificial-intelligence/triton_server_vllm/docker
```

Then edit the environment file: `./triton_server_vllm/docker/.env` and provide a Hugging Face Token:

```bash
HUGGING_FACE_HUB_TOKEN=<YOUR_HUGGING_FACE_ACCESS_TOKEN>
```

Next, give execution permissions to the `/triton_server_vllm/docker/start_services.sh` bash script by running the command:

```bash
chmod +x start_services.sh
```

Finally, build and start the Docker container:

```bash
docker compose build
docker compose up
```

```{note}
Starting the container and services will take some time since the `Mistral-7B-Instruct-v0.1`, `microsoft/phi-2` and `meta-llama/Meta-Llama-3-8B-Instruct` models are downloaded from Hugging Face Hub and served.
```

Upon executing the `docker compose up` command, the terminal will display an output similar to:

```text
[+] Running 2/1
 ✔ Network docker_default                 Created  0.1s 
 ✔ Container docker-triton_server_vllm-1  Created  0.0s 
Attaching to triton_server_vllm-1
...
triton_server_vllm-1  | [I 2024-08-27 15:33:39.976 ServerApp] Jupyter Server 2.14.2 is running at:
triton_server_vllm-1  | [I 2024-08-27 15:33:39.976 ServerApp] http://3dd761dca9b9:8888/lab
triton_server_vllm-1  | [I 2024-08-27 15:33:39.976 ServerApp]     http://127.0.0.1:8888/lab
triton_server_vllm-1  | [I 2024-08-27 15:33:39.976 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
...

triton_server_vllm-1  | INFO 08-27 22:22:12 llm_engine.py:68] Initializing an LLM engine (v0.3.3) with config: model='mistralai/Mistral-7B-Instruct-v0.1', tokenizer='mistralai/Mistral-7B-Instruct-v0.1', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=2, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto, device_config=cuda, seed=0)

triton_server_vllm-1  | INFO 08-27 22:22:13 llm_engine.py:68] Initializing an LLM engine (v0.3.3) with config: model='microsoft/phi-2', tokenizer='microsoft/phi-2', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=2, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto, device_config=cuda, seed=0)

triton_server_vllm-1  | INFO 08-27 22:22:13 llm_engine.py:68] Initializing an LLM engine (v0.3.3) with config: model='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=2, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto, device_config=cuda, seed=0)
```

When Triton Inference Server is ready, the console will display the following:

```text
triton_server_vllm-1  | I0827 22:27:53.490967 15 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
triton_server_vllm-1  | I0827 22:27:53.491185 15 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
```

On console's output we see:

* A Jupyter Server that is running at `http://127.0.0.1:8888/lab`
* The model `mistralai/Mistral-7B-Instruct-v0.1` is being initialized. Requests can be made at `http://localhost:8000/v2/models/mistral-7b-instruct/generate`
* The model `microsoft/phi-2` is being initialized. Requests can be made at `http://localhost:8000/v2/models/phi2/generate`
* The model `meta-llama/Meta-Llama-3-8B-Instruct` is being initialized. Requests can be made at `http://localhost:8000/v2/models/llama3-8b-instruct/generate`

With the models ready for inference, we can proceed with some tests.

## Understanding model_repository structure and configurations

The `docker-compose.yaml` file contained the necessary configurations to create a Docker container that serves the `phi-2`, `Mistral-7B-Instruct-v0.1`, and `Meta-Llama-3-8B-Instruct` models. The specific configurations used to serve and perform inference with each model are located in the `./triton_server_vllm/src/model_repository` folder. In our case, this `model_repository` folder has the following structure:

```text
model_repository/
    ├── llama3-8b-instruct/
    │   ├── config.pbtxt    # Configuration file that describes the model
    │   ├── 1/              # Version directory
    │       └── model.json  # The actual model file
    ├── mistral-7b-instruct/
    │   ├── config.pbtxt
    │   ├── 1/
    │       └── model.json
    ├── phi2/
    │   ├── config.pbtxt
    │   ├── 1/
    │       └── model.json
```

Each model's `model.json` file contains its own configuration. For the `llama3-8b-instruct` model its `model.json` is:

```json
{
    "model":"meta-llama/Meta-Llama-3-8B-Instruct",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "trust_remote_code": true,
    "disable_log_requests": true,
    "enforce_eager": true,
    "max_model_len": 2048
}
```

For the `Mistral-7B-Instruct-v0.1` model its `model.json` is:

```json
{
    "model":"mistralai/Mistral-7B-Instruct-v0.1",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 2,
    "trust_remote_code": true,    
    "disable_log_requests": true,
    "enforce_eager": true,
    "max_model_len": 2048
}
```

While for the `phi2` model its `model.json` is:

```json
{
    "model":"microsoft/phi-2",
    "gpu_memory_utilization": 0.8,
    "tensor_parallel_size": 1,
    "trust_remote_code": true,
    "disable_log_requests": true,
    "enforce_eager": true,
    "max_model_len": 2048
}
```

The value for the `tensor_parallel_size` parameter in each `model.json` file specifies how many GPUs will be used for each model's parallel computation. Since we want to run these 3 models concurrently and have access to 8 AMD Instinct MI210 GPUs, this implies that `Meta-Llama-3-8B-Instruct` will use 2 of the 8 GPUs, `Mistral-7B-Instruct-v0.1` will use 2 of the remaining 6 GPUs and `phi-2` will use 1 of the remaining 4 GPUs. If more GPUs are needed for a particular model, we need to adjust the value of the `tensor_parallel_size` for one or more models to fit within the available GPUs.

For more information on the additional parameters and configuration files see the Triton Inference Server-vLLM documentation: [Start Triton Inference Server](https://github.com/AI-General/triton-tutorials/blob/main/Quick_Deploy/vLLM/README.md#step-2-start-triton-inference-server)

## Inference with phi-2, Mistral-7B-Instruct-v0.1, and Meta-Llama-3-8B-Instruct

With our Jupyter Lab and Triton Inference Server running, navigate to `http://127.0.0.1:8888/lab/tree/src/triton_server_vllm.ipynb` to perform inference with these models.

Let's begin testing `microsoft/phi-2` as follows:

```python
# Define the URL for endpoint
url = "http://localhost:8000/v2/models/phi2/generate"

# Define payload
payload = {
    "text_input": "What is triton inference server?",
    "parameters": {
        "stream": False,
        "temperature": 0,
        "max_tokens": 100
    }
}

# Set the headers (optional)
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Print the response
print(response.status_code)
print(response.json())
```

We are making a `POST` request to the Triton Inference Server with the payload containing the prompt: `"What is triton inference server?"`. The output consists of the response status `200` and a `json` object:

```text
200
{'model_name': 'phi2', 'model_version': '1', 'text_output': 'What is triton inference server?\n\nTriton inference server is a software that helps to run machine learning models on a computer. It is like a helper that makes sure the models work correctly and gives us the results we need.\n\nWhat is the purpose of triton inference server?\n\nThe purpose of triton inference server is to help us use machine learning models in our daily lives. It makes it easier for us to use these models and get the results we need.\n\nHow does triton inference server'}

```

With `Mistral-7B-Instruct-v0.1` we have:

```python
# Define the URL for endpoint
url = "http://localhost:8000/v2/models/mistral-7b-instruct/generate"

# Define payload
payload = {
    "text_input": "What is triton inference server?",
    "parameters": {
        "stream": False,
        "temperature": 0,
        "max_tokens": 100
    }
}

# Set the headers (optional)
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Print the response
print(response.status_code)
print(response.json())
```

```text
200
{'model_name': 'mistral-7b-instruct', 'model_version': '1', 'text_output': 'What is triton inference server?\n\nTriton Inference Server is an open-source, high-performance, and scalable inference engine for deep learning models. It supports a wide range of deep learning frameworks, including TensorFlow, PyTorch, and MXNet, and can be used to deploy deep learning models in various environments, such as edge devices, cloud services, and on-premises data centers.\n\nTriton Inference Server provides a unified API for accessing'}
```

Finally, performing inference with `meta-llama/Meta-Llama-3-8B-Instruct` we have:

```python
# Define the URL for endpoint
url = "http://localhost:8000/v2/models/llama3-8b-instruct/generate"

# Define payload
payload = {
    "text_input": "What is triton inference server?",
    "parameters": {
        "stream": False,
        "temperature": 0,
        "max_tokens": 100
    }
}

# Set the headers (optional)
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Print the response
print(response.status_code)
print(response.json())
```

Where the response to the `POST` request is:

```text
200
{'model_name': 'llama3-8b-instruct', 'model_version': '1', 'text_output': 'What is triton inference server?¶\n\nTriton Inference Server is an open-source, high-performance, scalable, and extensible deep learning inference server developed by NVIDIA. It is designed to serve as a production-ready inference engine for deep learning models, allowing developers to deploy and manage their models in a scalable and efficient manner.\n\nTriton Inference Server provides a number of features that make it an attractive choice for deploying deep learning models in production environments, including:\n\n1. **Model serving**: Triton Inference Server can'}
```

Deploying all three models (`microsoft/phi-2`, `Mistral-7B-Instruct-v0.1`, and `meta-llama/Meta-Llama-3-8B-Instruct`) simultaneously allowed us to serve multiple LLMs. Triton Inference Server with the vLLM backend managed the necessary resources and optimized memory utilization to run these models concurrently.

## Summary

In this blog, we presented the deployment and serving of three LLMs using Triton Inference Server with a vLLM backend, all powered by AMD GPUs and the ROCm software platform. We provided a step-by-step guide on using Triton Inference Server to efficiently handle multiple LLMs, showcasing robust performance and reliability of AMD hardware in high-demand AI applications.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
