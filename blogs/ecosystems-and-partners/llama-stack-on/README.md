---
blogpost: true
blog_title: "A step-by-step guide on how to deploy Llama Stack on AMD Instinct™ GPU"
date: 22 Apr 2025
author: 'Alex He'
thumbnail: '3.jpg'
tags: AI/ML
category: Ecosystems and Partners
target_audience: AI enthusiast and developers
key_value_propositions: All developers who use LLama Stack to build end application will be able to use AMD GPUs with docker
language: English
myst:
    html_meta:
        "author": "Alex He"
        "description lang=en": "Learn how to use Meta’s Llama Stack with AMD ROCm and vLLM to scale inference, integrate APIs, and streamline production-ready AI workflows on AMD Instinct™ GPU"
        "keywords": "Llama Stack, AMD GPU, Dockers, vLLM"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        
---
# A Step-by-Step Guide On How To Deploy Llama Stack on AMD Instinct™ GPU

As a leader in high-performance computing, AMD empowers AI innovation by providing open-source tools and hardware acceleration for scalable model deployment. In this blog we will show you how this foundation can be leveraged to deploy Meta’s LLMs efficiently on AMD Instinct™ GPUs. Meta’s Llama series has democratized access to large language models, empowering developers worldwide. The Llama Stack—Meta’s all-in-one deployment framework—extends this vision by enabling seamless transitions from research to production through built-in tools for optimization, API integration, and scalability. This unified platform is ideal for teams requiring robust support to deploy Meta’s models at scale across diverse applications.

ROCm equips developers with a robust foundation to build high-throughput AI solutions tailored for production environments. Developers can leverage the Llama Stack framework and APIs to build AI applications such as Retrieval-Augmented Generation (RAG) systems and intelligent agents. This blog guides developers in deploying Llama Stack on AMD GPUs, creating a production-ready infrastructure for large language model (LLM) inference. We’ll also demonstrate programmatic interactions via the Llama Stack CLI and Python SDK, ensuring seamless server integration. To streamline this journey, we’ll first preview the core components involved—such as ROCm’s optimization tools, Llama Stack’s deployment workflows, and scalable GPU configurations—before diving into the hands-on session.

## Llama Stack and Remote vLLM Distribution

Llama Stack defines and standardizes the core building blocks needed to bring generative AI applications to market. It provides a unified set of APIs with implementations from leading service providers, enabling seamless transitions between development and production environments. [[1](#Reference)]

The Llama Stack’s Inference API is interoperable with a wide range of LLM inference providers, including vLLM, TGI, Ollama, and OpenAI APIs, ensuring seamless integration and flexibility for deployment. And it also provide 4 types client SDK, Python, Swift, Node and Kotlin.

For this tutorial, we’ve selected vLLM as the inference provider and the Llama Stack’s Python Client SDK to showcase scalable deployment workflows and illustrate hands-on, low-latency LLM integration into production-ready services.

## ROCm and vLLM Docker Images: Choosing the Right Environment for Development and Production

ROCm is an open-source software optimized to extract HPC and AI workload performance from AMD Instinct accelerators and AMD Radeon GPUs while maintaining compatibility with industry software frameworks. For more information, see [What is ROCm?](https://rocm.docs.amd.com/en/latest/what-is-rocm.html) [[2](#Reference)]

AMD collaborates with vLLM to deliver a streamlined, high-performance LLM inference engine and production-ready deployment solutions for enterprise-grade AI workloads.

**Available vLLM Containers** [[3](#Reference)]

AMD provides two main vLLM container options:

- [ROCm/vllm](https://hub.docker.com/r/ROCm/vllm): Production-ready container

  - Pinned to a specific version, for example: ROCm/vllm-dev:ROCm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6

  - Designed for stability

  - Optimized for deployment

- [ROCm/vllm-dev](https://hub.docker.com/r/ROCm/vllm-dev): Development container with the latest vLLM features

  - nightly, main and other specialized builds available:

    - nightly tags are built daily from the latest code, but may contain bugs

    - main tags are more stable builds, updated after testing

  - Includes development tools

  - Best for testing new features or custom modifications

## Deployment Llama Stack with ROCm

We chose [Remote vLLM Distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm.html#remote-vllm-distribution "Link to this heading") running with ROCm/vllm-dev docker image on Instinct™ MI300X. In addition to supporting many LLM inference providers (e.g., Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, etc.), Llama Stack also allows users to choose safety providers as an option (e.g., Meta’s Llama Guard, AWS Bedrock Guardrails, vLLM, etc.). In this tutorial, we use 2×MI300X GPUs: one for deploying LLM inference APIs, and another for Safety/Shield APIs deployment.

### Prerequist

#### Start ROCm/vllm vLLM server container

```shell
export HF_TOKEN=<your huggingface token>
export VLLM_DIMG="ROCm™/vllm-dev:main"
```

Set an alias to simplify the container launch command:

```shell
alias drun="docker run -it --rm \
    --ipc=host \
    --network host \
    --privileged \
    --shm-size 16g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ${HOME}:/ws -w /ws \
"
```

Launch the vLLM inference container using `meta-llama/Llama-3.2-3B-Instruct` as the LLM API provider.

```shell

drun --name lstack-vllm-serve \
	-e IPORT=8000 \
	-e IMODEL=meta-llama/Llama-3.2-3B-Instruct \
	-e HIP_VISIBLE_DEVICES=0 \
	$VLLM_DIMG
```

Launch the LLM model service within the container,

```shell
vllm serve $IMODEL --trust-remote-code --port $IPORT
```

[Optional] Using Llama Stack Safety/Shield APIs by,

```shell
drun --name lstack-vllm-guard \
	-e SAFETY_PORT=8081 \
	-e SAFETY_MODEL=meta-llama/Llama-Guard-3-1B \
	-e HIP_VISIBLE_DEVICES=1 \
	$VLLM_DIMG
```

Launch the guard model service within the container,

```shell
vllm serve $SAFETY_MODEL --port $SAFETY_PORT
```

Test the vLLM serve by `curl`,

```shell
 curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

 curl http://localhost:8081/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-Guard-3-1B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

```

Now the two remote API providers are ready.

#### Install llama-stack

Install llama-stack in conda env.

```shell
cd $HOME
yes | conda create -n lstack python=3.11
conda activate lstack
pip install llama-stack llama-stack-client
```

In this blog, I use llama-stack v0.1.9, the latest stable version at the time of writing.

```shell
$ pip list | grep llama_stack
llama_stack                       0.1.9
llama_stack_client                0.1.9
```

llama-stack provides templates that help create the distribution. Copy these templates to configure the remote-vllm version of the server.

```shell
git clone https://github.com/meta-llama/llama-stack.git
# checkout this branch what I used for this tutorial
git checkout -b release-0.1.19  origin/release-0.1.9

mkdir ws
cd ws/

# Copy the template yaml of remote-vllm distro 
cp ../llama-stack/llama_stack/templates/remote-vllm/run.yaml .
cp ../llama-stack/llama_stack/templates/remote-vllm/run-with-safety.yaml .
```

### Running Llama Stack

**NOTE:** Make sure all the commands bellow are running in the conda env `lstack` we created.

#### Via Conda

Use the template to build out the conda type of the remote-vllm distribution server.

```shell
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLAMA_STACK_PORT=8321

llama stack build --template remote-vllm --image-type conda

llama stack run ./run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://localhost:$INFERENCE_PORT/v1
```

If you are using Llama Stack Safety / Shield APIs, use:

```shell
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLAMA_STACK_PORT=8321
export SAFETY_PORT=8081
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

llama stack run ./run-with-safety.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://localhost:$INFERENCE_PORT/v1 \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env SAFETY_VLLM_URL=http://localhost:$SAFETY_PORT/v1
```

Then you can list the providers within another terminal while the llama-stack server running.

```shell
$ llama-stack-client providers list
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ API          ┃ Provider ID            ┃ Provider Type                  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ inference    │ vllm-inference         │ remote::vllm                   │
│ inference    │ sentence-transformers  │ inline::sentence-transformers  │
│ vector_io    │ faiss                  │ inline::faiss                  │
│ safety       │ llama-guard            │ inline::llama-guard            │
│ agents       │ meta-reference         │ inline::meta-reference         │
│ eval         │ meta-reference         │ inline::meta-reference         │
│ datasetio    │ huggingface            │ remote::huggingface            │
│ datasetio    │ localfs                │ inline::localfs                │
│ scoring      │ basic                  │ inline::basic                  │
│ scoring      │ llm-as-judge           │ inline::llm-as-judge           │
│ scoring      │ braintrust             │ inline::braintrust             │
│ telemetry    │ meta-reference         │ inline::meta-reference         │
│ tool_runtime │ brave-search           │ remote::brave-search           │
│ tool_runtime │ tavily-search          │ remote::tavily-search          │
│ tool_runtime │ code-interpreter       │ inline::code-interpreter       │
│ tool_runtime │ rag-runtime            │ inline::rag-runtime            │
│ tool_runtime │ model-context-protocol │ remote::model-context-protocol │
│ tool_runtime │ wolfram-alpha          │ remote::wolfram-alpha          │
└──────────────┴────────────────────────┴────────────────────────────────┘
```

The `vllm-inference` (Provider ID) as `remote::vllm`(Provider Type) as well as `llama-guard` (Provider ID) as `inline::llama-guard` (Provider ID) are listed.

Get the mode list by `llama-stack-client models list`

```shell

$llama-stack-client models list

Available Models

┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ model_type       ┃ identifier                                        ┃ provider_resource_id                              ┃ metadata                                       ┃ provider_id                     ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ embedding        │ all-MiniLM-L6-v2                                  │ all-MiniLM-L6-v2                                  │ {'embedding_dimension': 384.0}                 │ sentence-transformers           │
├──────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────┤
│ llm              │ meta-llama/Llama-3.2-3B-Instruct                  │ meta-llama/Llama-3.2-3B-Instruct                  │                                                │ vllm-inference                  │
├──────────────────┼───────────────────────────────────────────────────┼───────────────────────────────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────┤
│ llm              │ meta-llama/Llama-Guard-3-1B                       │ meta-llama/Llama-Guard-3-1B                       │                                                │ vllm-safety                     │
└──────────────────┴───────────────────────────────────────────────────┴───────────────────────────────────────────────────┴────────────────────────────────────────────────┴─────────────────────────────────┘

Total models: 3
```

Do the inference by using `llama-stack-client`.

```shell
$ llama-stack-client inference chat-completion \
	--message "hello, what model are you?"
```

You will get the rensponse from the ternminal as bellow,

```shell

ChatCompletionResponse(
    completion_message=CompletionMessage(
        content='Hello! I\'m an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."',
        role='assistant',
        stop_reason='end_of_turn',
        tool_calls=[]
    ),
    logprobs=None,
    metrics=[Metric(metric='prompt_tokens', value=17.0, unit=None), Metric(metric='completion_tokens', value=34.0, unit=None), Metric(metric='total_tokens', value=51.0, unit=None)]
)
```

#### Via Docker

This method allows you to get started quickly without having to build the distribution code. Let's use the prebuild docker image `llamastack/distribution-remote-vllm` from Llama Stack to run the server.

```shell
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLAMA_STACK_PORT=8321

docker run \
  -it \
  --pull always \
  --network=host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-remote-vllm \
  --config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://0.0.0.0:$INFERENCE_PORT/v1
```

If you are using Llama Stack Safety/Shield APIs, use:

```shell
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLAMA_STACK_PORT=8321
export SAFETY_PORT=8081
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

docker run \
  --pull always \
  --network=host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -v ./run-with-safety.yaml:/root/my-run.yaml \
  llamastack/distribution-remote-vllm \
  --config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://0.0.0.0:$INFERENCE_PORT/v1 \
  --env SAFETY_MODEL=$SAFETY_MODEL \
  --env SAFETY_VLLM_URL=http://0.0.0.0:$SAFETY_PORT/v1
```

Then do the client test as same as we did `Via Conda`.

```shell
$ llama-stack-client inference chat-completion \
	--message "hello, what model are you?"
```

### Python Client SDK

Llama-stack provides the Python Client SDK for the application development.
<!-- markdownlint-disable -->
**Example: An example application based on Python Client SDK**
<!-- markdownlint-restore -->

```python

# ls-inference.py
import os
import sys

def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}"
    )


def create_library_client(template="remote-vllm"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    if not client.initialize():
        print("llama stack not built properly")
        sys.exit(1)
    return client


client = (
    create_library_client()
)  # or create_http_client() depending on the environment you picked

# List available models
models = client.models.list()
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
print(response.completion_message.content)
```

Let's execute the application using the Llama-Stack server set up earlier with Conda or Docker as described in the preceding steps.

```shell
export INFERENCE_PORT=8000
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export LLAMA_STACK_PORT=8321
export VLLM_URL=http://localhost:$INFERENCE_PORT/v1

export SAFETY_PORT=8081
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

python ls-inference.py
```

We will get the inference with log as bellow,

```shell
...
--- Available models: ---
- all-MiniLM-L6-v2
- meta-llama/Llama-3.2-3B-Instruct
- meta-llama/Llama-Guard-3-1B

INFO     2025-03-21 06:29:35,699 openai._base_client:1685 uncategorized: Retrying request to /chat/completions in       
         0.471254 seconds                                                                                               
Lines of code descend
Logic's gentle, guiding hand
Beauty in the code
```

Note : We support Llama framework on ROCm version 6.3.1 other version of ROCm have not been validated.

## Summary

This tutorial has walked you through the complete workflow of deploying a Llama Stack server using ROCm/vLLM containers on AMD Instinct™ MI300X GPUs. It also demonstrated how to interact with the stack using the CLI and Python SDK, providing a robust foundation for scalable LLM applications on AMD hardware. This provides a solid foundation for leveraging these open-source technologies from AMD and Meta in further AI applications. Furthermore, these workflows are fully compatible with AMD Radeon™ Series GPUs, enabling streamlined development of AI-centric applications on local desktop platforms.

## Reference

[1] [Llama Stack Docunmentation](https://llama-stack.readthedocs.io/en/latest/index.html)

[2] [AMD ROCm documentation](https://ROCm.docs.amd.com/en/latest/)

[3] [How to Build a vLLM Container for Inference and Benchmarking](https://rocm.blogs.amd.com/software-tools-optimization/vllm-container/README.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
