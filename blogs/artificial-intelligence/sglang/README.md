---
blogpost: true
blog_title: 'SGLang: Fast Serving Framework for Large Language and Vision-Language Models on AMD GPUs'
thumbnail: 'cat.jpeg'
date: 13 November 2024
author: Michael Zhang, Hai Xiao, Hui Liu
tags: AI/ML, GenAI, LLM, PyTorch
category: Software tools & optimizations
language: English
myst:
  html_meta:
    "description lang=en": "Discover SGLang, a fast serving framework designed for large language and vision-language models on AMD GPUs, supporting efficient runtime and a flexible programming interface."
    "author": "Michael Zhang"
    "keywords": "LLM, vision-language models, backend optimization, SGLang, RadixAttention, Tensor Parallelism, multi-GPU, PyTorch, AI models, AMD, GPU, MI300, MI250, ROCm, gpt, llama"
    "property=og:locale": "en_US"
---

# SGLang: Fast Serving Framework for Large Language and Vision-Language Models on AMD GPUs

In the rapidly evolving landscape of artificial intelligence, the ability to deploy large language models (LLMs) and vision-language models (VLMs) efficiently is crucial for real-time applications. [SGLang](https://github.com/sgl-project/sglang) is an open-source framework designed to meet these demands by delivering fast backend runtime, a flexible frontend language, and extensive model support for a variety of LLMs and VLMs.

This blog will dive into the core features of SGLang, highlight its performance-optimized backend, and showcase its flexible serving capabilities—giving you the tools to maximize the potential of generative AI models in your applications using ROCm on AMD GPUs.

Here’s what you will learn in this blog: we will first introduce you to SGLang, highlighting its use cases and key features and advantages. We will then show you how to setup and deploy SGLang using ROCm on AMD GPUs. Following, you will learn how SGLang can help you optimize your inference speed and efficiency using quantized models, and how to use SGLang with multi-modal models like LLaVA NeXT. We will then cover the SGLang frontend language, its support of JSON Decoding, tensor parallelism (TP), data parallelism (DP), and various quantization options.

## What is SGLang?

[SGLang](https://github.com/sgl-project/sglang) is a comprehensive framework developed for efficient serving of large-scale language and vision-language models. It is designed to enhance user control and interaction speed with the model through an optimized backend runtime and a user-friendly frontend language (see also the [SGLang original paper](https://arxiv.org/abs/2312.07104), and the [LMSYS SGLang blog post](https://lmsys.org/blog/2024-01-17-sglang/)).

### Why SGLang?

As generative AI models grow in complexity and size, efficiently serving them becomes a significant challenge, especially for low latency and high throughput. SGLang addresses these challenges by providing a highly optimized runtime that leverages GPU capabilities, making it possible to deploy state-of-the-art LLMs and VLMs effectively. SGLang is designed to simplify the deployment process and reduces operational costs.

### Use Cases

SGLang is suitable for a wide range of applications, including:

1. **Interactive AI Assistants**: Leverage fast response times for real-time interaction.

1. **Multimodal Applications**: Seamlessly integrate vision and language capabilities to enable applications such as video captioning, interactive storytelling, and more.

1. **Scalable Backend for Generative AI**: Deploy multimodal AI models in the cloud, to scale and support high throughput and large user bases.

## Key Features of SGLang

### Fast Backend Runtime

- **RadixAttention for Prefix Caching:**
  
  RadixAttention structures and automates the reuse of Key-Value (KV) caches during runtime by storing them in a radix tree data structure. It retains and manages the KV cache after each generation request, allowing different prompts with shared prefixes to reuse the KV cache. The radix tree enables efficient prefix search, insertion, and eviction. This approach reduces redundant memory usage and computation time, enhancing performance without requiring manual configurations. In addition, Least Recently Used (LRU) eviction policy and a cache-aware scheduling policy implemented in RadixAttention manages the GPU memory constraints and improves cache hit rate.

- **Jump-Forward Constrained Decoding for Efficiency:**

  Jump-Forward Constrained Decoding allows the model to skip unnecessary computations during generation.

- **Continuous Batching for Optimal GPU Utilization:**

  Continuous Batching dynamically adjusts batch sizes to optimize GPU utilization.

- **Optimizations:**
  
  - **Paged Attention:** Reduces memory usage by partitioning the attention matrix into manageable blocks or pages, allowing the model to handle longer sequences efficiently without exceeding memory limitations.
  
  - **Tensor Parallelism:** Distributes the computation of neural network layers across multiple GPUs by splitting tensors along specific dimensions, enabling parallel processing of large models that exceed the memory capacity of a single GPU.
  
  - **FlashInfer Kernels:** Highly optimized GPU kernels designed for rapid inference. They leverage low-level hardware optimizations to accelerate common neural network operations, significantly reducing latency and improving throughput.
  
  - **Quantization:** Reduces the precision of model parameters and computations (e.g., using 8-bit integers instead of 32-bit floats), model size, and computational load with minimal impact on accuracy. SGLang supports various quantization methods, including Activation-Aware Weight Quantization (AWQ) and Generative Pre-trained Transformer Quantization (GPTQ), and utilizes data types such as INT4 and FP8. These quantization methods and data types can be mixed and matched to optimize model performance and efficiency.

**Note:** *FlashInfer support for ROCm (AMD GPUs) is currently under development.*

### Flexible Frontend Language

- **Intuitive Interface:** Provides an intuitive interface for programming Large Language Models (LLMs) with advanced prompting, control flows, multimodal inputs, and parallelism.
  
- **Complex Interactions:** Enables complex interactions with LLMs including chained generation calls, integration with external tools, and control flow management.

### Extensive Model Support

- **Supported Models:** Supports major generative models like Llama, Mistral, Grok, LLaVA, and QWen, as well as embedding models like e5-mistral.
  
- **Extensibility:** Easily extendable to support new models.

For more detailed information, see [SGLang blog post](https://lmsys.org/blog/2024-01-17-sglang/).

## Docker Setup on AMD GPUs

To run SGLang on AMD GPUs, you will need:

- [AMD GPU Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

- [ROCm 6.2+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)

- [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)

The simplest way to deploy SGLang on AMD GPUs is by using the prebuilt Docker image. Latest instructions are available in the [SGLang Installation Guide](https://sgl-project.github.io/start/install.html#method-3-using-docker).

### Generic Build Steps for ROCm Docker Image

To build your own Docker image with ROCm support, follow these steps:

1. Clone the SGLang Repository:

    ```bash
    git clone https://github.com/sgl-project/sglang.git
    ```

1. Build the Docker Image:

    Navigate to the `docker` directory in the cloned repository and run:

    ```bash
    cd sglang/docker
    docker build -t sglang-rocm:latest -f Dockerfile.rocm .
    ```

1. Launch the Docker Container:

    Start the container with the following command:

    ```bash
    docker run -it --ipc=host --cap-add=SYS_PTRACE --network=host \
    --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
    --group-add video --privileged -w /workspace sglang-rocm:latest
    ```

For prebuilt options and further details, refer to the [official SGLang documentation](https://sgl-project.github.io/start/install.html).

## Quick Start with SGLang

To help you get started with SGLang, we'll use the Meta-Llama-3.1-8B-Instruct model as an example. This model is a powerful language model developed by Meta and is well-suited for various language generation tasks. By following the steps below, you can quickly set up SGLang and start interacting with the Llama model on your local machine.

To run `Meta-Llama-3.1-8B-Instruct`, you need:

- Hugging Face Access Token: This blog requires a [Hugging Face](https://huggingface.co/) account with a generated [User Access Token](https://huggingface.co/docs/hub/security-tokens).

- Access to the `Llama-3.1` Models on Hugging Face: These are [gated models](https://huggingface.co/docs/hub/models-gated) on Hugging Face. To request access, see [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Steps to Get Started

1. Login to Hugging Face:

   After gaining access to the model, log in to Hugging Face using the CLI:

   ```bash
   huggingface-cli login
   ```

1. Start the SGLang Server:

   Launch an SGLang server with a simple command to host an LLM on your local machine:

   ```bash
   python -m sglang.launch_server \
     --model-path meta-llama/Llama-3.1-8B-Instruct \
     --port 30000
   ```

1. Generate Text:

   Once the server is running, open another terminal and send requests to generate text:

   ```bash
   curl http://localhost:30000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Once upon a time,",
       "sampling_params": {
         "max_new_tokens": 16,
         "temperature": 0
       }
     }'
   ```

   **Sample Output:**

   ```json
   {
     "text": " in a small village nestled in the rolling hills of the countryside, there lived a",
     "meta_info": {
       "prompt_tokens": 6,
       "completion_tokens": 16,
       "completion_tokens_wo_jump_forward": 16,
       "cached_tokens": 5,
       "finish_reason": {
         "type": "length",
         "length": 16
       },
       "id": "ab5db50b58734d5884b57b3e52a52302"
     }
   }
   ```

### Using Quantized Models

To enhance performance and leverage AMD's advancements, you can use the following optimized models provided by AMD Quark:

- [amd/Meta-Llama-3.1-405B-Instruct-FP8-KV](https://huggingface.co/amd/Meta-Llama-3.1-405B-Instruct-FP8-KV)
- [amd/Meta-Llama-3.1-70B-Instruct-FP8-KV](https://huggingface.co/amd/Meta-Llama-3.1-70B-Instruct-FP8-KV)
- [amd/Meta-Llama-3.1-8B-Instruct-FP8-KV](https://huggingface.co/amd/Meta-Llama-3.1-8B-Instruct-FP8-KV)

These models are quantized from the original models using AMD's Quark tool, resulting in improved inference speed and efficiency. By utilizing FP8-KV quantization with these models, you can achieve better performance while maintaining high accuracy.

**Example with AMD Optimized Model:**

```bash
python -m sglang.launch_server \
  --model-path amd/Meta-Llama-3.1-405B-Instruct-FP8-KV \
  --tp 8 \
  --quant fp8 \
  --port 30000
```

This command launches the SGLang server using the Meta-Llama-3.1-405B-Instruct-FP8-KV model, enabling tensor parallelism across 8 GPUs (--tp 8) and applying FP8 quantization (--quant fp8). This configuration is ideal for showcasing the performance advantages of AMD's MI300X GPUs, which are designed to handle large-scale models efficiently.

Note: The `--tp 8` flag specifies that the model will be split across 8 GPUs using tensor parallelism. The `--quant` fp8 flag enables FP8 weight quantization, reducing memory usage and computational load.

## Serving LLaVA NeXT Model

SGLang supports serving multi-modal models like [LLaVA NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), which integrates vision and language capabilities. Here's how you can set up and run the LLaVA NeXT 8B model using SGLang:

1. Start the SGLang Server with the LLaVA NeXT Model:

   ```bash
   python -m sglang.launch_server \
     --model-path lmms-lab/llama3-llava-next-8b \
     --port 30000 \
     --tp-size 1 \
     --chat-template llava_llama_3
   ```

   This command launches the server on port `30000`, using the specified model path and chat template suitable for LLaVA NeXT.

1. Interact with the Model Using an API Request:

   You can send a request to the server using `curl`:

   ```bash
   curl http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "default",
       "messages": [
         {
           "role": "user",
           "content": [
             {
               "type": "image_url",
               "image_url": {
                 "url": "https://www.ilankelman.org/stopsigns/australia.jpg"
               }
             },
             {
               "type": "text",
               "text": "What is the content of the image?"
             }
           ]
         }
       ],
       "temperature": 0
     }'
   ```

   Note: The `Authorization` header has been omitted since it's not required for local testing. If your setup requires authentication, you can include the `Authorization` header with your API key.

   This request asks the model to analyze an image provided via a URL and answer the question, "What is the content of the image?"

   ![Example Image](https://www.ilankelman.org/stopsigns/australia.jpg)

   Sample Output:

   ```text
   USER: What is the content of the image?
   ASSISTANT: The image shows a stop sign at an intersection, with a black car driving past it. The stop sign is located on a street corner, and there are buildings and shops in the background. The architecture suggests an urban setting.
   ```

This example demonstrates how SGLang can serve multi-modal models, allowing you to build applications that understand and interpret both text and images.

## Frontend: Structured Generation Language (SGLang)

The frontend language in SGLang can be used with both local models and API models. It serves as an alternative to the OpenAI API, offering an intuitive interface for complex prompting workflows.

### Language Features

To begin with, import `sglang`:

```python
import sglang as sgl
```

SGLang provides simple primitives such as `gen`, `select`, `fork`, and `image`. You can implement your prompt flow in a function decorated by `@sgl.function`. You can then invoke the function with `run` or `run_batch`. The system will manage the state, chat template, parallelism, and batching for you.

### Multi-Modality

Use `sgl.image` to pass an image as input:

```python
@sgl.function
def image_qa(s, image_file, question):
    s += sgl.user(sgl.image(image_file) + question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))
```

For a complete example, see `local_example_llava_next.py` located under the `src` folder in the `sglang` directory. The images used are under the `images` folder within the same directory.

#### Running the Multi-Modality Example

First, clone this blog repository and navigate to the sglang directory:

```bash
git clone https://github.com/ROCm/rocm-blogs.git
cd rocm-blogs/blogs/artificial-intelligence/sglang
```

You can then run the multi-modality example by executing:

```bash
python3 src/local_example_llava_next.py
```

**Code:**

```python

import sglang as sgl
from sglang.lang.chat_template import get_chat_template

@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path) + question)
    s += sgl.assistant(sgl.gen("answer"))

def single():
    state = image_qa.run(
        image_path="images/cat.jpeg", question="What is this?", max_new_tokens=128
    )
    print(state["answer"], "\n")

def stream():
    state = image_qa.run(
        image_path="images/cat.jpeg",
        question="What is this?",
        max_new_tokens=64,
        stream=True,
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()

def batch():
    states = image_qa.run_batch(
        [
            {"image_path": "images/cat.jpeg", "question": "What is this?"},
            {"image_path": "images/dog.jpeg", "question": "What is this?"},
        ],
        max_new_tokens=128,
    )
    for s in states:
        print(s["answer"], "\n")

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    runtime = sgl.Runtime(model_path="lmms-lab/llama3-llava-next-8b")
    runtime.endpoint.chat_template = get_chat_template("llama-3-instruct-llava")

    # Or you can use the 72B model
    # runtime = sgl.Runtime(model_path="lmms-lab/llava-next-72b", tp_size=8)
    # runtime.endpoint.chat_template = get_chat_template("chatml-llava")

    sgl.set_default_backend(runtime)
    print(f"chat template: {runtime.endpoint.chat_template.name}")

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()

    runtime.shutdown()
```

**Explanation:**

- **Function `image_qa`:**
  - Decorated with `@sgl.function`, it defines the prompt flow for image question-answering.
  - Uses `sgl.image(image_path)` to include the image in the prompt.
  - Generates an answer using `sgl.gen("answer")`.

- **Functions `single`, `stream`, and `batch`:**
  - `single()`: Runs a single request and prints the answer.
  - `stream()`: Streams the output as it is generated.
  - `batch()`: Runs a batch of requests and prints the answers.

**Sample Output:**

  ```bash
  chat template: llama-3-instruct-llava

  ========== single ==========

  This is a cartoon-style or digitally rendered image of a cat wearing sunglasses and a pink jacket with the hood up, set against a pink background. The design appears to be playful and eccentric, using bright colors and human-like accessories on the cat to create a whimsical character.

  ========== stream ==========

  This is an image of an anthropomorphized cat character designed to look like a young person wearing a vibrant pink hoodie with the hood on and cool shades on its eyes. The cat has human-like facial features, but its body retains characteristics typical of a feline. The image has a playful and imaginative aesthetic,

  ========== batch ==========

  This is an image of an anthropomorphized cat character. It's been given human-like features such as clothing (a pink hoodie) and accessories (sunglasses). The cat has a cute and somewhat whimsical style, with the hoodie and sunglasses giving it a playful, almost rebellious look as if it is embracing a human cool style. It's a creative representation not seen in real life for entertainment or artistic purposes.

  This is a stylized image showing a dog dressed up as a person, wearing a hoodie and sunglasses. The dog appears to be creatively photoshopped or digitally created to mimic the pose and clothing style of a human. The image is likely meant to be humorous or artistic, depicting the dog as if it were a person enjoying the outfit.
  ```

### JSON Decoding

SGLang allows you to specify a JSON schema using regular expressions, enabling precise control over the format of the generated JSON data. This is particularly useful when you need the model to generate outputs that conform to a specific structure.

**Example:**

  ```python
  character_regex = (
      r"""{\n"""
      + r"""    "name": "[\w\d\s]{1,16}",\n"""
      + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
      + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
      + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
      + r"""    "wand": {\n"""
      + r"""        "wood": "[\w\d\s]{1,16}",\n"""
      + r"""        "core": "[\w\d\s]{1,16}",\n"""
      + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
      + r"""    },\n"""
      + r"""    "alive": "(Alive|Deceased)",\n"""
      + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
      + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
      + r"""}"""
  )
  ```

In this example, `character_regex` defines a JSON schema for a character in the Harry Potter universe, specifying the expected fields and acceptable values using regular expressions.

**Function Definition:**

  ```python
  @sgl.function
  def character_gen(s, name):
      s += (
          name
          + " is a character in Harry Potter. Please fill in the following information about this character.\n"
      )
      s += "The constrained regex is:\n"
      s += character_regex + "\n"
      s += "The JSON output is:\n"
      s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
  ```

This function prompts the model to generate a JSON object that matches the specified schema for the given character name.

**Complete Example (`json_decode.py`):**

The code is located under the `src` folder in the `sglang` directory.

  ```python
  """
  Usage:
  python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
  python src/json_decode.py
  """

  from enum import Enum
  from pydantic import BaseModel
  import sglang as sgl
  from sglang.srt.constrained import build_regex_from_object

  character_regex = (
      r"""{\n"""
      + r"""    "name": "[\w\d\s]{1,16}",\n"""
      + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
      + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
      + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
      + r"""    "wand": {\n"""
      + r"""        "wood": "[\w\d\s]{1,16}",\n"""
      + r"""        "core": "[\w\d\s]{1,16}",\n"""
      + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
      + r"""    },\n"""
      + r"""    "alive": "(Alive|Deceased)",\n"""
      + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
      + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
      + r"""}"""
  )

  @sgl.function
  def character_gen(s, name):
      s += (
          name
          + " is a character in Harry Potter. Please fill in the following information about this character.\n"
      )
      s += "The constrained regex is:\n"
      s += character_regex + "\n"
      s += "The JSON output is:\n"
      s += sgl.gen("json_output", max_tokens=256, regex=character_regex)

  def driver_character_gen():
      state = character_gen.run(name="Hermione Granger")
      print(state.text())

  class Weapon(str, Enum):
      sword = "sword"
      axe = "axe"
      mace = "mace"
      spear = "spear"
      bow = "bow"
      crossbow = "crossbow"

  class Wizard(BaseModel):
      name: str
      age: int
      weapon: Weapon

  @sgl.function
  def pydantic_wizard_gen(s):
      s += "Give me a description about a wizard in the JSON format.\n"
      s += sgl.gen(
          "character",
          max_tokens=128,
          temperature=0,
          regex=build_regex_from_object(Wizard),  # Requires pydantic >= 2.0
      )

  def driver_pydantic_wizard_gen():
      state = pydantic_wizard_gen.run()
      print(state.text())

  if __name__ == "__main__":
      sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
      driver_character_gen()
      # Uncomment the following line to run the Pydantic example
      # driver_pydantic_wizard_gen()
  ```

**Usage:**

 Run the `json_decode.py` Script:

   ```bash
   python src/json_decode.py
   ```

**Sample Output:**

  ```json
  Hermione Granger is a character in Harry Potter. Please fill in the following information about this character.
  The constrained regex is:
  {
      "name": "[\w\d\s]{1,16}",
      "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",
      "blood status": "(Pure-blood|Half-blood|Muggle-born)",
      "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",
      "wand": {
          "wood": "[\w\d\s]{1,16}",
          "core": "[\w\d\s]{1,16}",
          "length": [0-9]{1,2}\.[0-9]{0,2}
      },
      "alive": "(Alive|Deceased)",
      "patronus": "[\w\d\s]{1,16}",
      "bogart": "[\w\d\s]{1,16}"
  }
  The JSON output is:
  {
      "name": "Hermione Granger",
      "house": "Gryffindor",
      "blood status": "Muggle-born",
      "occupation": "student",
      "wand": {
          "wood": "Vine",
          "core": "Dragon heartstring",
          "length": 10.75
      },
      "alive": "Alive",
      "patronus": "Otter",
      "bogart": "Failure"
  }
  ```

**Note:**

- Using Regular Expressions:
  - The `regex` parameter in `sgl.gen` ensures that the generated output matches the specified pattern, helping to enforce the structure and format of the JSON data.

- Using Pydantic Models:
  - `build_regex_from_object` automatically generates a regex pattern from a Pydantic model, simplifying the process of defining complex schemas.
  - Requires Pydantic version 2.0 or higher.

## Advanced Multi-GPU Deployment

SGLang supports both **tensor parallelism (TP)** and **data parallelism (DP)** for large-scale deployment. For more information on features of SGLang, see [SGLang documentation](https://sgl-project.github.io/)

- To enable multi-GPU tensor parallelism with two GPUs, add `--tp 2`.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --tp 2
  ```

- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if each GPU has enough memory to fit the entire model. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --dp 2 --tp 2
  ```

These deployment options allow you to easily scale your serving architecture in line with your needs, ensuring that large models can run efficiently across multiple GPUs and even across different servers.

## Enable Quantization

SGLang offers various quantization options to optimize model performance and efficiency. Quantization reduces the precision of model parameters and computations, decreasing model size and computational load with minimal impact on accuracy. Here's how to enable and configure quantization:

- **Enable FP8 Weight Quantization:**

  To enable FP8 weight quantization, add `--quantization fp8` when using an FP16 checkpoint or directly load an FP8 checkpoint without specifying any arguments.

  ```bash
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8
  ```

- **Enable FP8 KV Cache Quantization:**

  To enable FP8 quantization for the KV cache, add `--kv-cache-dtype fp8_e5m2`.

  ```bash
  python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8_e5m2
  ```

## Summary

In this blog post we introduced you to SGLang and its features, and showed you how to setup SGLang using ROCm on AMD GPUs. We demonstrated how you can use SGLang to optimize your inference using quantized models, and how to use SGLang with the LLaVA NeXT multi-modal model. We also presented the SGLang frontend language, its support of JSON Decoding, tensor parallelism (TP), data parallelism (DP), and various quantization options.

SGLang provides a highly optimized and scalable solution for serving large language models and vision-language models on AMD GPUs. Its powerful runtime and flexible frontend simplify the deployment of large-scale AI models and reduce complexity and cost. This makes SGLang an excellent tool for developers aiming to bring generative AI capabilities to production using AMD hardware.

## Resources

- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang Documentation](https://sgl-project.github.io/)
- [AMD ROCm Platform](https://rocm.docs.amd.com/)

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
