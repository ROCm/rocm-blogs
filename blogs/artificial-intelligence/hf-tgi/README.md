<head>
  <meta charset="UTF-8">
  <meta name="description" content="Efficient deployment of large language models with Hugging
  Face text generation inference empowered by AMD GPUs">
  <meta name="author" content="Douglas Jia">
  <meta name="keywords" content="AMD GPU, MI300, MI250, ROCm, blog, Hugging Face, LLM,
  language model, TGI, Text Generation Inference">
</head>

# Efficient deployment of large language models with Text Generation Inference on AMD GPUs

**Author:** [Douglas Jia](../../authors/douglas-jia.md)\
**First published:** 26 Jan 2024

[Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) is a
toolkit for deploying and serving Large Language Models (LLMs) with unparalleled efficiency. TGI is
tailored for popular open-source LLMs, such as Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and T5.
Optimizations include tensor parallelism, token streaming using Server-Sent Events (SSE), continuous
batching, and optimized transformers code. It has a robust feature set that includes quantization,
safetensors, watermarking (for determining if text is generated from language models), logits warper,
and support for custom prompt generation and fine-tuning.

TGI is a critical framework component of projects like Hugging Chat, Open Assistant, and nat.dev,
which speaks to its performance in production environments.

In this tutorial, we show you how to deploy and serve LLMs with TGI on AMD GPUs. Adapted from the
[official Hugging Face tutorial](https://huggingface.co/docs/text-generation-inference/index), this
tutorial incorporates additional insights for a more comprehensive learning experience. Contributions
from the original tutorial are duly acknowledged.

## Deploy LLMs with TGI

To leverage the TGI framework on ROCm-enabled AMD GPUs, you can choose between using the
official Docker container or building TGI from source code. We use the Docker approach, as it
streamlines the setup process and mitigates software compatibility issues. Alternatively, if you prefer
building from source, you can find detailed instructions on
[Hugging Face](https://huggingface.co/docs/text-generation-inference/installation).

On your Linux machine with ROCm-enabled AMD GPUs, run the following commands in the terminal
to deploy the Docker container with our specified model, `tiiuae/falcon-7b-instruct`:

```sh
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.3-rocm --model-id $model
```

These commands build a TGI server with the specified model that is ready to handle your requests. For
a comprehensive list of supported models, refer to
[supported models](https://huggingface.co/docs/text-generation-inference/supported_models).

If the model size exceeds the capacity of a single GPU and cannot be accommodated entirely, consider incorporating the `--num-shard n` flag in the `docker run` command for text-generation-inference. Here, `n` represents the number
of GPUs at your disposal. This flag activates tensor parallelism, effectively dividing the model into
shards that are distributed across the available GPUs.

## Query LLMs deployed on server

In the preceding step, you set up a server that actively listens for incoming requests. Now you can
open a new terminal to interact with this server (be sure to keep the original server running throughout
this process). To query the server, you can use various methods; we demonstrate two commonly used
options: Python `requests` package and `curl` command line.

### Python requests package

Start a Python session by running the `python3` command in your terminal. Then run the following
Python code:

```python
import requests

headers = {
    "Content-Type": "application/json",
}

data = {
    'inputs': 'What is the best way to learn Deep Learning?',
    'parameters': {
        'max_new_tokens': 200,
        'temperature': 0.1,
    },
}

response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)
print(response.json())
```

Output:

```sh
{'generated_text': '\nThe best way to learn Deep Learning is through a combination of hands-on practice and structured learning. Some popular resources for learning Deep Learning include online courses, such as those offered by Coursera or edX, and textbooks such as "Deep Learning" by Goodfellow, Bengio, and Courville. Additionally, participating in online coding challenges and competitions can help reinforce your knowledge and improve your skills.'}
```

You can change the `inputs` field to test different prompts. To try different generation configurations,
tune the parameters (e.g., `max_new_tokens` and `temperature`). To view all tunable parameters, refer
to [this list](https://huggingface.co/docs/transformers/main_classes/text_generation).

### Curl command line

In the terminal, you can directly query the server with the following Curl command:

```sh
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is the best way to learn Deep Learning?","parameters":{"max_new_tokens":200,"temperature":0.1}}' \
    -H 'Content-Type: application/json'
```

Output:

```sh
{"generated_text":"\nThe best way to learn Deep Learning is through a combination of hands-on practice and structured learning. Some popular ways to learn Deep Learning include taking online courses, attending workshops, and working on personal projects. It's also important to stay up-to-date on the latest research and developments in the field."}
```

You might observe variations in the output between the two methods. This is because we are using a
temperature value of `0.1`, which encourages diversity of the text generation. To get more deterministic
output, you can increase the temperature value to `1`.
