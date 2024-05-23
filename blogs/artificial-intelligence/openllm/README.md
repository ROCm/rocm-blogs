---
blogpost: true
date: 1 May 2024
author: Fabricio Flores
tags: LLM, Serving
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Step-by-Step Guide to OpenLLM on AMD GPUs"
    "keywords": "LLM, OpenLLM, vLLM, ROCm, AMD, GPU, MI300, MI250"
    "property=og:locale": "en_US"
---

# Step-by-Step Guide to Use OpenLLM on AMD GPUs

## Introduction

[OpenLLM](https://github.com/bentoml/OpenLLM) is an open-source platform designed to facilitate the deployment and utilization of large language models (LLMs), supporting a wide range of models for diverse applications, whether in cloud environments or on-premises. In this tutorial, we will guide you through the process of starting an LLM server using OpenLLM, enabling interaction with the server from your local machine, with special emphasis on leveraging the capabilities of AMD GPUs.

You can find files related to this blog post in the
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/main/docs/artificial-intelligence/openllm).

## Requirements

### Operating system and Hardware and Software requirements

* AMD GPU: List of supported OS and hardware on the [ROCm documentation page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

* Anaconda: [Install anaconda for Linux](https://www.anaconda.com/download/).

* ROCm version: 6.0 Refer to [ROCm installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html).

* Docker: [Docker engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

* OpenLLM: Version 0.4.44 [official documentation](https://github.com/bentoml/OpenLLM).

* vLLM: For using vLLM as runtime. [vLLM Official documentation](https://docs.vllm.ai/en/latest/index.html)

## Preliminaries

To ensure a smooth and efficient development process, we divide the process into two steps. First, create a dedicated Python environment for API testing and second using a Docker image for hosting our OpenLLM server.

### Create a conda environment

Let's begin with setting up a Conda environment tailored for OpenLLM. Launch your Linux terminal and execute the following command.

```bash
conda create --name openllm_env python=3.11
```

Let's also install OpenLLM and JupyterLab inside the environment. First, activate the environment:

```bash
conda activate openllm_env
```

and then run:

```bash
pip install openllm==0.4.44
pip install jupyterlab
```

### OpenLLM runtimes: PyTorch and vLLM

Different LLMs may support multiple runtime implementations that allow for faster computations or reduced memory footprint. A runtime is the underlying framework that provides the computational resources to run LLMs while also handling tasks required for processing the input and generating the response.

OpenLLM provides integrated support for various model runtimes, including **PyTorch** and **vLLM**. When using PyTorch runtime (backend), OpenLLM performs computations within the PyTorch framework.

Conversely, with vLLM as backend, OpenLLM uses a runtime specifically created to deliver high throughput and efficient memory management for executing and serving LLMs. vLLM is optimized for inference, incorporating enhancements, such as continuous batching and PagedAttention, that result in fast prediction times.

OpenLLM allows us to select the desired runtime by setting the option `--backend pt` or `--backend vllm` for PyTorch or vLLM respectively when starting an OpenLLM server.

For additional information about the available options for OpenLLM you can run the command `openllm -h` and also read OpenLLM's [official documentation](https://github.com/bentoml/OpenLLM?tab=readme-ov-file#install-openllm).

### Building a custom Docker image for OpenLLM with Pytorch and vLLM backend support

Let's begin by creating a custom Docker image that will serve as the operational environment for our OpenLLM server. We are leveraging [vLLM ROCm support](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#option-3-build-from-source-with-docker) to build a custom Docker image for our OpenLLM server.

Start by cloning the official [vLLM GitHub repository](https://github.com/vllm-project/vllm). On the same terminal that we used before, to create the Python environment (or a new one), run the command:

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
```

Inside the `vllm` directory, let's modify the content on the `Dockerfile.rocm` file. Let's add the following code before the final instruction `CMD ["/bin/bash"]` (in between line 107 and 109) on the `Dockerfile.rocm` file.

```bash
# Installing OpenLLM and additional Python packages
RUN python3 -m pip install openllm==0.4.44 
RUN python3 -m pip install -U pydantic

#Setting the desired visible devices when running OpenLLM
ENV CUDA_VISIBLE_DEVICES=0

# OpenLLM server runs in port 3000 by default
EXPOSE 3000
```

so that our custom `Dockerfile.rocm` would look like

```bash
# default base image
ARG BASE_IMAGE="rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1"

FROM $BASE_IMAGE
...
#The rest of the original Dockerfile.rocm file 
...

# Installing OpenLLM and additional Python packages
RUN python3 -m pip install openllm==0.4.44 
RUN python3 -m pip install -U pydantic

#Setting the desired visible devices when running OpenLLM
ENV CUDA_VISIBLE_DEVICES=0

# OpenLLM server runs in port 3000 by default
EXPOSE 3000

CMD ["/bin/bash"]
```

Let's create a new Docker image using the Dockerfile above. Alternatively, you can also get the `Dockerfile.rocm` file from [here](https://github.com/ROCm/rocm-blogs/tree/main/docs/artificial-intelligence/openllm/src) and replace the one located on the `vllm` folder.

```bash
docker build -t openllm_vllm_rocm -f Dockerfile.rocm .
```

we are naming the new image `openllm_vllm_rocm`.

Building the image might take a few minutes. If the process completes without any errors, a new Docker image will be available on your local system. To verify this run the following command:

```bash
sudo docker images
```

where the output will contain something similar to:

```bash
REPOSITORY                TAG       IMAGE ID       CREATED       SIZE
openllm_vllm_rocm         latest    695ed0675edf   2 hours ago   56.3GB
```

## Starting the server and testing different models

First, let's return to our original work directory:

```bash
cd ..
```

Start a container using the following command:

```bash
sudo docker run -it --rm -p 3000:3000 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --shm-size 8G -v $(pwd):/root/bentoml openllm_vllm_rocm
```

Let's describe some of the options for the command used to start the container:

* `-p 3000:3000`: This option publishes the container's port to the host. In this case it maps the port 3000 on the container to the port 3000 on the host. This allow us to access the OpenLLM server that runs on port 3000 inside the container.

* `--device=/dev/kfd --device=/dev/dri`: These options give access to specific devices on the host. `--device=/dev/kfd` is associated with AMD GPU devices and `--device=/dev/dri` is related to devices with direct access to the graphics hardware.

* `--group-add=video`: This options allows the container to have necessary permissions to access video hardware directly.

* `-v $(pwd):/root/bentoml`: This mounts the volume from the host to the container. It maps the current directory (`$(pwd)`) on the host to `/root/bentoml` inside the container. When OpenLLM downloads new models they are stored inside the `/root/bentoml` directory. Setting the volume saves the models on the host to avoid downloading them again.

* `openllm_vllm_rocm`: This is the name of our custom Docker image.

The rest of the options configure security preferences, grant more privileges and adjust resources usage.

### Serving facebook/opt-1.3b model

Let's start an OpenLLM server with the `facebook/opt-1.3b` model and PyTorch backend. Inside our running container use the following command:

```bash
openllm start facebook/opt-1.3b --backend pt
```

The command above starts an OpenLLM server with the `facebook/opt-1.3b` model and PyTorch backend (`--backend pt`). The command also automatically downloads the model if it doesn't already exist. OpenLLM supports several models, you can take a look at the [official documentation](https://github.com/bentoml/OpenLLM?tab=readme-ov-file#-supported-models) for the list of supported models.

If the server is running successfully you will see an output similar to this:

```bash
🚀Tip: run 'openllm build facebook/opt-1.3b --backend pt --serialization legacy' to create a BentoLLM for 'facebook/opt-1.3b'
2024-04-11T17:04:18+0000 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "_service:svc" can be accessed at http://localhost:3000/metrics.
2024-04-11T17:04:20+0000 [INFO] [cli] Starting production HTTP BentoServer from "_service:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

The previous command starts the server at http://0.0.0.0:3000/ (with port 3000 by default) and OpenLLM downloads the model to `/root/bentoml` inside the container if the model is not present.

With the server running, we can interact with it by either using the web UI at http://0.0.0.0:3000/ or using the OpenLLM’s built-in Python client.

Let's try it using the Python client. For this purpose, we are using the Python environment we created before. Open a new terminal and activate our environment:

```bash
conda activate openllm_env
```

Finally, start a new JupyterLab session:

```bash
jupyter lab
```

In your running notebook, you can paste the instructions below, alternatively you can download the [openllm_test notebook](https://github.com/ROCm/rocm-blogs/tree/main/docs/artificial-intelligence/openllm).

Inside the notebook run:

```python
import openllm

#Sync API
client = openllm.HTTPClient('http://localhost:3000', timeout=120)

#generate streaming
for it in client.generate_stream('What is a Large Language Model?', max_new_tokens=120,):
  print(it.text, end="")
```

and the output will be something similar to:

```text
A Large Language Model (LLM) is a model that uses a large number of input languages to produce a large number of output languages. The number of languages used in the model is called the language model size.

In a large language model, the number of input languages is typically large, but not necessarily unlimited. For example, a large language model can be used to model the number of languages that a person can speak, or the number of languages that a person can read, or the number of languages that a person can understand.

A large language model can be used to model the number of languages that a
```

### Serving databricks/dolly-v2-3b model

Let's try a different model and also let's make the generation less random by setting a lower `temperature` value. First, stop the previous server (you might also need to kill all the processes on port 3000 using `kill -9 $(lsof -ti :3000)`) and then start a new one with the command:

```bash
openllm start databricks/dolly-v2-3b --backend pt --temperature=0.1
```

Let's test this model now. Go back to the Jupyter notebook and run:

```python
import openllm

#Sync API
client = openllm.HTTPClient('http://localhost:3000', timeout=120)

#generate streaming
for it in client.generate_stream('What industry is Advanced Micro Devices part of?', max_new_tokens=120,):
  print(it.text, end="")
```

the output will be something similar to:

```text
AMD is a semiconductor company based in Sunnyvale, California. AMD designs and manufactures microprocessors, GPUs (graphics processing units), and memory controllers. AMD's largest product line is its microprocessors for personal computers and video game consoles. AMD's Radeon graphics processing unit (GPU) is used in many personal computers, video game consoles, and televisions.
```

### Serving Mistral-7B-Instruct-v0.1 model

Finally, lets serve a more capable model and use vLLM as backend (`--backend vllm`). Stop the previous server and run the following command

```bash
openllm start mistralai/Mistral-7B-Instruct-v0.1 --backend vllm
```

and test it by running the following:

```python
import openllm

#Sync API
client = openllm.HTTPClient('http://localhost:3000', timeout=120)

#generate streaming
for it in client.generate_stream('Create the python code for an autoencoder neural network', max_new_tokens=1000,):
  print(it.text, end="")
```

where the output will be something similar to:

```text
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the autoencoder model
model = Sequential([
    Flatten(input_shape=(28, 28)), # Flatten the input image of size 28x28
    Dense(128, activation='relu'), # Add a dense layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'), # Add another dense layer with 64 neurons and ReLU activation
    Dense(128, activation='relu'), # Add a third dense layer with 128 neurons and ReLU activation
    Flatten(input_shape=(128,)), # Flatten the output of the previous dense layer
    Dense(10, activation='softmax') # Add a final dense layer with 10 neurons (for each digit) and softmax activation
])

# Define the optimizer, loss function, and metric
optimizer = Adam(learning_rate=0.001)
loss_function = SparseCategoricalCrossentropy(from_logits=True) # From logits
metric = SparseCategoricalAccuracy()

# Compile the model
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=[metric])

# Train the model on the training data
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
```

## Conclusion

This blog has highlighted the process of leveraging OpenLLM on AMD GPUs, demonstrating its potential as a powerful tool for deploying Large Language Models. Our testing on AMD hardware showcases the technical compatibility and the scalability offered by this combination.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
