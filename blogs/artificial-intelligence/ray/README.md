---
blogpost: true
date: 1 Apr 2024
author: Vicky Tsang, Logan Grado, Eliot Li
tags: LLM, AI/ML, Generative AI, Tuning, Stable Diffusion
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Scale AI applications with Ray"
    "keywords": "AMD GPU, ROCm, Model Serving, Ray, ML Platform, Workload Orchestration, Generative AI, Tuning"
    "property=og:locale": "en_US"
---

# Scale AI applications with Ray

Most machine-learning (ML) workloads today require multiple GPUs or nodes to achieve the
performance or scale required by applications. However, scaling workloads beyond single node/single GPU workloads is difficult and require some expertise in distributed processing.

[Ray](https://github.com/ray-project/ray/) has developed a platform that allows AI practitioners to
scale their code from a laptop to a cluster of nodes with just a few lines of code. The platform is
designed to support a variety of common ML use cases (described in the next section). Ray is an
open-source project under the Apache 2.0 license.

AMD has been working with Ray to provide support on the ROCm platform. In this blog, we'll describe
how to use Ray to easily scale your AI applications from your laptop to multiple AMD GPUs. You can
find files related to this blog in the
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/ray).

## Use cases

Ray can be applied to many use cases for scaling ML applications, such as:

* LLMs and Gen AI​
* Batch inference​
* Model serving​
* Hyperparameter tuning​
* Distributed training​
* Reinforcement learning​
* ML platform​
* E2E ML workflows​
* Large-scale workload orchestration​

Refer to the [Ray documentation](https://docs.ray.io/en/latest/ray-overview/use-cases.html) for
detailed tutorials on these use cases. We'll explore a few common use cases using AMD GPUs and
ROCm.

## Installing Ray with ROCm support

Below, we briefly describe how to install Ray with ROCm support on a single node.

### Prerequisites

* You'll need a node of [ROCm-supported]( https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) AMD GPUs.
* A [supported Linux distribution](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
* ROCm - see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

### Setup required to run the examples in this blog

In order to facilitate running the examples in this blog, we also provide a dockerfile which will install the prerequisites necessary to run these examples.

* Start from a ROCm base image
* Create a python virtual environment
* Install Ray with ROCm support
* Install other python dependencies (PyTorch, etc)

The `dockerfile` and `docker-compose.yaml` files can be found at this [github folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/ray/docker).  We also include them in Appendix for your convenience.  Just create a folder `docker` and add those two files.  Then you can build and start the docker container, with this blog's directory mounted at `/root`, with:

```shell
cd docker
docker compose run ray-blog
```

```{tip}
Because the blog's directory is mounted into the container, any files you create or edit in the blog directory will be immediately available within the container
```

Alternatively you can build your own python environment (see [running on host](#running-on-host) in the appendix).

### Installing Ray only

If you just want to install `ray` with AMD support and test it with your own code, you can `pip install` the [appropriate wheel here](https://github.com/ROCm/ray/releases/tag/v3.0.0-dev0%2Brocm). These wheels are built with additional ROCm support.

```shell
pip install "ray[data,train,tune,serve] @ https://github.com/ROCm/ray/releases/download/v3.0.0-dev0%2Brocm/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl"
```

```{tip}
Once Ray releases its version with official ROCm support, you'll be able to use the
[Ray installation](https://docs.ray.io/en/latest/ray-overview/installation.html) instructions.
```

## Examples

Now that we have a node ready to use Ray to scale our applications, we can illustrate how it works
for these use cases:

* [Use Ray Train to fine-tune a transformer model](#use-ray-train-to-fine-tune-a-transformer-model)
* [Convert an LLM model into a Ray Serve application](#convert-an-llm-model-into-a-ray-serve-application)
* [Use Ray Serve to serve a Stable Diffusion model](#use-ray-serve-to-serve-a-stable-diffusion-model)
* [Use Ray Tune to tune an XGBoost classifier](#use-ray-tune-to-tune-an-xgboost-classifier)

### Use Ray Train to fine-tune a transformer model

Download the script
[```transformers_torch_trainer_basic.py```](https://github.com/ROCm/ray/blob/master/python/ray/train/examples/transformers/transformers_torch_trainer_basic.py), which uses the Ray Train library to scale the fine-tuning of a
[BERT base model](https://huggingface.co/google-bert/bert-base-cased) using the
[Yelp review dataset](https://huggingface.co/datasets/yelp_review_full) on Hugging Face.

``` tip
You can quickly download this script using `curl`:

    curl https://raw.githubusercontent.com/ROCm/ray/master/python/ray/train/examples/transformers/transformers_torch_trainer_basic.py > transformers_torch_trainer_basic.py
```

Let's use two GPUs to tune the model by setting `num_workers=2` in the last part of the
`transformers_torch_trainer_basic.py` script:

```python
# [4] Build a Ray TorchTrainer to launch `train_func` on all workers
# ==================================================================
trainer = TorchTrainer(
    train_func, scaling_config=ScalingConfig(num_workers=2, use_gpu=True)
)

trainer.fit()
```

We're now ready to run the script:

```bash
python transformers_torch_trainer_basic.py
```

The output should look like this:

```bash
Usage stats collection is enabled by default for nightly wheels. To disable this, run the following command: `ray disable-usage-stats` before starting Ray. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2024-03-06 23:34:17,106 INFO worker.py:1754 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
2024-03-06 23:34:18,177 INFO tune.py:220 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `Trainer(...)`.
2024-03-06 23:34:18,178 INFO tune.py:592 -- [output] This will use the new output engine with verbosity 1. To disable the new output and use the legacy output engine, set the environment variable RAY_AIR_NEW_OUTPUT=0. For more information, please see https://github.com/ray-project/ray/issues/36949

View detailed results here: /root/ray_results/TorchTrainer_2024-03-06_23-34-14
To visualize your results with TensorBoard, run: `tensorboard --logdir /root/ray_results/TorchTrainer_2024-03-06_23-34-14`

Training started without custom configuration.
(RayTrainWorker pid=75298) Setting up process group for: env:// [rank=0, world_size=2]
(TorchTrainer pid=71127) Started distributed worker processes:
(TorchTrainer pid=71127) - (ip=10.216.70.82, pid=75298) world_rank=0, local_rank=0, node_rank=0
(TorchTrainer pid=71127) - (ip=10.216.70.82, pid=75299) world_rank=1, local_rank=1, node_rank=0
Downloading readme: 100%|██████████| 6.72k/6.72k [00:00<00:00, 39.9MB/s]
Downloading data:   0%|          | 0.00/299M [00:00<?, ?B/s]
Downloading data:   1%|▏         | 4.19M/299M [00:00<00:39, 7.45MB/s]
Downloading data:   4%|▍         | 12.6M/299M [00:00<00:20, 14.0MB/s]
Downloading data:   7%|▋         | 21.0M/299M [00:01<00:14, 19.3MB/s]

...
...
...

(RayTrainWorker pid=75298) {'eval_loss': 0.9538511633872986, 'eval_accuracy': 0.589, 'eval_runtime': 3.1428, 'eval_samples_per_second': 318.185, 'eval_steps_per_second': 20.046, 'epoch': 3.0}
(RayTrainWorker pid=75298) {'train_runtime': 39.3335, 'train_samples_per_second': 76.271, 'train_steps_per_second': 4.805, 'train_loss': 1.1883205837673612, 'epoch': 3.0}
100%|██████████| 189/189 [00:39<00:00,  4.81it/s]

Training completed after 0 iterations at 2024-03-06 23:38:20. Total running time: 4min 2s
```

With two GPUs, it takes about 4 minutes to finish the fine-tuning. Let's try to run the same job with
four GPUs (assuming you have at least four GPUs in your system). Do this by changing `num_workers`
from `2` to `4`:

```python
# [4] Build a Ray TorchTrainer to launch `train_func` on all workers
# ==================================================================
trainer = TorchTrainer(
    train_func, scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
)

trainer.fit()
```

Running the script again should produce this output:

```text
Usage stats collection is enabled by default for nightly wheels. To disable this, run the following command: `ray disable-usage-stats` before starting Ray. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2024-03-06 23:49:10,338 INFO worker.py:1754 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
2024-03-06 23:49:11,468 INFO tune.py:220 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `Trainer(...)`.
2024-03-06 23:49:11,469 INFO tune.py:592 -- [output] This will use the new output engine with verbosity 1. To disable the new output and use the legacy output engine, set the environment variable RAY_AIR_NEW_OUTPUT=0. For more information, please see https://github.com/ray-project/ray/issues/36949

View detailed results here: /root/ray_results/TorchTrainer_2024-03-06_23-49-08
To visualize your results with TensorBoard, run: `tensorboard --logdir /root/ray_results/TorchTrainer_2024-03-06_23-49-08`

Training started without custom configuration.
(RayTrainWorker pid=83857) Setting up process group for: env:// [rank=0, world_size=4]
(TorchTrainer pid=83721) Started distributed worker processes:
(TorchTrainer pid=83721) - (ip=10.216.70.82, pid=83857) world_rank=0, local_rank=0, node_rank=0
(TorchTrainer pid=83721) - (ip=10.216.70.82, pid=83858) world_rank=1, local_rank=1, node_rank=0
(TorchTrainer pid=83721) - (ip=10.216.70.82, pid=83859) world_rank=2, local_rank=2, node_rank=0
(TorchTrainer pid=83721) - (ip=10.216.70.82, pid=83860) world_rank=3, local_rank=3, node_rank=0
Map:   0%|          | 0/50000 [00:00<?, ? examples/s]
Map:   2%|▏         | 1000/50000 [00:00<00:13, 3510.56 examples/s]
Map:   0%|          | 0/50000 [00:00<?, ? examples/s] [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)
Map:  40%|████      | 20000/50000 [00:05<00:07, 3945.42 examples/s] [repeated 75x across cluster]
Map:  80%|████████  | 40000/50000 [00:10<00:02, 4170.23 examples/s] [repeated 82x across cluster]
Map:  92%|█████████▏| 46000/50000 [00:11<00:00, 4377.90 examples/s]
Map: 100%|██████████| 50000/50000 [00:12<00:00, 3899.61 examples/s]

...
...
...

(RayTrainWorker pid=83857) 01<00:00, 20.58it/s]
100%|██████████| 96/96 [00:20<00:00,  4.59it/s]
(RayTrainWorker pid=83857) {'eval_loss': 0.9640204906463623, 'eval_accuracy': 0.578, 'eval_runtime': 1.7283, 'eval_samples_per_second': 578.6, 'eval_steps_per_second': 18.515, 'epoch': 3.0}
(RayTrainWorker pid=83857) {'train_runtime': 20.9123, 'train_samples_per_second': 143.456, 'train_steps_per_second': 4.591, 'train_loss': 1.1330984433492024, 'epoch': 3.0}

Training completed after 0 iterations at 2024-03-06 23:49:59. Total running time: 47s
```

With the two additional GPUs, it took 47 seconds for the same job. Using Ray, scaling a job with
additional resources is as simple as changing one parameter in your code.

### Convert an LLM model into a Ray Serve application

We can develop a Ray Serve application locally and deploy it in production on a cluster of AMD GPUs
using just a few lines of code. You can find detailed instructions on the official
[Ray documentation page](https://docs.ray.io/en/latest/serve/develop-and-deploy.html).

We use English-to-French translation as an example for deploying an ML application. First, we create
the Python script, `RayServe_En2Fr_translation_local.py`, based on the script `model.py` from the
[Ray documentation page](https://docs.ray.io/en/latest/serve/develop-and-deploy.html), which can be used to translate English text to French.

```python
# File name: RayServe_En2Fr_translation_local.py
from transformers import pipeline


class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation


translator = Translator()

translation = translator.translate("Hello world!")
print(translation)
```

We can test this script by running it locally:

```bash
python RayServe_En2Fr_translation_local.py
```

```bash
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.21k/1.21k [00:00<00:00, 612kB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 242M/242M [00:02<00:00, 115MB/s]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 147/147 [00:00<00:00, 93.6kB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.32k/2.32k [00:00<00:00, 2.76MB/s]
spiece.model: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 792k/792k [00:00<00:00, 11.7MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.39M/1.39M [00:00<00:00, 5.76MB/s]
Bonjour monde!
```

Next, we convert this script into `RayServe_En2Fr_translation.py`, which supports a Ray Serve
application with [FastAPI](https://docs.ray.io/en/latest/serve/http-guide.html#serve-set-up-fastapi-http) based on the instructions on the [Ray documentation page](https://docs.ray.io/en/latest/serve/develop-and-deploy.html).

```python
# File name: RayServe_En2Fr_translation.py
import ray
from ray import serve
from fastapi import FastAPI

from transformers import pipeline

app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    @app.post("/")
    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation


translator_app = Translator.bind()
```

We stand up the `translator_app` application in the background to serve an LLM model that translates
English to French. We run the script with the `serve run` CLI command, which takes in an import path
formatted as `<module>:<application>`.

Run the command from a directory that contains a local copy of the `RayServe_En2Fr_translation.py`,
script so it can import the application:

```bash
serve run RayServe_En2Fr_translation:translator_app &
```

The expected output is:

```text
2024-02-01 16:30:33,699        INFO scripts.py:413 -- Running import path: 'RayServe_En2Fr_translation:translator_app'.
Usage stats collection is enabled by default for nightly wheels. To disable this, run the following command: `ray disable-usage-stats` before starting Ray. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2024-02-01 16:30:37,719 INFO worker.py:1753 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8266
(ProxyActor pid=333236) INFO 2024-02-01 16:30:41,275 proxy 10.216.70.84 proxy.py:1145 - Proxy actor d37bfb4a9e388b83347f768101000000 starting on node a5c7469f3d4184bfb971897878429cc42cfd73d4fb484fab478f6215.
(ProxyActor pid=333236) INFO 2024-02-01 16:30:41,278 proxy 10.216.70.84 proxy.py:1357 - Starting HTTP server on node: a5c7469f3d4184bfb971897878429cc42cfd73d4fb484fab478f6215 listening on port 8000
(ProxyActor pid=333236) INFO:     Started server process [333236]
(ServeController pid=333144) INFO 2024-02-01 16:30:41,379 controller 333144 deployment_state.py:1580 - Deploying new version of deployment Translator in application 'default'. Setting initial target number of replicas to 2.
(ServeController pid=333144) INFO 2024-02-01 16:30:41,481 controller 333144 deployment_state.py:1865 - Adding 2 replicas to deployment Translator in application 'default'.
2024-02-01 16:30:45,323 SUCC scripts.py:457 -- Deployed app successfully.
```

After the server is set up in our cluster, we can test the application locally using the `model_client.py`
script from the
[Ray documentation page](https://docs.ray.io/en/latest/serve/develop-and-deploy.html), which we
renamed to `RayServe_En2Fr_tranlation_client.py`. It sends a `POST` request (in JSON) containing the
English text.

```python
# File name: RayServe_En2Fr_tranlation_client.py
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

This client script requests a translation for the phrase “*Hello world!*”:

```bash
python RayServe_En2Fr_tranlation_client.py
```

The expected output is:

```text
Bonjour monde!
(ServeReplica:default:Translator pid=333328) INFO 2024-02-01 16:38:02,251 default_Translator hxagjhct 6625dbbe-cbba-40cd-ba86-5cbff9cb6aa2 / replica.py:380 - __CALL__ OK 192.1ms
```

### Use Ray Serve to serve a Stable Diffusion model

Stable Diffusion is one of the most popular image generation models. It takes a text prompt and
generates an image according to the meaning of the prompt.

In this example, we use Ray to stand up a server for a
[stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
model with an API powered by [FastAPI](https://fastapi.tiangolo.com/).

To run this example, install the following:

```bash
pip install requests diffusers==0.12.1 transformers
```

Create a Python script with the `Serve` code per the
[Ray documentation](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html) and save it as
`RayServe_StableDiffusion.py`.

```python
# File name: RayServe_StableDiffusion.py
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch

from ray import serve
from ray.serve.handle import DeploymentHandle

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 2},
)
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image

entrypoint = APIIngress.bind(StableDiffusionV2.bind())
```

Start the Serve application with the following command:

```bash
serve run RayServe_StableDiffusion:entrypoint &
```

The expected output is:

```text
2024-03-12 23:00:21,290     INFO scripts.py:502 -- Running import path: 'RayServe_StableDiffusion:entrypoint'.
2024-03-12 23:00:22,443 WARNING api.py:424 -- The default value for `max_ongoing_requests` is currently 100, but will change to 5 in the next upcoming release.
2024-03-12 23:00:22,453 WARNING api.py:364 -- The default value for `target_ongoing_requests` is currently 1.0, but will change to 2.0 in an upcoming release.
2024-03-12 23:00:22,453 WARNING api.py:424 -- The default value for `max_ongoing_requests` is currently 100, but will change to 5 in the next upcoming release.
Usage stats collection is enabled by default for nightly wheels. To disable this, run the following command: `ray disable-usage-stats` before starting Ray. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2024-03-12 23:00:24,606 INFO worker.py:1743 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
(ProxyActor pid=121115) INFO 2024-03-12 23:00:28,216 proxy 10.216.70.82 proxy.py:1160 - Proxy starting on node 724ae23b5a7bfda10da26d36de8efc5af99cd7d7b1ddbb4379810ff5 (HTTP port: 8000).
(ServeController pid=121023) INFO 2024-03-12 23:00:28,316 controller 121023 deployment_state.py:1581 - Deploying new version of Deployment(name='StableDiffusionV2', app='default') (initial target replicas: 0).
(ServeController pid=121023) INFO 2024-03-12 23:00:28,317 controller 121023 deployment_state.py:1581 - Deploying new version of Deployment(name='APIIngress', app='default') (initial target replicas: 1).
(ServeController pid=121023) INFO 2024-03-12 23:00:28,419 controller 121023 deployment_state.py:1883 - Adding 1 replica to Deployment(name='APIIngress', app='default').
2024-03-12 23:00:30,262 INFO api.py:601 -- Deployed app 'default' successfully.
```

Now we can send requests to the server through the API. Create the script
`RayServe_StableDiffusion_client.py` using the client code from the
[Ray documentation](https://docs.ray.io/en/latest/serve/tutorials/stable-diffusion.html).

```python
# File name: RayServe_StableDiffusion_client.py
import requests

prompt = "a cute cat is dancing on the grass."
input = "%20".join(prompt.split(" "))
resp = requests.get(f"http://127.0.0.1:8000/imagine?prompt={input}")
with open("output.png", 'wb') as f:
    f.write(resp.content)
```

Running the `RayServe_StableDiffusion_client.py` script sends a request to this application with prompt
"*a cute cat is dancing on the grass.*".

```bash
python RayServe_StableDiffusion_client.py
```

The generated image is saved locally as `output.png`.

The expected output is:

```text
(ServeController pid=108630) INFO 2024-03-12 22:58:24,938 controller 108630 deployment_state.py:1648 - Autoscaling Deployment(name='StableDiffusionV2', app='default') to 1 replicas. Current num requests: 1, current num running replicas: 0.
(ServeController pid=108630) INFO 2024-03-12 22:58:24,939 controller 108630 deployment_state.py:1883 - Adding 1 replica to Deployment(name='StableDiffusionV2', app='default').
Fetching 12 files: 100%|██████████| 12/12 [00:00<00:00, 158774.91it/s]
(ServeReplica:default:StableDiffusionV2 pid=113302) Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with:
(ServeReplica:default:StableDiffusionV2 pid=113302) ```
(ServeReplica:default:StableDiffusionV2 pid=113302) pip install accelerate
(ServeReplica:default:StableDiffusionV2 pid=113302) ```
(ServeReplica:default:StableDiffusionV2 pid=113302) .
(ServeReplica:default:StableDiffusionV2 pid=113302) /opt/conda/envs/ray_py3.8/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
(ServeReplica:default:StableDiffusionV2 pid=113302)   return self.fget.__get__(instance, owner)()
  0%|          | 0/50 [00:00<?, ?it/s]2 pid=113302)
  2%|▏         | 1/50 [00:00<00:13,  3.54it/s]3302)
  6%|▌         | 3/50 [00:01<00:30,  1.56it/s]3302)
 10%|█         | 5/50 [00:01<00:15,  2.89it/s]3302)
 14%|█▍        | 7/50 [00:02<00:09,  4.39it/s]3302)
 18%|█▊        | 9/50 [00:02<00:06,  5.97it/s]3302)
 22%|██▏       | 11/50 [00:02<00:05,  7.55it/s]302)
 26%|██▌       | 13/50 [00:02<00:04,  9.02it/s]302)
 30%|███       | 15/50 [00:02<00:03, 10.32it/s]302)
 34%|███▍      | 17/50 [00:02<00:02, 11.42it/s]302)
 38%|███▊      | 19/50 [00:02<00:02, 12.30it/s]302)
 42%|████▏     | 21/50 [00:03<00:02, 12.98it/s]302)
 46%|████▌     | 23/50 [00:03<00:02, 13.49it/s]302)
 50%|█████     | 25/50 [00:03<00:01, 13.87it/s]302)
 54%|█████▍    | 27/50 [00:03<00:01, 14.14it/s]302)
 58%|█████▊    | 29/50 [00:03<00:01, 14.33it/s]302)
 62%|██████▏   | 31/50 [00:03<00:01, 14.47it/s]302)
 66%|██████▌   | 33/50 [00:03<00:01, 14.57it/s]302)
 70%|███████   | 35/50 [00:03<00:01, 14.65it/s]302)
 74%|███████▍  | 37/50 [00:04<00:00, 14.70it/s]302)
 78%|███████▊  | 39/50 [00:04<00:00, 14.73it/s]302)
 82%|████████▏ | 41/50 [00:04<00:00, 14.76it/s]302)
 86%|████████▌ | 43/50 [00:04<00:00, 14.78it/s]302)
 90%|█████████ | 45/50 [00:04<00:00, 14.79it/s]302)
 94%|█████████▍| 47/50 [00:04<00:00, 14.80it/s]302)
 98%|█████████▊| 49/50 [00:04<00:00, 14.81it/s]302)
100%|██████████| 50/50 [00:04<00:00, 10.03it/s]302)
(ServeReplica:default:StableDiffusionV2 pid=113302) INFO 2024-03-12 22:58:41,668 default_StableDiffusionV2 jmk29epw 6178f374-a0db-4f82-890e-d97ac108a456 /imagine replica.py:366 - GENERATE OK 5452.7ms
(ServeReplica:default:APIIngress pid=108814) INFO 2024-03-12 22:58:41,775 default_APIIngress aj3gvtvh 6178f374-a0db-4f82-890e-d97ac108a456 /imagine replica.py:366 - __CALL__ OK 16842.9ms
```

Here is the generated image:

![A cat playing on grass](images/output.png)

### Use Ray Tune to tune an XGBoost classifier

In this section, we use XGBoost to train an image classifier on Ray. XGBoost is an optimized library for
distributed gradient boosting. It's become the leading ML library for solving regression and
classification problems. For a deeper dive into how gradient boosting works, we recommend reading
[Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

In this example, the script,
[`xgboost_example.py`](https://github.com/ROCm/ray/blob/master/python/ray/tune/examples/xgboost_example.py),
trains an XGBoost image classifier to detect breast cancer. Ray Tune samples 10 different
hyperparameter settings and trains an XGBoost classifier on all of them. The `TrialScheduler` can stop
the low-performing trials early to reduce training time, thereby focusing all resources on the
high-performing trials. Please refer to the official [Ray documentation](https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html#tune-xgboost-ref) for details.

``` tip
You can quickly download this script using `curl`:

    curl https://raw.githubusercontent.com/ROCm/ray/master/python/ray/tune/examples/xgboost_example.py > xgboost_example.py
```

Install `scikit-learn` and `xgboost`.  Then run the script.

```bash
pip install scikit-learn
pip install xgboost
python xgboost_example.py
```

```text
Usage stats collection is enabled by default for nightly wheels. To disable this, run the following command: `ray disable-usage-stats` before starting Ray. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.
2024-03-07 00:31:55,362 INFO worker.py:1754 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265
2024-03-07 00:31:56,477 INFO tune.py:220 -- Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `Tuner(...)`.
2024-03-07 00:31:56,478 INFO tune.py:592 -- [output] This will use the new output engine with verbosity 1. To disable the new output and use the legacy output engine, set the environment variable RAY_AIR_NEW_OUTPUT=0. For more information, please see https://github.com/ray-project/ray/issues/36949
╭────────────────────────────────────────────────────────────────────────────╮
│ Configuration for experiment     train_breast_cancer_2024-03-07_00-31-53   │
├────────────────────────────────────────────────────────────────────────────┤
│ Search algorithm                 BasicVariantGenerator                     │
│ Scheduler                        AsyncHyperBandScheduler                   │
│ Number of trials                 10                                        │
╰────────────────────────────────────────────────────────────────────────────╯

View detailed results here: /root/ray_results/train_breast_cancer_2024-03-07_00-31-53
To visualize your results with TensorBoard, run: `tensorboard --logdir /root/ray_results/train_breast_cancer_2024-03-07_00-31-53`

Trial status: 10 PENDING
Current time: 2024-03-07 00:31:57. Total running time: 0s
Logical resource usage: 10.0/128 CPUs, 0/8 GPUs (0.0/1.0 accelerator_type:AMD-Instinct-MI210)
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                        status       max_depth     min_child_weight     subsample           eta │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_breast_cancer_1a600_00000   PENDING              7                    1      0.98392    0.000278876 │
│ train_breast_cancer_1a600_00001   PENDING              6                    1      0.909173   0.00555103  │
│ train_breast_cancer_1a600_00002   PENDING              4                    2      0.78322    0.0216884   │
│ train_breast_cancer_1a600_00003   PENDING              4                    2      0.528893   0.0112016   │
│ train_breast_cancer_1a600_00004   PENDING              7                    2      0.541909   0.0597606   │
│ train_breast_cancer_1a600_00005   PENDING              6                    1      0.938674   0.00347829  │
│ train_breast_cancer_1a600_00006   PENDING              8                    1      0.883378   0.00682739  │
│ train_breast_cancer_1a600_00007   PENDING              1                    2      0.972819   0.0197333   │
│ train_breast_cancer_1a600_00008   PENDING              6                    2      0.576396   0.00416918  │
│ train_breast_cancer_1a600_00009   PENDING              4                    1      0.697624   0.00031904  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Trial train_breast_cancer_1a600_00000 started with configuration:
╭───────────────────────────────────────────────────────────────────────╮
│ Trial train_breast_cancer_1a600_00000 config                          │
├───────────────────────────────────────────────────────────────────────┤
│ eta                                                           0.00028 │
│ eval_metric                                      ['logloss', 'error'] │
│ max_depth                                                           7 │
│ min_child_weight                                                    1 │
│ objective                                             binary:logistic │
│ subsample                                                     0.98392 │
╰───────────────────────────────────────────────────────────────────────╯

...
...
...

Trial train_breast_cancer_1a600_00004 completed after 10 iterations at 2024-03-07 00:31:59. Total running time: 2s
╭────────────────────────────────────────────────────────────────────╮
│ Trial train_breast_cancer_1a600_00004 result                       │
├────────────────────────────────────────────────────────────────────┤
│ checkpoint_dir_name                              checkpoint_000009 │
│ time_this_iter_s                                            0.0017 │
│ time_total_s                                               0.19301 │
│ training_iteration                                              10 │
│ test-error                                                 0.06993 │
│ test-logloss                                                0.3826 │
╰────────────────────────────────────────────────────────────────────╯

Trial status: 10 TERMINATED
Current time: 2024-03-07 00:31:59. Total running time: 2s
Logical resource usage: 1.0/128 CPUs, 0/8 GPUs (0.0/1.0 accelerator_type:AMD-Instinct-MI210)
Current best trial: 1a600_00004 with test-logloss=0.38260495725211563 and params={'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error'], 'max_depth': 7, 'min_child_weight': 2, 'subsample': 0.5419086005804928, 'eta': 0.05976055309805102}
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                        status         max_depth     min_child_weight     subsample           eta     iter     total time (s)     test-logloss     test-error │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_breast_cancer_1a600_00000   TERMINATED             7                    1      0.98392    0.000278876        1          0.158494          0.65572       0.363636  │
│ train_breast_cancer_1a600_00001   TERMINATED             6                    1      0.909173   0.00555103         1          0.164058          0.678485      0.412587  │
│ train_breast_cancer_1a600_00002   TERMINATED             4                    2      0.78322    0.0216884          1          0.141597          0.629168      0.335664  │
│ train_breast_cancer_1a600_00003   TERMINATED             4                    2      0.528893   0.0112016         10          0.141248          0.560342      0.300699  │
│ train_breast_cancer_1a600_00004   TERMINATED             7                    2      0.541909   0.0597606         10          0.193006          0.382605      0.0699301 │
│ train_breast_cancer_1a600_00005   TERMINATED             6                    1      0.938674   0.00347829         1          0.175483          0.703689      0.447552  │
│ train_breast_cancer_1a600_00006   TERMINATED             8                    1      0.883378   0.00682739         2          0.163225          0.639305      0.34965   │
│ train_breast_cancer_1a600_00007   TERMINATED             1                    2      0.972819   0.0197333          1          0.167176          0.678488      0.426573  │
│ train_breast_cancer_1a600_00008   TERMINATED             6                    2      0.576396   0.00416918         1          0.0118954         0.652763      0.363636  │
│ train_breast_cancer_1a600_00009   TERMINATED             4                    1      0.697624   0.00031904         1          0.0119591         0.691917      0.426573  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best model parameters: {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'error'], 'max_depth': 7, 'min_child_weight': 2, 'subsample': 0.5419086005804928, 'eta': 0.05976055309805102}
Best model total accuracy: 0.9301
(train_breast_cancer pid=100759) [00:31:59] WARNING: /xgboost/src/c_api/c_api.cc:1348: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats. [repeated 27x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)
(train_breast_cancer pid=100759) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/root/ray_results/train_breast_cancer_2024-03-07_00-31-53/train_breast_cancer_1a600_00006_6_eta=0.0068,max_depth=8,min_child_weight=1,subsample=0.8834_2024-03-07_00-31-57/checkpoint_000001) [repeated 27x across cluster]
```

Notice that eight of the trials stopped after only a couple iterations instead of finishing the 10
iterations. Only the two best performing ones completed the full 10 iterations.

## Summary

In this blog we described how to scale AI applications using Ray on multiple AMD GPUs. We'll explore
how to use Ray to scale AI applications to a multi-node cluster in a future blog.

## Appendix

### Docker files

`docker-compose.yaml`

```shell
version: "3.7"
services:
  ray-blog:
    build:
      context: ..
      dockerfile: ./docker/dockerfile
    volumes:
      - ..:/root/
    devices:
      - /dev/kfd
      - /dev/dri
    command: /bin/bash
```

`dockerfile`

```shell
FROM rocm/dev-ubuntu-22.04:5.7-complete
ARG PY_VERSION=3.8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python${PY_VERSION} python${PY_VERSION}-venv git

# Prepare python venv, add to path
ENV PATH=/ray_venv/bin:$PATH
RUN python${PY_VERSION} -m venv ray_venv && python -m pip install --upgrade pip wheel

# Install Ray
RUN pip install "ray[data,train,tune,serve] @ https://github.com/ROCm/ray/releases/download/v3.0.0-dev0%2Brocm/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl"

# Install torch
RUN --mount=type=cache,target=/root/.cache pip3 install torch==2.0.1 torchvision==0.15.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-5.7/

# Install additional dependencies
RUN pip3 install evaluate==0.4.1 \
    transformers==4.39.3 \
    accelerate==0.28.0 \
    scikit-learn==1.3.2 \
    requests==2.31.0 \
    diffusers==0.12.1

# Build XGBoost
RUN git clone --depth=1 --recurse-submodules https://github.com/ROCmSoftwarePlatform/xgboost xgboost \
    && cd xgboost \
    && mkdir build && cd build\
    && export GFXARCH="$(rocm_agent_enumerator | tail -1)" \
    && export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/rocm/lib/cmake:/opt/rocm/lib/cmake/AMDDeviceLibs/ \
    && cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=${GFXARCH} -DUSE_RCCL=1 ../ \
    && make -j \
    && pip3 install ../python-package/

WORKDIR /root
```

### Running on host

If you don't want to use docker, you can also run this blog directly on your machine - although it takes a little more work.

* Prerequisits:
  * Install ROCm 5.7.x
  * Ensure you have python 3.8 installed

* Create and activate a python virtual environment
  
    ```shell
    python3.8 -m venv venv
    source ./venv/bin/activate
    ```

* Install Ray whl:

    ```shell
    pip install ray[data,train,tune,serve] @ https://github.com/ROCm/ray/releases/download/v3.0.0-dev0%2Brocm/ray-3.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl
    ```

* Install dependencies:

    ```shell
    # Install torch
    pip3 install torch==2.0.1 torchvision==0.15.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-5.7/

    # Install additional dependencies
    pip3 install evaluate==0.4.1 \
        transformers==4.39.3 \
        accelerate==0.28.0 \
        scikit-learn==1.3.2 \
        requests==2.31.0 \
        diffusers==0.12.1
    ```

* Install XGBoost for ROCm (must be built from source)

    ```bash
    cd $HOME
    git clone --depth=1 --recurse-submodules https://github.com/ROCmSoftwarePlatform/xgboost
    cd xgboost
    mkdir build && cd build
    export GFXARCH="$(rocm_agent_enumerator | tail -1)"
    export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/rocm/lib/cmake:/opt/rocm/lib/cmake/AMDDeviceLibs/
    cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=${GFXARCH} -DUSE_RCCL=1 ../
    make -j
    pip3 install ../python-package/
    ```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
