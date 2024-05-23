---
blogpost: true
date: 25 Jan 2024
author: Douglas Jia
tags: LLM, AI/ML, GenAI, Fine-Tuning, JAX
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "LLM distributed supervised fine-tuning with JAX"
    "keywords": "AMD GPU, MI300, MI250, train models, LLM, JAX, fine-tuning,
  BERT, GLUE, ROCm, SFT, AAC, Natural Language Processing"
    "property=og:locale": "en_US"
---

# LLM distributed supervised fine-tuning with JAX

In this article, we review the process for fine-tuning a Bidirectional Encoder Representations
from Transformers (BERT)-based large language model (LLM) using JAX for a text classification task. We
explore techniques for parallelizing this fine-tuning procedure across multiple AMD GPUs, then
evaluate our model's performance on a holdout dataset. For this, we use a
[(BERT)-base-cased](https://huggingface.co/bert-base-cased) transformer model with a General
Language Understanding Evaluation (GLUE) benchmark dataset on multiple AMD GPUs.

We focus on two
[Single Program, Multiple Data (SPMD)](https://en.wikipedia.org/wiki/Single_program,_multiple_data)
parallelism methods in JAX. These are:

* Using a `pmap` function for straightforward data distribution over a single leading axis.
* Using `jit`, `Mesh`, and `mesh_utils` functions to shard data across devices, providing greater control
  over parallelization.

Our emphasis is on the first method, and we provide details on the second method in the
[final section](#using-jax-device-mesh-to-achieve-parallelism).

In developing this article, we referenced
[this tutorial](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb),
which we highly recommend.

## What is supervised fine-tuning?

In the era of artificial intelligence (AI), transformer architecture-based models like BERT, GPT-3, and
their successors have provided a sturdy foundation for achieving cutting-edge performance across
various natural language processing (NLP) tasks, including text classification, text generation, and
sentiment analysis. Nonetheless, when applied in isolation to these specific tasks, these large,
pre-trained models often exhibit limitations. Supervised fine-tuning (SFT) provides a solution to these
limitations.

Unlike pre-trained models, which undergo broad, unsupervised training on massive and diverse
datasets, SFT adopts a focused and resource-efficient approach. Typically, this requires a relatively
compact, high-quality dataset that is precisely tailored to the given task. SFT can improve model
performance to a state-of-the-art level without the need for protracted training periods, as it is able to
leverage the extensive knowledge acquired by pre-trained models.

The SFT process consists of fine-tuning the model's existing weights or adding extra parameters to
ensure alignment with the intricacies of the designated task. Often, this adaptation incorporates
task-specific layers, such as the addition of a softmax layer for classification, which enhances the
model's ability to address supervised tasks.

## What is JAX?

JAX is a high-performance numerical computation library for Python. In contrast to traditional machine
learning frameworks, such as TensorFlow and PyTorch, JAX has remarkable speed and efficiency. JAX
utilizes Just-in-Time (JIT) compilation, seamless automatic differentiation, and an inherent capability
to efficiently vectorize and parallelize code, which allows for simple adaptation for AI accelerators
(GPUs and TPUs).

## Why use AMD GPUs?

AMD GPUs stand out for their robust open-source support--featuring tools like ROCm and
HIP--making them easily adaptable to AI workflows. AMD's competitive price-to-performance ratio
caters to anyone seeking cost-effective solutions for AI and deep learning tasks. As AMD's presence in
the market grows, more machine learning libraries and frameworks are adding AMD GPU support.

## Hardware requirements and running environment

To harness the computational capabilities required for this task, we leverage the
[AMD Accelerator Cloud (AAC)](https://aac.amd.com/). AAC is a platform that offers on-demand cloud
computing resources and APIs on a pay-as-you-go basis. Specifically, we use a
[JAX docker container](https://hub.docker.com/layers/rocm/jax-build/rocm5.5.0-jax0.4.14-py3.10.0-fusion-imp/images/sha256-02a74b2f1992607adade6bbc0afb8db46fc901e442d67377facebd5c713a2ef6?context=explore) with 8 GPUs (on AAC) to utilize the full
potential of cutting-edge GPU parallel computing.

This article is hardware-agnostic, meaning that access to AAC is **not** a requirement for successfully
running the code examples provided. As long as you have access to accelerator devices, such as GPUs
or TPUs, you should be able to run the code examples with minimal code modifications. If you're using
AMD GPUs, make sure you have ROCm and its compatible versions of JAX and Jaxlib installed correctly.
Refer to the following tutorials for installation instructions:

* [ROCm installation](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
* [JAX and Jaxlib installation](https://github.com/ROCmSoftwarePlatform/jax/releases): You can also
  directly pull a JAX Docker image in the link.

## Code example on SFT of a transformer model

For this demonstration, we fine-tune a transformer-based LLM
([bert-base-cased](https://huggingface.co/bert-base-cased)) using a General Language Understanding
Evaluation (GLUE) [benchmark](https://gluebenchmark.com/) dataset, Quora Question Pairs (QQP). This
dataset consists of over 400,000 pairs of questions, each accompanied by a binary annotation that
indicates if the two questions are paraphrases of each other. The input variables are the sentences of
the two questions, while the output variable is a binary indicator denoting whether the questions share
the same meaning.

### Installation

First, install the required packages (`%%capture` is a *cell magic* that will suppress the output of the cell).

```python
%%capture
!pip install datasets
!pip install git+https://github.com/huggingface/transformers.git
!pip install flax
!pip install git+https://github.com/deepmind/optax.git
!pip install evaluate
!pip install ipywidgets
!pip install black isort # Jupyter Notebook code formatter; optional
```

Import the remaining packages and functionalities.

```python
import os
from itertools import chain
from typing import Callable

import evaluate
import flax
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from datasets import load_dataset
from flax import traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from ipywidgets import IntProgress as IProgress
from tqdm.notebook import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

JAX pre-allocates 75% of total GPU memory to reduce overhead and fragmentation when running the
first JAX operation, but may trigger out-of-memory (OOM) errors. To avoid the OOM issues, suppress
the default behavior by setting the `XLA_PYTHON_CLIENT_PREALLOCATE` flag to false.

Check if the GPU devices are detectable by JAX. If not, you may need to re-install ROCm, JAX, and
Jaxlib. If JAX is installed correctly, you can see all the GPU devices you requested, which in our case is 8
GPUs.

```python
jax.local_devices()
```

```sh
[gpu(id=0),
 gpu(id=1),
 gpu(id=2),
 gpu(id=3),
 gpu(id=4),
 gpu(id=5),
 gpu(id=6),
 gpu(id=7)]
```

### Get the fine-tuning dataset and pre-trained model checkpoint

Specify the settings for your fine-tuning process: the dataset, the pre-trained model, and how many
samples you want processed per batch and per device.

```python
task = "qqp"
model_checkpoint = "bert-base-cased"
per_device_batch_size = 64
```

Load the dataset and evaluation metric module.

```python
raw_dataset = load_dataset("glue", task)
metric = evaluate.load("glue", task)
```

The next few code blocks show how to tokenize the text data with the model-specific tokenizer and
load the tokenized training and validation data. Using the same tokenizer as used in the pre-trained
model ensures that the same words will be converted to the same embedding vector in the fine-tuning
process.

It's important to highlight that we've performed a 10% subsampling on the training and evaluation
datasets from the original training data. Despite this reduction, the QQP dataset still provides sufficient
data for achieving commendable performance and allows us to observe metric improvements after
each epoch. This subsampling approach also expedites our training process for illustration.

Process the training and evaluation datasets using the data preprocessing function and the map
wrapper's batch and parallel processing features. You can view the tokenized dataset in the following
output.

```python
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

```python
def preprocess_function(examples):
    texts = (examples["question1"], examples["question2"])
    processed = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    processed["labels"] = examples["label"]
    return processed
```

```python
# Details about how to handle and process huggingface dataset:
# https://huggingface.co/docs/datasets/process
data = raw_dataset["train"].shuffle(seed=0)
train_data = data.select(list(range(int(data.shape[0] * 0.1))))
eval_data = data.select(list(range(int(data.shape[0] * 0.1), int(data.shape[0] * 0.2))))
print(f"Shape of the original training dataset is: {data.shape}")
print(f"Shape of the current training dataset is: {train_data.shape}")
print(f"Shape of the current evaluation dataset is: {eval_data.shape}")
```

```sh
Shape of the original training dataset is: (363846, 4)
Shape of the current training dataset is: (36384, 4)
Shape of the current evaluation dataset is: (36385, 4)
```

```python
train_dataset = train_data.map(
    preprocess_function, batched=True, remove_columns=train_data.column_names
)
eval_dataset = eval_data.map(
    preprocess_function, batched=True, remove_columns=eval_data.column_names
)
```

```python
# You can view the tokenized dataset with the output of this cell.
pd.DataFrame(train_dataset[:3])
```

Download the pre-trained model configurations and checkpoint from Hugging Face. Note that you'll
see a warning message stating that some of the model weights weren't used. This is expected because
the BERT model checkpoint is a `PreTraining` model class and you're initializing a
`SequenceClassification` model. The warning message states:
*You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.*.
This is what we'll focus on in the rest of this blog.

```python
num_labels = 2
seed = 0
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)
model = FlaxAutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, config=config, seed=seed
)
```

```sh
Some weights of the model checkpoint at bert-base-cased were not used when initializing FlaxBertForSequenceClassification: {('cls', 'predictions', 'bias'), ('cls', 'predictions', 'transform', 'dense', 'kernel'), ('cls', 'predictions', 'transform', 'LayerNorm', 'bias'), ('cls', 'predictions', 'transform', 'LayerNorm', 'scale'), ('cls', 'predictions', 'transform', 'dense', 'bias')}
- This IS expected if you are initializing FlaxBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing FlaxBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of FlaxBertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: {('classifier', 'kernel'), ('classifier', 'bias'), ('bert', 'pooler', 'dense', 'kernel'), ('bert', 'pooler', 'dense', 'bias')}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

### Define the state of your fine-tuning model

The following code blocks show you how to set up training parameters, such as number of training
epochs and initial learning rate. A learning rate schedule is needed in order to have the learning rate
 linearly decay as the training progresses, which ensures learning efficiency and stability.

```python
num_train_epochs = 6
learning_rate = 2e-5
```

```python
total_batch_size = per_device_batch_size * jax.local_device_count()
print("The overall batch size (both for training and eval) is", total_batch_size)
```

```sh
The overall batch size (both for training and eval) is 512
```

```python
num_train_steps = len(train_dataset) // total_batch_size * num_train_epochs

learning_rate_function = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps
)
```

Next, you'll need to establish the training state, encompassing the optimizer and loss function
responsibilities, and oversee the update of the model's parameters throughout the training process.

With the state object, initialize and update the models. When invoking the model, provide the state as
input, the model then returns the updated state by adding information from the new batch of data
while preserving the model instance.

Flax offers a user-friendly class (`flax.training.train_state.TrainState`) that takes in the model parameters,
the loss function, and the optimizer. When supplied with data, it can update the model parameters
using the `apply_gradients` function.

The following code blocks show how to define and establish the training state, optimizer, and loss
function.

```python
class TrainState(train_state.TrainState):
    logits_function: Callable = flax.struct.field(pytree_node=False)
    loss_function: Callable = flax.struct.field(pytree_node=False)
```

```python
# Create a decay_mask_fn function to make sure that weight decay is not applied to any bias or
# LayerNorm weights, as it may not improve model performance and even be harmful.
def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {
        path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale"))
        for path in flat_params
    }
    return traverse_util.unflatten_dict(flat_mask)
```

```python
# Standard Adam optimizer with weight decay
def adamw(weight_decay):
    return optax.adamw(
        learning_rate=learning_rate_function,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
        mask=decay_mask_fn,
    )
```

```python
def loss_function(logits, labels):
    xentropy = optax.softmax_cross_entropy(
        logits, onehot(labels, num_classes=num_labels)
    )
    return jnp.mean(xentropy)


def eval_function(logits):
    return logits.argmax(-1)
```

```python
# Instantiate the TrainState
state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=adamw(weight_decay=0.01),
    logits_function=eval_function,
    loss_function=loss_function,
)
```

### Define how to train, evaluate the model, and enable parallelization

The `train_step` and `eval_step` parameters define how the model should be trained and evaluated. The
train step follows the standard training process:

1. Calculate the loss with the current weights.
2. Calculate the gradients of the loss function with respect to the weights.
3. Update the weights with the gradients and learning rate.
4. Repeat the above steps until the stopping criteria has been met.

It's important to highlight that the `lax.pmean` function computes the mean of gradients from data
batches across all 8 GPU devices. This crucial step guarantees the synchronization of model parameters
across all GPU devices.

```python
def train_step(state, batch, dropout_rng):
    targets = batch.pop("labels")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_function(params):
        logits = state.apply_fn(
            **batch, params=params, dropout_rng=dropout_rng, train=True
        )[0]
        loss = state.loss_function(logits, targets)
        return loss

    grad_function = jax.value_and_grad(loss_function)
    loss, grad = grad_function(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": learning_rate_function(state.step)},
        axis_name="batch",
    )
    return new_state, metrics, new_dropout_rng
```

```python
def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.logits_function(logits)
```

Next, apply the `jax.pmap` function to the defined `train_step` and `eval_step` functions. Applying
`pmap()` to a function compiles that function with XLA (similar to `jit()`), then runs it in parallel on XLA
devices, such as multiple GPUs or multiple TPU cores. Simply put, this step sends the training and
evaluation functions to all GPU devices. You'll also need to send the training state to all GPU devices
via `flax.jax_utils.replicate`. These steps ensure you're updating the state, via distributed training, on all
GPU devices.

```python
parallel_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))
parallel_eval_step = jax.pmap(eval_step, axis_name="batch")
state = flax.jax_utils.replicate(state)
```

Define the data loader functions that return a data batch generator. A new batch of data is fed into
each step of the final training and evaluation loops.

```python
def glue_train_data_loader(rng, dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch
```

```python
def glue_eval_data_loader(dataset, batch_size):
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch
```

A pseudo-random number generator (PRNG) key is generated based on an integer seed, and is then
split into 8 new keys so that each GPU device gets a different key. Then run the training steps to
update the `state` based on the pre-defined training parameters, such as number of epochs and
`total_batch_size`. After finishing each epoch, run the evaluation step on the eval dataset to see the
accuracy and f1 metrics. Because you used a smaller dataset than the original training dataset in the
benchmark, you can see that the eval metrics (train loss and eval accuracy) steadily improved in the
first few epochs.

```python
rng = jax.random.PRNGKey(seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())
```

```python
for i, epoch in enumerate(
    tqdm(range(1, num_train_epochs + 1), desc=f"Epoch ...", position=0, leave=True)
):
    rng, input_rng = jax.random.split(rng)

    # train
    with tqdm(
        total=len(train_dataset) // total_batch_size, desc="Training...", leave=True
    ) as progress_bar_train:
        for batch in glue_train_data_loader(input_rng, train_dataset, total_batch_size):
            state, train_metrics, dropout_rngs = parallel_train_step(
                state, batch, dropout_rngs
            )
            progress_bar_train.update(1)

    # evaluate
    with tqdm(
        total=len(eval_dataset) // total_batch_size, desc="Evaluating...", leave=False
    ) as progress_bar_eval:
        for batch in glue_eval_data_loader(eval_dataset, total_batch_size):
            labels = batch.pop("labels")
            predictions = parallel_eval_step(state, batch)
            metric.add_batch(
                predictions=list(chain(*predictions)), references=list(chain(*labels))
            )
            progress_bar_eval.update(1)

    eval_metric = metric.compute()

    loss = round(flax.jax_utils.unreplicate(train_metrics)["loss"].item(), 3)
    eval_score1 = round(list(eval_metric.values())[0], 3)
    metric_name1 = list(eval_metric.keys())[0]
    eval_score2 = round(list(eval_metric.values())[1], 3)
    metric_name2 = list(eval_metric.keys())[1]
    print(
        f"{i+1}/{num_train_epochs} | Train loss: {loss} | Eval {metric_name1}: {eval_score1}, {metric_name2}: {eval_score2}"
    )
```

```sh
Epoch ...:   0%|          | 0/6 [00:00<?, ?it/s]
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
1/6 | Train loss: 0.475 | Eval accuracy: 0.799, f1: 0.762
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
2/6 | Train loss: 0.369 | Eval accuracy: 0.834, f1: 0.789
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
3/6 | Train loss: 0.299 | Eval accuracy: 0.846, f1: 0.797
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
4/6 | Train loss: 0.239 | Eval accuracy: 0.846, f1: 0.806
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
5/6 | Train loss: 0.252 | Eval accuracy: 0.849, f1: 0.802
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
6/6 | Train loss: 0.212 | Eval accuracy: 0.849, f1: 0.805
```

## Using JAX device mesh to achieve parallelism

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
```

```python
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=num_labels)
model = FlaxAutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, config=config, seed=seed
)
state = TrainState.create(
    apply_fn=model.__call__,
    params=model.params,
    tx=adamw(weight_decay=0.01),
    logits_function=eval_function,
    loss_function=loss_function,
)
```

```sh
Some weights of the model checkpoint at bert-base-cased were not used when initializing FlaxBertForSequenceClassification: {('cls', 'predictions', 'bias'), ('cls', 'predictions', 'transform', 'dense', 'kernel'), ('cls', 'predictions', 'transform', 'LayerNorm', 'bias'), ('cls', 'predictions', 'transform', 'LayerNorm', 'scale'), ('cls', 'predictions', 'transform', 'dense', 'bias')}
- This IS expected if you are initializing FlaxBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing FlaxBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of FlaxBertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: {('classifier', 'kernel'), ('classifier', 'bias'), ('bert', 'pooler', 'dense', 'kernel'), ('bert', 'pooler', 'dense', 'bias')}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

```python
@jax.jit
def train_step(state, batch, dropout_rng):
    targets = batch.pop("labels")
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_function(params):
        logits = state.apply_fn(
            **batch, params=params, dropout_rng=dropout_rng, train=True
        )[0]
        loss = state.loss_function(logits, targets)
        return loss

    grad_function = jax.value_and_grad(loss_function)
    loss, grad = grad_function(state.params)
    new_state = state.apply_gradients(grads=grad)
    metrics = {"loss": loss, "learning_rate": learning_rate_function(state.step)}
    return new_state, metrics, new_dropout_rng
```

```python
@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(**batch, params=state.params, train=False)[0]
    return state.logits_function(logits)
```

```python
num_devices = len(jax.local_devices())
devices = mesh_utils.create_device_mesh((num_devices,))

# Data will be split along the batch axis
data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
data_sharding = NamedSharding(
    data_mesh,
    P(
        "batch",
    ),
)  # naming axes of the sharded partition


def glue_train_data_loader(rng, dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {
            k: jax.device_put(jnp.array(v), data_sharding) for k, v in batch.items()
        }

        yield batch


def glue_eval_data_loader(dataset, batch_size):
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {
            k: jax.device_put(jnp.array(v), data_sharding) for k, v in batch.items()
        }

        yield batch
```

```python
# Replicate the model and optimizer variable on all devices
def get_replicated_train_state(devices, state):
    # All variables will be replicated on all devices
    var_mesh = Mesh(devices, axis_names=("_"))
    # In NamedSharding, axes not mentioned are replicated (all axes here)
    var_replication = NamedSharding(var_mesh, P())

    # Apply the distribution settings to the model variables
    state = jax.device_put(state, var_replication)

    return state


state = get_replicated_train_state(devices, state)
```

```python
rng = jax.random.PRNGKey(seed)
dropout_rng = jax.random.PRNGKey(seed)
```

```python
for i, epoch in enumerate(
    tqdm(range(1, num_train_epochs + 1), desc=f"Epoch ...", position=0, leave=True)
):
    rng, input_rng = jax.random.split(rng)

    # train
    with tqdm(
        total=len(train_dataset) // total_batch_size, desc="Training...", leave=True
    ) as progress_bar_train:
        for batch in glue_train_data_loader(input_rng, train_dataset, total_batch_size):
            state, train_metrics, dropout_rng = train_step(state, batch, dropout_rng)
            progress_bar_train.update(1)

    # evaluate
    with tqdm(
        total=len(eval_dataset) // total_batch_size, desc="Evaluating...", leave=False
    ) as progress_bar_eval:
        for batch in glue_eval_data_loader(eval_dataset, total_batch_size):
            labels = batch.pop("labels")
            predictions = eval_step(state, batch)
            metric.add_batch(predictions=list(predictions), references=list(labels))
            progress_bar_eval.update(1)

    eval_metric = metric.compute()

    loss = round(train_metrics["loss"].item(), 3)
    eval_score1 = round(list(eval_metric.values())[0], 3)
    metric_name1 = list(eval_metric.keys())[0]
    eval_score2 = round(list(eval_metric.values())[1], 3)
    metric_name2 = list(eval_metric.keys())[1]
    print(
        f"{i+1}/{num_train_epochs} | Train loss: {loss} | Eval {metric_name1}: {eval_score1}, {metric_name2}: {eval_score2}"
    )
```

```sh
Epoch ...:   0%|          | 0/6 [00:00<?, ?it/s]
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
1/6 | Train loss: 0.469 | Eval accuracy: 0.796, f1: 0.759
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
2/6 | Train loss: 0.376 | Eval accuracy: 0.833, f1: 0.788
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
3/6 | Train loss: 0.296 | Eval accuracy: 0.844, f1: 0.795
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
4/6 | Train loss: 0.267 | Eval accuracy: 0.846, f1: 0.805
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
5/6 | Train loss: 0.263 | Eval accuracy: 0.848, f1: 0.804
Training...:   0%|          | 0/71 [00:00<?, ?it/s]
Evaluating...:   0%|          | 0/71 [00:00<?, ?it/s]
6/6 | Train loss: 0.222 | Eval accuracy: 0.849, f1: 0.805
```
