---
blogpost: true
date: 4 Apr 2024
author: Fabricio Flores
tags: LLM, AI/ML, GenAI, Computer Vision
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Building semantic search with SentenceTransformers on AMD"
    "keywords": "AMD GPU, MI300, MI250, ROCm, blog, SentenceTransformers
    semantic Search, text embeddings, BERT, transformers, Generative AI"
    "property=og:locale": "en_US"
---

# Building semantic search with SentenceTransformers on AMD

<span style="font-size:0.7em;">4 Apr, 2024 by {hoverxref}`Fabricio Flores<fabrflor>`. </span>

In this blog, we explain how to train a SentenceTransformers model on the Sentence Compression
dataset to perform semantic search. We use the BERT base model (uncased) as the base transformer
and apply Hugging Face PyTorch libraries.

Our goal in training this custom model is to use it for performing semantic search. Semantic search is
an information retrieval method that understands the intent and context of a search query, rather than
simply matching keywords. For example, searching for _apple pie recipes_ (the query) would return
results (the documents) about how to make apple pie, not just pages containing the words _apple_ and
_pie_.

You can find files related to this blog post in the
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/sentence_transformers_amd/).

## Introduction to SentenceTransformers

Training a SentenceTransformers model from scratch involves a process where the model is taught to
understand and encode sentences into meaningful, high-dimensional vectors. In this blog, we focus on
a dataset that contains pairs of equivalent sentences. Overall, the training process aims to have the
model learn how to map semantically similar sentences, which are close to each other within a vector
space, while moving the dissimilar ones farther apart. In contrast to generic pre-trained models, which
might not be able to capture the specificities of certain domains or use cases, a custom trained model
ensures that the model is finely tuned to understand the context and semantics relevant to a particular
domain or application.

We're interested in performing asymmetric semantic search. In this approach, the model acknowledges
that the query and the documents can be different in nature. For example, having short queries and
long documents. Asymmetric semantic search uses encodings that make search more effective, even
when there's a disparity in the type or length of text. This is useful for applications like information
retrieval from large documents or databases where queries are often shorter and less detailed than the
content that they are searching through. Here's an example of how semantic search works:

```text
Query: Is Paris located in France?

Most similar sentences in corpus:
The capital of France is Paris (Score: 0.6829)
Paris, which is a city in Europe with traditions and remarkable food, is the capital of France (Score: 0.6044)
Australia is known for its traditions and remarkable food (Score: -0.0159)
```

## Implementation on an AMD GPU

This blog has been tested on ROCm 6.0 and Pytorch 2.0.1 versions.

To get started, first install these prerequisites:

1. [ROCm for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html)

2. [Latest PyTorch Docker image](https://hub.docker.com/r/rocm/pytorch)

Run the PyTorch Docker container with:

```bash
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G rocm/pytorch:latest
```

Let's describe some the options for the command used to start the container:

* `-it`: This combination of `-i` and `-t` allows to interact with the container through the terminal.

* `--device=/dev/kfd --device=/dev/dri`: These options give access to specific devices on the host. `--device=/dev/kfd` is associated with AMD GPU devices and `--device=/dev/dri` is related to devices with direct access to the graphics hardware.

* `--group-add=video`: This option allows the container to have necessary permissions to access video hardware directly.

* `rocm/pytorch:latest`: This is the name of the latest PyTorch Docker image.

The rest of the options configure security options, grant more privileges and adjust resource usage.

Then install the following Python packages:

```python
!pip install datasets ipywidgets -U transformers sentence-transformers
```

### Importing Python packages

```python
from datasets import load_dataset
from sentence_transformers import InputExample, util
from torch.utils.data import DataLoader
from torch import nn
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, models
```

### The dataset

The [Sentence Compression](https://huggingface.co/datasets/embedding-data/sentence-compression)
dataset consists of 180,000 pairs of equivalent sentences. These sentence pairs demonstrate how a
longer sentence can be compressed into a shorter one, while retaining the same meaning.

### Preparing the dataset

```python
dataset_id = "embedding-data/sentence-compression"
dataset = load_dataset(dataset_id)
```

Let's explore one sample in the dataset:

```python
# Explore one sample
print(dataset['train']['set'][1])
```

```python
['Major League Baseball Commissioner Bud Selig will be speaking at St. Norbert College next month.',
'Bud Selig to speak at St. Norbert College']
```

The SentenceTransformers library requires our dataset to be in a specific format. This ensures the data
are compatible with the model architecture.

Let's create a list of the training samples (using half of the dataset for illustrative purposes). This
approach reduces the computational load and accelerates the training process.

```python
#convert dataset in required format
train_examples = []
train_data = dataset['train']['set']

n_examples = dataset['train'].num_rows//2 #select half of the dataset for training

for example in train_data[:n_examples]:
    original_sentence = example[0]
    compressed_sentence = example[1]

    input_example = InputExample(texts = [original_sentence, compressed_sentence])

    train_examples.append(input_example)
```

Now, let's instantiate the `DataLoader` class. This class provides an efficient way to iterate over our
dataset.

```python
#Instantiate Dataloader with training examples
train_dataloader = DataLoader(train_examples, shuffle = True, batch_size = 16)
```

### Implementation

In a sentence transformer model, we want to map a variable-length input sentence to a fixed size
vector. We start by first passing in the input sentence through a transformer model. In this example, we
use the [BERT base model (uncased)](https://huggingface.co/bert-base-uncased) model as the base
transformer model, which outputs the contextualized embeddings for each token in the input
sentence. After getting the embeddings for each token, we use the pooling layer to aggregate the
embeddings into a single sentence embedding. Lastly, we perform an additional transformation by
adding a dense layer (with a Tanh Activation function). This layer reduces the dimensionality of the
pooled sentence embeddings while enabling the model to capture more complex patterns in the data
by using a non-linear activation function.

```python
# Create a custom model
# Use an existing embedding model
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)

# Pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# Dense function
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

# Define the overall model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
```

### Training

During training, the selection of an appropriate loss function is crucial and depends on the specific
application and the structure of the dataset. For our purposes, we employ the
`MultipleNegativesRankingLoss` function. This function is particularly useful in semantic search
applications where the model needs to rank sentences based on their relevance. It operates by
contrasting a pair of semantically similar sentences (a positive pair) against multiple dissimilar ones.
This function is especially well-suited to our Sentence Compression dataset, as it distinguishes between
semantically similar and dissimilar sentences.

```python
#Given the dataset of equivalent sentences, choose MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model = model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs = 5)
```

### Inference

Let's evaluate the model.

```python
# from sentence_transformers import SentenceTransformer, util
import torch

# Sentences (documents/corpus) to encode
sentences = [
    'Paris, which is a city in Europe with traditions and remarkable food, is the capital of France',
    'The capital of France is Paris',
    'Australia is known for its traditions and remarkable food',
    """
        Despite the heavy rains that lasted for most of the week, the outdoor music festival,
        which featured several renowned international artists, was able to proceed as scheduled,
        much to the delight of fans who had traveled from all over the country
    """,
    """
        Photosynthesis, a process used by plans and other organisms to convert light into
        chemical energy, plays a crucial role in maintaining the balance of oxygen and carbon
        dioxide in the Earth's atmosphere.
    """
]

# Encode the sentences
sentences_embeddings = model.encode(sentences, convert_to_tensor=True)


# Query sentences:
queries = ['Is Paris located in France?', 'Tell me something about Australia',
           'music festival proceeding despite heavy rains',
           'what is the process that some organisms use to transform light into chemical energy?']


# Find the closest sentences of the corpus for each query using cosine similarity
for query in queries:

    # Encode the current query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Cosine similarity and closest document to query
    cos_scores = util.cos_sim(query_embedding, sentences_embeddings)[0]

    top_results = torch.argsort(cos_scores, descending = True)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nSimilar sentences in corpus:")

    for idx in top_results:
        print(sentences[idx], "(Score: {:.4f})".format(cos_scores[idx]))

```

We demonstrate the efficacy of the model by testing it with a few new examples.

```text
======================


Query: Is Paris located in France?

Similar sentences in corpus:
The capital of France is Paris (Score: 0.7907)
Paris, which is a city in Europe with traditions and remarkable food, is the capital of France (Score: 0.7081)

        Photosynthesis, a process used by plans and other organisms to convert light into
        chemical energy, plays a crucial role in maintaining the balance of oxygen and carbon
        dioxide in the Earth's atmosphere.
     (Score: 0.0657)
Australia is known for its traditions and remarkable food (Score: 0.0162)

        Despite the heavy rains that lasted for most of the week, the outdoor music festival,
        which featured several renowned international artists, was able to proceed as scheduled,
        much to the delight of fans who had traveled from all over the country
     (Score: -0.0934)


======================


Query: Tell me something about Australia

Similar sentences in corpus:
Australia is known for its traditions and remarkable food (Score: 0.6730)
Paris, which is a city in Europe with traditions and remarkable food, is the capital of France (Score: 0.1489)
The capital of France is Paris (Score: 0.1146)

        Despite the heavy rains that lasted for most of the week, the outdoor music festival, 
        which featured several renowned international artists, was able to proceed as scheduled, 
        much to the delight of fans who had traveled from all over the country
     (Score: 0.0694)

        Photosynthesis, a process used by plans and other organisms to convert light into
        chemical energy, plays a crucial role in maintaining the balance of oxygen and carbon
        dioxide in the Earth's atmosphere.
     (Score: -0.0241)


======================


Query: music festival proceeding despite heavy rains

Similar sentences in corpus:

        Despite the heavy rains that lasted for most of the week, the outdoor music festival,
        which featured several renowned international artists, was able to proceed as scheduled,
        much to the delight of fans who had traveled from all over the country
     (Score: 0.7855)
Paris, which is a city in Europe with traditions and remarkable food, is the capital of France (Score: 0.0700)

        Photosynthesis, a process used by plans and other organisms to convert light into
        chemical energy, plays a crucial role in maintaining the balance of oxygen and carbon
        dioxide in the Earth's atmosphere.
     (Score: 0.0351)
The capital of France is Paris (Score: 0.0037)
Australia is known for its traditions and remarkable food (Score: -0.0552)


======================


Query: what is the process that some organisms use to transform light into chemical energy?

Similar sentences in corpus:

        Photosynthesis, a process used by plans and other organisms to convert light into
        chemical energy, plays a crucial role in maintaining the balance of oxygen and carbon
        dioxide in the Earth's atmosphere.
     (Score: 0.6085)

        Despite the heavy rains that lasted for most of the week, the outdoor music festival,
        which featured several renowned international artists, was able to proceed as scheduled,
        much to the delight of fans who had traveled from all over the country
     (Score: 0.1370)
Paris, which is a city in Europe with traditions and remarkable food, is the capital of France (Score: 0.0141)
Australia is known for its traditions and remarkable food (Score: 0.0102)
The capital of France is Paris (Score: -0.0128)
```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
