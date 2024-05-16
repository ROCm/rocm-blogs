---
blogpost: true
date: 8 Feb 2024
author: Phillip Dang
tags: PyTorch, AI/ML, Tuning
category: Applications & models
language: English

myst:
  html_meta:
    "description lang=en": "Simplifying deep learning: A guide to PyTorch Lightning"
    "keywords": "PyTorch, PyTorch Lightning, train models, Tuning"
    "property=og:locale": "en_US"
---

# Simplifying deep learning: A guide to PyTorch Lightning

PyTorch Lightning is a higher-level wrapper built on top of PyTorch. Its purpose is to simplify and
abstract the process of training PyTorch models. It provides a structured and organized approach to
machine learning (ML) tasks by abstracting away the repetitive boilerplate code, allowing you to focus
more on model development and experimentation. PyTorch Lightning works out-of-the-box with AMD
GPUs and ROCm.

For more information on PyTorch Lightning, refer to
[this article](https://lightning.ai/docs/pytorch/stable/tutorials.html).

In this blog, we train a model on the IMDb movie review data set and demonstrate how to simplify and
organize code with PyTorch Lightning. We also demonstrate how to train models faster with GPUs.

You can find files related to this blog post in the
[GitHub folder](https://github.com/ROCm/rocm-blogs/tree/main/data/artificial-intelligence/pytorch-lightning).

## Prerequisites

To follow along with this blog, you must have the following software:

* [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Linux OS

Next, make sure your system recognizes both AMD GPUs:

```cpp
! rocm-smi --showproductname
================= ROCm System Management Interface ================
========================= Product Info ============================
GPU[0] : Card series: Instinct MI210
GPU[0] : Card model: 0x0c34
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] : Card SKU: D67301
GPU[1] : Card series: Instinct MI210
GPU[1] : Card model: 0x0c34
GPU[1] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1] : Card SKU: D67301
===================================================================
===================== End of ROCm SMI Log =========================
```

Make sure PyTorch also recognizes these GPUs:

```python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

```bash
number of GPUs: 2
['AMD Radeon Graphics', 'AMD Radeon Graphics']
```

Once you've confirmed that your system recognizes your devices, you're ready to go through a typical
ML workflow on PyTorch. This includes loading and processing the data, setting up
a training loop, a validation loop, and optimizers. Afterwards, you can see how PyTorch Lightning does
all this for you by providing a framework that can wrap all such modules in a scalable, easy-to-use way.

### Libraries

Before you begin, make sure you have all the necessary libraries installed:

```python
pip install lightning transformers datasets torchmetrics
```

Next import the modules you'll be working with for this blog:

```python
import collections
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
```

## Data processing

Our data set for this blog is the IMDb movie reviews, and our task is to classify whether a review is
positive (1) or negative (0). Load the data set and look at a few examples:

```python
# Load IMDb data set

imdb = load_dataset("imdb")
print(imdb)

for i in range(3):
    label = imdb['train']['label'][i]
    review = imdb['train']['text'][i]
    print('label: ', label)
    print('review:', review[:100])
    print()

counts = collections.Counter(imdb['train']['label'])
print(counts)
```

```python
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})

label:  0
review: I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it w

label:  0
review: "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's poli

label:  0
review: If only to avoid making this type of film in the future. This film is interesting as an experiment b

Counter({0: 12500, 1: 12500})
```

Our training and test data sets each consist of 25,000 samples, 50% positive and 50% negative. Next,
process and tokenize the texts and build a very simple model to classify the reviews.

Our goal isn't to build a super accurate model for this data set, but rather to demonstrate a typical ML
workflow, and how it can be organized and simplified with PyTorch Lightning to run seamlessly with
our AMD hardware.

### Build custom data set class

As typical in a PyTorch workflow, we define a custom
[data set class](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) that helps organize
movie reviews and their sentiments for our model. Specifically, it tokenizes the text, handles the
different sequence lengths, and returns the input IDs and labels our model will learn from.

```python
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.texts = data['text']
        self.labels = data['label']
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = float(self.labels[idx])
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
```

### Splitting the data

With the custom data class defined, split the data into training and validation sets and create a
DataLoader wrapper. We typically use PyTorch DataLoader to support efficient data handling/batching,
parallelization, shuffling, and sampling.

```python
# Split data into 2 sets

train_data = imdb['train']
val_data = imdb['test']

# Create data set objects that handle data processing

train_dataset = SentimentDataset(train_data, tokenizer, max_length)
val_dataset = SentimentDataset(val_data, tokenizer, max_length)

# Wrap these data set objects around DataLoader for efficient data handling and batching

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
```

## Modeling

Next, create the model. We'll be using the pre-trained
[Bert](https://huggingface.co/docs/transformers/model_doc/bert) model from Hugging Face, which
we'll fine-tune on our text classification task.

Our model consists of:

* A Bert model with multiple Transformers layers. These layers perform self-attention contextual
  embedding of words within a sentence.
* A fully connected linear layer to help fine-tune the model on the classification task by reducing the
  output dimensionality from the embedding dimensions to 1.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', freeze_bert=True):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.embedding_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(self.embedding_dim, 1)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits.squeeze(1)
```

### Initialize model

Set the computation device to the AMD GPU, initialize the model, set create an optimizer, and set up a
criterion for the loss function.

```python
# Set device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model

model = SentimentClassifier()

# Set up optimizers

optimizer = AdamW(model.parameters(), lr=learning_rate)

# set up  loss function
criterion = nn.BCEWithLogitsLoss()
```

### Training without Lightning

This is our typical training and validation loop without using Lightning:

```python
# Training loop

model.to(device)
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

# Evaluation

model.eval()
predictions = []
actual_labels = []
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        preds = model(input_ids, attention_mask)
        predictions.extend(torch.round(preds).cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())

accuracy = accuracy_score(actual_labels, predictions)
print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Accuracy: {accuracy:.4f}")
```

```python
Epoch [1/10], Train Loss: 0.3400, Val Accuracy: 0.3130

```

## PyTorch Lightning

There are two main components required to train a model with PyTorch Lightning:

1. `LightningDataModule`: Helps organize and encapsulate all data-related logic for your model. It's
   responsible for preparing your data, handling data loading, preprocessing, splitting into
   training/validation/test sets, and setting up data loaders. By separating data-related code from the
   rest of your model, it enhances code readability, reusability, and scalability.

2. `LightningModule`: Encapsulates all aspects of your model--the model architecture, optimization, loss
   functions, and training/validation steps. It separates the code for these components, providing
   specific methods like training_step and validation_step, which define what happens in each phase of
   the training process.

Import the Lightning library to see these two components in action:

```python
import lightning as L
```

### Data processing with Lightning

The methods of a `LightningDataModule` typically include:

1. `init`: Initializes the data module and sets up its initial parameters. This is where you typically define
   attributes like batch size, data set paths, transforms, etcetera. It's also where you might initialize
   variables used across different data-related methods.

2. `prepare_data`: Used for any data-related setup that happens only once, such as downloading data
   sets or preprocessing raw data. It's meant to be run in a single process before training starts.

3. `setup`: Handles any data-related logic that might be dependent on the state of the current process
   or GPU, such as splitting the data set into train, validation, and test sets, or applying specific
   transformations.

4. `train_dataloader`: Defines how your training data set should be loaded, batched, and shuffled.

5. `val_dataloader`: Defines how your validation data set should be loaded, batched, and shuffled.

```python
class SentimentDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, max_length=128):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data(self):
        self.dataset = load_dataset("imdb")

    def setup(self, stage=None):
        train_data = self.dataset['train']
        val_data = self.dataset['test']

        self.train_dataset = SentimentDataset(train_data, self.tokenizer, self.max_length)
        self.val_dataset = SentimentDataset(val_data, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
```

### Training with Lightning

The core component in PyTorch Lightning is a lightweight PyTorch wrapper that simplifies neural
network training. It encapsulates the essential parts of a PyTorch model into a structured and
organized format, making it easier to build, train, and test deep learning models. These methods
include:

1. `init`: Define the components of your model, such as layers, loss functions, optimizers, and any other
   attributes needed for training.

2. `forward` method: Similar to a PyTorch model, this method defines the forward pass of the neural
   network. It takes input tensors and computes the output or predictions.

3. `training_step`: This method defines what happens in a single training iteration. It takes a batch of
   data and computes the loss. It removes the need to explicitly define the loop for batches, loss
   computation, gradients, and optimizer steps.

4. `configure_optimizers`: Define the optimizers and, optionally, the learning rate schedulers to use
   during training.

5. `validation_step`: Similar to `training_step()`, this method defines the computation for a validation
   iteration.

```python
class SentimentClassifier(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.001) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        embeds = self.embedding(x)
        out = torch.mean(embeds, dim=1)  # Average the embeddings along the sequence length
        # sigmoid activation for binary classification
        out = torch.sigmoid(self.fc(out)).squeeze(1)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch):
        x = train_batch['input_ids']
        y = train_batch['label']
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch):
        x = val_batch['input_ids']
        y = val_batch['label']
        outputs = self(x)
        accuracy = Accuracy(task='binary').to(torch.device('cuda'))
        acc = accuracy(outputs, y)
        self.log('accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
        return
```

Once you've defined the data and model components, you can train the model using the following
code:

```python
model = SentimentClassifier()
dm = SentimentDataModule()

trainer = L.Trainer(max_epochs=num_epochs, accelerator='gpu')
trainer.fit(model, dm)
```

```python
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name      | Type              | Params
------------------------------------------------
0 | bert      | BertModel         | 109 M
1 | fc        | Linear            | 769
2 | criterion | BCEWithLogitsLoss | 0
------------------------------------------------
109 M     Trainable params
0         Non-trainable params
109 M     Total params
437.932   Total estimated model params size (MB)
```

### Complete code with PyTorch Lightning

Here is our complete code that you can run in a notebook or as a script in the terminal:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import lightning as L
from torchmetrics import Accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.texts = data['text']
        self.labels = data['label']
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = float(self.labels[idx])
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class SentimentDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, max_length=128):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data(self):
        self.dataset = load_dataset("imdb")

    def setup(self, stage=None):
        train_data = self.dataset['train']
        val_data = self.dataset['test']

        self.train_dataset = SentimentDataset(train_data, self.tokenizer, self.max_length)
        self.val_dataset = SentimentDataset(val_data, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)


class SentimentClassifier(L.LightningModule):
    def __init__(self, pretrained_model_name='bert-base-uncased', learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.embedding_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(self.embedding_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits.squeeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        preds = self(input_ids, attention_mask)
        accuracy = Accuracy(task='binary').to(torch.device('cuda'))
        acc = accuracy(preds, labels)
        self.log('accuracy', acc, prog_bar=True, on_step=False, on_epoch=True)
        return

num_epochs = 5
model = SentimentClassifier()
dm = SentimentDataModule()

trainer = L.Trainer(max_epochs=num_epochs, accelerator='gpu',limit_val_batches=0.1)
trainer.fit(model, dm)
```

Finally, as our purpose was to demonstrate typical ML workflow, rather than building the best model,
we encourage you to experiment further with model improvements. We suggest trying out other data
processing techniques and model architectures, and tinkering with hyperparameters.
