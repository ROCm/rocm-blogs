import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

from accelerate import Accelerator

# Instantiate the Accelerator class
accelerator = Accelerator()

print("Loading the data")

# Dataset preparation stage
# For illustration purposes we are fine-tuning the model on the first 1% of data. The dataset has 5 labels
dataset = load_dataset("yelp_review_full", split={'train': 'train[:1%]', 'test': 'test[:1%]'}) 

print("Tokenizing the data")

llm_model = "microsoft/Phi-3.5-mini-instruct"

# Tokenize the data with map method
tokenizer = AutoTokenizer.from_pretrained(llm_model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create train and evaluation dataloader
print("Instantiating the Dataloader")
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=8)

# Load the model for classification
print("Loading model for classification")
model = AutoModelForSequenceClassification.from_pretrained(llm_model, num_labels=5)

# Optimizer 
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# The model, optimizer and data loaders are passed to the Accelerator instance
train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer, lr_scheduler)

print("Begin Training...")

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print("Training completed.")

print("\nSaving model weights after training")

model_save_path = "model_classification_finetuned_phi35.pth"
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to: {model_save_path}")