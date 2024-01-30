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