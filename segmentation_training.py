import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
import torch
from transformers import Trainer
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets_segmentation import DocumentDataset, ParagraphDataset
from segmentation_models import ParagraphBERTClassifier, ParagraphClassifier
from utils import set_seed

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

set_seed(0)

folder_path = "labeled_csvs/" 
model_name = "allenai/scibert_scivocab_uncased"

dataset = DocumentDataset(folder_path, model_name, max_tokens=100)

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(dataset.label_encoder.classes_)

model = ParagraphClassifier(
    model_name="allenai/scibert_scivocab_uncased",
    num_labels=num_classes,
    lstm_hidden_size=256,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()



num_epochs = 6
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)             # [batch=1, paragraphs, tokens]
        attention_mask = batch["attention_mask"].to(device)   # [batch=1, paragraphs, tokens]
        positional_values = batch["positional_value"].to(device) # [batch=1, paragraphs]
        labels = batch["label"].to(device)                   # [batch=1, paragraphs]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
          outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask.squeeze(0),
              positional_values=positional_values.squeeze(0),
              labels=labels.squeeze(0)
          )

          loss = outputs["loss"]

        total_train_loss += loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Average Training Loss = {avg_train_loss:.4f}")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader):  # this loader yields one document per batch
        input_ids = batch["input_ids"].to(device)             # [1, paragraphs, tokens]
        attention_mask = batch["attention_mask"].to(device)   # [1, paragraphs, tokens]
        positional_values = batch["positional_value"].to(device)  # [1, paragraphs]
        labels = batch["label"].to(device)                   # [1, paragraphs]


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask.squeeze(0),
            positional_values=positional_values.squeeze(0),
            labels=None
        )

        logits = outputs["logits"]        # [1, paragraphs, num_classes]
        preds = torch.argmax(logits, dim=-1)  # [1, paragraphs]

        all_preds.append(preds.squeeze(0).cpu())
        all_labels.append(labels.squeeze(0).cpu())

# Stack and evaluate
all_preds = torch.cat(all_preds, dim=0)   # [total_paragraphs]
all_labels = torch.cat(all_labels, dim=0) # [total_paragraphs]

accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Validation Accuracy: {accuracy:.4f}")

wait = input("Waiting...")

torch.save(model.state_dict(), 'model_weights.pth')

cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), normalize='true')

classes = ["PRELIM/RELATED" if x == "LIT REVIEW" else x for x in dataset.label_encoder.classes_]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples", cbar=True, xticklabels=classes, yticklabels=classes)

plt.title("Confusion Matrix - Validation Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png")