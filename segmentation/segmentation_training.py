from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from segmentation_datasets import DocumentDataset
from segmentation_models import ParagraphClassifier
from utils import set_seed

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

set_seed(42)

folder_path = "segmentation_labeled_csvs/" 
model_name = "allenai/scibert_scivocab_uncased"

dataset = DocumentDataset(folder_path, model_name, max_tokens=124)

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
    lstm_hidden_size=128,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 6
scaler = torch.cuda.amp.GradScaler()

# TRAINING

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        positional_values = batch["positional_value"].to(device)
        labels = batch["label"].to(device)

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



# EVALUATION

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        positional_values = batch["positional_value"].to(device)
        labels = batch["label"].to(device)


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask.squeeze(0),
            positional_values=positional_values.squeeze(0),
            labels=None
        )

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.squeeze(0).cpu())
        all_labels.append(labels.squeeze(0).cpu())


all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Validation Accuracy: {accuracy:.4f}")

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels.numpy(), all_preds.numpy(), average=None
)

print("\nPer-class Precision, Recall, F1:")
for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
    print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

for avg in ['micro', 'macro']:
    p, r, f, _ = precision_recall_fscore_support(
        all_labels.numpy(), all_preds.numpy(), average=avg
    )
    print(f"\n{avg.capitalize()} Average:")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")


print("\nClassification Report:\n")
print(classification_report(all_labels.numpy(), all_preds.numpy()))

cm = confusion_matrix(all_labels.numpy(), all_preds.numpy(), normalize='true')

classes = ["PRELIM/RELATED" if x == "LIT REVIEW" else x for x in dataset.label_encoder.classes_]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples", cbar=True, xticklabels=classes, yticklabels=classes)

plt.tight_layout()
plt.savefig("confusion_matrix.png")

torch.save(model.state_dict(), 'model_weights.pth')