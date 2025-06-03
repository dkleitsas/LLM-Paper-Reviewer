import os
import glob
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# === CONFIGURATION ===
csv_folder = "classification_csvs"  # Folder where all CSV files are located
text_column = "Paragraph"   # Column name in your CSV
label_column = "Label" # Column with binary labels
model_name = "allenai/scibert_scivocab_uncased"
max_length = 512
output_dir = "classifier"

# === DYNAMICALLY LOAD ALL CSV FILES ===
csv_files = glob.glob(os.path.join(csv_folder, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files.")

datasets_list = []

for i, file in enumerate(csv_files):
    dataset = load_dataset("csv", data_files=file)["train"]
    paper_id = os.path.splitext(os.path.basename(file))[0]
    dataset = dataset.map(lambda ex: {**ex, "PaperID": paper_id})
    datasets_list.append(dataset)

combined_dataset = concatenate_datasets(datasets_list)

# === LABEL ENCODING ===
label_values = combined_dataset.unique(label_column)
label_values.sort()

# Adjust depending on your actual labels (0/1 or strings like 'positive'/'negative')
if set(label_values) == {0, 1}:
    label2id = {0: 0, 1: 1}
    id2label = {0: 'accepted', 1: 'rejcted'}
else:
    label2id = {'rejected': 0, 'accepted': 1}
    id2label = {0: 'rejected', 1: 'accepted'}

def encode_labels(example):
    example["label"] = label2id[example[label_column]]
    return example

combined_dataset = combined_dataset.map(encode_labels)

# === GROUPED TRAIN/TEST SPLIT BY PAPER ===
all_paper_ids = list(set(combined_dataset['PaperID']))
train_ids, test_ids = train_test_split(all_paper_ids, test_size=0.2, random_state=42)

train_dataset = combined_dataset.filter(lambda ex: ex['PaperID'] in train_ids)
test_dataset = combined_dataset.filter(lambda ex: ex['PaperID'] in test_ids)

dataset = {"train": train_dataset, "test": test_dataset}

print(f"Train papers: {len(set(train_dataset['PaperID']))}")
print(f"Test papers : {len(set(test_dataset['PaperID']))}")

# === TOKENIZATION ===
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# === LOAD MODEL ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# === COMPUTE METRICS ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,  # adjust based on available VRAM
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir=os.path.join(output_dir, "logs"),
    fp16=True,  # set to False if using CPU or older GPU
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === TRAIN ===
trainer.train()

predictions = trainer.predict(tokenized_test)
logits = predictions.predictions
predicted_labels = np.argmax(logits, axis=-1)

# === SECTION-LEVEL CONFUSION MATRIX ===
section_cm = confusion_matrix(predictions.label_ids, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(section_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(id2label.values()), yticklabels=list(id2label.values()))
plt.title("Section-Level Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("section_level_confusion_matrix.png")
plt.close()

# === PAPER-LEVEL AGGREGATION ===
test_with_preds = tokenized_test.add_column("PredictedLabel", predicted_labels)
df = pd.DataFrame(test_with_preds)
df["TrueLabel"] = df["label"]

# Optional: Save section-level results
df.to_csv("section_level_predictions.csv", index=False)

# Group and aggregate
paper_preds = defaultdict(list)
true_labels_per_paper = {}

for _, row in df.iterrows():
    pid = row["PaperID"]
    paper_preds[pid].append(row["PredictedLabel"])
    if pid not in true_labels_per_paper:
        true_labels_per_paper[pid] = row["TrueLabel"]

paper_level_preds = {
    pid: Counter(preds).most_common(1)[0][0]
    for pid, preds in paper_preds.items()
}

y_true = [true_labels_per_paper[pid] for pid in paper_level_preds]
y_pred = [paper_level_preds[pid] for pid in paper_level_preds]

acc = accuracy_score(y_true, y_pred)
prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("\n=== Paper-Level Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


paper_cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(paper_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(id2label.values()), yticklabels=list(id2label.values()))
plt.title("Paper-Level Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("paper_level_confusion_matrix.png")
plt.close()

# === SAVE FINAL MODEL ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)