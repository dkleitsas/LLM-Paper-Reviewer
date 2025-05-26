import os
import glob
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

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

# Load and merge datasets
datasets_list = [load_dataset("csv", data_files=file)["train"] for file in csv_files]
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

# === TRAIN/TEST SPLIT ===
dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)

# === TOKENIZATION ===
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

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
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === TRAIN ===
trainer.train()

# === SAVE FINAL MODEL ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)