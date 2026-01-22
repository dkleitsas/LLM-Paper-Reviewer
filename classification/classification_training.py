import os
import glob
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from scipy.special import softmax
from sklearn.metrics import classification_report

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

csv_folder = "classification_csvs"          # Folder where all CSV files are located
text_column = "Paragraph"                   # Column name in CSV
label_column = "Label"                      # Column with labels
model_name = "allenai/scibert_scivocab_uncased"
max_length = 512
output_dir = "classifier"

csv_files = glob.glob(os.path.join(csv_folder, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files.")

datasets_list = []

for i, file in enumerate(csv_files):
    dataset = load_dataset("csv", data_files=file)["train"]
    paper_id = os.path.splitext(os.path.basename(file))[0]
    dataset = dataset.map(lambda ex: {**ex, "PaperID": paper_id})
    datasets_list.append(dataset)

combined_dataset = concatenate_datasets(datasets_list)

label_values = combined_dataset.unique(label_column)
label_values.sort()

label2id = {'rejected': 0, 'accepted': 1}
id2label = {0: 'rejected', 1: 'accepted'}

def encode_labels(example):
    example["label"] = label2id[example[label_column]]
    return example

combined_dataset = combined_dataset.map(encode_labels)

all_paper_ids = list(set(combined_dataset['PaperID']))
train_ids, test_ids = train_test_split(all_paper_ids, test_size=0.2, random_state=42)

train_dataset = combined_dataset.filter(lambda ex: ex['PaperID'] in train_ids)
test_dataset = combined_dataset.filter(lambda ex: ex['PaperID'] in test_ids)

dataset = {"train": train_dataset, "test": test_dataset}

print(f"Train papers: {len(set(train_dataset['PaperID']))}")
print(f"Test papers : {len(set(test_dataset['PaperID']))}")

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

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

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

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir=os.path.join(output_dir, "logs"),
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(tokenized_test)
logits = predictions.predictions
predicted_labels = np.argmax(logits, axis=-1)


# Section-level confusion matrix
section_cm = confusion_matrix(predictions.label_ids, predicted_labels, normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(section_cm, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=["Rejected", "Accepted"], yticklabels=["Rejected", "Accepted"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("section_level_confusion_matrix.png")
plt.close()


# Paper-level aggregation
test_with_preds = tokenized_test.add_column("PredictedLabel", predicted_labels)
df = pd.DataFrame(test_with_preds)
df["TrueLabel"] = df["label"]


probs = softmax(logits, axis=1)

# Attach predictions
test_with_preds = tokenized_test.add_column("PredictedLabel", predicted_labels)
test_with_preds = test_with_preds.add_column("ConfidenceRejected", probs[:, label2id['rejected']])
test_with_preds = test_with_preds.add_column("ConfidenceAccepted", probs[:, label2id['accepted']])
df = pd.DataFrame(test_with_preds)
df["TrueLabel"] = df["label"]
df.to_csv("section_level_predictions.csv", index=False)

# Group by paper
grouped = df.groupby("PaperID")
paper_results = {}

for pid, group in grouped:
    section_preds = list(group["PredictedLabel"])
    confidences_rej = list(group["ConfidenceRejected"])
    confidences_acc = list(group["ConfidenceAccepted"])
    true_label = group["TrueLabel"].iloc[0]

    # Aggregation strategies
    majority_label = Counter(section_preds).most_common(1)[0][0]
    any_rejected_label = label2id['rejected'] if label2id['rejected'] in section_preds else label2id['accepted']
    all_rejected_label = label2id['rejected'] if all(p == label2id['rejected'] for p in section_preds) else label2id['accepted']
    confidence_weighted_label = label2id['rejected'] if sum(confidences_rej) > sum(confidences_acc) else label2id['accepted']

    paper_results[pid] = {
        "TrueLabel": true_label,
        "MajorityVote": majority_label,
        "AnyRejected": any_rejected_label,
        "AllRejected": all_rejected_label,
        "ConfidenceWeighted": confidence_weighted_label
    }

# Evaluate each strategy
strategy_names = ["MajorityVote", "AnyRejected", "AllRejected", "ConfidenceWeighted"]

print("\n=== Paper-Level Aggregation Metrics ===")
for strategy in strategy_names:
    y_true = [v["TrueLabel"] for v in paper_results.values()]
    y_pred = [v[strategy] for v in paper_results.values()]
    print(f"\n--- {strategy} ---")
    print(classification_report(y_true, y_pred, target_names=list(id2label.values())))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples',
                xticklabels=["Rejected", "Accepted"], yticklabels=["Rejected", "Accepted"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{strategy.lower()}_confusion_matrix.png")
    plt.close()

# Save paper-level predictions
paper_df = pd.DataFrame.from_dict(paper_results, orient="index").reset_index().rename(columns={"index": "PaperID"})
paper_df["TrueLabelName"] = paper_df["TrueLabel"].map(id2label)
for strategy in strategy_names:
    paper_df[strategy + "Name"] = paper_df[strategy].map(id2label)
paper_df.to_csv("paper_level_predictions.csv", index=False)

strategy_label_map = {
    "MajorityVote": "Majority Voting",
    "AnyRejected": "Any Rejected",
    "AllRejected": "All Rejected",
    "ConfidenceWeighted": "Confidence Weighted"
}

# Metrics per strategy
strategy_scores = {
    "Strategy": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}

for strategy in strategy_names:
    y_true = [v["TrueLabel"] for v in paper_results.values()]
    y_pred = [v[strategy] for v in paper_results.values()]
    
    strategy_scores["Strategy"].append(strategy_label_map[strategy])
    strategy_scores["Accuracy"].append(accuracy_score(y_true, y_pred))
    strategy_scores["Precision"].append(precision_score(y_true, y_pred))
    strategy_scores["Recall"].append(recall_score(y_true, y_pred))
    strategy_scores["F1"].append(f1_score(y_true, y_pred))


scores_df = pd.DataFrame(strategy_scores)
scores_df = scores_df.set_index("Strategy").reset_index().melt(id_vars="Strategy", var_name="Metric", value_name="Score")

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=scores_df, x="Strategy", y="Score", hue="Metric")
plt.ylim(0, 1.05)
plt.legend(title="Metric")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("paper_level_strategy_comparison.png")
plt.close()

# Save final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)


