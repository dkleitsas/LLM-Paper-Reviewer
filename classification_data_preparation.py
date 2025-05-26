import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
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
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, random_split

from segmentation_datasets import DocumentDataset
from segmentation_models import ParagraphClassifier

from collections import Counter, defaultdict


import csv
from tqdm import tqdm
import os


def filter_short_runs(labels, min_run_length=6):
    corrected = labels.copy()
    i = 0
    while i < len(labels):
        current = labels[i]
        run_start = i
        while i + 1 < len(labels) and labels[i + 1] == current:
            i += 1
        run_end = i
        run_length = run_end - run_start + 1

        if run_length < min_run_length:
            # Look at neighbors
            prev_label = labels[run_start - 1] if run_start > 0 else None
            next_label = labels[run_end + 1] if run_end + 1 < len(labels) else None

            if prev_label == next_label and prev_label is not None:
                for j in range(run_start, run_end + 1):
                    corrected[j] = prev_label

        i += 1
    return corrected


def sliding_window_smooth(labels, window_size=8):
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        window = labels[start:end]
        majority_label = Counter(window).most_common(1)[0][0]
        smoothed.append(majority_label)
    return smoothed

def clean_references_and_below(paragraphs, section_labels, references_labels):
    """
    Remove all paragraphs starting from the first 'References' section label (case-insensitive).
    """
    for i, label in enumerate(section_labels):
        if label.lower() in (r.lower() for r in references_labels):
            return paragraphs[:i], section_labels[:i]
    return paragraphs, section_labels


def aggregate_sections_global(paragraphs, section_labels, acceptance_label):
    section_map = defaultdict(list)

    for para, label in zip(paragraphs, section_labels):
        section_map[label].append(para)

    aggregated_labels = []
    aggregated_sections = []
    aggregated_paragraphs = []

    for label, paras in section_map.items():
        aggregated_labels.append(label)
        aggregated_sections.append(acceptance_label)
        aggregated_paragraphs.append(" ".join(paras))

    return aggregated_paragraphs, aggregated_labels, aggregated_sections


def run_segmentation_inference(model, dataset, output_folder, label_for_doc):

    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            positional_value = batch["positional_value"].to(device)
            original_paragraphs = batch["original_paragraphs"]
            print("before model")

            outputs = model(input_ids, attention_mask, positional_value)  # [1, P, num_labels]
            print("after model")
            preds = torch.argmax(outputs["logits"], dim=-1)  # [1, P]
            print("after argmax")
            pred_labels = preds[0].cpu().tolist()
            print("after pred labels")
            filtered_labels = filter_short_runs(pred_labels, min_run_length=6)
            smoothed_labels = sliding_window_smooth(filtered_labels, window_size=8)

            string_labels = dataset.label_encoder.inverse_transform(smoothed_labels)

            paragraphs_clean, section_labels_clean = clean_references_and_below(
                original_paragraphs[0], string_labels, references_labels=["REFERENCES"]
            )

            aggregated_pars, aggregated_sections, aggregated_labels = aggregate_sections_global(
                paragraphs_clean, section_labels_clean, label_for_doc
            )
            

            # Save to CSV
            original_path = dataset.file_paths[batch_idx]
            filename = os.path.basename(original_path)
            output_path = os.path.join(output_folder, filename)

            with open(output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Paragraph", "Section", "Label"])
                for para, sec, lab in zip(aggregated_pars, aggregated_sections, aggregated_labels):
                    writer.writerow([para, sec, lab])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accepted_folder = "paper_csvs/NeurIPS/accepted"
rejected_folder = "paper_csvs/NeurIPS/rejected"
output_folder = "classification_csvs/"

tokenizer_name = "allenai/scibert_scivocab_uncased"
model = ParagraphClassifier(
    model_name="allenai/scibert_scivocab_uncased",
    num_labels=8,
    lstm_hidden_size=128,
)

model.load_state_dict(torch.load("model_weights.pth", weights_only=True, map_location=device))

model.to(device)

for folder, label in [(accepted_folder, "accepted"), (rejected_folder, "rejected")]:
    dataset = DocumentDataset(folder, tokenizer_name, max_tokens=124, training=False)
    run_segmentation_inference(model, dataset, output_folder, label)


