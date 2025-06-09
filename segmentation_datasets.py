import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder


class DocumentDataset(Dataset):
    def __init__(self, folder_path, tokenizer_name, max_tokens=128, training=True):
        self.file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.csv')]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.training = training

        self.label_encoder = LabelEncoder()

        # Fit label encoder to ALL labels across all documents
        all_labels = ['ABSTRACT', 'CONCLUSION/DISCUSSION', 'IMPLEMENTATION/METHODS',
                        'INTRODUCTION', 'LIT REVIEW', 'OTHER', 'REFERENCES', 'RESULTS/EXPERIMENTS']
        self.label_encoder.fit(all_labels)

        print("Classes:", self.label_encoder.classes_)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load one document (one CSV)
        df = pd.read_csv(file_path)

        paragraphs = df["Paragraph"].astype(str).fillna("").tolist()
        paragraphs = [p for p in paragraphs if p.strip()]
        positional_values = df["Section Appearance Order"].tolist()
        if self.training:
            labels = df["Section"].astype(str).str.strip().tolist()

            labels = self.label_encoder.transform(labels)
        else:
            original_paragraphs = paragraphs

        encoding = self.tokenizer(
            paragraphs,
            truncation=True,
            padding="max_length",
            max_length=self.max_tokens,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"]           # Shape: [num_paragraphs, num_tokens]
        attention_mask = encoding["attention_mask"] # Shape: [num_paragraphs, num_tokens]
        positional_values = torch.tensor(positional_values, dtype=torch.float)  # Shape: [num_paragraphs]
        if self.training:
            labels = torch.tensor(labels, dtype=torch.long)                         # Shape: [num_paragraphs]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "positional_value": positional_values,
                "label": labels
            }
        
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "positional_value": positional_values,
                "original_paragraphs": original_paragraphs
            }

    # 2. Collate function
    def collate_fn(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        positional_values = [item['positional_value'] for item in batch]
        if self.training:
            labels = [item['label'] for item in batch]
        else:
            original_paragraphs = [item['original_paragraphs'] for item in batch]

        # Pad on number of paragraphs (batch dimension)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        positional_values = pad_sequence(positional_values, batch_first=True, padding_value=0.0)
        if self.training:
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 to ignore padding in loss

            return {
                "input_ids": input_ids,            # Shape: [batch_size, max_paragraphs, num_tokens]
                "attention_mask": attention_mask,
                "positional_value": positional_values,
                "label": labels
            }
        else:
            return {
                "input_ids": input_ids,            # Shape: [batch_size, max_paragraphs, num_tokens]
                "attention_mask": attention_mask,
                "positional_value": positional_values,
                "original_paragraphs": original_paragraphs  
            }
