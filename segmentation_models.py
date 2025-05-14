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

class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class ParagraphClassifier(nn.Module):
    def __init__(self, model_name, lstm_hidden_size, num_labels, attention_size="None", window_size="None"):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.classifier_bert = nn.Linear(bert_hidden_size + 1, num_labels)

        self.lstm = nn.LSTM(input_size=bert_hidden_size + 1 + num_labels,
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            bidirectional=True)
        
        self.attention = SelfAttentionLayer(feature_size=lstm_hidden_size * 2)

        self.classifier_lstm = nn.Linear(lstm_hidden_size * 2, num_labels)


    def forward(self, input_ids, attention_mask, positional_values, labels=None):
        batch_size, num_paragraphs, num_tokens = input_ids.size()

        # Flatten paragraphs into a big batch for BERT
        input_ids_flat = input_ids.view(batch_size * num_paragraphs, num_tokens)
        attention_mask_flat = attention_mask.view(batch_size * num_paragraphs, num_tokens)
        positional_scalar_flat = positional_values.view(batch_size * num_paragraphs)

        # BERT forward
        bert_outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Concatenate positional scalar
        pos_scalar_expanded = positional_scalar_flat.unsqueeze(1)  # [batch_size * num_paragraphs, 1]
        bert_with_positional = torch.cat([cls_embeddings, pos_scalar_expanded], dim=1)

        # Initial paragraph logits from BERT
        logits_bert = self.classifier_bert(bert_with_positional)

        # Prepare LSTM input
        bert_with_positional = bert_with_positional.view(batch_size, num_paragraphs, -1)
        logits_bert_reshaped = logits_bert.view(batch_size, num_paragraphs, -1)

        lstm_input = torch.cat([bert_with_positional, logits_bert_reshaped], dim=-1)

        # LSTM processing
        lstm_output, _ = self.lstm(lstm_input)  # [batch_size, num_paragraphs, hidden_dim * 2]



        # Apply classifier to each paragraph
        logits_lstm = self.classifier_lstm(lstm_output)  # [batch_size, num_paragraphs, num_labels]

        # Loss calculation
        total_loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()

            # Flatten predictions and labels for loss computation
            logits_bert_flat = logits_bert.view(batch_size * num_paragraphs, -1)
            logits_lstm_flat = logits_lstm.view(batch_size * num_paragraphs, -1)
            labels_flat = labels.view(batch_size * num_paragraphs)

            loss_bert = loss_fn(logits_bert_flat, labels_flat)
            loss_lstm = loss_fn(logits_lstm_flat, labels_flat)
            total_loss = 0.3 * loss_bert + 0.7 * loss_lstm

        return {"logits": logits_lstm, "loss": total_loss} if labels is not None else {"logits": logits_lstm}
    

class ParagraphBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        input_size = hidden_size + 1
        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, input_ids, attention_mask, positional_values, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        positional_values = positional_values.unsqueeze(1)  # [batch, 1]
        cls_output = torch.cat([cls_output, positional_values], dim=1)

        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}
    

class DocumentLSTMClassifier(nn.Module):
    def __init__(self, model_name, lstm_hidden_size, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels

        # Freeze classifier from Stage 1
        self.bert_classifier = nn.Linear(self.hidden_size + 1, num_labels)
        self.bert_classifier.load_state_dict(torch.load("bert_classifier_head.pth"))
        for param in self.bert_classifier.parameters():
            param.requires_grad = False

        # New input size is CLS + logits
        self.lstm = nn.LSTM(
            input_size=self.hidden_size + num_labels + 1,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, positional_values, labels=None):
        print(input_ids.shape)
        batch_size, num_paragraphs, num_tokens = input_ids.shape

        input_ids = input_ids.view(-1, num_tokens)
        attention_mask = attention_mask.view(-1, num_tokens)
        positional_values = positional_values.view(-1, 1)

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B*P, hidden]
            cls_output = torch.cat([cls_embeddings, positional_values], dim=1)
            bert_logits = self.bert_classifier(cls_output)   # [B*P, num_labels]
        # Concatenate CLS + logits

        combined_input = torch.cat([cls_embeddings, bert_logits, positional_values], dim=1)  # [B*P, hidden+num_labels]
        combined_input = combined_input.view(batch_size, num_paragraphs, -1)

        # LSTM over paragraph sequence
        lstm_output, _ = self.lstm(combined_input)
        logits = self.classifier(lstm_output)  # [B, P, num_labels]


        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = loss_fn(logits_flat, labels_flat)

        return {"logits": logits, "loss": loss}