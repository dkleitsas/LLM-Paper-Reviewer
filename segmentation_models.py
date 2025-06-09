import torch
from torch import nn
from transformers import AutoModel


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
    
