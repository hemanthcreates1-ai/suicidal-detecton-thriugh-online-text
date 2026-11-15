import torch
import torch.nn as nn
from transformers import BertModel

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, encoder_outputs):
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        energy = self.projection(encoder_outputs) # (batch_size, seq_len, 1)
        weights = torch.softmax(energy.squeeze(-1), dim=1) # (batch_size, seq_len)
        
        # (batch_size, seq_len, 1) * (batch_size, seq_len, hidden_dim) -> sum over seq_len
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1) # (batch_size, hidden_dim)
        return outputs, weights

class BertCNNAttention(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, cnn_filters=128, filter_sizes=[2, 3, 4], dropout_prob=0.3):
        super(BertCNNAttention, self).__init__()
        
        # 1. Deep Semantic Context: BERT Transformer
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 2. Local Feature Extraction: 1D CNN Layer on BERT sequence outputs
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size, 
                      out_channels=cnn_filters, 
                      kernel_size=k)
            for k in filter_sizes
        ])
        
        # 3. Self-Attention Layer to capture hidden meanings
        # Attention over CNN grouped features
        self.attention = SelfAttention(hidden_dim=len(filter_sizes) * cnn_filters)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(len(filter_sizes) * cnn_filters, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT Outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence hidden state: (batch_size, seq_len, bert_hidden_size)
        sequence_output = bert_outputs.last_hidden_state
        
        # Permute for CNN: (batch_size, bert_hidden_size, seq_len)
        sequence_output = sequence_output.permute(0, 2, 1)
        
        # Apply CNNs
        conv_outputs = []
        for conv in self.convs:
            # (batch_size, cnn_filters, seq_len - kernel_size + 1)
            c = torch.relu(conv(sequence_output))
            # Pad to the same length so we can concatenate along the channel dimension
            # We pad the right side
            pad_size = sequence_output.size(-1) - c.size(-1)
            c = torch.nn.functional.pad(c, (0, pad_size))
            conv_outputs.append(c)
            
        # Concatenate CNN outputs along the filter dimension
        # (batch_size, len(filter_sizes)*cnn_filters, seq_len)
        cnn_features = torch.cat(conv_outputs, dim=1)
        
        # Permute back for attention: (batch_size, seq_len, hidden_dim)
        cnn_features_seq = cnn_features.permute(0, 2, 1)
        
        # Apply Self-Attention
        attention_out, attn_weights = self.attention(cnn_features_seq)
        
        # Classification Head
        out = self.dropout(attention_out)
        out = self.fc1(out)
        out = self.relu(out)
        logits = self.fc2(out)
        
        return logits
