import torch
import torch.nn as nn
import torch.nn.functional as F
from bert.bert_config import BertConfig
from dl import BaseModel

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        return self.out(attention_output)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config.hidden_size, config.num_attention_heads)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = FeedForward(config.hidden_size, config.intermediate_size, config.hidden_dropout_prob)
        self.feed_forward_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.attention_norm(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.feed_forward_norm(x + feed_forward_output)
        return x

class BertModel(BaseModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.mlm_head(x)
        return logits
