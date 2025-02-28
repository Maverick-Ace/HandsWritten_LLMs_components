import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "嵌入向量维度无法被h整除"

        self.d_K = self.d_model // self.h # 计算出每个头的大小
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(self, query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # query, key, value -> (batch, h, seq_len, d_k)

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = nn.Softmax(attention_scores, dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
    
    def multiattention(self, q, k, v, mask):
        # query, key, value -> (batch, h, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_K).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_K).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_K).transpose(1, 2)

        # x -> (batch, h, seq_len, d_k)
        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_K * self.h)

        return self.w_o(x)




class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, heads, dropout):
        assert hidden_size % heads

        self.d_k = hidden_size//heads
        self.heads = heads

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask, dropout):
        batch_size = q.shape[0]

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2).transpose(2, 3)
        v = v.view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        
        attention_scores = q @ k / torch.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = nn.Softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = self.dropout(attention_scores)

        attention_scores = attention_scores @ v

        attention_scores = attention_scores.transpose(1, 2).contiguous().view(batch_size, -1, self.heads*self.d_k)

        return self.w_o(attention_scores)

