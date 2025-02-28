import torch 
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.dropout(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.dorpout = nn.Dropout(dropout)

    @staticmethod
    def forward(self, q, k, v, mask, dropout):

        d_model = q.shape[-1]
        
        attention_scores = (q @ k) / math.sqrt(d_model)

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, 1e-9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ v), attention_scores

    def selfattention(self, x, mask):
        y = self.layer_norm(x)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        y, self.attention_scores = SelfAttention.forward(q, k, v, mask, self.dropout)
        x = x + y
        
        return x
    




class SelfAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.d_model = d_model

    def forward(self, x, mask, dropout):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attention_scores = q @ k.transpose(1, 2) / torch.sqrt(self.d_model)

        if mask:
            attention_scores.masked_fill_(0, -1e9)

        attention_scores = nn.Softmax(attention_scores)

        if dropout:
            attention_scores = self.dropout(attention_scores)

        return attention_scores @ v

        
