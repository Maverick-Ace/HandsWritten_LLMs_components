import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float):
        self.l1 = nn.Linear(d_model, hidden_size)
        self.l2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float):
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-8)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-8)

        self.out_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, hidden_size, dropout)

    def attention(self, q, k, v, mask):
        d_model = q.shape(-1)

        attention_scores = q @ k.transpose(1, 2) / torch.sqrt(d_model)
        # 做完mask后再softmax，使得被mask的位置变成0概率，同时不影响其他位置的概率和为1
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        
        attention_scores = self.attn_dropout(attention_scores)
        return (attention_scores @ v), attention_scores

    def forward(self, x, mask):
        # Attention Block
        x_norm = self.ln1(x)
        q, k, v = self.w_q(x_norm), self.w_k(x_norm), self.w_v(x_norm)
        attn, _ = self.attention(q, k, v, mask)

        attn = self.out_proj(attn)
        attn = self.out_dropout(attn)
        x = x + attn

        # Feed Forward Block
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x