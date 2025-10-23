import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)

        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model:int, hidden_size: int, dropout:float):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-8)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.out_project = nn.Linear(d_model, d_model)

        self.ffn = FeedForward(d_model, hidden_size, dropout)


    def attention(self, q, k, v, mask):
        d_model = q.shape[-1]
        
        attention_scores = (q @ k.transpose(1, 2)) / torch.sqrt(d_model)

        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, float('-inf'))

        attention_scores = attention_scores.softmax(dim=-1)

        attention_scores = self.attn_dropout(attention_scores)

        return (attention_scores @ v), attention_scores

    def forward(self, x, mask):
        # Attention Block
        x_norm = self.layer_norm1(x)

        q = self.w_q(x_norm)
        k = self.w_k(x_norm)
        v = self.w_v(x_norm)
        attn_out, self.attention_scores = self.attention(q, k, v, mask)
        attn_out = self.out_project(attn_out)
        attn_out = self.output_dropout(attn_out)

        x = x + attn_out

        # Feed Forward Block
        x_norm = self.layer_norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
    

