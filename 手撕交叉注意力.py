import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.self_attention_block = ...
        self.cross_attention_block = ... # (q, k, v)
        self.feed_forword_block = ...
        self.residual_connection = nn.ModuleList(
            [self.residual_connection]*3
        )

    def residual_connection(self, x, y):
        return nn.LayerNorm(x + self.dropout(y), dim=-1)

    def forward(self, x, encoder_out, src_mask, tgt_mask): 
        # src_mask 为源掩码，即用于处理源序列的padding掩码
        # tgt_mask 为目标掩码，即用于处理目标序列的padding掩码，防止未来信息泄露
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(encoder_out, encoder_out, x, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forword_block(x))

        return x
    

class CrossAttention(nn.Module):
    def __init__(self, d_k:int, dropout:float):
        self.w_q = nn.Linear(d_k, d_k, bias=False)
        self.w_k = nn.Linear(d_k, d_k, bias=False)
        self.w_v = nn.Linear(d_k, d_k, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask, dropout):
        q = self.w_q(enc_output['q'])
        k = self.w_k(enc_output['k'])
        v = self.w_v(x)

        attention_scores = q @ k.transpose(1, 2)

        if tgt_mask is not None:
            attention_scores.masked_fill_(0, -1e9)

        attention_scores = nn.Softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = self.dropout(attention_scores)

        return attention_scores @ v