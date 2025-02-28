import torch
import torch.nn as nn
import math

class PostionalEncodeing(nn.Moudle):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, ) -> (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model) 为了后续与x相加保持一致的维度

        self.register_buffer('pe', pe) # 将pe注册为buffer，这样在保存模型的时候pe不会被保存

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # x -> (batch, sqe_len, d_model)
        return self.dropout(x)
