import torch
import torch.nn as nn

##### 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model

    def forward(self, x, mask, dropout=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attention_scores = q @ k.transpose(1, 2) / torch.sqrt(self.d_model)
        if mask:
            attention_scores = attention_scores.masked_fill_(0, -1e9)
        
        attention_scores = nn.Softmax(attention_scores, dim=-1)

        if dropout:
            attention_scores = self.dropout(attention_scores)

        return attention_scores @ v

##### FFN
class FFN(nn.Module):
    def __init__(self, d_model, filter_size, dropout:float):
        super().__init__()

        self.d_model = d_model
        self.filter_size = filter_size
        self.dropout = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, filter_size)
        self.linera2 = nn.Linear(filter_size, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linera2(y)
        x = x + y
        return x
    
##### 位置编码
class PE(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-torch.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)

        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shpe[1], :]).requires_grad_(False)
        return self.dropout(x)
    
##### LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, features, eps:float=10**-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

        self.eps = eps
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        return ( (x - mean) / (std + self.eps) ) * self.gamma + self.beta
    
##### RMSNorm
class RMSNorm(nn.Module):
    def __init(self, hidden_size:int, eps:float=10**-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32) # 增加计算精度

        variance = x.pow(2).mean(-1, keep_dim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return self.gamma * x.to(input_dtype)

##### 交叉注意力
class CrossAttention(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        self.w_q = nn.Linear(d_model, bias=False)
        self.w_k = nn.Linear(d_model, bias=False)
        self.w_v = nn.Linear(d_model, bias=False)

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, i_mask):
        q = self.w_q(enc_out['q'])
        k = self.w_k(enc_out['k'])
        v = self.w_v(x)

        attention_scores = q @ k.transpose(1, 2) / torch.sqrt(self.d_model)

        if i_mask:
            attention_scores = attention_scores.masked_fill_(i_mask, -1e9)
        
        attention_scores = nn.Softmax(attention_scores, dim=-1)

        return attention_scores @ v   

