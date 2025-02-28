import torch
import torch.nn as nn
import math
from utils import utils

def initialize_weight(x):
    # 初始化权重
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0) # 以0填充x.bias

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float):
        super().__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.layer2 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size:int, dropout:float, heads:int=8):
        self.heads = heads
        self.att_size = att_size = hidden_size//heads
        self.dropout = nn.Dropout(dropout)

        self.scale = 1/torch.sqrt(att_size)

        self.w_q = nn.Linear(hidden_size, att_size*heads, bias=False)
        self.w_k = nn.Linear(hidden_size, att_size*heads, bias=False)
        self.w_v = nn.Linear(hidden_size, att_size*heads, bias=False)
        self.w_o = nn.Linear(att_size*heads, hidden_size, bias=False)

        initialize_weight(self.w_q)
        initialize_weight(self.w_k)
        initialize_weight(self.w_v)
        initialize_weight(self.w_o)

    def forward(self, q, k, v, mask, cache:None):
        orig_q_size = q.shape()

        batch_size = q.shape[0]
        d_k = self.att_size
        d_v = self.att_size
        
        # 权重矩阵映射
        q = self.w_q(q) 
        # (batch, seq_len, hidden_size) -> (batch, sqe_len, att_size*heads)
        k = self.w_k(k)
        v = self.w_v(v)

        # 拆分多头，先拆q，因为q一定是从输入来的
        q = q.view(batch_size, -1, self.heads, self.att_size)

        # 计算注意力分数
        if cache is not None and k in cache:
            # 如果缓存中有k,v直接加载，此时用于交叉注意力
            k, v = cache['k'], cache['v'] 
        else:
            # 如果缓存中没有k,v，作为encoder段计算k,v并缓存
            k = k.view(batch_size, -1, self.heads, self.att_size)
            v = v.view(batch_size, -1, self.heads, self.att_size)

            if cache is not None: # 如果传入了额cache则代表需要更新cache
                cache['k'] = k
                cache['v'] = v

        # 将q,k,v的头维度向前提
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3) # 将k的提前转置，便于与q做点积
        v = v.transpose(1, 2)

        # 计算注意力分数
        x = q @ k
        # mask
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        # scale
        x.mul_(self.scale)
        # sofmax
        x = nn.Softmax(x, dim=-1)
        # dropout
        x = self.dropout(x)
        # attention value
        x = x @ v

        # (batch, h, seq_len, seq_len) -> (batch, seq_len, h, seq_len)
        x = x.transpose(1, 2).contiguous()

        # merge heads
        x = x.view(batch_size, -1, self.att_size*self.heads)

        x = self.w_o(x)

        assert x.shape() == orig_q_size
        return x
        
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float):
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.self_attention_norm(x)
        y = self.self_attention(y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float):
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout)
        self.self_attention_dropout = nn.Dropout(dropout)

        self.cross_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attention = MultiHeadAttention(hidden_size, dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        # self_mask让模型只能看到前i-1个token输入
        # i_mask为处理padding token 的mask
        # 经过第一次自注意力
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask) 
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None: # 如果存在编码器输出，则进行交叉注意力
            y = self.cross_attention_norm(x)
            y = self.cross_attention(x, enc_output, enc_output, i_mask, cache)
            y = self.cross_attention_dropout(y)
            x = x + y
        
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x
    
class Encoder(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float, n_layers:int=8):
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(hidden_size, filter_size, dropout)
            for _ in range(n_layers)]
        )
        # 最后输出时补一层norm
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        for layer in self.enc_layers:
            x = layer(x)
        
        x = self.layer_norm(x, mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float, n_layers:int=8):
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, filter_size, dropout)
             for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, targets, enc_output, src_mask, i_mask, cache):
        decoder_outputs = targets
        for i, layer in enumerate(self.layers): # 遍历每一层decoder
            layer_cache = None # 创建当前层的k、v缓存字典
            if cache is not None: # 如果传入了cache，则进行更新
                if i not in cache: # 检查第i层的decoder是否有encoder的k、v缓存，没有则创建
                    cache[i] = {}
                layer_cache = cache[i] # 将该层的缓存字典赋给变量并传入decoder layer进行更新，cache的大字典会同时更新
            
            targets = layer(targets, enc_output, src_mask, i_mask, layer_cache)
    
        return self.layer_norm(targets)
    

