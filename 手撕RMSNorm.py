import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size:int, eps=1e-6):
        self.weights = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32) # 增加计算精度
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # 沿着嵌入向量维度计算方差
        """
        原始公式中并不是直接用的方法，而是 Sqrt(Mean(x^2)) , 但由于默认x的均值为0, x的方差 等价于 x^2的均值平方根
        """
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_eps) # torch.rsqrt为平方根倒数

        return self.weights * hidden_states.to(input_dtype)