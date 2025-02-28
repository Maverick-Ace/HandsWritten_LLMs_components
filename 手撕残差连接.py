import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        self.norm = nn.LayerNorm(features, eps=10**-8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(ResidualConnection.sublayer(self.norm(x, dim=-1)))
    
    @staticmethod
    def sublayer(self, x):
        ...
        return ...
    






















class ResidualConnection(nn.Module):
    def __init__(self, hidden_size:int, dropout:float):
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(ResidualConnection.sublayer(self.norm(x, dim=-1)))
    
    @staticmethod
    def sublayer(self, x):
        ...