import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps:float = 10**-8):
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta





















class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        self.hidden_size = hidden_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / std + self.beta