import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_filter:int, dropout:float):
        self.linear_1 = nn.Linear(d_model, d_filter)
        self.linear_2 = nn.Linear(d_filter, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(
            self.dropout(
                torch.relu(
                    self.linear_1(x)
                    )
                )
            )
    



















class FFN(nn.Module):
    def __init__(self, hidden_size:int, filter_size:int, dropout:float):
        super().__init__()
        self.l1 = nn.Linear(hidden_size, filter_size)
        self.l2 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        y = self.l1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.l2(y)
        x = x + y
        return self.norm(x)
