import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, merge, rank=16, lora_alpha=1.0, dropout=0.5):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha # 旁路参数合并到主路的权重
        self.dropout = dropout

        # 线性映射层
        self.linear = nn.Linear(in_features, out_features)
        if rank > 0: 
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank)) # 构建全零初始化的B矩阵
            self.lora_a = nn.Parameter(torch.normal(0, 1/math.sqrt(self.in_features), (self.in_features, self.rank)))
            # 定义dropout层
            self.scaling = self.lora_alpha / self.rank
            self.lora_dropout = nn.Dropout(p=dropout)
            # 是否合并LoRA权重到原始权重
            self.merge = merge
            self.merged = False

    def train(self, mode=True):
        """设置训练模式"""
        super(LoRALinear, self).train(mode)
        if mode:
            if self.merge and self.merged:
                # 如果之前合并过，需要恢复原始权重
                self.linear.weight.data -= self.get_lora_weight()
                self.merged = False
        else:
            if self.merge and not self.merged:
                # 在评估模式下合并权重
                self.linear.weight.data += self.get_lora_weight()
                self.merged = True
        
    def get_lora_weight(self):
        """获取LoRA权重"""
        return (self.lora_b @ self.lora_a.t()) * self.scaling

    def forward(self, x):
        if self.rank > 0:
            if self.merged:
                return self.linear(x)
            else:
                # 主路计算
                result = self.linear(x)
                # LoRA旁路计算
                lora_output = self.lora_dropout(x) @ self.lora_a @ self.lora_b.t()
                return result + lora_output * self.scaling
        else:
            return self.linear(x)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'rank={self.rank}, '
            f'lora_alpha={self.lora_alpha}, '
            f'dropout={self.dropout}, '
            f'merged={self.merged}'
        )

