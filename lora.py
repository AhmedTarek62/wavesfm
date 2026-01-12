"""
Source: https://github.com/mnikitin/timm-vit-lora
"""


import torch
from functools import partial
from timm.models.vision_transformer import VisionTransformer


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = self.alpha / self.rank * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def create_lora_model(model: VisionTransformer, lora_rank=8, lora_alpha=1.0):
    # Add LoRA adapters to self-attention blocks (query, value)
    assign_lora = partial(QkvWithLoRA, rank=lora_rank, alpha=lora_alpha)
    for block in model.blocks:
        block.attn.qkv = assign_lora(block.attn.qkv)
    return model
