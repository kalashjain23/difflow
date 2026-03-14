from torch import nn
from torch import Tensor
import torch


class SelfAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        
        self.dim = dim
        
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        
    def forward(self, x: Tensor):
        # x: (B, C, H*W)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # softmax(q * k.T / d ** 0.5) * v
        weights = torch.nn.functional.softmax((q @ k.transpose(-1, -2)) / self.dim ** 0.5, dim=-1)
        embeddings = weights @ v
        
        return self.proj(embeddings)