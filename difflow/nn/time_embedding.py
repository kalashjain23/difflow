from torch import nn
from torch import Tensor
import torch
from typing import Optional

import math

class TimeEmbedding(nn.Module):
    def __init__(self, model_channels: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        
        self.model_channels = model_channels
        out_dim = out_dim if out_dim else model_channels * 4
        
        self.linear1 = nn.Linear(self.model_channels, self.model_channels * 4)
        self.linear2 = nn.Linear(4 * self.model_channels, out_dim)
        
    def sinusoidal(self, t: Tensor) -> Tensor:
        # t: (B,)
        half_dim = self.model_channels // 2
        
        # (half_dim,)
        frequencies = torch.exp(-(torch.arange(half_dim) / half_dim) * math.log(10000)).to(t.device)
        
        # args: (B, half_dim)
        args = torch.outer(t, frequencies)
        
        # (B, dim)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embeddings
        
    def forward(self, t: Tensor) -> Tensor:
        x = self.sinusoidal(t)
        x = self.linear1(x)
        x = torch.nn.functional.silu(x)
        x = self.linear2(x)
        
        return x
        