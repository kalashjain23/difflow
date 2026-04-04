import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Optional


class FlowMatching(nn.Module):
    def __init__(self, embed_dim: Optional[int] = None, device: str = "cpu"):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear((3 + embed_dim) if embed_dim else 3, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
        self.device = device
        
    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None):
        # x: (B, X, Y)
        # x is a collection of data points in the 2d space
        xt = torch.cat([x, t, c], dim=-1) if c is not None else torch.cat([x, t], dim=-1)
            
        return self.model(xt)
        
    def sample(self, size: Tuple, steps: int, c: Optional[Tensor] = None):
        # full noise initial sample
        x = torch.randn(size).to(self.device)
        dt = 1.0 / steps

        # euler solver
        for i in range(steps):
            t_val = i / steps
            t = torch.full((size[0], 1), t_val, device=self.device)
            x = x + self.forward(x, t, c) * dt
        
        return x
        