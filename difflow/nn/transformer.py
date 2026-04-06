from torch import nn, Tensor
from .attention import Attention
from typing import Optional, Tuple
import torch


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        
        self.self_attn = Attention(dim=dim, heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x: Tensor):
        attn_x = self.self_attn(self.norm1(x))
        x = x + attn_x
        
        ffn_x = self.ffn(self.norm2(x))
        x = x + ffn_x
        
        return x
    

class MOETransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 1, experts: int = 1):
        super().__init__()
        
        self.self_attn = Attention(dim=dim, heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.expert_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(experts)
        ])
        
    def forward(self, x: Tensor, token_split: Tuple, mask: Optional[Tensor] = None):
        attn_x = self.self_attn(self.norm1(x), mask=mask)
        x = x + attn_x
        residual = x
        x = self.norm2(x)
        
        chunks = torch.split(x, token_split, dim=1)
        
        out = []
        for chunk, ffn in zip(chunks, self.expert_ffn):
            out.append(ffn(chunk))
        
        x = torch.cat(out, dim=1)
        x = x + residual
        
        return x
        