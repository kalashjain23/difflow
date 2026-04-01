from torch import nn
from torch import Tensor
import torch
from typing import Optional


class Attention(nn.Module):
    def __init__(self, dim: int, context_dim: Optional[int] = None, heads: int = 1) -> None:
        super().__init__()
        
        self.dim = dim
        context_dim = context_dim if context_dim else dim
        self.heads = heads
        
        # projection layers
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(context_dim, self.dim)
        self.v_proj = nn.Linear(context_dim, self.dim)
        self.final_proj = nn.Linear(self.dim, self.dim)
        
    def forward(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        B, seq_len, dim = x.shape
        context = context if context is not None else x
        context_len = context.shape[1]
        
        # project q, k and v (acts as cross attn when context is provided)
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # break into heads
        q = q.reshape(B, seq_len, self.heads, self.dim // self.heads).transpose(1, 2)
        k = k.reshape(B, context_len, self.heads, self.dim // self.heads).transpose(1, 2)
        v = v.reshape(B, context_len, self.heads, self.dim // self.heads).transpose(1, 2)
        
        # compute and scale attention weights
        attn_weights = (q @ k.transpose(-1, -2))
        scaled_weights = attn_weights / ((self.dim // self.heads) ** 0.5)
        
        # aplpy masking
        if mask is not None: scaled_weights += mask
        
        # apply softmax to convert them into probability distribution
        weights = torch.nn.functional.softmax(scaled_weights, dim=-1)
        
        # convert into embeddings
        embeddings = weights @ v
        embeddings = embeddings.transpose(1, 2).reshape(B, seq_len, self.dim)
        
        return self.final_proj(embeddings)