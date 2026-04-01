from torch import Tensor, nn
from .attention import Attention
from einops import rearrange


class UNetAttention(nn.Module):
    def __init__(self, num_groups: int, channels: int) -> None:
        super().__init__()
        
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.attention = Attention(channels)
        
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.attention(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x + residual
        