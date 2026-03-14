from torch import nn
from torch import Tensor
import torch


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_groups: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        # x: (B, C, H, W), time_emb: (B, 4 * model_channels)
        residual = x
        x = self.block1(x)
        
        # (B, out_channels)
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        
        # x: (B, out_channels, H, W)
        x = torch.add(x, time_emb)
        
        # x: (B, out_channels, H, W)
        x = self.block2(x)
        
        # x: (B, out_channels, H, W)
        x = torch.add(x, self.residual_conv(residual))
        
        return x
        
        