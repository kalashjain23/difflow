from torch import nn
from typing import Tuple
from torch import Tensor
import torch

from .resnet import ResNet
from .time_embedding import TimeEmbedding
from .unet_attention import Attention


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_resnet: int,
        attention_channels: Tuple[int],
        channel_mult: Tuple[int],
        time_emb_dim: int,
        num_groups: int,
        dropout: float
    ):
        super().__init__()
        
        self.attention_channels = attention_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        
        self.time_embedding = TimeEmbedding(model_channels)
        
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        prev_channel = model_channels
        self.encoder = nn.ModuleList()
        for i in range(len(channel_mult)):
            channel = model_channels * channel_mult[i]
            
            for _ in range(num_resnet):
                self.encoder.append(ResNet(prev_channel, channel, time_emb_dim, num_groups, dropout))
                if channel in attention_channels:
                    self.encoder.append(Attention(num_groups, channel))
                
                prev_channel = channel
                
            # Downsampling
            if i != len(channel_mult)-1:
                self.encoder.append(nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1))
            
        self.neck = nn.ModuleList([
            ResNet(prev_channel, prev_channel, time_emb_dim, num_groups, dropout),
            Attention(num_groups, prev_channel),
            ResNet(prev_channel, prev_channel, time_emb_dim, num_groups, dropout)
        ])
        
        self.decoder = nn.ModuleList()
        prev_channel = model_channels * channel_mult[-1]
        for i in reversed(range(len(channel_mult))):
            channel = model_channels * channel_mult[i]
            
            if i != len(channel_mult) - 1:
                self.decoder.append(nn.ConvTranspose2d(
                    model_channels * channel_mult[i+1],
                    channel,
                    kernel_size=4, stride=2, padding=1
                ))
                prev_channel = channel

            for j in range(num_resnet):
                in_ch = prev_channel + channel if (j == 0 and i != len(channel_mult) - 1) else channel
                self.decoder.append(ResNet(in_ch, channel, time_emb_dim, num_groups, dropout))
                if channel in attention_channels:
                    self.decoder.append(Attention(num_groups, channel))
                
                prev_channel = channel
                
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, prev_channel),
            nn.SiLU(),
            nn.Conv2d(prev_channel, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x: Tensor, t: Tensor):
        skips = []
        t_emb = self.time_embedding(t)
        
        x = self.init_conv(x)
        
        for module in self.encoder:
            if isinstance(module, ResNet):
                x = module(x, t_emb)
            elif isinstance(module, Attention):
                x = module(x)
            else:
                skips.append(x)
                x = module(x)
                
        for module in self.neck:
            if isinstance(module, ResNet):
                x = module(x, t_emb)
            elif isinstance(module, Attention):
                x = module(x)
            else:
                x = module(x)

        is_first_resnet = False
        for module in self.decoder:
            if isinstance(module, ResNet):
                if is_first_resnet:
                    x = torch.cat([x, skips.pop()], dim=1)
                    is_first_resnet = False
                x = module(x, t_emb)
            elif isinstance(module, Attention):
                x = module(x)
            else:
                x = module(x)
                is_first_resnet = True
                
        x = self.out_conv(x)
        
        return x
        