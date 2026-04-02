import torch
from torch import nn
from torch import Tensor
from .unet import UNet

from typing import Tuple


class DDPM(nn.Module):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        steps: int = 1000,
        in_channels: int = 3,
        noise_model_channels: int = 128,
        num_resnet: int = 2,
        attention_channels: Tuple[int] = (256,),
        channel_mult: Tuple[int] = (1, 2, 2, 2),
        time_emb_dim: int = 128,
        num_groups: int = 32,
        dropout: int = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        
        # total steps to run 
        self.steps = steps
        
        # model to predict the noise
        self.noise_model = UNet(in_channels, noise_model_channels, num_resnet, attention_channels, channel_mult, 4 * time_emb_dim, num_groups, dropout)
        
        # equally spaced beta between the specified range
        self.beta = torch.linspace(beta_start, beta_end, steps, device=device)
        # converting into (1 - beta**2) ** 0.5
        self.alpha = torch.sqrt(1 - torch.square(self.beta)).to(device)
        
        # cum alpha is a tensor containing the cumulative product of the alpha tensor
        self.cum_alpha = torch.cumprod(self.alpha, dim=0).to(device)
        
        self.device = device
        
    def forward(self, x: Tensor, t: Tensor):
        cum_alpha = self.cum_alpha[t].reshape(x.shape[0], 1, 1, 1)
        noise = torch.randn_like(x)
        
        x = cum_alpha * x + torch.sqrt(1 - torch.square(cum_alpha)) * noise
        return x, noise
    
    def reverse(self, x: Tensor, t: int):
        t_batch = torch.full((x.shape[0],), t, device=self.device)
        pred_noise = self.noise_model(x, t_batch)
        
        alpha = self.alpha[t].reshape(1, 1, 1, 1)
        cum_alpha = self.cum_alpha[t].reshape(1, 1, 1, 1)
        
        mu = ((1 / alpha) * (x - ((1 - torch.square(alpha))/torch.sqrt(1 - torch.square(cum_alpha))) *  pred_noise))
        
        if t == 0: return mu
        
        sigma = torch.sqrt(self.beta[t]) * torch.randn_like(x)
        
        return mu + sigma
    
    def sample(self, shape: Tuple):
        x = torch.randn(shape, device=self.device)
        
        for t in reversed(range(self.steps)):
            x = self.reverse(x, t)
        
        return x