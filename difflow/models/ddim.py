import torch
from torch import nn
from torch import Tensor
from .unet import UNet

from typing import Tuple


class DDIM(nn.Module):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        steps: int = 1000,
        ddim_steps: int = 50,
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
        self.ddim_steps = ddim_steps
        
        # model to predict the noise
        self.noise_model = UNet(in_channels, noise_model_channels, num_resnet, attention_channels, channel_mult, 4 * time_emb_dim, num_groups, dropout)
        
        # equally spaced beta between the specified range
        self.beta = torch.linspace(beta_start, beta_end, steps, device=device)
        # alpha_t = 1 - beta_t
        self.alpha = (1 - self.beta).to(device)
        
        # cum alpha is a tensor containing the cumulative product of the alpha tensor
        self.cum_alpha = torch.cumprod(self.alpha, dim=0).to(device)
        
        self.device = device
        
    def forward(self, x: Tensor, t: Tensor):
        cum_alpha = self.cum_alpha[t].reshape(x.shape[0], 1, 1, 1)
        noise = torch.randn_like(x)
        
        x = torch.sqrt(cum_alpha) * x + torch.sqrt(1 - cum_alpha) * noise
        return x, noise
    
    def reverse(self, x: Tensor, t: int, t_prev: int, eta: float):
        t_batch = torch.full((x.shape[0],), t, device=self.device)
        pred_noise = self.noise_model(x, t_batch)
        
        cum_alpha = self.cum_alpha[t].reshape(1, 1, 1, 1)
        cum_alpha_prev = self.cum_alpha[t_prev].reshape(1, 1, 1, 1) if t_prev >= 0 else torch.ones_like(cum_alpha)
        
        x_0 = ((x - torch.sqrt(1 - cum_alpha) * pred_noise) / torch.sqrt(cum_alpha)) * torch.sqrt(cum_alpha_prev)
        sigma = eta * torch.sqrt((1 - cum_alpha_prev) / (1 - cum_alpha)) * torch.sqrt(1 - cum_alpha / cum_alpha_prev)
        dir_xt = torch.sqrt(1 - cum_alpha_prev - torch.square(sigma)) * pred_noise
        
        return x_0 + dir_xt + sigma * torch.randn_like(x)
    
    def sample(self, shape: Tuple, eta: float = 0.0):
        x = torch.randn(shape, device=self.device)
        timesteps = torch.linspace(self.steps - 1, 0, self.ddim_steps).long()
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else -1
            x = self.reverse(x, t, t_prev, eta)
        
        return x