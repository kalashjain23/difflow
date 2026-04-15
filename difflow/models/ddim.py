import torch
from torch import Tensor
from .ddpm import DDPM

from typing import Tuple


class DDIM(DDPM):
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
        super().__init__(
            beta_start,
            beta_end,
            steps,
            in_channels,
            noise_model_channels,
            num_resnet,
            attention_channels,
            channel_mult,
            time_emb_dim,
            num_groups,
            dropout,
            device
        )
        
        self.ddim_steps = ddim_steps
    
    def reverse(self, x: Tensor, t: int, t_prev: int):
        t_batch = torch.full((x.shape[0],), t, device=self.device)
        pred_noise = self.noise_model(x, t_batch)
        
        cum_alpha = self.cum_alpha[t].reshape(1, 1, 1, 1)
        cum_alpha_prev = self.cum_alpha[t_prev].reshape(1, 1, 1, 1) if t_prev >= 0 else torch.ones_like(cum_alpha)
        
        x_0 = ((x - torch.sqrt(1 - cum_alpha) * pred_noise) / torch.sqrt(cum_alpha))
        xt_prev = x_0 * torch.sqrt(cum_alpha_prev) + torch.sqrt(1 - cum_alpha_prev) * pred_noise
        
        return xt_prev
    
    def sample(self, shape: Tuple):
        x = torch.randn(shape, device=self.device)
        timesteps = torch.linspace(self.steps - 1, 0, self.ddim_steps).long()
        
        for i in range(len(timesteps)):
            t = timesteps[i].item()
            t_prev = timesteps[i+1].item() if i+1 < len(timesteps) else -1
            x = self.reverse(x, t, t_prev)
        
        return x