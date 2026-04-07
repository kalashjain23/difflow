from torch import nn, Tensor
import torch
from transformers import PaliGemmaForConditionalGeneration
from difflow.nn import TimeEmbedding, MOETransformerBlock
from typing import Tuple, Optional


class Pi0(nn.Module):
    def __init__(self, chunk_size: int, action_dim: int, embed_dim: int, state_dim: int, heads: int, n_layers: int, device: str = "cpu"):
        super().__init__()
        self.chunk_size = chunk_size
        self.device = device
        self.embed_dim = embed_dim
        
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-pt-224",
            dtype=torch.float16,
        )
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.vlm_proj = nn.Linear(2048, embed_dim)
        
        self.time_embedding = TimeEmbedding(embed_dim)
        self.w1 = nn.Linear(action_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim * 2, embed_dim)
        self.w3 = nn.Linear(embed_dim, embed_dim)
        
        self.state_proj = nn.Linear(state_dim, embed_dim)
        
        self.action_expert = nn.ModuleList([
            MOETransformerBlock(embed_dim, heads=heads, experts=2) for _ in range(n_layers)
        ])
        
        self.action_out_proj = nn.Linear(embed_dim, action_dim)
        
        
    def forward(self, images: Tensor, prompt: Tensor, token_type_ids: Optional[Tensor], states: Tensor, actions: Tensor, time: Tensor):
        outputs = self.vlm(
            input_ids=prompt,
            pixel_values=images,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        vlm_tokens = outputs.hidden_states[-1]
        vlm_tokens = self.vlm_proj(vlm_tokens.float())
        
        state_tokens = self.state_proj(states).unsqueeze(1)
        
        time_embed = self.time_embedding.sinusoidal(time).unsqueeze(1).expand(-1, actions.shape[1], -1)
        action_tokens = self.w3(
            nn.functional.silu(
                self.w2(
                    torch.cat([self.w1(actions), time_embed], dim=-1)
                )
            )
        )
        
        x = torch.cat([vlm_tokens, state_tokens, action_tokens], dim=1)
        
        n_vlm = vlm_tokens.shape[1]
        n_state = state_tokens.shape[1]
        n_action = action_tokens.shape[1]
        total = n_vlm + n_state + n_action
        
        mask = torch.zeros(total, total, device=x.device)
        mask[:n_vlm, n_vlm:] = -1e9
        mask[n_vlm:n_vlm+n_state, n_vlm+n_state:] = -1e9
        
        for block in self.action_expert:
            x = block(x, (n_vlm, n_state + n_action), mask=mask)
            
        action_out = x[:, -self.chunk_size:, :]
        
        return self.action_out_proj(action_out)
            
    def sample(self, action_size: Tuple, steps: int, images: Tensor, prompt: Tensor, states: Tensor, token_type_ids: Optional[Tensor] = None):
        action = torch.randn(action_size).to(self.device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = torch.full((action_size[0],), t_val, device=self.device)
            action = action + self.forward(images, prompt, token_type_ids, states, action, t) * dt
            
        return action