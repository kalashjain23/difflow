from torch.utils.data import DataLoader
from .trainer import Trainer
from difflow.nn import EMA
from torch import nn, Tensor
from torch.optim.adam import Adam
import torch
from typing import Optional


class Pi0Trainer(Trainer):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        model: nn.Module,
        checkpoint: int = 50,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.device = model.device
        self.checkpoint = checkpoint
        self.beta_dist = torch.distributions.Beta(1.5, 1.0)
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(list(p for p in model.parameters() if p.requires_grad), lr=2e-4)
        self.ema = EMA(self.model, ema_decay)
        
    def loss(self, v_field: Tensor, pred_v_field: Tensor):
        return self.criterion(v_field, pred_v_field)
    
    def train(self, data: DataLoader, start_epoch: int = 0):
        losses = []
        
        for epoch in range(start_epoch+1, self.epochs+1):
            epoch_loss = []
            for batch in data:
                images, prompt, token_type_ids, state, action = batch
                images = images.to(self.device)
                prompt = prompt.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                state = state.to(self.device)
                action = action.to(self.device)
                
                action_0 = torch.randn_like(action)
                
                t = self.beta_dist.sample((action.shape[0], 1, 1)).to(self.device) * 0.999
                
                action_t = ((1-t) * action_0) + (t * action)
                
                self.optimizer.zero_grad()

                pred_v_field = self.model.forward(images, prompt, token_type_ids, state, action_t, t.view(-1))
                v_field = action - action_0
                
                loss = self.loss(v_field, pred_v_field)
                epoch_loss.append(loss.item())
                loss.backward()
                
                self.optimizer.step()
                self.ema.update()
            
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(avg_loss)
            print(f"epoch {epoch}, loss: {avg_loss:.4f}")
                
            if epoch % self.checkpoint == 0:
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': losses,
                    'epoch': epoch,
                    'ema': self.ema.ema_weights
                }
                    
                torch.save(checkpoint, f'pi0_checkpoint_{epoch}.pth')
                print(f"epoch {epoch} saved")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['losses'], checkpoint['epoch']
    