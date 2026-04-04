from torch.utils.data import DataLoader
from .trainer import Trainer
from difflow.nn import EMA
from torch import nn, Tensor
from torch.optim.adam import Adam
import torch
from typing import Optional


class FMTrainer(Trainer):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        model: nn.Module,
        num_embed: Optional[int] = None,
        embed_dim: Optional[int] = None,
        lr=2e-4,
        checkpoint: int = 50,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.device = model.device
        self.checkpoint = checkpoint
        self.embedding = nn.Embedding(num_embed, embed_dim).to(self.device) if num_embed is not None else None
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            (list(model.parameters()) + list(self.embedding.parameters())) if self.embedding is not None else model.parameters(),
            lr=lr
        )
        self.ema = EMA(self.model, ema_decay)
        
    def loss(self, v_field: Tensor, pred_v_field: Tensor):
        # mse loss between the actual and predicted velocity field
        return self.criterion(v_field, pred_v_field)
    
    def train(self, data: DataLoader, start_epoch: int = 0):
        losses = []
        
        for epoch in range(start_epoch+1, self.epochs+1):
            epoch_loss = []
            for sample in data:
                # handling dataset with and without condition
                if isinstance(sample, (list, tuple)) and len(sample) == 2:
                    batch, label = sample
                    label = label.to(self.device)
                else:
                    batch = sample[0]
                    label = None
                    
                # initial noisy data distribution
                batch = batch.to(self.device)
                x0 = torch.randn_like(batch)
                t = torch.rand(batch.shape[0], 1, device=self.device)
                c_embed = self.embedding(label.to(self.device)) if self.embedding is not None else None
                
                # interpolated data distribution (mixture of signal and noise)
                batch_interpolated = ((1-t) * x0) + (t * batch)
                
                self.optimizer.zero_grad()

                # prediction of the velocity field based on the interpolated data distribution and timestep
                pred_v_field = self.model(batch_interpolated, t, c_embed)
                v_field = batch - x0
                
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
                if self.embedding is not None:
                    checkpoint['embedding'] = self.embedding.state_dict()
                    
                torch.save(checkpoint, f'fm_checkpoint_{epoch}.pth')
                print(f"epoch {epoch} saved")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.embedding is not None and 'embedding' in checkpoint:
            self.embedding.load_state_dict(checkpoint['embedding'])
        return checkpoint['losses'], checkpoint['epoch']
    