from torch.utils.data import DataLoader
from .trainer import Trainer
from difflow.nn.ema import EMA
from torch import nn, Tensor
from torch.optim.adam import Adam
import torch


class DDPMTrainer(Trainer):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        model: nn.Module,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.device = model.device
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(model.parameters(), lr=2e-4)
        self.ema = EMA(self.model, ema_decay)
        
    def loss(self, noise: Tensor, pred_noise: Tensor):
        return self.criterion(noise, pred_noise)
    
    def train(self, data: DataLoader):
        losses = []
        
        for epoch in range(1, self.epochs+1):
            epoch_loss = []
            for batch, _ in data:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                t = torch.randint(0, self.model.steps, (batch.shape[0],)).to(self.device)
                noisy_image, noise = self.model.forward(batch, t)
                
                pred_noise = self.model.noise_model.forward(noisy_image, t)
                
                loss = self.loss(noise, pred_noise)
                epoch_loss.append(loss.item())
                loss.backward()
                
                self.optimizer.step()
                self.ema.update()
            
            avg_loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(avg_loss)
            print(f"epoch {epoch}, loss: {avg_loss:.4f}")
                
            if epoch % 10 == 0:
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': losses,
                    'epoch': epoch,
                    'ema': self.ema.ema_weights
                }, f'ddpm_checkpoint_{epoch}.pth')
                print(f"epoch {epoch} saved")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['losses'], checkpoint['epoch']
    