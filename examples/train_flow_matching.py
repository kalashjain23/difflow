from torch.utils.data import DataLoader, TensorDataset
import torch
from difflow.models import FlowMatching
from difflow.training import FMTrainer

# dummy spiral dataset
def multi_spiral_dataset(n_spirals=2, n=50000, noise=0.05):
    all_data, all_labels = [], []
    
    for i in range(n_spirals):
        theta = torch.linspace(0, 4 * torch.pi, n // n_spirals)
        r = theta / (4 * torch.pi)
        offset = (2 * torch.pi * i) / n_spirals  # rotate each spiral
        x = r * torch.cos(theta + offset) + torch.randn(n // n_spirals) * noise
        y = r * torch.sin(theta + offset) + torch.randn(n // n_spirals) * noise
        data = torch.stack([x, y], dim=1)
        labels = torch.full((n // n_spirals,), i, dtype=torch.long)
        all_data.append(data)
        all_labels.append(labels)
    
    return TensorDataset(torch.cat(all_data), torch.cat(all_labels))


if __name__ == '__main__':
    batch_size = 128
    embed_dim = 8
    num_embed = 2
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = multi_spiral_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FlowMatching(embed_dim=embed_dim, device=device).to(device)
    model.train()
    trainer = FMTrainer(batch_size, epochs, num_embed=num_embed, embed_dim=embed_dim, model=model, checkpoint=250)
    
    trainer.train(loader)