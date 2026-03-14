from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from difflow.training.ddpm import DDPMTrainer
from difflow.models.ddpm import DDPM

def get_cifar10(batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    batch_size = 128
    epochs = 200
    steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_cifar10(batch_size)
    model = DDPM(beta_start, beta_end, steps, device=device).to(device)
    model.train()
    trainer = DDPMTrainer(batch_size, epochs, model=model)
    
    trainer.train(dataset)
