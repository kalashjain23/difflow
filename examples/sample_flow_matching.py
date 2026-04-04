import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from difflow.models import FlowMatching
from torch import nn


# generating the original data distribution
def multi_spiral_dataset(n_spirals=2, n=1000, noise=0.05):
    all_data, all_labels = [], []
    for i in range(n_spirals):
        theta = torch.linspace(0, 4 * torch.pi, n // n_spirals)
        r = theta / (4 * torch.pi)
        offset = (2 * torch.pi * i) / n_spirals
        x = r * torch.cos(theta + offset) + torch.randn(n // n_spirals) * noise
        y = r * torch.sin(theta + offset) + torch.randn(n // n_spirals) * noise
        data = torch.stack([x, y], dim=1)
        labels = torch.full((n // n_spirals,), i, dtype=torch.long)
        all_data.append(data)
        all_labels.append(labels)
    return torch.cat(all_data).numpy(), torch.cat(all_labels).numpy()

# sampling for each label (spiral1 and spiral2)
def sample(checkpoint_path, n_spirals=2, embed_dim=8, n_samples=1000, steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowMatching(embed_dim=embed_dim, device=str(device)).to(device)
    embedding = nn.Embedding(n_spirals, embed_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['ema'])
    embedding.load_state_dict(checkpoint['embedding'])
    model.eval()
    embedding.eval()

    samples = {}
    with torch.no_grad():
        for spiral_id in range(n_spirals):
            label = torch.full((n_samples // n_spirals,), spiral_id, dtype=torch.long, device=device)
            c_embed = embedding(label)
            samples[spiral_id] = model.sample((n_samples // n_spirals, 2), steps=steps, c=c_embed).cpu().numpy()

    losses = checkpoint['losses']
    return samples, losses

# visualizing the original data distribution, the generated dsitribution from the model and the loss curve
def visualize(original, original_labels, samples, losses, n_spirals=2):
    colors = ['orange', 'green', 'red']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for spiral_id in range(n_spirals):
        mask = original_labels == spiral_id
        axes[0].scatter(original[mask, 0], original[mask, 1], s=2, alpha=0.5, color=colors[spiral_id])
        axes[1].scatter(samples[spiral_id][:, 0], samples[spiral_id][:, 1], s=2, alpha=0.5, color=colors[spiral_id], label=f'spiral {spiral_id}')

    axes[0].set_title("Original Distribution")
    axes[0].set_aspect('equal')
    axes[1].set_title("Generated Distribution")
    axes[1].set_aspect('equal')
    axes[1].legend()

    axes[2].plot(losses)
    axes[2].set_title("Training Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("flow_matching_viz.png", dpi=150)
    plt.show()


if __name__ == '__main__':
    original, original_labels = multi_spiral_dataset()
    samples, losses = sample("fm_checkpoint_500.pth")
    visualize(original, original_labels, samples, losses)
