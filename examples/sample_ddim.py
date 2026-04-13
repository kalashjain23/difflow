from difflow.models import DDIM
from torchvision.utils import save_image
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("ddpm_checkpoint_300.pth", map_location=device) # using the existing ddpm checkpoint

model = DDIM(
    beta_start=1e-4,
    beta_end=0.02,
    ddim_steps=200
)
model.load_state_dict(checkpoint['ema'])
model.eval()

n_samples = 4

with torch.no_grad():
    samples = model.sample((n_samples, 3, 32, 32))
    
save_image(samples.cpu(), "samples.png", nrow=4, normalize=True)
