from difflow.models.ddpm import DDPM
from torchvision.utils import save_image
import torch
from difflow.nn.ema import EMA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("ddpm_checkpoint_100.pth", map_location=device)

model = DDPM(
    beta_start=1e-4,
    beta_end=0.02,
)
model.load_state_dict(checkpoint['model'])
model.eval()

ema = EMA(model)
ema.ema_weights = checkpoint['ema']
ema.apply()

n_samples = 4

with torch.no_grad():
    samples = model.sample((n_samples, 3, 32, 32))
    
ema.restore()
    
save_image(samples.cpu(), "samples.png", nrow=4, normalize=True)
