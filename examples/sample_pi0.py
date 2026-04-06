from difflow.models import Pi0
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("pi0_checkpoint_50.pth", map_location=device)

chunk_size = 10
action_dim = 7
state_dim = 7
embed_dim = 256
heads = 4
n_layers = 2

model = Pi0(chunk_size, action_dim, embed_dim, state_dim, heads, n_layers, device=str(device)).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(text="<image> move arm forward", images=img, return_tensors="pt")

with torch.no_grad():
    actions = model.sample(
        action_size=(1, chunk_size, action_dim),
        steps=10,
        images=inputs["pixel_values"].to(device),
        prompt=inputs["input_ids"].to(device),
        states=torch.randn(1, state_dim).to(device),
    )

print(f"sampled actions shape: {actions.shape}")
print(actions)
