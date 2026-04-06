"""
Smoke test for Pi0 model training.
Uses dummy PIL images processed through PaliGemma's processor.
"""
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from difflow.models import Pi0
from difflow.training import Pi0Trainer


def dummy_dataset(processor, n_samples=16, state_dim=7, action_dim=7, chunk_size=10):
    prompt = "<image> move arm forward"
    all_input_ids = []
    all_pixel_values = []

    for _ in range(n_samples):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        all_input_ids.append(inputs["input_ids"].squeeze(0))
        all_pixel_values.append(inputs["pixel_values"].squeeze(0))

    input_ids = torch.stack(all_input_ids)
    pixel_values = torch.stack(all_pixel_values)
    states = torch.randn(n_samples, state_dim)
    actions = torch.randn(n_samples, chunk_size, action_dim)

    return TensorDataset(pixel_values, input_ids, states, actions)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunk_size = 10
    action_dim = 7
    state_dim = 7
    embed_dim = 256
    heads = 4
    n_layers = 2
    batch_size = 4
    epochs = 2

    processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

    model = Pi0(
        chunk_size=chunk_size,
        action_dim=action_dim,
        embed_dim=embed_dim,
        state_dim=state_dim,
        heads=heads,
        n_layers=n_layers,
        device=str(device),
    ).to(device)

    dataset = dummy_dataset(
        processor,
        n_samples=16,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainer = Pi0Trainer(
        batch_size=batch_size,
        epochs=epochs,
        model=model,
        checkpoint=1,
    )

    model.train()
    trainer.train(loader)

    # quick sampling test
    # model.eval()
    # with torch.no_grad():
    #     img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    #     inputs = processor(text="<image> move arm forward", images=img, return_tensors="pt")
    #     sampled_actions = model.sample(
    #         action_size=(1, chunk_size, action_dim),
    #         steps=10,
    #         images=inputs["pixel_values"].to(device),
    #         prompt=inputs["input_ids"].to(device),
    #         states=torch.randn(1, state_dim).to(device),
    #     )
    # print(f"sampled actions shape: {sampled_actions.shape}")
    print("done")
