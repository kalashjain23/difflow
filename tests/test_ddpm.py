import torch
from difflow.models.ddpm import DDPM


def test_forward_shape():
    model = DDPM(beta_start=1e-4, beta_end=0.02, steps=10, noise_model_channels=32, channel_mult=(1, 2), attention_channels=(), time_emb_dim=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    noisy, noise = model(x, t)
    assert noisy.shape == (2, 3, 32, 32)
    assert noise.shape == (2, 3, 32, 32)


def test_reverse_shape():
    model = DDPM(beta_start=1e-4, beta_end=0.02, steps=10, noise_model_channels=32, channel_mult=(1, 2), attention_channels=(), time_emb_dim=32)
    x = torch.randn(2, 3, 32, 32)
    out = model.reverse(x, t=5)
    assert out.shape == (2, 3, 32, 32)


def test_sample_shape():
    model = DDPM(beta_start=1e-4, beta_end=0.02, steps=3, noise_model_channels=32, channel_mult=(1, 2), attention_channels=(), time_emb_dim=32)
    with torch.no_grad():
        out = model.sample((1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_backward():
    model = DDPM(beta_start=1e-4, beta_end=0.02, steps=10, noise_model_channels=32, channel_mult=(1, 2), attention_channels=(), time_emb_dim=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 10, (2,))
    noisy, noise = model(x, t)
    pred = model.noise_model(noisy, t)
    loss = torch.nn.functional.mse_loss(pred, noise)
    loss.backward()
    assert any(p.grad is not None for p in model.noise_model.parameters())
