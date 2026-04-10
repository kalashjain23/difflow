import torch
from difflow.models.flow_matching import FlowMatching


def test_forward_shape():
    model = FlowMatching()
    x = torch.randn(4, 2)
    t = torch.rand(4, 1)
    out = model(x, t)
    assert out.shape == (4, 2)


def test_forward_with_conditioning():
    model = FlowMatching(embed_dim=8)
    x = torch.randn(4, 2)
    t = torch.rand(4, 1)
    c = torch.randn(4, 8)
    out = model(x, t, c)
    assert out.shape == (4, 2)


def test_sample_shape():
    model = FlowMatching()
    with torch.no_grad():
        out = model.sample((4, 2), steps=5)
    assert out.shape == (4, 2)


def test_sample_with_conditioning():
    model = FlowMatching(embed_dim=8)
    c = torch.randn(4, 8)
    with torch.no_grad():
        out = model.sample((4, 2), steps=5, c=c)
    assert out.shape == (4, 2)


def test_backward():
    model = FlowMatching()
    x = torch.randn(4, 2)
    t = torch.rand(4, 1)
    out = model(x, t)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
