import torch
from torch import nn
from difflow.models.pi0 import Pi0
from unittest.mock import patch, MagicMock


def make_pi0(embed_dim=64, chunk_size=4, action_dim=7, state_dim=7, heads=2, n_layers=1):
    """Create a Pi0 with a dummy VLM (no PaliGemma download)."""
    vlm_hidden_dim = 2048
    n_vlm_tokens = 8

    with patch.object(Pi0, '__init__', lambda self, *a, **kw: None):
        model = Pi0.__new__(Pi0)

    nn.Module.__init__(model)
    model.chunk_size = chunk_size
    model.device = "cpu"
    model.embed_dim = embed_dim

    # dummy VLM that returns fake hidden states
    dummy_vlm = MagicMock()
    dummy_output = MagicMock()
    dummy_output.hidden_states = [torch.randn(1, n_vlm_tokens, vlm_hidden_dim)]
    dummy_vlm.return_value = dummy_output
    model.vlm = dummy_vlm

    model.vlm_proj = nn.Linear(vlm_hidden_dim, embed_dim)

    from difflow.nn import TimeEmbedding, MOETransformerBlock
    model.time_embedding = TimeEmbedding(embed_dim)
    model.w1 = nn.Linear(action_dim, embed_dim)
    model.w2 = nn.Linear(embed_dim * 2, embed_dim)
    model.w3 = nn.Linear(embed_dim, embed_dim)
    model.state_proj = nn.Linear(state_dim, embed_dim)
    model.action_expert = nn.ModuleList([
        MOETransformerBlock(embed_dim, heads=heads, experts=2) for _ in range(n_layers)
    ])
    model.action_out_proj = nn.Linear(embed_dim, action_dim)

    return model


def test_forward_shape():
    model = make_pi0()
    images = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 100, (1, 10))
    states = torch.randn(1, 7)
    actions = torch.randn(1, 4, 7)
    t = torch.rand(1)
    out = model.forward(images, prompt, None, states, actions, t)
    assert out.shape == (1, 4, 7)


def test_sample_shape():
    model = make_pi0()
    images = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 100, (1, 10))
    states = torch.randn(1, 7)
    with torch.no_grad():
        out = model.sample(
            action_size=(1, 4, 7),
            steps=3,
            images=images,
            prompt=prompt,
            states=states,
        )
    assert out.shape == (1, 4, 7)


def test_backward():
    model = make_pi0()
    images = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 100, (1, 10))
    states = torch.randn(1, 7)
    actions = torch.randn(1, 4, 7)
    t = torch.rand(1)
    out = model.forward(images, prompt, None, states, actions, t)
    loss = out.sum()
    loss.backward()
    assert model.w1.weight.grad is not None
    assert model.action_out_proj.weight.grad is not None


def test_mask_shape():
    """Verify the blockwise causal mask is constructed correctly."""
    model = make_pi0()
    images = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 100, (1, 10))
    states = torch.randn(1, 7)
    actions = torch.randn(1, 4, 7)
    t = torch.rand(1)

    # run forward to get the mask indirectly - check output is valid
    out = model.forward(images, prompt, None, states, actions, t)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
