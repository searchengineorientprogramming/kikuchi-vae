"""Small shape and finite-value smoke test for the VAE."""

import sys
from pathlib import Path

import torch

# Allow `python tests/test_model.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import VAE
from losses import vae_loss


def test_forward_and_backward():
    model = VAE(latent_dim=16, in_channels=1)
    x = torch.rand(4, 1, 28, 28) * 2.0 - 1.0

    recon, mu, logvar, z = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)
    assert z.shape == (4, 16)

    loss, _ = vae_loss(recon, x, mu, logvar, beta=1e-3)
    assert torch.isfinite(loss)

    loss.backward()


if __name__ == "__main__":
    test_forward_and_backward()
    print("Model smoke test passed.")
