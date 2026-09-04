"""Visualize reconstructions and random samples from a trained VAE.

Run after training:

    python visualize.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from data import get_dataloaders
from model import VAE
from utils import ensure_directories, get_device, set_seed


def to_display_range(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1,1] tensors to [0,1] for plotting."""
    return ((x + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def save_reconstructions(model, loader, device, n=8):
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n].to(device)
    labels = labels[:n]

    recon, _, _, _ = model(images)

    images = to_display_range(images.cpu())
    recon = to_display_range(recon.cpu())

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        axes[0, i].imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Input {labels[i].item()}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")

    fig.tight_layout()
    path = config.RESULTS_DIR / "reconstructions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


@torch.no_grad()
def save_generated_samples(model, device, n=16):
    model.eval()

    z = torch.randn(n, model.latent_dim, device=device)
    generated = to_display_range(model.decode(z).cpu())

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(8, 8))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(generated[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.tight_layout()
    path = config.RESULTS_DIR / "generated_samples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    set_seed(config.SEED)
    ensure_directories(config.RESULTS_DIR)

    device = get_device()
    _, test_loader = get_dataloaders()

    checkpoint = torch.load(config.BEST_CHECKPOINT, map_location=device)

    model = VAE(
        latent_dim=checkpoint["latent_dim"],
        in_channels=config.IN_CHANNELS,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    save_reconstructions(model, test_loader, device)
    save_generated_samples(model, device)

if __name__ == "__main__":
    main()
