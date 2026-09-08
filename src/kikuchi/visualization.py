from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def plot_reconstructions(
    model,
    dataset,
    device,
    output_path,
    num_images=8,
):
    """
    Randomly select images from a dataset and compare
    originals with VAE reconstructions.

    Top row: original images
    Bottom row: reconstructed images
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Randomly select samples from the whole dataset
    indices = random.sample(
        range(len(dataset)),
        k=min(num_images, len(dataset)),
    )

    images = []

    for idx in indices:
        sample = dataset[idx]

        # Handles datasets returning (image, label)
        if isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample

        images.append(image)

    images = torch.stack(images).to(device)

    # Remember whether model was training
    was_training = model.training

    model.eval()

    # Deterministic reconstruction using your encode() -> mu
    reconstructions = model.reconstruct(images)

    if was_training:
        model.train()

    # Move to CPU for plotting
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    n = len(images)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(2 * n, 4),
    )

    for i in range(n):

        original = images[i].squeeze()
        reconstructed = reconstructions[i].squeeze()

        # If your model uses [-1, 1] normalization,
        # convert back to [0, 1] for visualization.
        original = (original + 1) / 2
        reconstructed = (reconstructed + 1) / 2

        original = original.clamp(0, 1)
        reconstructed = reconstructed.clamp(0, 1)

        axes[0, i].imshow(
            original,
            cmap="gray",
        )

        axes[1, i].imshow(
            reconstructed,
            cmap="gray",
        )

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=12,
    )

    axes[1, 0].set_ylabel(
        "Reconstructed",
        fontsize=12,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)