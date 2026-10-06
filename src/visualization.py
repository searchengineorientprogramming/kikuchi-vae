import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from utils import save_json


def save_history(history, output, plot=True, dpi=140):
    output = Path(output)
    save_json(history, output / "training_history.json")
    keys = list(dict.fromkeys(key for row in history for key in row))
    temp = output / "metrics.csv.tmp"
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    temp.replace(output / "metrics.csv")
    if not plot: return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("loss", "recon_loss", "kl_loss")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history],
                    label=split, marker=".")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=dpi)
    plt.close(fig)


@torch.inference_mode()
def plot_reconstructions(model, dataset, device, output_path, num_images=8, indices=None, dpi=140):
    ids = list(range(min(num_images, len(dataset)))) if indices is None else list(indices)
    if not ids: return ids
    images = torch.stack([dataset[i][0] for i in ids])
    was_training = model.training
    model.eval()
    recon = model.reconstruct(images.to(device)).cpu()
    model.train(was_training)
    n = len(ids)
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4), squeeze=False)
    for i in range(n):
        for row, values in enumerate((images, recon)):
            axes[row, i].imshow((values[i, 0].numpy()+1)/2, cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")
        axes[0, i].set_title(f"ID {int(dataset.indices[ids[i]])}", fontsize=8)
    fig.suptitle("Validation: input (top), decode(mu) (bottom)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return ids
