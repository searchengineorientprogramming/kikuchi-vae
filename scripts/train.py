"""Run from the project root: python train.py --up2 /path/to/patterns.up2"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from kikuchi import config
from kikuchi.data import get_dataloaders
from kikuchi.model import VAE, count_parameters


def vae_loss(recon, target, mu, logvar, beta):
    # Reduction is explicit because the original vae_loss helper was not supplied.
    # Mean squared error over batch/channel/pixels + beta * mean per-image KL.
    logvar = logvar.clamp(-10, 10)  # matches model.reparameterize
    rec = (recon - target).square().mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()
    loss = rec + beta * kl
    return loss, dict(loss=loss.detach(), recon_loss=rec.detach(), kl_loss=kl.detach())


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = dict(loss=0.0, recon_loss=0.0, kl_loss=0.0)
    count = 0
    progress = tqdm(loader, desc="Train" if training else "Evaluate", leave=False)
    with torch.set_grad_enabled(training):
        for images, _ in progress:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            if training:
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar, _ = model(images)
            else:
                # Repeatable evaluation: use the latent mean, not a random z.
                mu, logvar = model.encoder(images)
                recon = model.decode(mu)
            loss, parts = vae_loss(recon, images, mu, logvar, beta)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite VAE loss; inspect data and learning rate.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM,
                                               error_if_nonfinite=True)
                optimizer.step()
            n = images.size(0)
            count += n
            for key in totals:
                totals[key] += float(parts[key]) * n
            progress.set_postfix(loss=f"{float(parts['loss']):.5f}")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def plot_reconstructions(model, dataset, device, path):
    # Same validation images each epoch; never access test data here.
    images = torch.stack([dataset[i][0] for i in range(min(8, len(dataset)))])
    model.eval()
    recon = model.reconstruct(images.to(device)).cpu()
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4), squeeze=False)
    for i in range(n):
        for row, values in enumerate((images, recon)):
            axes[row, i].imshow((values[i, 0].numpy() + 1) / 2,
                                cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")
        axes[0, i].set_title(f"ID {int(dataset.indices[i])}", fontsize=8)
    fig.suptitle("Validation: originals (top), reconstructions (bottom)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_history(history, output):
    (output / "training_history.json").write_text(json.dumps(history, indent=2))
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("loss", "recon_loss", "kl_loss")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=140)
    plt.close(fig)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--up2", type=Path, default=config.UP2_PATH)
    parser.add_argument("--offset", default=config.UP2_OFFSET,
                        help="Explicit byte offset (default 16 for inspected file), or 'header'.")
    parser.add_argument("--split-file", type=Path, default=config.SPLIT_FILE)
    parser.add_argument("--max-patterns", type=int, default=config.MAX_PATTERNS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=config.BETA)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--normalization", choices=("per_pattern_minmax", "uint16"),
                        default=config.NORMALIZATION)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default=config.DEVICE)
    parser.add_argument("--output-root", type=Path, default=config.OUTPUT_ROOT)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lr <= 0 or args.beta < 0:
        parser.error("epochs/batch-size/lr must be positive; workers/beta must be nonnegative.")
    try:
        args.offset = None if args.offset in (None, "header") else int(args.offset)
    except ValueError:
        parser.error("--offset must be an integer or 'header'.")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        args.up2, offset_override=args.offset, split_file=args.split_file,
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
        max_patterns=args.max_patterns, normalization=args.normalization,
        pin_memory=(device.type == "cuda"))

    output = args.output_root.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (output / "reconstructions").mkdir(parents=True, exist_ok=False)
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(data_info)
    run_config.update(device_used=str(device), split_fractions=list(config.SPLIT_FRACTIONS),
                      loss="mean_pixel_MSE + beta * mean_per_image_sum_latent_KL",
                      evaluation="deterministic decode(mu), with KL regularizer")
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2))
    model = VAE(latent_dim=args.latent_dim, in_channels=config.IN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Trainable parameters: {count_parameters(model):,}\nResults: {output}")
    history, best_val = [], float("inf")
    best_path = output / "vae_best.pt"
    try:
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, device, args.beta, optimizer)
            val = run_epoch(model, val_loader, device, args.beta)
            history.append(dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()},
                                **{f"val_{k}": v for k, v in val.items()}))
            print(f"Epoch {epoch}/{args.epochs}: train={train['loss']:.6f}, val={val['loss']:.6f}")
            if val["loss"] < best_val:
                best_val = val["loss"]
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                latent_dim=args.latent_dim, in_channels=config.IN_CHANNELS,
                                beta=args.beta, val_loss=best_val, run_config=run_config), best_path)
            save_history(history, output)
            plot_reconstructions(model, val_loader.dataset, device,
                                 output / "reconstructions" / f"epoch_{epoch:03d}.png")
        # Select on validation; evaluate the held-out test set once at the end.
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test = run_epoch(model, test_loader, device, args.beta)
        test["best_epoch"] = checkpoint["epoch"]
        (output / "test_metrics.json").write_text(json.dumps(test, indent=2))
        print(f"Best epoch: {checkpoint['epoch']} | final test loss: {test['loss']:.6f}")
        print(f"Checkpoint: {best_path}")
    finally:
        for loader in (train_loader, val_loader, test_loader):
            loader.dataset.close()


if __name__ == "__main__":
    main()
