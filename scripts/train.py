import torch
from tqdm.auto import tqdm

from kikuchi import config
from kikuchi import get_dataloaders
from kikuchi import VAE, count_parameters
from kikuchi import vae_loss
from kikuchi import ensure_directories, get_device, save_history, set_seed

import csv
from pathlib import Path
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, device, beta):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, _ in progress:
        images = images.to(device)

        optimizer.zero_grad(set_to_none=True)

        recon, mu, logvar, _ = model(images)

        loss, parts = vae_loss(
            recon=recon,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.GRAD_CLIP_NORM,
        )

        optimizer.step()

        batch_size = images.size(0)
        total_loss += parts["loss"].item() * batch_size
        total_recon += parts["recon_loss"].item() * batch_size
        total_kl += parts["kl_loss"].item() * batch_size

        progress.set_postfix(loss=f"{parts['loss'].item():.4f}")

    n = len(loader.dataset)

    return {
        "loss": total_loss / n,
        "recon_loss": total_recon / n,
        "kl_loss": total_kl / n,
    }


@torch.no_grad()
def evaluate(model, loader, device, beta):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for images, _ in loader:
        images = images.to(device)

        recon, mu, logvar, _ = model(images)

        _, parts = vae_loss(
            recon=recon,
            target=images,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        batch_size = images.size(0)
        total_loss += parts["loss"].item() * batch_size
        total_recon += parts["recon_loss"].item() * batch_size
        total_kl += parts["kl_loss"].item() * batch_size

    n = len(loader.dataset)

    return {
        "loss": total_loss / n,
        "recon_loss": total_recon / n,
        "kl_loss": total_kl / n,
    }

def save_metrics_csv(history, output_path):
    output_path = Path(output_path)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_recon_loss",
            "train_kl_loss",
            "val_loss",
            "val_recon_loss",
            "val_kl_loss",
        ])
    
        num_epochs = len(history["train_loss"])

        for i in range(num_epochs):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_recon_loss"][i],
                history["train_kl_loss"][i],
                history["test_loss"][i],
                history["test_recon_loss"][i],
                history["test_kl_loss"][i],
            ])

def plot_training_history(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train loss",
    )

    plt.plot(
        epochs,
        history["test_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_metric(
    history,
    train_key,
    val_key,
    ylabel,
    output_path,
):
    epochs = range(1, len(history[train_key]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history[train_key],
        label="Train",
    )

    plt.plot(
        epochs,
        history[val_key],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    set_seed(config.SEED)
    ensure_directories(config.DATA_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR)

    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = VAE(
        latent_dim=config.LATENT_DIM,
        in_channels=config.IN_CHANNELS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    print(f"Trainable parameters: {count_parameters(model):,}")
    print(
        f"latent_dim={config.LATENT_DIM} | "
        f"lr={config.LEARNING_RATE} | "
        f"beta={config.BETA} | "
        f"batch_size={config.BATCH_SIZE}"
    )

    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "test_loss": [],
        "test_recon_loss": [],
        "test_kl_loss": [],
    }

    best_test_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=config.BETA,
        )

        test_metrics = evaluate(
            model,
            test_loader,
            device,
            beta=config.BETA,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_recon_loss"].append(test_metrics["recon_loss"])
        history["test_kl_loss"].append(test_metrics["kl_loss"])

        weighted_kl = config.BETA * test_metrics["kl_loss"]

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={test_metrics['loss']:.4f} | "
            f"train_recon={train_metrics['recon_loss']:.4f} | "
            f"val_recon={test_metrics['recon_loss']:.4f} | "
            f"val_KL={test_metrics['kl_loss']:.4f}"
        )

        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_dim": config.LATENT_DIM,
                    "beta": config.BETA,
                    "test_loss": best_test_loss,
                },
                config.BEST_CHECKPOINT,
            )

        save_history(history, config.HISTORY_FILE)
        save_metrics_csv(
            history,
            config.RESULTS_DIR / "metrics.csv",
        )
        plot_training_history(
            history,
            config.RESULTS_DIR / "training_curves.png",
        )
        plot_metric(
            history,
            "train_loss",
            "test_loss",
            "Total loss",
            config.RESULTS_DIR / "loss_curve.png",
        )

        plot_metric(
            history,
            "train_recon_loss",
            "test_recon_loss",
            "Reconstruction loss",
            config.RESULTS_DIR / "reconstruction_curve.png",
        )

        plot_metric(
            history,
            "train_kl_loss",
            "test_kl_loss",
            "KL divergence",
            config.RESULTS_DIR / "kl_curve.png",
)

    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Checkpoint: {config.BEST_CHECKPOINT}")
    print(f"History: {config.HISTORY_FILE}")


if __name__ == "__main__":
    main()
