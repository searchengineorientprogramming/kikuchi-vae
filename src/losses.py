from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
):

    recon_loss = F.mse_loss(recon, target, reduction="mean")

    safe_logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl_loss = -0.5 * torch.mean(
        1.0 + safe_logvar - mu.pow(2) - safe_logvar.exp()
    )

    total = recon_loss + beta * kl_loss

    return total, {
        "loss": total.detach(),
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
    }
