import torch
import torch.nn.functional as F


def vae_loss(recon, target, mu, logvar, beta, kl_reduction="sum"):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    safe_logvar = logvar.clamp(-10.0, 10.0)
    kl_terms = -0.5 * (1.0 + safe_logvar - mu.square() - safe_logvar.exp())
    if kl_reduction == "sum":
        kl_loss = kl_terms.sum(dim=1).mean()
    elif kl_reduction == "mean":
        kl_loss = kl_terms.mean()
    else:
        raise ValueError("KL_REDUCTION must be 'sum' or 'mean'.")
    loss = recon_loss + beta * kl_loss
    return loss, dict(loss=loss.detach(), recon_loss=recon_loss.detach(), kl_loss=kl_loss.detach())
