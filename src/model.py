from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class ResidualConv2d(nn.Module):
    """Same-shape residual convolution."""

    def __init__(self, channels: int, kernel_size: int = 3, negative_slope: float = 0.2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ResidualConv2d expects an odd kernel size.")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + x)


class Encoder(nn.Module):
    """Encode [B,1,28,28] into mu and logvar."""

    def __init__(self, latent_dim: int = 16, in_channels: int = 1):
        super().__init__()
        if latent_dim < 8 or latent_dim % 8 != 0:
            raise ValueError("latent_dim must be >= 8 and divisible by 8 (8,16,32,64,...).")

        c1 = latent_dim // 4
        c2 = latent_dim // 2
        c3 = latent_dim
        c4 = latent_dim * 2

        self.latent_dim = latent_dim
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # 28 -> 14
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2)
        self.res1 = ResidualConv2d(c1, kernel_size=5)

        # 14 -> 7
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=5, stride=2, padding=2)
        self.res2 = ResidualConv2d(c2, kernel_size=5)

        # 7 -> 4
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.res3 = ResidualConv2d(c3, kernel_size=3)

        # 4 -> 1
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=0)

        self.fc = nn.Linear(c4, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[-2:] != (28, 28):
            raise ValueError(f"Expected 28x28 images, got {tuple(x.shape[-2:])}")

        x = self.act(self.conv1(x))  # [B,c1,14,14]
        x = self.res1(x)
        x = self.act(self.conv2(x))  # [B,c2,7,7]
        x = self.res2(x)
        x = self.act(self.conv3(x))  # [B,c3,4,4]
        x = self.res3(x)
        x = self.act(self.conv4(x))  # [B,c4,1,1]

        x = torch.flatten(x, start_dim=1)
        stats = self.fc(x)
        mu, logvar = torch.chunk(stats, chunks=2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """Decode z back to [B,1,28,28]."""

    def __init__(self, latent_dim: int = 16, out_channels: int = 1):
        super().__init__()
        if latent_dim < 8 or latent_dim % 8 != 0:
            raise ValueError("latent_dim must be >= 8 and divisible by 8.")

        c1 = latent_dim
        c2 = latent_dim // 2
        c3 = latent_dim // 4
        c4 = latent_dim // 8

        self.latent_dim = latent_dim
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Same projection idea as original model.
        self.proj = nn.Linear(latent_dim, latent_dim * 4 * 4)

        # 4 -> 7
        self.deconv1 = nn.ConvTranspose2d(c1, c1, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualConv2d(c1, kernel_size=3)

        # 7 -> 14
        self.deconv2 = nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualConv2d(c2, kernel_size=3)

        # 14 -> 28
        self.deconv3 = nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.res3 = ResidualConv2d(c3, kernel_size=3)

        # Keep a fourth decoder stage; spatial size stays 28x28.
        self.deconv4 = nn.ConvTranspose2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.res4 = ResidualConv2d(c4, kernel_size=3)

        self.out_conv = nn.Conv2d(c4, out_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}")

        x = self.proj(z)
        x = x.view(z.shape[0], self.latent_dim, 4, 4)

        x = self.act(self.deconv1(x))  # [B,z,7,7]
        x = self.res1(x)
        x = self.act(self.deconv2(x))  # [B,z/2,14,14]
        x = self.res2(x)
        x = self.act(self.deconv3(x))  # [B,z/4,28,28]
        x = self.res3(x)
        x = self.act(self.deconv4(x))  # [B,z/8,28,28]
        x = self.res4(x)

        return torch.tanh(self.out_conv(x))


class VAE(nn.Module, PyTorchModelHubMixin):
    """Variational autoencoder for MNIST."""

    def __init__(self, latent_dim: int = 16, in_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.encoder = Encoder(latent_dim, in_channels)
        self.decoder = Decoder(latent_dim, in_channels)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, sample: bool=False) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        if sample:
            return self.reparameterize(mu, logvar)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for latent_dim in (8, 16, 32, 64):
        model = VAE(latent_dim=latent_dim)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            recon, mu, logvar, z = model(x)

        print(
            f"latent={latent_dim:3d} | "
            f"input={tuple(x.shape)} | "
            f"recon={tuple(recon.shape)} | "
            f"mu={tuple(mu.shape)} | "
            f"z={tuple(z.shape)} | "
            f"params={count_parameters(model):,}"
        )
