from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class ResidualConv2d(nn.Module):
    """Same-shape residual convolution used for the shortcut layers."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("ResidualConv2d expects an odd kernel size.")

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut connection: x -> conv(x) + x -> activation
        return self.act(self.conv(x) + x)


class Encoder(nn.Module):
    """
    Encoder for 120x120 single-channel Kikuchi patterns.

    For latent_dim=128, the architecture follows the 128-D column
    in the paper table:

        120x120x1
        -> 56x56x32   : Conv 9x9, stride 2
        -> 56x56x32   : residual Conv 9x9
        -> 24x24x64   : Conv 9x9, stride 2
        -> 24x24x64   : residual Conv 9x9
        -> 8x8x128    : Conv 9x9, stride 2
        -> 8x8x128    : residual Conv 9x9
        -> 1x1x256    : Conv 8x8
        -> 256-D FC   : [mu(128), logvar(128)]
    """

    def __init__(self, latent_dim: int = 128, in_channels: int = 1):
        super().__init__()

        if latent_dim < 16 or latent_dim % 16 != 0:
            raise ValueError(
                "latent_dim must be >= 16 and divisible by 16 "
                "(e.g. 16, 32, 64, 128, 256)."
            )

        c1 = latent_dim // 4   # 32 for latent_dim=128
        c2 = latent_dim // 2   # 64
        c3 = latent_dim        # 128
        c4 = latent_dim * 2    # 256

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # 120 -> 56
        self.conv1 = nn.Conv2d(
            in_channels, c1, kernel_size=9, stride=2, padding=0
        )
        self.res1 = ResidualConv2d(c1, kernel_size=9)

        # 56 -> 24
        self.conv2 = nn.Conv2d(
            c1, c2, kernel_size=9, stride=2, padding=0
        )
        self.res2 = ResidualConv2d(c2, kernel_size=9)

        # 24 -> 8
        self.conv3 = nn.Conv2d(
            c2, c3, kernel_size=9, stride=2, padding=0
        )
        self.res3 = ResidualConv2d(c3, kernel_size=9)

        # 8 -> 1
        self.conv4 = nn.Conv2d(
            c3, c4, kernel_size=8, stride=1, padding=0
        )

        # One FC layer produces both VAE statistics:
        # first latent_dim values = mu
        # second latent_dim values = logvar
        self.fc = nn.Linear(c4, 2 * latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x.ndim != 4:
            raise ValueError(
                f"Expected input [B,C,H,W], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), "
                f"got {x.shape[1]}"
            )

        if x.shape[-2:] != (120, 120):
            raise ValueError(
                f"Expected 120x120 images, got {tuple(x.shape[-2:])}"
            )

        x = self.act(self.conv1(x))  # [B, 32, 56, 56]
        x = self.res1(x)

        x = self.act(self.conv2(x))  # [B, 64, 24, 24]
        x = self.res2(x)

        x = self.act(self.conv3(x))  # [B, 128, 8, 8]
        x = self.res3(x)

        x = self.act(self.conv4(x))  # [B, 256, 1, 1]

        x = torch.flatten(x, start_dim=1)
        stats = self.fc(x)

        mu, logvar = torch.chunk(stats, chunks=2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder for 120x120 single-channel Kikuchi patterns.

    For latent_dim=128:

        z(128)
        -> projection + reshape -> 4x4x128
        -> 11x11x128 : ConvTranspose 9x9, stride 2
        -> 11x11x128 : residual Conv 9x9
        -> 27x27x64  : ConvTranspose 9x9, stride 2
        -> 27x27x64  : residual Conv 9x9
        -> 61x61x32  : ConvTranspose 9x9, stride 2
        -> 61x61x32  : residual Conv 9x9
        -> 120x120x16: ConvTranspose 9x9, stride 2
        -> 120x120x16: residual Conv 9x9
        -> 120x120x1 : Conv 1x1 + tanh
    """

    def __init__(self, latent_dim: int = 128, out_channels: int = 1):
        super().__init__()

        if latent_dim < 16 or latent_dim % 16 != 0:
            raise ValueError(
                "latent_dim must be >= 16 and divisible by 16."
            )

        c1 = latent_dim        # 128
        c2 = latent_dim // 2  # 64
        c3 = latent_dim // 4  # 32
        c4 = latent_dim // 8  # 16

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # z -> 4x4 feature map
        self.proj = nn.Linear(
            latent_dim,
            latent_dim * 4 * 4,
        )

        # 4 -> 11
        self.deconv1 = nn.ConvTranspose2d(
            c1,
            c1,
            kernel_size=9,
            stride=2,
            padding=2,
        )
        self.res1 = ResidualConv2d(c1, kernel_size=9)

        # 11 -> 27
        self.deconv2 = nn.ConvTranspose2d(
            c1,
            c2,
            kernel_size=9,
            stride=2,
            padding=1,
        )
        self.res2 = ResidualConv2d(c2, kernel_size=9)

        # 27 -> 61
        self.deconv3 = nn.ConvTranspose2d(
            c2,
            c3,
            kernel_size=9,
            stride=2,
            padding=0,
        )
        self.res3 = ResidualConv2d(c3, kernel_size=9)

        # 61 -> 120
        # PyTorch needs padding=5 and output_padding=1 to reproduce
        # the 120x120 spatial size listed in the paper table.
        self.deconv4 = nn.ConvTranspose2d(
            c3,
            c4,
            kernel_size=9,
            stride=2,
            padding=5,
            output_padding=1,
        )
        self.res4 = ResidualConv2d(c4, kernel_size=9)

        self.out_conv = nn.Conv2d(
            c4,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}"
            )

        x = self.proj(z)
        x = x.view(
            z.shape[0],
            self.latent_dim,
            4,
            4,
        )

        x = self.act(self.deconv1(x))  # [B, 128, 11, 11]
        x = self.res1(x)

        x = self.act(self.deconv2(x))  # [B, 64, 27, 27]
        x = self.res2(x)

        x = self.act(self.deconv3(x))  # [B, 32, 61, 61]
        x = self.res3(x)

        x = self.act(self.deconv4(x))  # [B, 16, 120, 120]
        x = self.res4(x)

        # Paper uses tanh at the decoder output.
        return torch.tanh(self.out_conv(x))


class VAE(nn.Module, PyTorchModelHubMixin):
    """VAE for 120x120 Kikuchi patterns."""

    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.encoder = Encoder(
            latent_dim=latent_dim,
            in_channels=in_channels,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=in_channels,
        )

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        # Helps avoid numerical overflow in exp(logvar).
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(
        self,
        x: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor:
        """
        Return the latent representation.

        sample=False -> deterministic representation mu
        sample=True  -> sampled latent z
        """
        mu, logvar = self.encoder(x)

        if sample:
            return self.reparameterize(mu, logvar)

        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic reconstruction using mu.
        z = self.encode(x, sample=False)
        return self.decode(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar, z


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":
    model = VAE(
        latent_dim=128,
        in_channels=1,
    )

    x = torch.randn(
        2,
        1,
        120,
        120,
    )

    with torch.no_grad():
        recon, mu, logvar, z = model(x)

    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)

    print(f"input:          {tuple(x.shape)}")
    print(f"reconstruction: {tuple(recon.shape)}")
    print(f"mu:             {tuple(mu.shape)}")
    print(f"logvar:         {tuple(logvar.shape)}")
    print(f"z:              {tuple(z.shape)}")
    print(f"encoder params: {encoder_params:,}")
    print(f"decoder params: {decoder_params:,}")
    print(f"total params:   {count_parameters(model):,}")
