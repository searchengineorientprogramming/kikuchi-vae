"""MNIST dataset and DataLoader construction."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import config


def get_transform():
    """Map MNIST pixels from [0, 1] to [-1, 1].

    The VAE decoder ends with tanh, so its reconstruction is also in [-1, 1].
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def get_dataloaders():
    transform = get_transform()

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
