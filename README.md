# MNIST Variational Autoencoder

A small PyTorch VAE project for learning and debugging variational autoencoders on MNIST.

The architecture is adapted from a larger convolutional VAE originally designed for 120×120 Kikuchi patterns. The overall encoder/decoder structure is preserved, while the spatial convolution parameters are adjusted for native 28×28 MNIST images.

## Repository structure

```text
mnist-vae/
├── config.py              # Hyperparameters and paths
├── model.py
├── losses.py               # Encoder, decoder, VAE, and VAE loss
├── data.py                # MNIST preprocessing and DataLoaders
├── train.py               # Training + validation + checkpoint saving
├── visualize.py           # Reconstructions and random VAE samples
├── utils.py               # Seed, device, directory, history helpers
├── requirements.txt       # Python dependencies
├── tests/
│   └── test_model.py      # Forward/backward smoke test
├── data/                  # MNIST download location (gitignored)
├── checkpoints/           # Saved model weights (gitignored)
└── results/               # Training history and plots (gitignored)
```

## Current baseline

The working baseline found during debugging is:

```python
LATENT_DIM = 16
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
BETA = 1e-3
EPOCHS = 10
GRAD_CLIP_NORM = 1.0
```

MNIST is kept at its native 28×28 size and normalized from `[0, 1]` to `[-1, 1]` because the decoder ends in `tanh`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## 1. Test the model

Before training, verify the tensor shapes and backward pass:

```bash
python tests/test_model.py
```

Expected output:

```text
Model smoke test passed.
```

## 2. Train

```bash
python train.py
```

The script downloads MNIST automatically, trains the VAE, and saves the best checkpoint to:

```text
checkpoints/vae_mnist_best.pt
```

Training metrics are saved to:

```text
results/training_history.json
```

The training log prints both the raw KL loss and its actual weighted contribution `beta * KL`. This makes it easier to diagnose the balance between reconstruction and latent regularization.

## 3. Visualize results

After training:

```bash
python visualize.py
```

This creates:

```text
results/reconstructions.png
results/generated_samples.png
```

## Why beta is small

The objective is

```text
loss = reconstruction_loss + beta * KL_loss
```

With mean-reduced pixel MSE, the reconstruction loss is much smaller in scale than the raw KL loss. `BETA = 1e-3` was therefore used as the current stable baseline rather than assuming `beta = 1` is universally appropriate.

The useful quantity to monitor is not only the raw KL loss, but:

```text
beta * KL_loss
```

relative to the reconstruction loss.

## Debugging result

The model was debugged in two stages:

1. Temporarily remove stochastic sampling and KL regularization by using `z = mu`. The deterministic autoencoder successfully reduced reconstruction MSE from roughly 0.52 to 0.05, confirming that the encoder and decoder could learn MNIST.
2. Restore VAE sampling and KL loss with `beta = 1e-3`. Reconstruction continued to improve while KL remained nonzero and stable, confirming that the variational latent representation was working.

This baseline can now be used before experimenting with latent dimension, beta, or transferring the workflow back to Kikuchi patterns.


### Code organization

- `model.py`: Encoder, Decoder, VAE architecture, and model utilities.
- `losses.py`: VAE objective, including reconstruction and KL-divergence terms.
- `train.py`: Optimization and evaluation loops.
- `config.py`: Experiment hyperparameters.
