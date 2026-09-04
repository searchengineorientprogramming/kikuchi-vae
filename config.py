"""Central configuration for the MNIST VAE experiment.

Change experiment hyperparameters here instead of editing train.py.
"""

from pathlib import Path

# Reproducibility
SEED = 42

# Model
LATENT_DIM = 16
IN_CHANNELS = 1

# Data
BATCH_SIZE = 64
NUM_WORKERS = 0
DATA_DIR = Path("data")

# Optimization
LEARNING_RATE = 1e-4
EPOCHS = 10
BETA = 1e-3
GRAD_CLIP_NORM = 1.0

# Output
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
BEST_CHECKPOINT = CHECKPOINT_DIR / "vae_mnist_best.pt"
HISTORY_FILE = RESULTS_DIR / "training_history.json"
