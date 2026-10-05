"""Defaults for the 120x120 UP2 VAE; CLI arguments override these values."""
from pathlib import Path

SEED = 42
LATENT_DIM = 64
IN_CHANNELS = 1

UP2_PATH = Path("data/718RX_1um_120x120.up2")
# Explicit override for the file inspected in this conversation ONLY.
# Set None to use the header's offset for a different, standard UP2 file.
UP2_OFFSET = 16
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
SPLIT_FILE = None  # None -> <UP2 filename>.vae_splits.npz beside the source
NORMALIZATION = "per_pattern_minmax"  # or "uint16"; both output [-1, 1]
BATCH_SIZE = 16  # lower this if model activations exhaust RAM
NUM_WORKERS = 0  # safe default for Mac / notebooks
MAX_PATTERNS = None  # e.g. 1000 for a small trial; random subset before splitting

LEARNING_RATE = 1e-4
EPOCHS = 10
BETA = 1e-3
GRAD_CLIP_NORM = 1.0
DEVICE = "auto"  # CUDA, then Apple MPS, then CPU
OUTPUT_ROOT = Path("runs")  # a new timestamped directory for each invocation
