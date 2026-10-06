# Kikuchi VAE

A PyTorch variational autoencoder for reconstructing EBSD Kikuchi patterns and learning compact latent representations. Patterns are read directly from a UP2 file on demand, without loading the entire dataset into RAM or exporting separate images.

## Setup

Run commands from the project root:

```bash
python -m pip install -r requirements.txt
```

## Configure and train

Edit `src/config.py` to set the input file and experiment settings. For example:

```python
UP2_PATH = Path("data/718RX_1um_120x120.up2")
UP2_OFFSET = 16  
MAX_PATTERNS = 80000
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
BETA = 1e-5  # Example value to tune for reconstruction quality.
KL_REDUCTION = "sum"
RESUME = None
RUN_TEST_AT_END = False  # Enable for final evaluation after tuning.
```

Start training:

```bash
python src/train.py
```

The script reports the pattern count and image dimensions, creates or reuses saved train/validation/test indices, and trains the model. Validation runs after each epoch. The current loader supports grayscale 120 × 120 patterns and normalizes inputs to `[-1, 1]`.

Splits are saved beside the UP2 file and reused for matching settings. Training order is shuffled each epoch, while split membership stays fixed. Random pattern splits do not guarantee separation between grains or scans.

## Main files

| File | Purpose |
| --- | --- |
| `src/config.py` | Data paths, hyperparameters, device, and output settings |
| `src/data.py` | UP2 inspection, splitting, normalization, and batch loading |
| `src/model.py` | Encoder, latent sampling, and decoder |
| `src/losses.py` | Reconstruction MSE plus beta-weighted KL loss |
| `src/train.py` | Training, validation, checkpoints, and optional testing |
| `src/visualization.py` | Loss curves and reconstruction comparisons |
| `src/utils.py` | Device selection, seeds, and saving helpers |

## Results

Each run saves its results in `runs/<run_name_timestamp>/`:

- `run_config.json`: effective settings and dataset details.
- `metrics.csv` and `training_history.json`: epoch losses and timing.
- `training_curves.png`: training and validation loss curves.
- `reconstructions/`: fixed validation inputs and their reconstructions.
- `vae_best.pt`: checkpoint with the lowest validation total loss.
- `vae_last.pt`: latest completed epoch.
- `test_metrics.json`: best-model test results when final testing is enabled.

Training uses sampled latent vectors; validation and reconstruction plots use the latent mean. Inspect reconstruction images alongside losses when comparing experiments.

## Resume training

```bash
python src/train.py --resume runs/<previous_run>/vae_last.pt --epochs 20
```

`--epochs` specifies the total target epoch count. Keep the original data and learning settings, and retain `vae_best.pt` beside `vae_last.pt`. Use a fresh run when changing hyperparameters.
