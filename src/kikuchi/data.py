"""Read one UP2 image at a time; persist disjoint, reproducible index splits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from . import config


def inspect_up2(path, offset_override=None):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    with path.open("rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("UP2 file is shorter than its initial header.")
        version, width, height, declared_offset = struct.unpack("<4I", header)
        # Small fingerprint to detect common source changes without scanning 26 GB.
        f.seek(0)
        first = f.read(65536)
        f.seek(max(0, stat.st_size - 65536))
        last = f.read(65536)
    if version not in (1, 3, 4):
        raise ValueError(f"Unsupported UP2 version {version}; inspect the format first.")
    if (height, width) != (120, 120):
        raise ValueError(f"Model requires 120x120; header says {height}x{width}.")
    offset = declared_offset if offset_override is None else int(offset_override)
    if not 16 <= offset < stat.st_size:
        raise ValueError(f"Invalid data offset: {offset}")
    count, remainder = divmod(stat.st_size - offset, height * width * 2)
    if remainder or count < 3:
        raise ValueError(
            f"Offset {offset} gives {count} records and {remainder} leftover bytes. "
            "Do not discard leftovers automatically; verify the file layout."
        )
    if offset != declared_offset:
        warnings.warn(
            f"Using explicit offset {offset}; header reports {declared_offset}. "
            "This override was visually checked for the supplied 718RX file; "
            "it is not a general version-4 rule.", stacklevel=2
        )
    meta = dict(source_path=str(path), source_size_bytes=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                sample_sha256=hashlib.sha256(first + last).hexdigest(),
                version=version, header_offset=declared_offset, offset=offset,
                count=count, height=height, width=width, dtype="<u2")
    print(f"UP2: {path}\nPatterns: {count:,}\nSize (H x W): {height} x {width}"
          f"\nDtype: uint16 | offset: {offset} | file: {stat.st_size / 1024**3:.2f} GiB")
    return meta


def create_or_load_splits(meta, split_file, seed, fractions, max_patterns=None):
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.all(fractions > 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive split fractions must sum to one.")
    n = meta["count"]
    if max_patterns is not None and not 3 <= max_patterns <= n:
        raise ValueError(f"max_patterns must be between 3 and {n}.")
    selected_n = n if max_patterns is None else int(max_patterns)
    nt, nv = int(selected_n * fractions[0]), int(selected_n * fractions[1])
    expected_lengths = (nt, nv, selected_n - nt - nv)
    if min(expected_lengths) < 1:
        raise ValueError("Too few patterns for three nonempty splits.")
    signature = dict(schema=1, source=meta, seed=int(seed),
                     fractions=fractions.tolist(), max_patterns=max_patterns)
    split_file = Path(split_file).expanduser().resolve()
    names = ("train_idx", "val_idx", "test_idx")
    if split_file.exists():
        with np.load(split_file, allow_pickle=False) as saved:
            if "metadata_json" not in saved or json.loads(str(saved["metadata_json"].item())) != signature:
                raise ValueError(
                    f"Split metadata mismatch: {split_file}. "
                    "Use a different --split-file for changed data/settings."
                )
            splits = [saved[name].copy() for name in names]
        print(f"Reusing splits: {split_file}")
    else:
        ids = np.random.default_rng(seed).permutation(n)[:selected_n]
        splits = [ids[:nt], ids[nt:nt + nv], ids[nt + nv:]]
        split_file.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create prevents accidental overwriting of an existing split.
        with split_file.open("xb") as f:
            np.savez(f, **dict(zip(names, splits)), metadata_json=json.dumps(signature))
        print(f"Created splits: {split_file}")
    for ids, expected in zip(splits, expected_lengths):
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != expected:
            raise ValueError("Invalid split index array.")
        if int(ids.min()) < 0 or int(ids.max()) >= n:
            raise ValueError("Split indices out of bounds.")
    combined = np.concatenate(splits)
    if np.unique(combined).size != selected_n:
        raise ValueError("Duplicate or overlapping split indices.")
    for name, ids in zip(("Train", "Validation", "Test"), splits):
        print(f"{name}: {len(ids):,}")
    return splits, split_file


class UP2Dataset(Dataset):
    """Return (float32 image [1,H,W], original pattern ID).

    A per-process file handle and seek/read keep image allocations bounded.
    No full-file memory map, image export, or image cache is needed.
    """
    def __init__(self, metadata, indices, normalization="per_pattern_minmax"):
        if normalization not in ("per_pattern_minmax", "uint16"):
            raise ValueError("Unknown normalization mode.")
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalization = normalization
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = open(self.metadata["source_path"], "rb", buffering=0)
            self._pid = os.getpid()
        pattern_id = int(self.indices[index])
        h, w = self.metadata["height"], self.metadata["width"]
        self._file.seek(self.metadata["offset"] + pattern_id * h * w * 2)
        raw = np.fromfile(self._file, dtype="<u2", count=h * w)
        if raw.size != h * w:
            raise IOError(f"Incomplete pattern {pattern_id}; source may have changed.")
        image = raw.reshape(h, w).astype(np.float32)
        if self.normalization == "per_pattern_minmax":
            image -= image.min()
            image /= max(float(image.max()), 1.0)
        else:
            image /= 65535.0
        image *= 2.0
        image -= 1.0
        return torch.from_numpy(image[None]), pattern_id


def get_dataloaders(path=None, offset_override=config.UP2_OFFSET,
                    split_file=None, seed=config.SEED,
                    fractions=config.SPLIT_FRACTIONS,
                    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                    max_patterns=None, normalization=config.NORMALIZATION,
                    pin_memory=False):
    meta = inspect_up2(config.UP2_PATH if path is None else path, offset_override)
    if split_file is None:
        suffix = "" if max_patterns is None else f".subset{max_patterns}"
        split_file = str(meta["source_path"]) + suffix + ".vae_splits.npz"
    splits, split_file = create_or_load_splits(meta, split_file, seed, fractions, max_patterns)
    datasets = [UP2Dataset(meta, ids, normalization) for ids in splits]
    generator = torch.Generator().manual_seed(seed)
    options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        options.update(prefetch_factor=1, persistent_workers=True)
    loaders = [DataLoader(ds, shuffle=(i == 0), generator=generator if i == 0 else None,
                          **options) for i, ds in enumerate(datasets)]
    return (*loaders, {"up2": meta, "split_file": str(split_file),
                       "normalization": normalization,
                       "split_counts": [len(ds) for ds in datasets]})
