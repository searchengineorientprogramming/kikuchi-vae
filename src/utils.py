from pathlib import Path
import json
import os
import random
import numpy as np
import torch


def jsonable(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False))
    os.replace(temporary, path)


def save_checkpoint(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def capture_rng(loader, device):
    ns = np.random.get_state()
    state = dict(python=random.getstate(), numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
                torch=torch.get_rng_state(), shuffle=loader.generator.get_state())
    if device.type == "cuda": state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps": state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, loader, device):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    loader.generator.set_state(state["shuffle"].cpu())
    if device.type == "cuda" and "cuda" in state: torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    if device.type == "mps" and "mps" in state: torch.mps.set_rng_state(state["mps"].cpu())
