from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from tqdm.auto import tqdm
import config
from data import get_dataloaders
from model import VAE, count_parameters
from losses import vae_loss
from utils import (jsonable, save_json, save_checkpoint, set_seed, get_device,
                    synchronize, capture_rng, restore_rng)
from visualization import save_history, plot_reconstructions


def settings_from_cli():
    s = {k: v for k, v in vars(config).items() if k.isupper()}
    p = argparse.ArgumentParser(description=__doc__)
    mapping = {"up2": ("UP2_PATH", Path), "offset": ("UP2_OFFSET", str),
               "split-file": ("SPLIT_FILE", Path), "max-patterns": ("MAX_PATTERNS", int),
               "epochs": ("EPOCHS", int), "batch-size": ("BATCH_SIZE", int),
               "eval-batch-size": ("EVAL_BATCH_SIZE", int), "num-workers": ("NUM_WORKERS", int),
               "latent-dim": ("LATENT_DIM", int), "lr": ("LEARNING_RATE", float),
               "beta": ("BETA", float), "seed": ("SEED", int), "split-seed": ("SPLIT_SEED", int),
               "normalization": ("NORMALIZATION", str), "device": ("DEVICE", str),
               "output-root": ("OUTPUT_ROOT", Path), "run-name": ("RUN_NAME", str),
               "resume": ("RESUME", Path), "log-every": ("LOG_EVERY_STEPS", int)}
    for flag, (key, dtype) in mapping.items():
        p.add_argument("--"+flag, dest=key, type=dtype, default=argparse.SUPPRESS)
    p.add_argument("--no-test", dest="RUN_TEST_AT_END", action="store_false", default=argparse.SUPPRESS)
    s.update(vars(p.parse_args()))
    s["UP2_OFFSET"] = None if s["UP2_OFFSET"] in (None, "header") else int(s["UP2_OFFSET"])
    for key in ("EPOCHS", "BATCH_SIZE", "EVAL_BATCH_SIZE", "PREFETCH_FACTOR", "LOG_EVERY_STEPS",
                "CHECK_FINITE_EVERY", "PLOT_EVERY_EPOCHS", "NUM_RECON_IMAGES", "PLOT_DPI"):
        if s[key] < 1: p.error(f"{key} must be positive.")
    if s["IN_CHANNELS"] != 1: p.error("UP2 loader requires IN_CHANNELS=1.")
    if s["NUM_WORKERS"] < 0 or s["RECON_EVERY_EPOCHS"] < 0: p.error("Worker/plot counts cannot be negative.")
    if s["LEARNING_RATE"] <= 0 or s["BETA"] < 0 or s["WEIGHT_DECAY"] < 0 or s["ADAM_EPS"] <= 0:
        p.error("Invalid optimization settings.")
    if len(s["ADAM_BETAS"]) != 2 or any(not 0 <= b < 1 for b in s["ADAM_BETAS"]): p.error("Invalid ADAM_BETAS.")
    if s["KL_REDUCTION"] not in ("sum", "mean"): p.error("Invalid KL_REDUCTION.")
    if s["CPU_THREADS"] is not None and s["CPU_THREADS"] < 1: p.error("CPU_THREADS must be positive.")
    if s["GRAD_CLIP_NORM"] is not None and s["GRAD_CLIP_NORM"] <= 0: p.error("Invalid GRAD_CLIP_NORM.")
    if s["PIN_MEMORY"] not in ("auto", True, False): p.error("PIN_MEMORY must be auto, True, or False.")
    if Path(s["RUN_NAME"]).name != s["RUN_NAME"] or s["RUN_NAME"] in ("", ".", ".."):
        p.error("RUN_NAME must be a plain name, not a path.")
    return s


def run_epoch(model, loader, device, settings, optimizer=None, label=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    chunk = torch.zeros(3, device=device)
    count = chunk_count = 0
    synchronize(device)
    started = time.perf_counter()
    progress = tqdm(loader, desc=label or ("Train" if training else "Validation"),
                    mininterval=settings["PROGRESS_MININTERVAL"], leave=False)
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for step, (images, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            if training:
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar, _ = model(images)
            else:
                mu, logvar = model.encoder(images)
                recon = model.decode(mu)
            loss, parts = vae_loss(recon, images, mu, logvar, settings["BETA"], settings["KL_REDUCTION"])
            check = step == 1 or step % settings["CHECK_FINITE_EVERY"] == 0 or step == len(loader)
            if check and not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss; no checkpoint saved for this epoch.")
            if training:
                loss.backward()
                if settings["GRAD_CLIP_NORM"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["GRAD_CLIP_NORM"],
                                                    error_if_nonfinite=check)
                optimizer.step()
            n = images.size(0)
            count += n
            chunk_count += n
            chunk += torch.stack([parts[k] for k in ("loss", "recon_loss", "kl_loss")]) * n
            if step % settings["LOG_EVERY_STEPS"] == 0 or step == len(loader):
                values = chunk.cpu().numpy().astype(np.float64)
                if not np.isfinite(values).all(): raise FloatingPointError("Non-finite metrics.")
                totals += values
                progress.set_postfix(loss=f"{values[0]/chunk_count:.6f}")
                chunk.zero_()
                chunk_count = 0
    synchronize(device)
    seconds = time.perf_counter() - started
    result = dict(zip(("loss", "recon_loss", "kl_loss"), (totals/count).tolist()))
    result.update(seconds=seconds, images_per_second=count/max(seconds, 1e-9))
    return result


def check_resume(checkpoint, settings, data_info):
    old = checkpoint.get("effective_config")
    if old is not None:
        keys = ("LATENT_DIM", "IN_CHANNELS", "LEARNING_RATE", "BETA", "KL_REDUCTION", "ADAM_BETAS",
                "ADAM_EPS", "WEIGHT_DECAY", "GRAD_CLIP_NORM", "BATCH_SIZE", "SEED", "SPLIT_SEED",
                "NORMALIZATION", "MAX_PATTERNS", "SPLIT_FRACTIONS", "UP2_OFFSET")
        for key in keys:
            if jsonable(settings[key]) != old[key]:
                raise ValueError(f"Resume setting differs: {key}. Start a fresh run for changed learning settings.")
    else:
        legacy = checkpoint.get("run_config", {})
        old = dict(LATENT_DIM=checkpoint["latent_dim"], IN_CHANNELS=checkpoint.get("in_channels", 1),
                    BETA=checkpoint["beta"], KL_REDUCTION="sum", ADAM_BETAS=[0.9,0.999],
                    ADAM_EPS=1e-8, WEIGHT_DECAY=0.0, GRAD_CLIP_NORM=1.0,
                    BATCH_SIZE=legacy.get("batch_size",16), LEARNING_RATE=legacy.get("lr",1e-4),
                    NORMALIZATION=legacy.get("normalization","per_pattern_minmax"),
                    MAX_PATTERNS=legacy.get("max_patterns"), SEED=legacy.get("seed",42),
                    SPLIT_SEED=legacy.get("seed",42))
        for key, value in old.items():
            if jsonable(settings[key]) != value: raise ValueError(f"Legacy resume setting differs: {key}.")
    old_data = checkpoint.get("data_info", checkpoint.get("run_config", {}))
    if old_data.get("up2") != data_info["up2"]:
        raise ValueError("Resume source metadata differs from checkpoint.")
    if old_data.get("split_file") != data_info["split_file"]:
        raise ValueError("Resume split file differs from checkpoint.")
    if old_data.get("split_counts") != data_info["split_counts"]:
        raise ValueError("Resume split counts differ from checkpoint.")


def main():
    s = settings_from_cli()
    if s["CPU_THREADS"] is not None: torch.set_num_threads(s["CPU_THREADS"])
    set_seed(s["SEED"])
    device = get_device(s["DEVICE"])
    pin = device.type == "cuda" if s["PIN_MEMORY"] == "auto" else bool(s["PIN_MEMORY"])
    if device.type != "cuda" and pin:
        warnings.warn("Pinned memory is disabled here for non-CUDA devices.")
        pin = False
    print(f"Device: {device} | workers: {s['NUM_WORKERS']} | batch: {s['BATCH_SIZE']}")
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        s["UP2_PATH"], offset_override=s["UP2_OFFSET"], split_file=s["SPLIT_FILE"],
        seed=s["SPLIT_SEED"], fractions=s["SPLIT_FRACTIONS"], batch_size=s["BATCH_SIZE"],
        num_workers=s["NUM_WORKERS"], max_patterns=s["MAX_PATTERNS"], normalization=s["NORMALIZATION"],
        pin_memory=pin, eval_batch_size=s["EVAL_BATCH_SIZE"], prefetch_factor=s["PREFETCH_FACTOR"],
        persistent_workers=s["PERSISTENT_WORKERS"], shuffle_seed=s["SEED"])

    model = VAE(latent_dim=s["LATENT_DIM"], in_channels=s["IN_CHANNELS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], betas=s["ADAM_BETAS"],
                                    eps=s["ADAM_EPS"], weight_decay=s["WEIGHT_DECAY"])
    history, start_epoch, best_val, best_epoch, inherited_best = [], 1, float("inf"), 0, None
    if s["RESUME"]:
        resume_path = Path(s["RESUME"]).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        check_resume(checkpoint, s, data_info)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if s["EPOCHS"] < start_epoch: raise ValueError("EPOCHS must exceed the checkpoint epoch.")
        history = checkpoint.get("history", [])
        if not history and (resume_path.parent / "training_history.json").exists():
            history = [r for r in json.loads((resume_path.parent / "training_history.json").read_text())
                        if r["epoch"] <= checkpoint["epoch"]]
        best_val = checkpoint["val_loss"]
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        if checkpoint["epoch"] == best_epoch:
            inherited_best = checkpoint
        else:
            best_file = resume_path.parent / "vae_best.pt"
            inherited_best = torch.load(best_file, map_location="cpu", weights_only=True)
            check_resume(inherited_best, s, data_info)
            if inherited_best["epoch"] != best_epoch or inherited_best["val_loss"] != best_val:
                raise ValueError("Companion vae_best.pt does not match the resumed checkpoint.")
        if "rng_state" in checkpoint:
            restore_rng(checkpoint["rng_state"], train_loader, device)
        else:
            warnings.warn("Legacy checkpoint has no RNG state: continuing weights/optimizer, not exact stochastic trajectory.")
        print(f"Continuing at epoch {start_epoch}; best validation epoch so far: {best_epoch}")
    output = Path(s["OUTPUT_ROOT"]).expanduser().resolve() / (s["RUN_NAME"]+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)
    saved_settings = jsonable(s)
    run_config = dict(settings=saved_settings, **data_info, device_used=str(device),
                        pin_memory_used=pin, torch_version=str(torch.__version__),
                        evaluation="decode(mu) + KL; train uses sampled z")
    save_json(run_config, output / "run_config.json")
    (output / "config_snapshot.py").write_text(Path(config.__file__).read_text())
    if inherited_best is not None:
        save_checkpoint(inherited_best, output / "vae_best.pt")
        del inherited_best, checkpoint
    print(f"Parameters: {count_parameters(model):,}\nOutputs: {output}")
    try:
        for epoch in range(start_epoch, s["EPOCHS"]+1):
            epoch_start = time.perf_counter()
            train = run_epoch(model, train_loader, device, s, optimizer, f"Train {epoch}/{s['EPOCHS']}")
            val = run_epoch(model, val_loader, device, s)
            row = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"], beta=s["BETA"],
                       **{f"train_{k}": v for k,v in train.items()}, **{f"val_{k}": v for k,v in val.items()})
            improved = val["loss"] < best_val
            if improved: best_val, best_epoch = val["loss"], epoch
            history.append(row)
            save_history(history, output, plot=False)
            state = dict(epoch=epoch, best_epoch=best_epoch, val_loss=best_val,
                            current_val_loss=val["loss"], model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(), latent_dim=s["LATENT_DIM"],
                            in_channels=s["IN_CHANNELS"], beta=s["BETA"], history=history,
                            effective_config=saved_settings, data_info=data_info,
                            rng_state=capture_rng(train_loader, device))
            if improved: save_checkpoint(state, output / "vae_best.pt")
            save_checkpoint(state, output / "vae_last.pt")
            plot = epoch % s["PLOT_EVERY_EPOCHS"] == 0 or epoch == s["EPOCHS"]
            save_history(history, output, plot=plot, dpi=s["PLOT_DPI"])
            if s["RECON_EVERY_EPOCHS"] and (epoch % s["RECON_EVERY_EPOCHS"] == 0 or epoch == s["EPOCHS"]):
                plot_reconstructions(model, val_loader.dataset, device,
                    output / "reconstructions" / f"epoch_{epoch:03d}.png",
                    num_images=s["NUM_RECON_IMAGES"], dpi=s["PLOT_DPI"])
            line = (f"Epoch {epoch}/{s['EPOCHS']} | train={train['loss']:.6f} val={val['loss']:.6f} | "
                    f"MSE={train['recon_loss']:.6f}/{val['recon_loss']:.6f} "
                    f"KL={train['kl_loss']:.6f}/{val['kl_loss']:.6f} | "
                    f"train {train['images_per_second']:.1f} images/s | "
                    f"train {train['seconds']:.1f}s, val {val['seconds']:.1f}s | "
                    f"epoch wall {time.perf_counter()-epoch_start:.1f}s")
            print(line, flush=True)
            with (output / "epochs.log").open("a") as f: f.write(line+"\n")
        if s["RUN_TEST_AT_END"]:
            checkpoint = torch.load(output / "vae_best.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            test = run_epoch(model, test_loader, device, s, label="Final test")
            test["best_epoch"] = best_epoch
            save_json(test, output / "test_metrics.json")
            print(f"Final test loss: {test['loss']:.6f}")
        print(f"Best validation epoch: {best_epoch} | loss: {best_val:.6f}\nSaved: {output}")
    finally:
        for loader in (train_loader, val_loader, test_loader): loader.dataset.close()


if __name__ == "__main__":
    main()
