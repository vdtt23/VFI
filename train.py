import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, USE_AMP, BASE_CHANNELS,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, SEED,
    MAX_TRAIN_BATCHES
)
from data_loader.vimeo_dataset import VimeoTripletDataset
from models.unet import FrameInterpolationUNet
from utils.visualization import save_training_curves


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, shuffle):
    kwargs = dict(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=shuffle,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def run_validation(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main():
    seed_everything(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_ds = VimeoTripletDataset(split="train", augment=True)
    val_ds = VimeoTripletDataset(split="val", augment=False)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    model = FrameInterpolationUNet(base_channels=BASE_CHANNELS).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch_idx, (inputs, targets, _) in enumerate(progress):
            if MAX_TRAIN_BATCHES is not None and batch_idx >= MAX_TRAIN_BATCHES:
                break

            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{loss.item():.5f}")

        train_loss = running_loss / max(seen, 1)
        val_loss = run_validation(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {
                "base_channels": BASE_CHANNELS,
            },
        }
        torch.save(checkpoint, CHECKPOINT_DIR / "last_model.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
            print("Saved new best checkpoint.")

        with (RESULTS_DIR / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_training_curves(
            history, FIGURES_DIR / "training_curves.png"
        )

    print("Training finished.")
    print("Best validation L1:", best_val)


if __name__ == "__main__":
    main()
