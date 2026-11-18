import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, USE_AMP, BASE_CHANNELS,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, SEED,
    MAX_TRAIN_BATCHES, MAX_TRIPLETS, TRIPLETS_DIR, SPLITS_DIR,
    PROJECT_ROOT, EXPERIMENT_NAME, RESUME_TRAINING,
    RESUME_FROM_CHECKPOINT, TRAIN_RATIO, VAL_RATIO,
)
from data_loader.vimeo_dataset import VimeoTripletDataset
from models.unet import FrameInterpolationUNet
from utils.visualization import save_training_curves
from experiment_logger import log_epoch, log_experiment


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the U-Net frame interpolation model"
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Experiment folder name under checkpoints/ and results/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint to resume from, relative to project root or absolute",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Train through this epoch number",
    )
    return parser.parse_args()


def check_dataset_size():
    if not TRIPLETS_DIR.is_dir():
        raise FileNotFoundError(
            f"Triplet dataset directory not found: {TRIPLETS_DIR}"
        )

    triplet_dirs = [
        path for path in TRIPLETS_DIR.iterdir()
        if path.is_dir()
        and all((path / name).is_file() for name in ("im1.png", "im2.png", "im3.png"))
    ]
    actual = len(triplet_dirs)

    print(f"Configured MAX_TRIPLETS: {MAX_TRIPLETS}")
    print(f"Complete triplet folders on disk: {actual}")

    if actual < MAX_TRIPLETS:
        print(
            "WARNING: The local dataset has fewer complete triplets than "
            f"MAX_TRIPLETS ({actual} < {MAX_TRIPLETS}). "
            "The script will not download or prepare data."
        )

    split_counts = {}
    for split in ("train", "val", "test"):
        split_file = SPLITS_DIR / f"{split}.txt"
        if not split_file.is_file():
            print(f"WARNING: Missing split file: {split_file}")
            continue

        ids = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing = [
            triplet_id for triplet_id in ids
            if not (TRIPLETS_DIR / triplet_id / "im1.png").is_file()
        ]
        split_counts[split] = len(ids)
        if missing:
            print(
                f"WARNING: {len(missing)} {split} split entries have missing "
                "or incomplete folders."
            )

    if split_counts:
        total_split = sum(split_counts.values())
        if total_split != MAX_TRIPLETS:
            print(
                "WARNING: Split file total "
                f"({total_split}) differs from MAX_TRIPLETS "
                f"({MAX_TRIPLETS}). "
                "Training will use the configured proportion from existing "
                "split files. Existing splits are not being modified."
            )
        print(
            "Current split counts: "
            + ", ".join(f"{name}={count}" for name, count in split_counts.items())
        )
        print(
            "NOTE: Increasing MAX_TRIPLETS does not automatically regenerate "
            "train/validation/test splits."
        )


def resolve_resume_checkpoint(args, experiment_checkpoint_dir):
    if args.resume_from:
        path = Path(args.resume_from)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    experiment_last = experiment_checkpoint_dir / "last_model.pth"
    if experiment_last.is_file():
        return experiment_last
    return CHECKPOINT_DIR / "last_model.pth"


def load_resume_checkpoint(model, optimizer, scaler, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    print(f"Resuming from checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Resume checkpoint must contain a dictionary.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(
            "Resume checkpoint is missing a valid model_state_dict."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint model weights are incompatible with FrameInterpolationUNet. "
            "Check base_channels and model architecture."
        ) from exc

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Checkpoint optimizer state is incompatible."
            ) from exc
    else:
        print("WARNING: checkpoint has no optimizer_state_dict; optimizer is new.")

    if "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("Loaded GradScaler state.")
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"WARNING: could not load GradScaler state: {exc}")
            print("Using a newly initialized GradScaler.")
    else:
        print(
            "WARNING: checkpoint has no scaler_state_dict; "
            "using a newly initialized GradScaler."
        )

    completed_epoch = int(checkpoint.get("epoch", 0))
    if "best_val" in checkpoint:
        best_val = float(checkpoint["best_val"])
    else:
        best_val = float(checkpoint.get("val_loss", float("inf")))

    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict):
        checkpoint_channels = checkpoint_config.get("base_channels")
        if checkpoint_channels is not None and int(checkpoint_channels) != BASE_CHANNELS:
            print(
                "WARNING: checkpoint base_channels="
                f"{checkpoint_channels}, current BASE_CHANNELS={BASE_CHANNELS}. "
                "Weights were loaded after instantiating the current architecture."
            )

    source_triplets = (
        checkpoint_config.get("max_triplets")
        if isinstance(checkpoint_config, dict)
        else None
    )
    if source_triplets is not None and int(source_triplets) != MAX_TRIPLETS:
        print(
            "WARNING: continuing training on a different dataset size "
            f"(checkpoint={source_triplets}, current={MAX_TRIPLETS}). "
            "This continues weights, but not necessarily on the same data split."
        )

    print(f"Checkpoint epoch: {completed_epoch}")
    print(f"Starting from epoch: {completed_epoch + 1}")
    print(f"Restored best validation loss: {best_val:.6f}")
    return checkpoint, completed_epoch, best_val


def load_history(history_path, experiment_name):
    if not history_path.is_file():
        return {
            "experiment": experiment_name,
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
        }

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, dict):
        raise RuntimeError(f"Invalid history format: {history_path}")

    history.setdefault("experiment", experiment_name)
    history.setdefault("epochs", [])
    history.setdefault("train_loss", [])
    history.setdefault("val_loss", [])
    return history


def save_epoch_history(history_path, history):
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def upsert_epoch(history, epoch_record):
    epoch = int(epoch_record["epoch"])
    records = history["epochs"]
    replaced = False
    for index, record in enumerate(records):
        if int(record["epoch"]) == epoch:
            records[index] = epoch_record
            replaced = True
            break
    if not replaced:
        records.append(epoch_record)
    records.sort(key=lambda record: int(record["epoch"]))

    history["train_loss"] = [
        record["train_loss"] for record in records
    ]
    history["val_loss"] = [
        record["val_loss"] for record in records
    ]


def main():
    args = parse_args()
    total_training_time = 0.0
    seed_everything(SEED)
    experiment_name = args.experiment_name
    experiment_checkpoint_dir = CHECKPOINT_DIR / experiment_name
    experiment_result_dir = RESULTS_DIR / experiment_name
    experiment_figure_dir = experiment_result_dir / "figures"
    history_path = experiment_result_dir / "history.json"

    experiment_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_result_dir.mkdir(parents=True, exist_ok=True)
    experiment_figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Checkpoint directory: {experiment_checkpoint_dir}")
    print(f"Results directory: {experiment_result_dir}")
    check_dataset_size()

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    expected_train = int(MAX_TRIPLETS * TRAIN_RATIO)
    expected_val = int(MAX_TRIPLETS * VAL_RATIO)

    train_ds = VimeoTripletDataset(
        split="train",
        augment=True,
        max_samples=expected_train,
    )
    val_ds = VimeoTripletDataset(
        split="val",
        augment=False,
        max_samples=expected_val,
    )

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

    history = load_history(history_path, experiment_name)
    completed_epoch = 0
    best_val = float("inf")
    best_train = float("inf")

    resume_requested = args.resume or RESUME_TRAINING
    if resume_requested:
        resume_path = resolve_resume_checkpoint(
            args, experiment_checkpoint_dir
        )
        if RESUME_FROM_CHECKPOINT and not args.resume_from:
            configured_path = Path(RESUME_FROM_CHECKPOINT)
            if not configured_path.is_absolute():
                configured_path = PROJECT_ROOT / configured_path
            resume_path = configured_path.resolve()
        _, completed_epoch, best_val = load_resume_checkpoint(
            model, optimizer, scaler, resume_path
        )
    else:
        print("Starting a new training run.")

    start_epoch = completed_epoch + 1
    if args.epochs < start_epoch:
        print(
            f"Nothing to train: --epochs={args.epochs} is at or below "
            f"completed checkpoint epoch={completed_epoch}."
        )
        return

    print(f"Training epochs: {start_epoch}-{args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
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

        elapsed = time.time() - start
        total_training_time += elapsed
        best_train = min(best_train, train_loss)
        print(
            f"Epoch {epoch}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, time={elapsed:.1f}s"
        )
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            print("New best validation L1.")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
            "best_val": best_val,
            "config": {
                "base_channels": BASE_CHANNELS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_triplets": MAX_TRIPLETS,
                "experiment_name": experiment_name,
            },
        }
        torch.save(checkpoint, experiment_checkpoint_dir / "last_model.pth")

        if is_best:
            torch.save(checkpoint, experiment_checkpoint_dir / "best_model.pth")
            print("Saved new best checkpoint.")

        upsert_epoch(history, {
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_time": elapsed,
            "best_val_loss": best_val,
        })
        save_epoch_history(history_path, history)

        save_training_curves(
            history, experiment_figure_dir / "training_curves.png"
        )
        log_epoch(
            experiment_name,
            epoch,
            train_loss,
            val_loss,
            elapsed,
        )

    print("Training finished.")
    print("Best validation L1:", best_val)
    log_experiment(
        config=experiment_name,
        triplets=MAX_TRIPLETS,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        train_l1=best_train,
        val_l1=best_val,
        psnr=None,
        ssim=None,
        total_time=total_training_time,
        checkpoint=str(experiment_checkpoint_dir / "best_model.pth"),
    )


if __name__ == "__main__":
    main()
