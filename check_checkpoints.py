"""Validate the best and last training checkpoints."""

import argparse
from pathlib import Path

import torch

from config import BASE_CHANNELS, CHECKPOINT_DIR, DEVICE
from models.unet import FrameInterpolationUNet


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def validate_checkpoint(path, device):
    print(f"\nChecking: {path}")

    if not path.is_file():
        print("  FAIL: file does not exist")
        return False

    try:
        checkpoint = load_checkpoint(path, device)
    except Exception as exc:
        print(f"  FAIL: cannot load checkpoint: {exc}")
        return False

    if not isinstance(checkpoint, dict):
        print("  FAIL: checkpoint must be a dictionary")
        return False

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        print("  FAIL: missing or empty model_state_dict")
        return False

    if not all(torch.is_tensor(value) for value in state_dict.values()):
        print("  FAIL: model_state_dict contains non-tensor values")
        return False

    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        print("  FAIL: config must be a dictionary")
        return False

    base_channels = config.get("base_channels", BASE_CHANNELS)
    try:
        model = FrameInterpolationUNet(
            base_channels=int(base_channels)
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        print(f"  FAIL: weights do not match the U-Net architecture: {exc}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print("  PASS: checkpoint is readable and matches FrameInterpolationUNet")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Validation L1: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Base channels: {base_channels}")
    print(
        "  Optimizer state: "
        + ("present" if "optimizer_state_dict" in checkpoint else "missing")
    )
    print(
        "  GradScaler state: "
        + ("present" if "scaler_state_dict" in checkpoint else "missing")
    )
    print(f"  Best validation L1: {checkpoint.get('best_val', 'not saved')}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check best_model.pth and last_model.pth"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory containing best_model.pth and last_model.pth",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used to instantiate and validate the model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = DEVICE
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    print(f"Device: {device}")
    results = [
        validate_checkpoint(
            args.checkpoint_dir / "best_model.pth",
            device,
        ),
        validate_checkpoint(
            args.checkpoint_dir / "last_model.pth",
            device,
        ),
    ]

    if all(results):
        print("\nAll checkpoints are valid.")
    else:
        raise SystemExit("\nOne or more checkpoints are invalid.")


if __name__ == "__main__":
    main()
