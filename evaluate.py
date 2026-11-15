import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    BASE_CHANNELS, CHECKPOINT_DIR, METRICS_DIR,
    FIGURES_DIR, NUM_VISUALIZATION_SAMPLES, USE_AMP
)
from data_loader.vimeo_dataset import VimeoTripletDataset
from models.unet import FrameInterpolationUNet
from baseline import linear_blending
from utils.metrics import batch_psnr, batch_ssim
from utils.visualization import save_comparison


def evaluate_model(model, loader, use_baseline=False):
    psnr_values = []
    ssim_values = []
    saved = 0

    for inputs, targets, ids in tqdm(loader, desc="Evaluating"):
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        if use_baseline:
            predictions = linear_blending(inputs)
        else:
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=USE_AMP,
                ):
                    predictions = model(inputs)

        psnr_values.append(batch_psnr(predictions, targets))
        ssim_values.append(batch_ssim(predictions, targets))

        if saved < NUM_VISUALIZATION_SAMPLES:
            for i in range(inputs.size(0)):
                if saved >= NUM_VISUALIZATION_SAMPLES:
                    break
                save_comparison(
                    inputs[i],
                    targets[i],
                    predictions[i],
                    FIGURES_DIR / f"{'baseline' if use_baseline else 'unet'}_{saved:03d}.png",
                    title=f"Sample {ids[i]}",
                )
                saved += 1

    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = VimeoTripletDataset(split="test", augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run python train.py first."
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )
    model = FrameInterpolationUNet(
        base_channels=checkpoint.get("config", {}).get(
            "base_channels", BASE_CHANNELS
        )
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Evaluating linear blending baseline...")
    base_psnr, base_ssim = evaluate_model(
        model, test_loader, use_baseline=True
    )

    print("Evaluating U-Net...")
    unet_psnr, unet_ssim = evaluate_model(
        model, test_loader, use_baseline=False
    )

    rows = [
        ["Linear blending", base_psnr, base_ssim],
        ["U-Net", unet_psnr, unet_ssim],
    ]

    with (METRICS_DIR / "test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PSNR_dB", "SSIM"])
        writer.writerows(rows)

    print("\nTest results")
    for name, psnr, ssim in rows:
        print(f"{name:20s} PSNR={psnr:.4f} dB | SSIM={ssim:.4f}")

    print(f"Saved metrics to {METRICS_DIR / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()
