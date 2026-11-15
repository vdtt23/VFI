from pathlib import Path
import matplotlib.pyplot as plt
import torch


def save_training_curves(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train L1")
    ax.plot(epochs, history["val_loss"], label="Validation L1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 loss")
    ax.set_title("Training and validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_comparison(inputs, target, prediction, output_path, title=""):
    """
    inputs: [6,H,W], target/prediction: [3,H,W], all [0,1].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = inputs[:3].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    last = inputs[3:].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    gt = target.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    pred = prediction.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [first, last, gt, pred]
    labels = ["Input frame 1", "Input frame 3", "Ground truth", "Prediction"]

    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
