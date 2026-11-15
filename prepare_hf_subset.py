"""
Download a small, local subset from the Hugging Face Vimeo-90K repository.

The repository's Dataset Viewer shows columns: image, label, and each label
appears on three consecutive rows. Labels may recur after streamed data-file
boundaries, so this script groups consecutive rows and FAILS if a run does
not contain exactly three images.

IMPORTANT:
Before training, inspect results/figures/hf_triplet_preview.png and confirm
that each saved triplet is ordered as first, middle, last frame.
"""
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    HF_DATASET_ID, MAX_TRIPLETS, TRIPLETS_DIR, SPLITS_DIR,
    TRAIN_RATIO, VAL_RATIO, SPLIT_SEED, FIGURES_DIR
)


def save_triplet(triplet_id: str, images: list):
    out_dir = TRIPLETS_DIR / triplet_id
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["im1.png", "im2.png", "im3.png"]
    for image, name in zip(images, names):
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL image for {triplet_id}/{name}, got {type(image)}"
            )
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(out_dir / name, format="PNG")


def make_splits(triplet_ids):
    ids = list(triplet_ids)
    random.Random(SPLIT_SEED).shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split_map = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, split_ids in split_map.items():
        with (SPLITS_DIR / f"{split}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(split_ids))

    print("\nSplit counts:")
    for split, split_ids in split_map.items():
        print(f"  {split}: {len(split_ids)}")


def save_preview(triplet_ids, count=4):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chosen = triplet_ids[:count]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.5 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    for row, triplet_id in enumerate(chosen):
        for col, filename in enumerate(("im1.png", "im2.png", "im3.png")):
            path = TRIPLETS_DIR / triplet_id / filename
            with Image.open(path) as img:
                axes[row][col].imshow(img)
            axes[row][col].set_title(
                f"{triplet_id} - {['First', 'Middle', 'Last'][col]}"
            )
            axes[row][col].axis("off")

    plt.tight_layout()
    output = FIGURES_DIR / "hf_triplet_preview.png"
    plt.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Saved preview: {output}")


def main():
    TRIPLETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming dataset: {HF_DATASET_ID}")
    stream = load_dataset(
        HF_DATASET_ID,
        split="train",
        streaming=True,
    )

    triplet_ids = []
    current_label = None
    current_images = []

    progress = tqdm(total=MAX_TRIPLETS, desc="Preparing triplets")

    def flush_group(label, images):
        if label is None:
            return
        if len(images) != 3:
            raise RuntimeError(
                f"Label {label!r} has {len(images)} images, expected exactly 3. "
                "Stop here; do not train on incorrectly grouped frames."
            )
        triplet_id = f"triplet_{len(triplet_ids):06d}"
        save_triplet(triplet_id, images)
        triplet_ids.append(triplet_id)
        progress.update(1)

    for row in stream:
        if "image" not in row or "label" not in row:
            raise KeyError(
                f"Expected columns 'image' and 'label'; got {list(row.keys())}"
            )

        label = row["label"]

        if current_label is None:
            current_label = label

        if label != current_label:
            flush_group(current_label, current_images)

            if len(triplet_ids) >= MAX_TRIPLETS:
                break

            current_label = label
            current_images = []

        current_images.append(row["image"])

        if len(current_images) > 3:
            raise RuntimeError(
                f"Label {current_label!r} has more than 3 consecutive rows. "
                "Dataset grouping assumption is not valid."
            )

    if len(triplet_ids) < MAX_TRIPLETS and current_label is not None:
        flush_group(current_label, current_images)

    progress.close()

    if not triplet_ids:
        raise RuntimeError("No triplets were prepared.")

    print(f"\nPrepared {len(triplet_ids)} triplets.")
    make_splits(triplet_ids)
    save_preview(triplet_ids)

    print("\nNEXT STEP:")
    print("1. Open results/figures/hf_triplet_preview.png")
    print("2. Confirm each row shows first -> middle -> last frames.")
    print("3. Only then run: python check_data.py")


if __name__ == "__main__":
    main()
