from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import TRIPLETS_DIR, SPLITS_DIR, IMAGE_HEIGHT, IMAGE_WIDTH


class VimeoTripletDataset(Dataset):
    """
    Reads locally prepared triplets:
      data/processed/triplets/<triplet_id>/im1.png
      data/processed/triplets/<triplet_id>/im2.png
      data/processed/triplets/<triplet_id>/im3.png

    Model input = concat(im1, im3), target = im2.
    """

    def __init__(
        self,
        split="train",
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        augment=False,
        max_samples=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        split_file = SPLITS_DIR / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Missing {split_file}. Run python prepare_hf_subset.py first."
            )

        self.triplet_ids = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if max_samples is not None:
            self.triplet_ids = self.triplet_ids[:max_samples]

        self.augment = augment
        self.image_height = image_height
        self.image_width = image_width
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(
            (image_height, image_width),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __len__(self):
        return len(self.triplet_ids)

    def _load_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.resize(image)
            return self.to_tensor(image)

    def __getitem__(self, index):
        triplet_id = self.triplet_ids[index]
        folder = TRIPLETS_DIR / triplet_id

        im1 = self._load_image(folder / "im1.png")
        im2 = self._load_image(folder / "im2.png")
        im3 = self._load_image(folder / "im3.png")

        # Same spatial augmentation must be applied to all three frames.
        if self.augment and random.random() < 0.5:
            im1 = TF.hflip(im1)
            im2 = TF.hflip(im2)
            im3 = TF.hflip(im3)

        # Temporal reversal is valid for midpoint interpolation.
        if self.augment and random.random() < 0.5:
            im1, im3 = im3, im1

        inputs = torch.cat((im1, im3), dim=0)  # [6, H, W]
        target = im2                                 # [3, H, W]
        return inputs, target, triplet_id
