from torch.utils.data import DataLoader
from data_loader.vimeo_dataset import VimeoTripletDataset
from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


def main():
    dataset = VimeoTripletDataset(split="train", augment=False, max_samples=16)
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=False,
        num_workers=0,  # simplest and safest for first Windows test
    )

    inputs, targets, ids = next(iter(loader))
    print("Number of samples:", len(dataset))
    print("Input shape:", tuple(inputs.shape))
    print("Target shape:", tuple(targets.shape))
    print("First IDs:", ids[:3])
    print("Input range:", inputs.min().item(), inputs.max().item())
    print("Target range:", targets.min().item(), targets.max().item())

    assert inputs.ndim == 4 and inputs.shape[1] == 6
    assert targets.ndim == 4 and targets.shape[1] == 3
    assert inputs.shape[-2:] == targets.shape[-2:]
    print("DataLoader check passed.")


if __name__ == "__main__":
    main()
