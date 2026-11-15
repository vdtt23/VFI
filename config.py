from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
TRIPLETS_DIR = DATA_ROOT / "triplets"
SPLITS_DIR = DATA_ROOT / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Hugging Face source
HF_DATASET_ID = "danjacobellis/vimeo90k_triplet"

# Data preparation
MAX_TRIPLETS = 5000       # Start small; increase after pipeline works
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
SPLIT_SEED = 42

# Images are originally 448x256 (W x H). Keep aspect ratio.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 448

# Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4             # Increase to 8 only after checking VRAM
NUM_WORKERS = 2            # Windows: start with 2, benchmark 4 later
PIN_MEMORY = DEVICE.type == "cuda"
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
USE_AMP = DEVICE.type == "cuda"
SEED = 42
MAX_TRAIN_BATCHES = None   # Set e.g. 50 for a quick smoke test; None = full epoch

# Model
BASE_CHANNELS = 32

# Evaluation / visualization
NUM_VISUALIZATION_SAMPLES = 12
