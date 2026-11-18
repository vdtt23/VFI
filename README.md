# Video Frame Interpolation Using Deep Learning

Course project implementation:
- Linear blending baseline
- U-Net-based midpoint frame interpolation
- Hugging Face Vimeo-90K Triplet subset preparation
- Training, validation, PSNR/SSIM evaluation and visualization

## 1. Environment (Windows)

Use Python 3.12 and a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with the CUDA build appropriate for your driver:
https://pytorch.org/get-started/locally/

Then install other dependencies:

```powershell
pip install -r requirements.txt
```

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Prepare a local subset from Hugging Face

The selected Hugging Face repository exposes one `train` split and columns
`image` and `label`. The Dataset Viewer shows each label repeated three times
consecutively. Labels may recur when the stream advances to another data file;
the preparation script groups only consecutive rows and requires exactly three
images per run.

Start with 5,000 triplets (config.py: MAX_TRIPLETS):

```powershell
python prepare_hf_subset.py
```

This writes:
- `data/processed/triplets/<triplet_id>/im1.png`
- `data/processed/triplets/<triplet_id>/im2.png`
- `data/processed/triplets/<triplet_id>/im3.png`
- `data/processed/splits/train.txt`
- `data/processed/splits/val.txt`
- `data/processed/splits/test.txt`

IMPORTANT: Open `results/figures/hf_triplet_preview.png`. Verify visually that
each row is first frame, middle frame, last frame in temporal order. If not,
do not train; inspect the dataset's grouping/ordering first.

The split is a random split of the selected Hugging Face `train` data. It is
NOT the official Vimeo-90K test split. State this clearly in the report.

## 3. Check DataLoader

```powershell
python check_data.py
```

Expected shapes:
- Input: `[B, 6, 256, 448]`
- Target: `[B, 3, 256, 448]`

## 4. Train

```powershell
python train.py
```

Default settings are conservative for a 16GB GPU:
- Batch size 4
- 256 x 448 input
- U-Net base channels 32
- Adam, learning rate 1e-4
- L1 loss
- FP16 autocast on CUDA
- 15 epochs
- 2 DataLoader workers

If CUDA out-of-memory occurs, reduce BATCH_SIZE to 2 in config.py.
If GPU memory is comfortable and utilization is low, try batch size 6 or 8.
On Windows, keep `if __name__ == "__main__":` guards in scripts.

## 5. Evaluate

```powershell
python evaluate.py
```

Outputs:
- `results/metrics/test_metrics.csv`
- `results/figures/baseline_*.png`
- `results/figures/unet_*.png`

## 6. Check checkpoints

To check whether both training checkpoints exist, can be loaded, and match the
current U-Net architecture:

```powershell
python check_checkpoints.py
```

The script checks:
- `checkpoints/best_model.pth`
- `checkpoints/last_model.pth`
- checkpoint structure and model weights
- compatibility with `FrameInterpolationUNet`
- saved epoch, validation loss and model configuration

To check checkpoints in another directory:

```powershell
python check_checkpoints.py --checkpoint-dir experiments\run_01\checkpoints
```

## 7. Experiments and resume training

Training creates separate folders based on `experiment_name`:

```text
checkpoints/
  experiment_5000/
  experiment_15000/
results/
  experiment_5000/
  experiment_15000/
```

Start a completely new 5,000-triplet experiment:

```powershell
python train.py --experiment-name experiment_5000 --epochs 15
```

Start a completely new 15,000-triplet experiment:

```powershell
python train.py --experiment-name experiment_15000 --epochs 15
```

The experiment name is independent of `MAX_TRIPLETS` in `config.py`. Set
`MAX_TRIPLETS` when preparing a dataset, but training does not automatically
download or prepare data.

Resume the 15,000 experiment from its own last checkpoint:

```powershell
python train.py --experiment-name experiment_15000 --epochs 30 --resume
```

Continue the weights from the 5,000-triplet experiment into the 15,000-triplet
experiment:

```powershell
python train.py --experiment-name experiment_15000 --epochs 30 --resume-from checkpoints\experiment_5000\last_model.pth
```

Resume also works through `config.py`:

```python
RESUME_TRAINING = True
RESUME_FROM_CHECKPOINT = "checkpoints/experiment_5000/last_model.pth"
```

Resume loads model and optimizer state, restores GradScaler state when present,
and starts after the checkpoint's completed epoch. Legacy checkpoints without
`scaler_state_dict` are supported and receive a newly initialized GradScaler.

When continuing onto a larger dataset, the project prints a warning because
new data or a new train/validation/test split changes the training distribution.
The script does not regenerate or delete splits.

Evaluate a specific experiment:

```powershell
python evaluate.py --experiment-name experiment_15000 --checkpoint best_model.pth
```

Evaluate another checkpoint directory explicitly:

```powershell
python evaluate.py --checkpoint-dir checkpoints\experiment_15000 --checkpoint last_model.pth
```

## 8. Project caveats

1. The Hugging Face repository has a single `train` split; this project creates
   its own train/validation/test split from the selected subset.
2. The preparation script assumes each consecutive group of three rows is a
   triplet and that those rows are temporally ordered. The script checks group
   size, but visual inspection is still required.
3. Do not use the test split for model selection or hyperparameter tuning.
4. Report actual sample counts, training time, hardware and measured metrics.
5. The U-Net is a course-project baseline, not a reproduction of RIFE,
   Super SloMo or another state-of-the-art method.
