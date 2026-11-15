# Video Frame Interpolation Using Deep Learning

Course project implementation:
- Linear blending baseline
- U-Net-based midpoint frame interpolation
- Hugging Face Vimeo-90K Triplet subset preparation
- Training, validation, PSNR/SSIM evaluation and visualization

## 1. Environment (Windows)

Use Python 3.11 and a virtual environment:

```powershell
py -3.11 -m venv .venv
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

## 6. Project caveats

1. The Hugging Face repository has a single `train` split; this project creates
   its own train/validation/test split from the selected subset.
2. The preparation script assumes each consecutive group of three rows is a
   triplet and that those rows are temporally ordered. The script checks group
   size, but visual inspection is still required.
3. Do not use the test split for model selection or hyperparameter tuning.
4. Report actual sample counts, training time, hardware and measured metrics.
5. The U-Net is a course-project baseline, not a reproduction of RIFE,
   Super SloMo or another state-of-the-art method.
