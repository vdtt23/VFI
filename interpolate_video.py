"""
Video frame interpolation using the trained U-Net model.

Run from the project root:
    python interpolate_video.py --input input.mp4 --output output_60fps.mp4

Assumes the model was trained with concatenated RGB input frames:
[B, 6, 256, 448] -> [B, 3, 256, 448].
"""

import argparse
import inspect
from pathlib import Path

import cv2
import numpy as np
import torch

from models.unet import FrameInterpolationUNet as UNet

HEIGHT, WIDTH = 256, 448


def parse_args():
    p = argparse.ArgumentParser(description="Interpolate video with trained U-Net")
    p.add_argument("--input", required=True, help="Input video")
    p.add_argument("--output", default="results/interpolated_video.mp4")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--keep-fps",
        action="store_true",
        help="Keep source FPS (output will play about twice as long)",
    )
    p.add_argument("--codec", default="mp4v")
    return p.parse_args()


def build_model():
    # Match the project's U-Net constructor while tolerating common arg names.
    params = inspect.signature(UNet).parameters
    candidates = {
        "in_channels": 6,
        "out_channels": 3,
        "base_channels": 32,
        "input_channels": 6,
        "output_channels": 3,
    }
    kwargs = {k: v for k, v in candidates.items() if k in params}
    try:
        return UNet(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Cannot instantiate UNet with 6 input and 3 output channels. "
            "Check the constructor in models/unet.py. " + str(exc)
        ) from exc


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError(
        "Unsupported checkpoint format. Expected a raw state_dict or a "
        "dictionary containing model_state_dict/state_dict/model/net."
    )


def load_model(checkpoint_path, device):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_model()
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # Compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state = get_state_dict(checkpoint)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device, non_blocking=True)


def postprocess(prediction, original_size):
    image = prediction[0].detach().float().clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.resize(
        image, original_size, interpolation=cv2.INTER_LINEAR
    )


@torch.inference_mode()
def infer_midpoint(model, frame_a, frame_b, device):
    a = preprocess(frame_a, device)
    b = preprocess(frame_b, device)
    x = torch.cat((a, b), dim=1)  # [1, 6, 256, 448]

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(x)
    else:
        pred = model(x)

    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if pred.ndim != 4 or pred.shape[1] != 3:
        raise RuntimeError(
            f"Unexpected model output shape {tuple(pred.shape)}; expected [1,3,H,W]."
        )
    return pred


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not determine input video dimensions.")

    output_fps = source_fps if args.keep_fps else source_fps * 2.0
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(
        str(output_path), fourcc, output_fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create {output_path}. Try --codec avc1, or use AVI/XVID."
        )

    ok, previous = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("The input video has no readable frames.")

    writer.write(previous)
    pairs = 0
    written = 1
    print(
        f"Input: {width}x{height}, {source_fps:.3f} FPS, "
        f"{total_frames} frames. Output: {output_fps:.3f} FPS."
    )

    while True:
        ok, current = cap.read()
        if not ok:
            break

        middle = infer_midpoint(model, previous, current, device)
        middle_bgr = postprocess(middle, (width, height))

        # Keep original frames and insert one predicted frame between them.
        writer.write(middle_bgr)
        writer.write(current)
        written += 2
        pairs += 1
        previous = current

        if pairs % 100 == 0:
            print(f"Processed {pairs} frame pairs; wrote {written} frames.")

    cap.release()
    writer.release()

    print("Done.")
    print(f"Frame pairs processed: {pairs}")
    print(f"Output frames: {written}")
    print(f"Saved: {output_path.resolve()}")
    if args.keep_fps:
        print("Note: keeping the original FPS makes playback approximately twice as long.")


if __name__ == "__main__":
    main()
