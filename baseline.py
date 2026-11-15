import torch


def linear_blending(inputs):
    """
    inputs: [B, 6, H, W], first 3 channels are first frame,
    last 3 channels are last frame.
    Returns midpoint linear blend [B, 3, H, W].
    """
    first = inputs[:, :3]
    last = inputs[:, 3:]
    return 0.5 * first + 0.5 * last
