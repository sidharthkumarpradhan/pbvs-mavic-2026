from typing import Tuple

import cv2
import numpy as np


def variance_weighted_fusion(img: np.ndarray) -> np.ndarray:
    """Fuse RGB channels into greyscale.

    Uses per-image inverse-variance weights.

    Instead of fixed ITU-R BT.601 luminance weights (0.299R + 0.587G + 0.114B),
    channels are fused with adaptive weights inversely proportional to each
    channel's variance.  Channels with lower variance (more stable signal) are
    up-weighted, avoiding over-reliance on any single channel under varying
    illumination conditions.

    Parameters
    ----------
    img : np.ndarray, shape (C, H, W), float64
        Multi-channel image read by rasterio (C >= 1).

    Returns
    -------
    np.ndarray, shape (H, W), float64
        Fused greyscale image.  Falls back to channel 0
        for single-channel input.
    """
    if img.shape[0] < 3:
        return img[0].astype(np.float64)

    channels = [img[c].astype(np.float64) for c in range(3)]

    variances = np.array([
        np.var(ch[ch > 0]) if (ch > 0).any() else 1e-8
        for ch in channels
    ])
    inv_var = 1.0 / (variances + 1e-8)
    weights = inv_var / inv_var.sum()

    return (
        weights[0] * channels[0]
        + weights[1] * channels[1]
        + weights[2] * channels[2]
    )


def clahe_enhance(
    gray_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Enhances local contrast by equalising histograms within small tiles.
    The clip limit prevents over-amplification of noise in flat regions —
    particularly important for aerial imagery with large homogeneous areas.

    Parameters
    ----------
    gray_u8 : np.ndarray, uint8, (H, W)
        Greyscale input.
    clip_limit : float
        Contrast clip limit (default 2.0).
    tile_grid : tuple of int
        Tile grid size in (cols, rows) (default 8x8).

    Returns
    -------
    np.ndarray, uint8, (H, W)
        CLAHE-enhanced image.
    """
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe_op.apply(gray_u8)
