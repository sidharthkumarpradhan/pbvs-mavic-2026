import math

import numpy as np


def adaptive_gamma_correct(
    img_u8: np.ndarray, target_mean: float = 128.0
) -> np.ndarray:
    """Per-image adaptive gamma correction.

    Computes gamma analytically so that applying I^gamma drives the image
    mean toward ``target_mean``.  Gamma is clamped to [0.6, 1.8] to prevent
    extreme corrections on unusually dark or bright outputs.

    A 256-entry look-up table (LUT) is built for fast per-pixel remapping.

    Parameters
    ----------
    img_u8 : np.ndarray, uint8
        Greyscale image to correct.
    target_mean : float
        Desired output mean luminance (default 128).

    Returns
    -------
    np.ndarray, uint8
        Gamma-corrected image.
    """
    curr = img_u8.mean()
    if curr < 2.0:
        return img_u8  # near-black — avoid log(0)

    gamma = (
        math.log(target_mean / 255.0 + 1e-8)
        / math.log(curr / 255.0 + 1e-8)
    )
    gamma = float(np.clip(gamma, 0.6, 1.8))

    lut = np.array(
        [min(255, int((v / 255.0) ** gamma * 255.0)) for v in range(256)],
        dtype=np.uint8,
    )
    return lut[img_u8]
