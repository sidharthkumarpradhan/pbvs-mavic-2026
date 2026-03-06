from typing import Optional

import cv2
import numpy as np


class AdaptiveHistogramTransfer:
    """Two-pass adaptive histogram transfer with bilateral residual correction.

    Pass 1 — Standard CDF-based histogram matching maps the source pixel
    distribution onto a reference distribution (global tone alignment).

    Pass 2 — The residual between source and Pass-1 output is bilateral-
    filtered and blended back in to recover edges and local texture that
    global matching tends to suppress.

    Parameters
    ----------
    alpha : float
        Residual blend weight (default 0.25).
    bilateral_d : int
        Bilateral filter neighbourhood diameter (default 9).
    bilateral_sc : float
        Bilateral colour sigma (default 50).
    bilateral_ss : float
        Bilateral spatial sigma (default 50).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        bilateral_d: int = 9,
        bilateral_sc: float = 50.0,
        bilateral_ss: float = 50.0,
    ):
        self.alpha = alpha
        self.bd = bilateral_d
        self.bsc = bilateral_sc
        self.bss = bilateral_ss

    def _cdf_match(
        self, source: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Map source pixel intensities to match the reference CDF via
        linear interpolation between CDF sample points.
        """
        s = source.flatten().astype(np.uint8)
        r = reference.flatten().astype(np.uint8)

        s_vals, s_idx, s_counts = np.unique(
            s, return_inverse=True, return_counts=True,
        )
        r_vals, r_counts = np.unique(r, return_counts=True)

        s_cdf = np.cumsum(s_counts).astype(np.float64) / s.size
        r_cdf = np.cumsum(r_counts).astype(np.float64) / r.size

        mapped = np.interp(s_cdf, r_cdf, r_vals.astype(np.float64))
        result = mapped[s_idx].reshape(source.shape)
        return np.clip(result, 0, 255).astype(np.uint8)

    def __call__(
        self, source: np.ndarray, reference: Optional[np.ndarray]
    ) -> np.ndarray:
        """Apply two-pass histogram transfer.

        Parameters
        ----------
        source : np.ndarray, uint8, (H, W)
            Input SAR image (ideally speckle-filtered beforehand).
        reference : np.ndarray or None
            Reference pixel distribution.  If None the source is returned
            unchanged (fallback mode).

        Returns
        -------
        np.ndarray, uint8, (H, W)
            Tone-matched and residual-corrected image.
        """
        if reference is None:
            return source

        # Pass 1 — global CDF histogram matching
        p1 = self._cdf_match(source, reference).astype(np.float32)

        # Pass 2 — bilateral-filtered residual blend
        residual = source.astype(np.float32) - p1
        res_smoothed = cv2.bilateralFilter(
            residual, self.bd, self.bsc, self.bss,
        )
        corrected = p1 + self.alpha * res_smoothed
        return np.clip(corrected, 0, 255).astype(np.uint8)
