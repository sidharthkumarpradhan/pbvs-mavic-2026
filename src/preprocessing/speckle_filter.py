import numpy as np
import cv2


class SpeckleFilterBank:
    """Three-stage adaptive speckle suppression for SAR imagery.

    Combines a Lee-filter approximation (Stage 1) with an edge-preserving
    bilateral filter (Stage 2), blended by a per-pixel coefficient-of-variation
    (CoV) mask (Stage 3).  Applied to every SAR image before model input.

    Stage 1 — Lee-approximation using 7x7 local statistics:
        w = sigma^2 / (sigma^2 + eta^2 * mu^2)
        I_lee = mu + w * (I - mu)

    Stage 2 — Bilateral filter for edge-preserving spatial smoothing.

    Stage 3 — CoV-weighted blend:
        alpha = CoV / max(CoV)  in [0, 1]
        I_out = alpha * I_lee + (1 - alpha) * I_bilateral

    High-CoV (textured/edge) regions prefer the Lee output to preserve
    structure, while low-CoV (homogeneous) regions prefer bilateral output
    for stronger smoothing.

    Parameters
    ----------
    noise_var : float
        Assumed multiplicative noise variance for the Lee filter.
    bilateral_d : int
        Diameter of the bilateral filter neighbourhood.
    bilateral_sc : float
        Bilateral filter sigma in colour/intensity space.
    bilateral_ss : float
        Bilateral filter sigma in coordinate space.
    """

    def __init__(
        self,
        noise_var: float = 0.05,
        bilateral_d: int = 9,
        bilateral_sc: float = 75.0,
        bilateral_ss: float = 75.0,
    ):
        self.noise_var = noise_var
        self.bd = bilateral_d
        self.bsc = bilateral_sc
        self.bss = bilateral_ss

    def _lee_approx(self, img_f32: np.ndarray) -> np.ndarray:
        """Approximate the Lee filter using 7x7 local statistics.

        Adaptive weight w = sigma^2 / (sigma^2 + eta^2 * (mu^2 + eps)) controls
        smoothing strength: edges (high sigma^2) stay sharp while flat areas
        (low sigma^2) get smoothed more aggressively.
        """
        mu = cv2.blur(img_f32, (7, 7))
        mu2 = cv2.blur(img_f32 ** 2, (7, 7))
        var = np.maximum(mu2 - mu ** 2, 0.0)
        w = var / (var + self.noise_var * (mu ** 2 + 1e-8))
        return mu + w * (img_f32 - mu)

    def _bilateral(self, img_u8: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter.

        Smooths flat regions aggressively while keeping sharp boundaries
        between land-cover types intact.
        """
        return cv2.bilateralFilter(img_u8, self.bd, self.bsc, self.bss)

    def _cov_mask(self, img_f32: np.ndarray) -> np.ndarray:
        """Per-pixel coefficient of variation (CoV = sigma / mu), normalised
        to [0, 1].  High CoV means textured/edge — trust Lee.  Low CoV means
        flat — trust bilateral.
        """
        mu = cv2.blur(img_f32, (7, 7)) + 1e-8
        std = np.sqrt(
            np.maximum(cv2.blur(img_f32 ** 2, (7, 7)) - mu ** 2, 0.0)
        )
        cov = std / mu
        return np.clip(cov / (cov.max() + 1e-8), 0.0, 1.0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply three-stage speckle suppression to a single SAR image.

        Parameters
        ----------
        img : np.ndarray, uint8, shape (H, W)
            Greyscale SAR image with pixel values in [0, 255].

        Returns
        -------
        np.ndarray, uint8, shape (H, W)
            Speckle-suppressed image, same shape and dtype as input.
        """
        f32 = img.astype(np.float32) / 255.0
        lee = self._lee_approx(f32)
        bil = self._bilateral(img).astype(np.float32) / 255.0
        alpha = self._cov_mask(f32)
        blended = alpha * lee + (1.0 - alpha) * bil
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
