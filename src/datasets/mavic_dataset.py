import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.speckle_filter import SpeckleFilterBank


class MAVICDataset(Dataset):
    """Paired SAR / EO dataset for MAVIC-T training.

    Loads matched SAR and EO image pairs by filename intersection.  SAR images
    are optionally preprocessed with SpeckleFilterBank before normalisation.
    Both modalities are normalised to [-1, 1] to match the generator's Tanh
    output range.

    Parameters
    ----------
    sar_dir : str
        Path to SAR training images.
    target_dir : str
        Path to EO training images.
    size : int
        Resize both images to (size x size) pixels.
    apply_speckle : bool
        Apply SpeckleFilterBank to SAR images before normalisation.
    """

    def __init__(
        self,
        sar_dir: str,
        target_dir: str,
        size: int = 256,
        apply_speckle: bool = True,
    ):
        self.sar_dir = sar_dir
        self.target_dir = target_dir
        self.size = size
        self.apply_speckle = apply_speckle
        self.sfb = SpeckleFilterBank()

        if os.path.exists(sar_dir) and os.path.exists(target_dir):
            self.files = sorted(
                set(os.listdir(sar_dir)) & set(os.listdir(target_dir))
            )
            print(f"Dataset: {len(self.files)} matched SAR/EO pairs")
        else:
            print(
                "WARNING: data directories not found."
                " Empty dataset."
            )
            self.files = []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]

        # Load and resize SAR image (greyscale)
        sar = Image.open(os.path.join(self.sar_dir, fname)).convert("L")
        sar = sar.resize((self.size, self.size), Image.LANCZOS)
        sar_np = np.array(sar)

        # Apply speckle suppression before normalisation
        if self.apply_speckle:
            sar_np = self.sfb(sar_np)

        # Load and resize EO image (greyscale target)
        eo = Image.open(os.path.join(self.target_dir, fname)).convert("L")
        eo = eo.resize((self.size, self.size), Image.LANCZOS)

        # Normalise both to [-1, 1]
        sar_t = torch.from_numpy(sar_np).float().unsqueeze(0) / 127.5 - 1.0
        eo_arr = np.array(eo)
        eo_t = (
            torch.from_numpy(eo_arr).float().unsqueeze(0)
            / 127.5 - 1.0
        )
        return sar_t, eo_t
