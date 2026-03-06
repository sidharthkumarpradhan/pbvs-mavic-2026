import torch
import torch.nn as nn


class SARToEODiscriminator(nn.Module):
    """Lightweight two-layer convolutional discriminator.

    Used for adversarial training.

    Receives a 2-channel input (SAR concatenated with real or generated EO)
    and outputs a patch-level realness score.  Trained with LSGAN (MSE) loss
    for stable gradient feedback to the generator.

    Input shape  : (B, 2, H, W) — channel 0: SAR, channel 1: EO
    Output shape : (B, 1, H', W') — patch-level score map
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            # No sigmoid — LSGAN uses MSE directly on raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
