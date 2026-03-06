import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Single downsampling block in the U-Net encoder.

    Applies Conv2d (stride=2) followed by optional InstanceNorm and
    LeakyReLU(0.2).  The first encoder layer omits normalisation to
    let the network see raw input statistics.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    norm : bool
        Whether to apply InstanceNorm (disabled on the first encoder layer).
    """

    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=4, stride=2,
                padding=1, bias=False,
            )
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single upsampling block in the U-Net decoder.

    Applies ConvTranspose2d (stride=2) -> InstanceNorm -> ReLU -> optional
    Dropout.  The output is concatenated with the corresponding encoder
    skip feature map to preserve spatial detail.

    Parameters
    ----------
    in_ch : int
        Input channels (from previous decoder layer).
    out_ch : int
        Output channels before skip concatenation.
    dropout : float
        Dropout probability (0.5 on the first three decoder layers, else 0).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample x then concatenate with the encoder skip connection."""
        return torch.cat([self.block(x), skip], dim=1)


class SARToEOGenerator(nn.Module):
    """Eight-level conditional U-Net generator for SAR -> EO image translation.

    Takes a speckle-filtered single-channel SAR image and produces a
    single-channel greyscale EO prediction.  Skip connections from every
    encoder level are concatenated into the corresponding decoder level,
    preserving fine spatial details across the domain gap.

    Architecture (encoder -> bottleneck -> decoder):
        D1(1->64)  D2(64->128)  D3(128->256)  D4(256->512)
        D5(512->512)  D6(512->512)  D7(512->512)  D8(512->512) [bottleneck]
        U1(512->512)+D7  U2(1024->512)+D6  U3(1024->512)+D5
        U4(1024->512)+D4  U5(1024->256)+D3  U6(512->128)+D2
        U7(256->64)+D1  Out(128->out_ch) + Tanh

    Total parameters: ~54.4 M

    Parameters
    ----------
    in_ch : int
        Input channels (1 for greyscale SAR).
    out_ch : int
        Output channels (1 for greyscale EO).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()

        # Encoder path — eight downsampling blocks
        self.enc1 = EncoderBlock(in_ch, 64, norm=False)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 512)
        self.enc6 = EncoderBlock(512, 512)
        self.enc7 = EncoderBlock(512, 512)
        self.enc8 = EncoderBlock(512, 512, norm=False)  # bottleneck

        # Decoder path — seven upsampling blocks with skip connections
        self.dec1 = DecoderBlock(512, 512, dropout=0.5)
        self.dec2 = DecoderBlock(1024, 512, dropout=0.5)
        self.dec3 = DecoderBlock(1024, 512, dropout=0.5)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec5 = DecoderBlock(1024, 256)
        self.dec6 = DecoderBlock(512, 128)
        self.dec7 = DecoderBlock(256, 64)

        # Output head
        self.output_head = nn.Sequential(
            nn.ConvTranspose2d(
                128, out_ch,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — cache each output for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder — upsample and fuse with encoder skip features
        d = self.dec1(e8, e7)
        d = self.dec2(d, e6)
        d = self.dec3(d, e5)
        d = self.dec4(d, e4)
        d = self.dec5(d, e3)
        d = self.dec6(d, e2)
        d = self.dec7(d, e1)
        return self.output_head(d)
