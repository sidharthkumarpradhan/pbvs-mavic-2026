"""Train the conditional U-Net GAN for SAR -> EO image translation.

Trains an 8-level encoder-decoder generator with LSGAN adversarial loss and
L1 reconstruction loss.  SpeckleFilterBank preprocessing is applied at data
load time, and CosineAnnealingWarmRestarts is used
for learning rate scheduling.

Usage
-----
    python train.py --sar_dir data/train/sar2eo/sar \\
                    --eo_dir  data/train/sar2eo/eo  \\
                    --epochs 5 --batch_size 16 --output_dir weights
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.mavic_dataset import MAVICDataset
from src.models.discriminator import SARToEODiscriminator
from src.models.generator import SARToEOGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SAR-to-EO conditional U-Net GAN."
    )
    parser.add_argument(
        "--sar_dir", type=str, required=True,
        help="SAR training images directory.",
    )
    parser.add_argument(
        "--eo_dir", type=str, required=True,
        help="EO training images directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="weights",
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Resize images to this square size.",
    )
    parser.add_argument(
        "--lr_g", type=float, default=2e-4,
        help="Generator learning rate.",
    )
    parser.add_argument(
        "--lr_d", type=float, default=1e-4,
        help="Discriminator learning rate.",
    )
    parser.add_argument(
        "--lambda_l1", type=float, default=100.0,
        help="L1 reconstruction weight.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--no_speckle", action="store_true",
        help="Disable SpeckleFilterBank.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build models
    generator = SARToEOGenerator().to(device)
    discriminator = SARToEODiscriminator().to(device)

    n_gen = sum(p.numel() for p in generator.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator     : {n_gen / 1e6:.1f}M parameters")
    print(f"Discriminator : {n_disc / 1e6:.3f}M parameters")

    # Data loader with optional speckle preprocessing
    dataset = MAVICDataset(
        sar_dir=args.sar_dir,
        target_dir=args.eo_dir,
        size=args.image_size,
        apply_speckle=not args.no_speckle,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimisers — asymmetric learning rates
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr_g, betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d, betas=(0.5, 0.999),
    )

    # Cosine annealing with warm restarts (T_0=2, constant period)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_g, T_0=2, T_mult=1,
    )
    sched_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_d, T_0=2, T_mult=1,
    )

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"L1 loss weight (lambda): {args.lambda_l1}")

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()

        for i, (sar, eo) in enumerate(train_loader):
            sar, eo = sar.to(device), eo.to(device)

            # --- Discriminator update ---
            opt_d.zero_grad()
            fake_eo = generator(sar).detach()
            real_pair = torch.cat([sar, eo], dim=1)
            fake_pair = torch.cat([sar, fake_eo], dim=1)
            real_pred = discriminator(real_pair)
            fake_pred = discriminator(fake_pair)

            loss_d = 0.5 * (
                F.mse_loss(real_pred, torch.ones_like(real_pred))
                + F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            )
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), max_norm=1.0,
            )
            opt_d.step()

            # --- Generator update ---
            opt_g.zero_grad()
            fake_eo = generator(sar)
            fake_pair = torch.cat([sar, fake_eo], dim=1)
            adv_pred = discriminator(fake_pair)

            loss_adv = F.mse_loss(adv_pred, torch.ones_like(adv_pred))
            loss_l1 = F.l1_loss(fake_eo, eo)
            loss_g = loss_adv + args.lambda_l1 * loss_l1

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), max_norm=1.0,
            )
            opt_g.step()

            if (i + 1) % 500 == 0:
                lr_now = sched_g.get_last_lr()[0]
                print(
                    f"  Ep {epoch + 1}/{args.epochs}"
                    f" [{i + 1}/{len(train_loader)}]"
                    f" G={loss_g.item():.4f}"
                    f" (adv={loss_adv.item():.3f},"
                    f" l1={loss_l1.item():.3f})"
                    f"  D={loss_d.item():.4f}"
                    f"  LR={lr_now:.2e}"
                )

        sched_g.step()
        sched_d.step()

        ckpt_path = os.path.join(
            args.output_dir, f"sar2eo_ep{epoch + 1}.pth",
        )
        torch.save(generator.state_dict(), ckpt_path)
        print(
            f">>> Epoch {epoch + 1} complete"
            f" — checkpoint saved to {ckpt_path}"
        )

    final_path = os.path.join(args.output_dir, "sar2eo_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\nTraining complete. Final weights saved to {final_path}")


if __name__ == "__main__":
    train(parse_args())
