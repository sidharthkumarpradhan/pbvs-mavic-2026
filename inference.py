"""Run inference for all four MAVIC-T sub-tasks.

Sub-tasks
---------
1. SAR -> EO  : Conditional U-Net GAN + adaptive gamma correction
2. RGB -> IR  : Variance-weighted channel fusion + CLAHE
3. SAR -> RGB : AdaptiveHistogramTransfer with UC Davis reference stats
4. SAR -> IR  : AdaptiveHistogramTransfer with UC Davis reference stats

Usage
-----
    python inference.py --test_dir data/test \\
                        --weights  weights/sar2eo_final.pth \\
                        --output_dir submission \\
                        --uc_davis_dir data/uc_davis
"""

import argparse
import os

import numpy as np
import rasterio
import torch
from PIL import Image

from src.models.generator import SARToEOGenerator
from src.postprocessing.channel_fusion import (
    clahe_enhance,
    variance_weighted_fusion,
)
from src.postprocessing.gamma_correction import (
    adaptive_gamma_correct,
)
from src.postprocessing.histogram_transfer import (
    AdaptiveHistogramTransfer,
)
from src.preprocessing.speckle_filter import (
    SpeckleFilterBank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAVIC-T inference for all four sub-tasks.",
    )
    parser.add_argument(
        "--test_dir", type=str, required=True,
        help="Root of test data (sar2eo/ etc.).",
    )
    parser.add_argument(
        "--weights", type=str,
        default="weights/sar2eo_final.pth",
        help="Path to trained generator weights.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="submission",
        help="Where to write output images.",
    )
    parser.add_argument(
        "--uc_davis_dir", type=str, default=None,
        help="UC Davis reference imagery dir.",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Output image size.",
    )
    return parser.parse_args()


# ── Task 1: SAR -> EO ──────────────────────────

def run_sar2eo(
    test_dir: str,
    out_dir: str,
    generator: SARToEOGenerator,
    speckle_filter: SpeckleFilterBank,
    device: torch.device,
    image_size: int = 256,
) -> int:
    """Translate SAR images to EO via generator."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    print(f"Running SAR -> EO inference on {test_dir} ...")

    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".png"):
            continue

        raw = Image.open(
            os.path.join(test_dir, fname),
        ).convert("L")
        raw = raw.resize(
            (image_size, image_size), Image.LANCZOS,
        )
        raw_filtered = speckle_filter(np.array(raw))

        inp = (
            torch.from_numpy(raw_filtered)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            / 127.5
            - 1.0
        )

        with torch.no_grad():
            out = generator(inp.to(device)).cpu()

        out_np = (
            (out[0, 0].numpy() + 1.0) * 127.5
        ).clip(0, 255).astype(np.uint8)
        out_np = adaptive_gamma_correct(
            out_np, target_mean=128.0,
        )

        out_img = Image.fromarray(out_np).convert("RGB")
        out_img.save(os.path.join(out_dir, fname))
        count += 1
        if count % 500 == 0:
            print(f"  {count} images processed...")

    print(f"SAR -> EO complete: {count} files saved to {out_dir}")
    return count


# ── Task 2: RGB -> IR ──────────────────────────

def run_rgb2ir(test_dir: str, out_dir: str, image_size: int = 256) -> int:
    """Convert RGB to pseudo-IR via fusion + CLAHE."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    print(f"Processing RGB -> IR in {test_dir} ...")

    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue

        with rasterio.open(
            os.path.join(test_dir, fname),
        ) as src:
            img = src.read().astype(np.float64)

        valid = (
            (img[0] > 0) | (img[1] > 0) | (img[2] > 0)
            if img.shape[0] >= 3
            else img[0] > 0
        )

        gray = variance_weighted_fusion(img)

        # Suppress water/sky regions (high blue ratio + dark pixels)
        if img.shape[0] >= 3:
            blue_ratio = img[2] / (img[0] + img[1] + img[2] + 1e-8)
            suppress_mask = (blue_ratio > 0.4) & (gray < 100) & valid
            gray[suppress_mask] *= 0.5

        gray[~valid] = 0

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        gray_u8 = clahe_enhance(gray_u8, clip_limit=2.0, tile_grid=(8, 8))

        Image.fromarray(gray_u8).convert("RGB").resize(
            (image_size, image_size), Image.LANCZOS
        ).save(os.path.join(out_dir, fname))
        count += 1

    print(f"RGB -> IR complete: {count} files saved to {out_dir}")
    return count


# ── Reference collection ──────────────────────────────────────

def collect_uc_davis_references(uc_base: str):
    """Gather reference pixel statistics from UC Davis.

    Returns (ref_rgb, ref_ir) where ref_rgb is a
    list of three uint8 arrays (one per channel)
    and ref_ir is a single uint8 array.  Either may
    be None if the data is unavailable.
    """
    all_rgb = [[], [], []]
    all_ir = []

    print(
        "Collecting reference pixel statistics"
        f" from {uc_base} ..."
    )

    for loc in sorted(os.listdir(uc_base)):
        loc_path = os.path.join(uc_base, loc)
        if not os.path.isdir(loc_path):
            continue

        for f in os.listdir(loc_path):
            fp = os.path.join(loc_path, f)
            try:
                if "rgb" in f.lower() and f.endswith(
                    (".tiff", ".tif"),
                ):
                    with rasterio.open(fp) as s:
                        data = s.read()
                    if data.shape[0] >= 3:
                        for c in range(3):
                            v = data[c].flatten()
                            v = v[v > 0]
                            if len(v):
                                n = min(50000, len(v))
                                idx = np.random.choice(
                                    len(v), n,
                                    replace=False,
                                )
                                all_rgb[c].append(v[idx])

                elif (
                    "ir" in f.lower()
                    and "rgb" not in f.lower()
                    and f.endswith(
                        (".tiff", ".tif"),
                    )
                ):
                    with rasterio.open(fp) as s:
                        data = s.read()
                    v = data[0].flatten()
                    v = v[v > 0]
                    if len(v):
                        n = min(50000, len(v))
                        idx = np.random.choice(
                            len(v), n,
                            replace=False,
                        )
                        all_ir.append(v[idx])
            except Exception:
                continue

    ref_rgb = (
        [np.concatenate(ch).astype(np.uint8) for ch in all_rgb]
        if any(all_rgb[0])
        else None
    )
    ref_ir = (
        np.concatenate(all_ir).astype(np.uint8)
        if all_ir else None
    )

    if ref_rgb is not None:
        sizes = [len(c) for c in ref_rgb]
        print(f"Reference RGB: {sizes} pixels")
    if ref_ir is not None:
        print(f"Reference IR : {len(ref_ir)} pixels")

    return ref_rgb, ref_ir


# ── Task 3: SAR -> RGB ─────────────────────────

def run_sar2rgb(
    test_dir: str,
    out_dir: str,
    speckle_filter: SpeckleFilterBank,
    aht: AdaptiveHistogramTransfer,
    ref_rgb,
    image_size: int = 256,
) -> int:
    """Translate SAR to pseudo-RGB via histogram transfer."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    print(f"Processing SAR -> RGB in {test_dir} ...")

    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue

        with rasterio.open(os.path.join(test_dir, fname)) as s:
            sar = s.read()

        sg = (
            sar[0] if sar.shape[0] == 1 else np.mean(sar[:3], axis=0)
        ).astype(np.uint8)
        sg = speckle_filter(sg)

        if ref_rgb:
            rgb = np.stack([aht(sg, ref_rgb[c]) for c in range(3)], axis=-1)
        else:
            rgb = np.stack([sg, sg, sg], axis=-1)

        Image.fromarray(rgb).resize(
            (image_size, image_size), Image.LANCZOS
        ).save(os.path.join(out_dir, fname))
        count += 1

    print(f"SAR -> RGB complete: {count} files saved to {out_dir}")
    return count


# ── Task 4: SAR -> IR ──────────────────────────

def run_sar2ir(
    test_dir: str,
    out_dir: str,
    speckle_filter: SpeckleFilterBank,
    aht: AdaptiveHistogramTransfer,
    ref_ir,
    image_size: int = 256,
) -> int:
    """Translate SAR to pseudo-IR via histogram transfer."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    print(f"Processing SAR -> IR in {test_dir} ...")

    for fname in sorted(os.listdir(test_dir)):
        if not (fname.endswith(".tiff") or fname.endswith(".tif")):
            continue

        with rasterio.open(os.path.join(test_dir, fname)) as s:
            sar = s.read()

        sg = (
            sar[0] if sar.shape[0] == 1 else np.mean(sar[:3], axis=0)
        ).astype(np.uint8)
        sg = speckle_filter(sg)
        ir = aht(sg, ref_ir)

        Image.fromarray(ir).convert("RGB").resize(
            (image_size, image_size), Image.LANCZOS
        ).save(os.path.join(out_dir, fname))
        count += 1

    print(f"SAR -> IR complete: {count} files saved to {out_dir}")
    return count


# ── Main entry point ───────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    speckle_filter = SpeckleFilterBank(noise_var=0.05)
    aht = AdaptiveHistogramTransfer(alpha=0.25)

    # Load generator for SAR -> EO
    generator = SARToEOGenerator().to(device)
    if os.path.exists(args.weights):
        generator.load_state_dict(
            torch.load(args.weights, map_location=device, weights_only=True)
        )
        generator.eval()
        print(f"Checkpoint loaded: {args.weights}")
    else:
        print(
            f"WARNING: {args.weights} not found."
            " Using random weights."
        )
        generator.eval()

    # Collect UC Davis reference statistics (if provided)
    ref_rgb, ref_ir = None, None
    if args.uc_davis_dir and os.path.exists(args.uc_davis_dir):
        ref_rgb, ref_ir = collect_uc_davis_references(args.uc_davis_dir)
    else:
        print(
            "UC Davis reference not found —"
            " using channel replication fallback."
        )

    # Run all four sub-tasks
    sar2eo_dir = os.path.join(args.test_dir, "sar2eo")
    rgb2ir_dir = os.path.join(args.test_dir, "rgb2ir")
    sar2rgb_dir = os.path.join(args.test_dir, "sar2rgb")
    sar2ir_dir = os.path.join(args.test_dir, "sar2ir")

    eo_out = os.path.join(args.output_dir, "sar2eo")
    ir_out = os.path.join(args.output_dir, "rgb2ir")
    rgb_out = os.path.join(args.output_dir, "sar2rgb")
    sir_out = os.path.join(args.output_dir, "sar2ir")

    if os.path.isdir(sar2eo_dir):
        run_sar2eo(
            sar2eo_dir, eo_out, generator,
            speckle_filter, device, args.image_size,
        )

    if os.path.isdir(rgb2ir_dir):
        run_rgb2ir(
            rgb2ir_dir, ir_out, args.image_size,
        )

    if os.path.isdir(sar2rgb_dir):
        run_sar2rgb(
            sar2rgb_dir, rgb_out, speckle_filter,
            aht, ref_rgb, args.image_size,
        )

    if os.path.isdir(sar2ir_dir):
        run_sar2ir(
            sar2ir_dir, sir_out, speckle_filter,
            aht, ref_ir, args.image_size,
        )

    print("\nAll inference tasks complete.")


if __name__ == "__main__":
    main(parse_args())
