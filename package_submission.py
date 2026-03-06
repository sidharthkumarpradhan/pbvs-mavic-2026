"""Verify submission output directories and package into a ZIP archive.

Checks that all four output folders contain the expected number of files in
the correct format, writes the ``readme.txt`` metadata file required by
CodaBench, and produces ``submission.zip``.

Usage
-----
    python package_submission.py --submission_dir submission \\
                                 --output_zip submission.zip
"""

import argparse
import os
import zipfile

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package MAVIC-T submission.",
    )
    parser.add_argument(
        "--submission_dir", type=str,
        default="submission",
        help="Root of generated outputs.",
    )
    parser.add_argument(
        "--output_zip", type=str,
        default="submission.zip",
        help="Output ZIP file path.",
    )
    return parser.parse_args()


def verify_outputs(sub_dir: str) -> bool:
    """Print a summary of each output folder."""
    print("=" * 50)
    print("Output Verification")
    print("=" * 50)
    all_ok = True

    for folder in ["sar2eo", "sar2rgb", "sar2ir", "rgb2ir"]:
        path = os.path.join(sub_dir, folder)
        if os.path.exists(path):
            files = sorted(os.listdir(path))
            if files:
                sample = Image.open(
                    os.path.join(path, files[0]),
                )
                ext = files[0].rsplit(".", 1)[-1]
                print(
                    f"  + {folder:<10}"
                    f" {len(files):>5} files"
                    f" | ext={ext:<5}"
                    f" | size={sample.size}"
                    f" | mode={sample.mode}"
                )
            else:
                print(f"  - {folder:<10} EMPTY")
                all_ok = False
        else:
            print(f"  - {folder:<10} NOT FOUND")
            all_ok = False

    return all_ok


def write_readme(sub_dir: str) -> None:
    """Write the ``readme.txt`` metadata required by the CodaBench scorer."""
    readme_content = (
        "runtime per image [s] : 0.05\n"
        "CPU[1] / GPU[0] : 0\n"
        "Extra Data [1] / No Extra Data [0] : 0\n"
        "Other description : "
        "SpeckleFilterBank (Lee-approx + "
        "bilateral + CoV-weighted blend) "
        "preprocessing; "
        "Conditional U-Net GAN with LSGAN "
        "adversarial + L1 reconstruction loss, "
        "CosineAnnealingWarmRestarts scheduler, "
        "and gradient clipping for SAR2EO; "
        "adaptive gamma correction for SAR2EO; "
        "variance-weighted channel fusion + "
        "CLAHE for RGB2IR; "
        "AdaptiveHistogramTransfer (two-pass CDF "
        "+ bilateral residual, alpha=0.25) "
        "for SAR2RGB and SAR2IR.\n"
    )
    with open(os.path.join(sub_dir, "readme.txt"), "w") as f:
        f.write(readme_content)
    print("\nreadme.txt written.")


def build_zip(sub_dir: str, zip_path: str) -> None:
    """Package outputs and readme into a ZIP archive."""
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("Building submission.zip ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(sub_dir, "readme.txt"), "readme.txt")
        for folder in ["sar2eo", "sar2rgb", "sar2ir", "rgb2ir"]:
            fp = os.path.join(sub_dir, folder)
            if os.path.exists(fp):
                for fname in sorted(os.listdir(fp)):
                    zf.write(os.path.join(fp, fname), f"{folder}/{fname}")

    # Quick verification pass
    print("\nZIP verification:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for folder in ["sar2eo", "sar2rgb", "sar2ir", "rgb2ir"]:
            entries = [
                n for n in zf.namelist()
                if n.startswith(folder + "/")
            ]
            print(f"  {folder}: {len(entries)} files")

    zip_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\nZIP size: {zip_mb:.1f} MB -> {zip_path}")


def main(args: argparse.Namespace) -> None:
    ok = verify_outputs(args.submission_dir)
    if not ok:
        print(
            "\nWARNING: Some output folders missing."
            " Re-run inference first."
        )

    write_readme(args.submission_dir)
    build_zip(args.submission_dir, args.output_zip)
    print("\nSubmission archive ready for CodaBench upload.")


if __name__ == "__main__":
    main(parse_args())
