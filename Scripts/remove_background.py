#!/usr/bin/env python3
"""
Remove image backgrounds using rembg (BiRefNet models).

Recommended library: rembg — free, offline, MIT-licensed, and supports
state-of-the-art BiRefNet models for high-quality portrait edges (hair, etc.).

Usage:
    python scripts/remove_background.py
    python scripts/remove_background.py -i photo.jpg -o photo_nobg.png
    python scripts/remove_background.py -i ./folder -o ./output --batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

DEFAULT_INPUT = Path("Assets/OneDrive_1_7-1-2026/MCC/MCC001.jpeg")
DEFAULT_OUTPUT = Path("Assets/OneDrive_1_7-1-2026/MCC/MCC001_nobg.png")

# birefnet-portrait: best for studio headshots / half-body portraits
# birefnet-general: strong all-purpose model when portrait model is not ideal
PORTRAIT_MODEL = "birefnet-portrait"
GENERAL_MODEL = "birefnet-general"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove image backgrounds with rembg (BiRefNet)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image or folder (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path or folder (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=[PORTRAIT_MODEL, GENERAL_MODEL, "u2net_human_seg"],
        default=PORTRAIT_MODEL,
        help=f"Segmentation model (default: {PORTRAIT_MODEL})",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every image in the input folder",
    )
    parser.add_argument(
        "-a",
        "--alpha-matting",
        action="store_true",
        help="Refine edges with alpha matting (slower, better hair edges)",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_nobg.png")


def collect_images(input_path: Path, batch: bool) -> list[Path]:
    if batch or input_path.is_dir():
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input folder not found: {input_path}")
        images = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No images found in folder: {input_path}")
        return images

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return [input_path]


def remove_background(
    input_path: Path,
    output_path: Path,
    session,
    alpha_matting: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        result = remove(
            image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        result.save(output_path, format="PNG")

    print(f"Saved: {output_path}")


def main() -> int:
    args = parse_args()

    try:
        images = collect_images(args.input, args.batch)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Model: {args.model} (first run downloads weights to ~/.u2net/)")
    session = new_session(args.model)

    if len(images) == 1 and not args.batch and not args.input.is_dir():
        output_path = args.output
        if output_path.suffix.lower() not in {".png"}:
            output_path = output_path.with_suffix(".png")
    elif args.output.suffix.lower() in SUPPORTED_EXTENSIONS:
        print(
            "Error: For batch mode, --output must be a folder, not a file.",
            file=sys.stderr,
        )
        return 1
    else:
        output_path = None

    for image_path in images:
        target = (
            output_path
            if output_path is not None
            else args.output / default_output_path(image_path).name
            if args.batch or args.input.is_dir()
            else args.output
        )
        if target.suffix.lower() not in {".png"}:
            target = target.with_suffix(".png")

        remove_background(image_path, target, session, args.alpha_matting)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
