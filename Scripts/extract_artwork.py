#!/usr/bin/env python3
"""
Extract embroidered artwork from fabric photos for mockup use.

Uses rembg with models tuned for objects/products on textured backgrounds
(not portraits). Default model: bria-rmbg (BRIA RMBG 2.0).

Usage:
    python scripts/extract_artwork.py
    python scripts/extract_artwork.py -i "Assets/attachment/绣花/绣花" -o "./extracted" --batch
    python scripts/extract_artwork.py -i "photo.png" -o "artwork.png" -a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

DEFAULT_INPUT = Path("Assets/attachment/绣花/绣花")
DEFAULT_OUTPUT = Path("Assets/attachment/绣花/绣花/extracted")

# bria-rmbg: best for product/object photos on fabric (e-commerce trained)
# birefnet-general: strong fallback for varied embroidery styles
# birefnet-dis: dichotomous segmentation — good for salient object isolation
ARTWORK_MODELS = ("bria-rmbg", "birefnet-general", "birefnet-dis", "isnet-general-use")
DEFAULT_MODEL = "bria-rmbg"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embroidered artwork from fabric photos (transparent PNG)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input image or folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path or folder",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=ARTWORK_MODELS,
        default=DEFAULT_MODEL,
        help=f"Segmentation model (default: {DEFAULT_MODEL})",
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
        help="Refine edges with alpha matting (slower; better for thread/bead edges)",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Crop transparent padding around the extracted artwork",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Pixels of padding when --trim is used (default: 8)",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_extracted.png")


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


def trim_transparent(image: Image.Image, padding: int) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return image

    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    return image.crop((left, top, right, bottom))


def extract_artwork(
    input_path: Path,
    output_path: Path,
    session,
    alpha_matting: bool,
    trim: bool,
    padding: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as image:
        result = remove(
            image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=250,
            alpha_matting_background_threshold=8,
            alpha_matting_erode_size=6,
        )
        if trim:
            result = trim_transparent(result, padding)
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

    batch_mode = args.batch or args.input.is_dir()
    if len(images) == 1 and not batch_mode:
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
        if output_path is not None:
            target = output_path
        elif batch_mode:
            target = args.output / default_output_path(image_path).name
        else:
            target = args.output
        if target.suffix.lower() not in {".png"}:
            target = target.with_suffix(".png")

        extract_artwork(
            image_path,
            target,
            session,
            args.alpha_matting,
            args.trim,
            args.padding,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
