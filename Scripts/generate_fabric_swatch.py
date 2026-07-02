#!/usr/bin/env python3
"""
Generate fabric swatch mockups — a cropped square of tee fabric with embroidery.

Like the reference samples in Assets/samples/: close-up fabric texture with
centered artwork, not the full garment.

Usage:
    python scripts/generate_fabric_swatch.py -t tee.png -a artwork.png -o swatch.png
    python scripts/generate_fabric_swatch.py --batch
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from generate_mockup import (
    MockupSettings,
    PreparedTee,
    collect_images,
    composite_print,
    load_artwork_rgba,
    prepare_tee,
    safe_name,
    save_image,
)

DEFAULT_TEE = Path("Assets/OneDrive_1_7-1-2026/MCC-edited/no-models/MCC001-2_nobg.png")
DEFAULT_ARTWORK = Path("Assets/attachment/绣花/绣花/extracted/3D Embroidery_extracted.png")
DEFAULT_OUTPUT = Path("Assets/fabric-swatches/sample.png")

DEFAULT_ARTWORKS = Path("Assets/attachment/绣花/绣花/extracted")
DEFAULT_TEES = Path("Assets/OneDrive_1_7-1-2026/MCC-edited/no-models")
DEFAULT_BATCH_OUTPUT = Path("Assets/fabric-swatches")

SWATCH_PLACEMENT = (0.50, 0.50, 0.72, 0.72)


@dataclass
class SwatchSettings:
    size: int = 600
    crop_scale: float = 1.8
    displacement: float = 8.0
    texture: float = 0.28
    opacity: float = 0.96
    blend: str = "auto"
    max_dimension: int = 2500
    jpeg_quality: int = 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fabric swatch mockups (cropped tee fabric + embroidery)."
    )
    parser.add_argument("-t", "--tee", type=Path, default=DEFAULT_TEE, help="Tee image")
    parser.add_argument("-a", "--artwork", type=Path, default=DEFAULT_ARTWORK, help="Artwork PNG")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
    parser.add_argument("--artworks", type=Path, default=DEFAULT_ARTWORKS, help="Artwork folder (batch)")
    parser.add_argument("--tees", type=Path, default=DEFAULT_TEES, help="Tee folder (batch)")
    parser.add_argument("--batch", action="store_true", help="Batch all artworks × all tees")
    parser.add_argument("--size", type=int, default=600, help="Output swatch size in pixels (default: 600)")
    parser.add_argument("--crop-scale", type=float, default=1.8, help="Source crop size multiplier (default: 1.8)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing outputs (batch)")
    parser.add_argument("--limit", type=int, default=0, help="Max mockups in batch mode (0 = all)")
    return parser.parse_args()


def swatch_settings_from_args(args: argparse.Namespace) -> SwatchSettings:
    return SwatchSettings(size=args.size, crop_scale=args.crop_scale)


def median_fabric_color(fabric_rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None and np.any(mask):
        pixels = fabric_rgb[mask]
    else:
        pixels = fabric_rgb.reshape(-1, 3)
    if len(pixels) == 0:
        return np.array([128, 128, 128], dtype=np.float32)
    return np.median(pixels, axis=0).astype(np.float32)


def extract_fabric_swatch(
    tee_rgb: np.ndarray,
    tee_alpha: np.ndarray | None,
    tee_region_mask: np.ndarray | None,
    placement: tuple[float, float, float, float],
    settings: SwatchSettings,
) -> np.ndarray:
    height, width = tee_rgb.shape[:2]
    cx = int(placement[0] * width)
    cy = int(placement[1] * height)

    base_size = int(min(width, height) * max(placement[2], placement[3]) * settings.crop_scale)
    crop_size = max(base_size, 120)
    half = crop_size // 2

    x0 = max(0, min(cx - half, width - crop_size))
    y0 = max(0, min(cy - half, height - crop_size))
    x1 = min(width, x0 + crop_size)
    y1 = min(height, y0 + crop_size)
    if x1 - x0 < crop_size:
        x0 = max(0, x1 - crop_size)
    if y1 - y0 < crop_size:
        y0 = max(0, y1 - crop_size)

    patch = tee_rgb[y0:y1, x0:x1].astype(np.float32)

    shirt_mask = None
    if tee_region_mask is not None:
        shirt_mask = tee_region_mask[y0:y1, x0:x1] > 0
    elif tee_alpha is not None:
        shirt_mask = tee_alpha[y0:y1, x0:x1] > 128

    if shirt_mask is not None:
        fill = median_fabric_color(patch.astype(np.uint8), shirt_mask)
        coverage = shirt_mask.astype(np.float32)[..., np.newaxis]
        patch = patch * coverage + fill * (1.0 - coverage)

    swatch = cv2.resize(
        patch.astype(np.uint8),
        (settings.size, settings.size),
        interpolation=cv2.INTER_AREA,
    )
    return swatch


def render_fabric_swatch(
    prepared_tee: PreparedTee,
    artwork_path: Path,
    output_path: Path,
    tee_settings: MockupSettings,
    swatch_settings: SwatchSettings,
) -> Path:
    swatch = extract_fabric_swatch(
        prepared_tee.tee_rgb,
        prepared_tee.tee_alpha,
        prepared_tee.tee_region_mask,
        prepared_tee.placement,
        swatch_settings,
    )
    artwork = load_artwork_rgba(artwork_path)
    result = composite_print(
        swatch,
        artwork,
        SWATCH_PLACEMENT,
        swatch_settings.displacement,
        swatch_settings.texture,
        swatch_settings.opacity,
        swatch_settings.blend,
        tee_has_alpha=False,
        use_blend_if=True,
    )
    opaque_settings = MockupSettings(
        transparent_output=False,
        jpeg_quality=swatch_settings.jpeg_quality,
    )
    return save_image(result, output_path, opaque_settings.jpeg_quality, transparent_output=False)


def batch_output_path(root: Path, tee_path: Path, artwork_path: Path) -> Path:
    return root / safe_name(tee_path) / f"{safe_name(artwork_path)}.png"


def run_batch(args: argparse.Namespace) -> int:
    try:
        artworks = collect_images(args.artworks)
        tees = collect_images(args.tees)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tee_settings = MockupSettings(
        background="auto",
        auto_placement=True,
        transparent_output=False,
        max_dimension=2500,
    )
    swatch_settings = swatch_settings_from_args(args)

    total = len(artworks) * len(tees)
    print(f"Artworks: {len(artworks)} | Tees: {len(tees)} | Swatches: {total} | Size: {swatch_settings.size}px")

    created = 0
    skipped = 0
    processed = 0

    for tee_index, tee_path in enumerate(tees, start=1):
        print(f"[{tee_index}/{len(tees)}] Tee: {safe_name(tee_path)}")
        prepared = prepare_tee(tee_path, tee_settings)
        print(f"  chest cx={prepared.placement[0]:.3f} cy={prepared.placement[1]:.3f}")

        for artwork_path in artworks:
            if args.limit and processed >= args.limit:
                break

            target = batch_output_path(args.output, tee_path, artwork_path)
            if args.skip_existing and target.is_file():
                skipped += 1
                continue

            saved = render_fabric_swatch(prepared, artwork_path, target, tee_settings, swatch_settings)
            created += 1
            processed += 1
            print(f"  Saved: {saved}")

        if args.limit and processed >= args.limit:
            break

    print(f"Done. Created: {created} | Skipped: {skipped}")
    return 0


def main() -> int:
    args = parse_args()
    swatch_settings = swatch_settings_from_args(args)

    if args.batch:
        args.output = DEFAULT_BATCH_OUTPUT if args.output == DEFAULT_OUTPUT else args.output
        return run_batch(args)

    if not args.tee.is_file():
        print(f"Error: Tee image not found: {args.tee}", file=sys.stderr)
        return 1
    if not args.artwork.is_file():
        print(f"Error: Artwork not found: {args.artwork}", file=sys.stderr)
        return 1

    tee_settings = MockupSettings(background="auto", auto_placement=True, transparent_output=False)
    prepared = prepare_tee(args.tee, tee_settings)
    saved = render_fabric_swatch(prepared, args.artwork, args.output, tee_settings, swatch_settings)
    print(f"Saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
