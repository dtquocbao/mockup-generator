#!/usr/bin/env python3
"""
Batch-generate mockups for every artwork × tee combination.

Defaults:
  Artworks: Assets/extracted-artworks
  Tees:     Assets/extracted-tee
  Output:   Assets/mockups/{tee_name}/{artwork_name}.png (transparent)

Usage:
    python scripts/batch_mockups.py
    python scripts/batch_mockups.py --skip-existing
    python scripts/batch_mockups.py --opaque
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generate_mockup import (
    MockupSettings,
    collect_images,
    parse_background_setting,
    prepare_tee,
    render_mockup,
    safe_name,
)

DEFAULT_ARTWORKS = Path("Assets/extracted-artworks")
DEFAULT_TEES = Path("Assets/extracted-tee")
DEFAULT_OUTPUT = Path("Assets/mockups")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mockups for all artworks on all tees."
    )
    parser.add_argument(
        "--artworks",
        type=Path,
        default=DEFAULT_ARTWORKS,
        help="Folder of extracted artwork PNGs",
    )
    parser.add_argument(
        "--tees",
        type=Path,
        default=DEFAULT_TEES,
        help="Folder of tee photos (nobg PNGs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output root folder",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="auto",
        help="Background for processing: auto, transparent, or R,G,B (default: auto)",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Save opaque JPEG with contrasting background instead of transparent PNG",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=2500,
        help="Resize long edge for processing speed (default: 2500)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG output quality when using --opaque (default: 92)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations that already have an output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N mockups (0 = all, useful for testing)",
    )
    return parser.parse_args()


def output_path(output_root: Path, tee_path: Path, artwork_path: Path, opaque: bool) -> Path:
    tee_name = safe_name(tee_path)
    artwork_name = safe_name(artwork_path)
    suffix = ".jpg" if opaque else ".png"
    return output_root / tee_name / f"{artwork_name}{suffix}"


def output_exists(target: Path, opaque: bool) -> bool:
    if target.is_file():
        return True
    alternate = target.with_suffix(".jpg" if not opaque else ".png")
    return alternate.is_file()


def main() -> int:
    args = parse_args()

    try:
        artworks = collect_images(args.artworks)
        tees = collect_images(args.tees)
        background = parse_background_setting(args.background)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total = len(artworks) * len(tees)
    print(f"Artworks: {len(artworks)} | Tees: {len(tees)} | Combinations: {total}")
    print(f"Output mode: {'opaque JPEG' if args.opaque else 'transparent PNG'}")

    settings = MockupSettings(
        background=background,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
        auto_placement=True,
        use_blend_if=True,
        transparent_output=not args.opaque,
    )

    created = 0
    skipped = 0
    processed = 0

    for tee_index, tee_path in enumerate(tees, start=1):
        tee_name = safe_name(tee_path)
        print(f"[{tee_index}/{len(tees)}] Preparing tee: {tee_name}")
        prepared_tee = prepare_tee(tee_path, settings)
        placement = prepared_tee.placement
        print(
            f"  placement cx={placement[0]:.3f} cy={placement[1]:.3f} "
            f"w={placement[2]:.3f} h={placement[3]:.3f} | "
            f"bg={prepared_tee.composite_background}"
        )

        for artwork_path in artworks:
            if args.limit and processed >= args.limit:
                break

            target = output_path(args.output, tee_path, artwork_path, args.opaque)
            if args.skip_existing and output_exists(target, args.opaque):
                skipped += 1
                continue

            saved_path = render_mockup(prepared_tee, artwork_path, target, settings)
            created += 1
            processed += 1
            print(f"  Saved: {saved_path}")

        if args.limit and processed >= args.limit:
            break

    print(f"Done. Created: {created} | Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
