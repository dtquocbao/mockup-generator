#!/usr/bin/env python3
"""
Generate realistic t-shirt mockups from a blank tee photo and artwork.

Techniques (standard print-mockup workflow):
  1. AI tee detection (rembg) to auto-center artwork on the shirt
  2. Displacement map from blurred shirt luminance (wrinkles follow fabric)
  3. Fabric-aware blend (multiply on light tees, overlay on dark)
  4. High-pass fabric texture reinjection
  5. Luminance-based blend-if so shadows/highlights show through the print

Usage:
    python scripts/generate_mockup.py -t blank_tee.jpg -a artwork.png -o mockup.jpg
    python scripts/generate_mockup.py --no-auto-placement --placement 0.5 0.40 0.26 0.30
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_TEE = Path("Assets/extracted-tee/MCC001-2.JPG")
DEFAULT_OUTPUT = Path("Assets/mockups/MCC001-2_mockup.png")
SEGMENTATION_MODEL = "birefnet-general"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

BLEND_MODES = ("auto", "multiply", "overlay", "soft_light", "normal")
DEFAULT_PLACEMENT = (0.50, 0.40, 0.26, 0.30)


@dataclass
class MockupSettings:
    background: str = "auto"
    placement: tuple[float, float, float, float] = DEFAULT_PLACEMENT
    displacement: float = 12.0
    texture: float = 0.22
    opacity: float = 0.94
    blend: str = "auto"
    max_dimension: int = 2500
    jpeg_quality: int = 92
    auto_placement: bool = True
    use_blend_if: bool = True
    transparent_output: bool = True


@dataclass
class PreparedTee:
    tee_rgb: np.ndarray
    tee_alpha: np.ndarray | None
    output_alpha: np.ndarray | None
    original_size: tuple[int, int]
    placement: tuple[float, float, float, float]
    tee_region_mask: np.ndarray | None
    tee_has_alpha: bool
    composite_background: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite artwork onto a blank tee photo as a realistic mockup."
    )
    parser.add_argument(
        "-t",
        "--tee",
        type=Path,
        default=DEFAULT_TEE,
        help="Blank t-shirt photo",
    )
    parser.add_argument(
        "-a",
        "--artwork",
        type=Path,
        required=False,
        help="Artwork image (PNG with transparency recommended)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path",
    )
    parser.add_argument(
        "--placement",
        type=float,
        nargs=4,
        metavar=("CX", "CY", "WIDTH", "HEIGHT"),
        default=(0.50, 0.40, 0.26, 0.30),
        help="Print area: center-x, center-y, width, height (0–1 fractions of tee image)",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        default=12.0,
        help="Displacement strength in pixels (default: 12)",
    )
    parser.add_argument(
        "--texture",
        type=float,
        default=0.22,
        help="Fabric texture reinjection amount 0–1 (default: 0.22)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.94,
        help="Artwork opacity 0–1 (default: 0.94)",
    )
    parser.add_argument(
        "--blend",
        choices=BLEND_MODES,
        default="auto",
        help="Blend mode (default: auto — overlay on dark fabric, multiply on light)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=2500,
        help="Resize long edge for processing speed (0 = full resolution, default: 2500)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG output quality (default: 92)",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="auto",
        help="Background for processing: auto (contrasting), transparent, or R,G,B (default: auto)",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Save opaque image with contrasting background instead of transparent PNG",
    )
    parser.add_argument(
        "--auto-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect tee with AI and center artwork (default: on)",
    )
    parser.add_argument(
        "--no-blend-if",
        action="store_true",
        help="Disable luminance-based edge fading",
    )
    return parser.parse_args()


def parse_background_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Background must be three comma-separated integers: R,G,B")
    return (int(np.clip(parts[0], 0, 255)), int(np.clip(parts[1], 0, 255)), int(np.clip(parts[2], 0, 255)))


def parse_background_setting(value: str) -> str | tuple[int, int, int]:
    lowered = value.strip().lower()
    if lowered in {"auto", "transparent"}:
        return lowered
    return parse_background_rgb(value)


def shirt_pixels(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is not None:
        mask = alpha > 128
        if np.any(mask):
            return rgb[mask]
    return rgb.reshape(-1, 3)


def contrasting_background(rgb: np.ndarray, alpha: np.ndarray | None) -> tuple[int, int, int]:
    pixels = shirt_pixels(rgb, alpha)
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    if float(np.mean(luminance)) < 128:
        return (248, 248, 248)
    return (32, 32, 32)


def resolve_composite_background(
    setting: str | tuple[int, int, int],
    rgb: np.ndarray,
    alpha: np.ndarray | None,
) -> tuple[int, int, int]:
    if setting in {"auto", "transparent"}:
        return contrasting_background(rgb, alpha)
    return setting


def load_tee_raw(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    image = Image.open(path)
    if image.mode == "RGBA":
        rgba = np.array(image)
        return rgba[:, :, :3], rgba[:, :, 3]
    return np.array(image.convert("RGB")), None


def composite_tee(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    background: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if alpha is not None:
        alpha_3d = alpha[..., np.newaxis].astype(np.float32) / 255.0
        bg = np.array(background, dtype=np.float32)
        composite = (rgb.astype(np.float32) * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)
        seg_rgb = composite_for_segmentation(rgb, alpha)
        return composite, alpha, seg_rgb
    return rgb, None, rgb


def load_tee_image(path: Path, background: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    rgb, alpha = load_tee_raw(path)
    return composite_tee(rgb, alpha, background)


def composite_for_segmentation(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Flatten cutouts onto white so the segmentation model sees a solid subject."""
    alpha_3d = alpha[..., np.newaxis].astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.float32)
    return (rgb.astype(np.float32) * alpha_3d + white * (1.0 - alpha_3d)).astype(np.uint8)


@lru_cache(maxsize=1)
def get_segmentation_session():
    from rembg import new_session

    return new_session(SEGMENTATION_MODEL)


def ai_foreground_mask(seg_rgb: np.ndarray) -> np.ndarray:
    from rembg import remove

    mask_image = remove(
        Image.fromarray(seg_rgb),
        session=get_segmentation_session(),
        only_mask=True,
    )
    mask = np.array(mask_image.convert("L"))
    return mask > 127


def clean_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned > 127


def refine_to_chest(mask: np.ndarray) -> np.ndarray:
    """Narrow a full-body mask to the chest print zone (skip head and lower torso)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return mask

    y0, y1 = int(ys.min()), int(ys.max())
    body_h = y1 - y0
    if body_h <= 0:
        return mask

    top = y0 + int(body_h * 0.18)
    bottom = y0 + int(body_h * 0.52)
    chest = np.zeros_like(mask)
    chest[top:bottom] = mask[top:bottom]
    return chest if np.any(chest) else mask


def is_portrait_subject(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return False
    height = ys.max() - ys.min()
    width = xs.max() - xs.min()
    return height > width * 1.05


def detect_tee_mask(seg_rgb: np.ndarray, tee_alpha: np.ndarray | None) -> np.ndarray:
    ai_mask = clean_mask(ai_foreground_mask(seg_rgb))

    if tee_alpha is not None:
        person_mask = tee_alpha > 128
        combined = ai_mask & person_mask
        mask = combined if np.count_nonzero(combined) > 0.01 * combined.size else person_mask
    else:
        mask = ai_mask

    if is_portrait_subject(mask):
        mask = refine_to_chest(mask)

    return clean_mask(mask)


def placement_from_tee_mask(mask: np.ndarray, image_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    if not np.any(mask):
        return DEFAULT_PLACEMENT

    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    shirt_w = x1 - x0
    shirt_h = y1 - y0
    cx = (x0 + x1) / 2 / width
    cy = (y0 + shirt_h * 0.42) / height
    w_frac = min(0.55, shirt_w / width * 0.50)
    h_frac = min(0.45, shirt_h / height * 0.42)
    return cx, cy, w_frac, h_frac



def load_artwork_rgba(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    return np.array(image)


def maybe_downscale_triple(
    tee_rgb: np.ndarray,
    tee_alpha: np.ndarray | None,
    seg_rgb: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    tee_rgb, tee_alpha = maybe_downscale_pair(tee_rgb, tee_alpha, max_dimension)
    seg_rgb = maybe_downscale(seg_rgb, max_dimension)
    return tee_rgb, tee_alpha, seg_rgb


def maybe_downscale_pair(
    tee_rgb: np.ndarray,
    tee_alpha: np.ndarray | None,
    max_dimension: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_dimension <= 0:
        return tee_rgb, tee_alpha

    height, width = tee_rgb.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_dimension:
        return tee_rgb, tee_alpha

    scale = max_dimension / long_edge
    new_size = (int(width * scale), int(height * scale))
    tee_rgb = cv2.resize(tee_rgb, new_size, interpolation=cv2.INTER_AREA)
    if tee_alpha is not None:
        tee_alpha = cv2.resize(tee_alpha, new_size, interpolation=cv2.INTER_AREA)
    return tee_rgb, tee_alpha


def maybe_downscale(image: np.ndarray, max_dimension: int) -> np.ndarray:
    if max_dimension <= 0:
        return image
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_dimension:
        return image
    scale = max_dimension / long_edge
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def rect_from_placement(
    image_shape: tuple[int, ...],
    placement: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    cx, cy, w_frac, h_frac = placement
    rect_w = max(1, int(width * w_frac))
    rect_h = max(1, int(height * h_frac))
    x0 = int(cx * width - rect_w / 2)
    y0 = int(cy * height - rect_h / 2)
    x0 = max(0, min(x0, width - rect_w))
    y0 = max(0, min(y0, height - rect_h))
    return x0, y0, rect_w, rect_h


def fit_artwork_to_rect(artwork: np.ndarray, rect_w: int, rect_h: int) -> np.ndarray:
    art_h, art_w = artwork.shape[:2]
    scale = min(rect_w / art_w, rect_h / art_h)
    new_w = max(1, int(art_w * scale))
    new_h = max(1, int(art_h * scale))
    resized = np.array(
        Image.fromarray(artwork).resize((new_w, new_h), Image.Resampling.LANCZOS)
    )

    canvas = np.zeros((rect_h, rect_w, 4), dtype=np.uint8)
    offset_x = (rect_w - new_w) // 2
    offset_y = (rect_h - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def displacement_map_from_fabric(fabric_bgr: np.ndarray, blur_sigma: float = 4.0) -> np.ndarray:
    gray = cv2.cvtColor(fabric_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return gray


def apply_displacement(
    image_bgra: np.ndarray,
    displacement: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 0:
        return image_bgra

    grad_y, grad_x = np.gradient(displacement)
    height, width = image_bgra.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = map_x + grad_x * strength
    map_y = map_y + grad_y * strength

    channels = cv2.split(image_bgra)
    warped = [cv2.remap(ch, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for ch in channels]
    return cv2.merge(warped)


def fabric_luminance(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def blend_if_mask(luminance: np.ndarray, shadow_floor: float = 0.04) -> np.ndarray:
    """Fade print only in the deepest fabric shadows — not across entire dark garments."""
    ramp_end = shadow_floor + 0.12
    return np.clip((luminance - shadow_floor) / max(ramp_end - shadow_floor, 1e-6), 0.0, 1.0)


def artwork_has_transparency(artwork_rgba: np.ndarray) -> bool:
    alpha = artwork_rgba[:, :, 3]
    return bool(np.any(alpha < 250))


def choose_blend_mode(
    fabric_rgb: np.ndarray,
    requested: str,
    artwork_rgba: np.ndarray,
    tee_has_alpha: bool,
) -> str:
    if requested != "auto":
        return requested
    if artwork_has_transparency(artwork_rgba) and (tee_has_alpha or np.mean(fabric_rgb) < 110):
        return "normal"
    return "overlay" if np.mean(fabric_rgb) < 110 else "multiply"


def multiply_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    return np.clip(base_f * overlay_f * 255.0, 0, 255).astype(np.uint8)


def overlay_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    mask = base_f < 0.5
    blended = np.where(mask, 2.0 * base_f * overlay_f, 1.0 - 2.0 * (1.0 - base_f) * (1.0 - overlay_f))
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def soft_light_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    blended = (1.0 - 2.0 * overlay_f) * base_f * base_f + 2.0 * overlay_f * base_f
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def blend_layers(
    fabric_rgb: np.ndarray,
    artwork_rgb: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "multiply":
        return multiply_blend(fabric_rgb, artwork_rgb)
    if mode == "overlay":
        return overlay_blend(fabric_rgb, artwork_rgb)
    if mode == "soft_light":
        return soft_light_blend(fabric_rgb, artwork_rgb)
    return artwork_rgb


def extract_fabric_texture(fabric_bgr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(fabric_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    texture = fabric_bgr.astype(np.float32) - blurred.astype(np.float32)
    return texture


def soften_alpha(alpha: np.ndarray, radius: float = 1.2) -> np.ndarray:
    if radius <= 0:
        return alpha
    k = max(3, int(radius * 2) | 1)
    return cv2.GaussianBlur(alpha, (k, k), sigmaX=radius, sigmaY=radius)


def composite_print(
    tee_rgb: np.ndarray,
    artwork_rgba: np.ndarray,
    placement: tuple[float, float, float, float],
    displacement_strength: float,
    texture_amount: float,
    opacity: float,
    blend_mode: str,
    tee_alpha: np.ndarray | None = None,
    tee_region_mask: np.ndarray | None = None,
    use_blend_if: bool = True,
    tee_has_alpha: bool = False,
) -> np.ndarray:
    result = tee_rgb.copy()
    x0, y0, rect_w, rect_h = rect_from_placement(tee_rgb.shape, placement)

    fabric_rgb = tee_rgb[y0 : y0 + rect_h, x0 : x0 + rect_w]
    fabric_bgr = cv2.cvtColor(fabric_rgb, cv2.COLOR_RGB2BGR)

    fitted = fit_artwork_to_rect(artwork_rgba, rect_w, rect_h)
    displacement = displacement_map_from_fabric(fabric_bgr)
    warped = apply_displacement(fitted, displacement, displacement_strength)

    art_rgb = warped[:, :, :3]
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    alpha = soften_alpha(alpha, radius=1.0)

    mode = choose_blend_mode(fabric_rgb, blend_mode, artwork_rgba, tee_has_alpha)
    blended = blend_layers(fabric_rgb, art_rgb, mode)

    lum_mask = np.ones(fabric_rgb.shape[:2], dtype=np.float32)
    if use_blend_if:
        lum_mask = blend_if_mask(fabric_luminance(fabric_rgb))

    shirt_mask = np.ones(fabric_rgb.shape[:2], dtype=np.float32)
    if tee_region_mask is not None:
        shirt_mask = (tee_region_mask[y0 : y0 + rect_h, x0 : x0 + rect_w] > 0).astype(np.float32)
    elif tee_alpha is not None:
        shirt_mask = (tee_alpha[y0 : y0 + rect_h, x0 : x0 + rect_w] > 0.05).astype(np.float32)

    effective_alpha = np.clip(alpha * opacity * lum_mask * shirt_mask, 0.0, 1.0)[..., np.newaxis]

    patch = (
        fabric_rgb.astype(np.float32) * (1.0 - effective_alpha)
        + blended.astype(np.float32) * effective_alpha
    )

    if texture_amount > 0:
        texture = extract_fabric_texture(fabric_bgr)
        texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        patch = np.clip(patch + texture_rgb * texture_amount * effective_alpha, 0, 255)

    result[y0 : y0 + rect_h, x0 : x0 + rect_w] = patch.astype(np.uint8)
    return result


def save_image(
    image_rgb: np.ndarray,
    path: Path,
    jpeg_quality: int,
    alpha: np.ndarray | None = None,
    transparent_output: bool = False,
) -> Path:
    path = Path(path)
    suffix = path.suffix.lower()
    use_transparent = alpha is not None and transparent_output

    if use_transparent and suffix not in {".png"}:
        path = path.with_suffix(".png")

    path.parent.mkdir(parents=True, exist_ok=True)

    if use_transparent:
        if alpha.shape[:2] != image_rgb.shape[:2]:
            alpha = cv2.resize(alpha, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgba = np.dstack([image_rgb, np.clip(alpha, 0, 255).astype(np.uint8)])
        Image.fromarray(rgba, mode="RGBA").save(path)
        return path

    image = Image.fromarray(image_rgb)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)
    return path


def prepare_tee(tee_path: Path, settings: MockupSettings) -> PreparedTee:
    raw_rgb, raw_alpha = load_tee_raw(tee_path)
    composite_background = resolve_composite_background(settings.background, raw_rgb, raw_alpha)
    tee, tee_alpha, seg_rgb = composite_tee(raw_rgb, raw_alpha, composite_background)
    tee_has_alpha = tee_alpha is not None
    output_alpha = raw_alpha.copy() if raw_alpha is not None else None
    original_size = tee.shape[:2][::-1]
    tee, tee_alpha, seg_rgb = maybe_downscale_triple(tee, tee_alpha, seg_rgb, settings.max_dimension)

    placement = settings.placement
    tee_region_mask = None
    if settings.auto_placement:
        tee_region_mask = detect_tee_mask(seg_rgb, tee_alpha)
        placement = placement_from_tee_mask(tee_region_mask, tee.shape)

    return PreparedTee(
        tee_rgb=tee,
        tee_alpha=tee_alpha,
        output_alpha=output_alpha,
        original_size=original_size,
        placement=placement,
        tee_region_mask=tee_region_mask,
        tee_has_alpha=tee_has_alpha,
        composite_background=composite_background,
    )


def render_mockup(
    prepared_tee: PreparedTee,
    artwork_path: Path,
    output_path: Path,
    settings: MockupSettings,
) -> Path:
    artwork = load_artwork_rgba(artwork_path)
    result = composite_print(
        prepared_tee.tee_rgb,
        artwork,
        prepared_tee.placement,
        settings.displacement,
        settings.texture,
        settings.opacity,
        settings.blend,
        tee_alpha=prepared_tee.tee_alpha,
        tee_region_mask=prepared_tee.tee_region_mask,
        use_blend_if=settings.use_blend_if,
        tee_has_alpha=prepared_tee.tee_has_alpha,
    )

    if settings.max_dimension > 0 and (result.shape[1], result.shape[0]) != prepared_tee.original_size:
        result = cv2.resize(result, prepared_tee.original_size, interpolation=cv2.INTER_LANCZOS4)

    output_alpha = None
    if settings.transparent_output and prepared_tee.output_alpha is not None:
        output_alpha = prepared_tee.output_alpha

    return save_image(
        result,
        output_path,
        settings.jpeg_quality,
        alpha=output_alpha,
        transparent_output=settings.transparent_output,
    )


def settings_from_args(args: argparse.Namespace) -> MockupSettings:
    return MockupSettings(
        background=parse_background_setting(args.background),
        placement=tuple(args.placement),
        displacement=args.displacement,
        texture=args.texture,
        opacity=args.opacity,
        blend=args.blend,
        max_dimension=args.max_dimension,
        jpeg_quality=args.jpeg_quality,
        auto_placement=args.auto_placement,
        use_blend_if=not args.no_blend_if,
        transparent_output=not args.opaque,
    )


def generate_single_mockup(tee_path: Path, artwork_path: Path, output_path: Path, settings: MockupSettings) -> tuple[PreparedTee, Path]:
    prepared = prepare_tee(tee_path, settings)
    saved_path = render_mockup(prepared, artwork_path, output_path, settings)
    return prepared, saved_path


def collect_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
        and "._" not in path.name
    )
    if not images:
        raise FileNotFoundError(f"No images found in folder: {folder}")
    return images


def safe_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_extracted"):
        stem = stem[: -len("_extracted")]
    return re.sub(r'[<>:"/\\|?*]', "_", stem)


def main() -> int:
    args = parse_args()

    if args.artwork is None:
        print("Error: --artwork is required.", file=sys.stderr)
        return 1
    if not args.tee.is_file():
        print(f"Error: Tee image not found: {args.tee}", file=sys.stderr)
        return 1
    if not args.artwork.is_file():
        print(f"Error: Artwork not found: {args.artwork}", file=sys.stderr)
        return 1

    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    prepared, saved_path = generate_single_mockup(args.tee, args.artwork, args.output, settings)
    if settings.auto_placement:
        placement = prepared.placement
        print(
            "AI tee placement: "
            f"cx={placement[0]:.3f} cy={placement[1]:.3f} "
            f"w={placement[2]:.3f} h={placement[3]:.3f}"
        )
    if prepared.tee_has_alpha:
        print(f"Composite background: {prepared.composite_background}")
    print(f"Saved: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
