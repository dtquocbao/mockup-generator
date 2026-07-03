#!/usr/bin/env python3
"""
Enhance wrinkled product photos into marketing-ready apparel shots.

Uses SDXL img2img + ControlNet (Canny) by default so the garment silhouette,
color, and label stay locked while wrinkles are ironed out. Optional FLUX.1-dev
backend for higher fidelity when VRAM allows.

Modes (match Assets/enhanced samples):
  pro        — catalog look: white studio background, fully ironed fabric
  realistic  — natural look: soft light-gray background, mild fabric texture kept

Usage:
    python scripts/product_photo_enhance.py -i tee.png -m pro
    python scripts/product_photo_enhance.py -i tee.png -m realistic
    python scripts/product_photo_enhance.py -i tee.png -m both
    python scripts/product_photo_enhance.py -i ./folder --batch -m pro
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_INPUT = Path("Assets/OneDrive_1_7-1-2026/MCC-edited/MCC003-2_nobg.png")
DEFAULT_OUTPUT_DIR = Path("Assets/enhanced")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MODES = ("pro", "realistic", "both")
BACKENDS = ("sdxl", "flux")

NEGATIVE_PROMPT = (
    "wrinkles, creases, folds, crumpled fabric, messy fabric, "
    "harsh shadows, dirty, stains, low quality, blurry, distorted, "
    "warped shape, extra sleeves, extra collar, text artifacts, watermark"
)


@dataclass(frozen=True)
class ModePreset:
    name: str
    prompt: str
    background: tuple[int, int, int]
    strength: float
    guidance: float
    controlnet_scale: float
    steps: int


PRESETS: dict[str, ModePreset] = {
    "pro": ModePreset(
        name="pro",
        prompt=(
            "professional ecommerce apparel product photography, plain solid-color "
            "crew-neck t-shirt laid flat, perfectly ironed smooth fabric, no wrinkles, "
            "crisp collar and hem, clean studio lighting, pure white background, "
            "commercial catalog quality, sharp edges, accurate garment color"
        ),
        background=(255, 255, 255),
        strength=0.42,
        guidance=6.0,
        controlnet_scale=0.85,
        steps=30,
    ),
    "realistic": ModePreset(
        name="realistic",
        prompt=(
            "realistic apparel product photography, plain solid-color crew-neck "
            "t-shirt laid flat, gently pressed fabric with natural soft drape, "
            "subtle knit texture preserved, soft diffused studio lighting, "
            "light gray background, high-end product photo, natural and clean"
        ),
        background=(245, 245, 245),
        strength=0.32,
        guidance=5.5,
        controlnet_scale=0.90,
        steps=28,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance product photos with SDXL/FLUX img2img (pro or realistic)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input product photo or folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output file or folder (default: Assets/enhanced)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=MODES,
        default="both",
        help="Enhancement style: pro, realistic, or both (default: both)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="sdxl",
        help="Model backend: sdxl (default) or flux",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Override denoise strength (0.2–0.55 recommended)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Long-edge size for generation (default: 1024)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every image in the input folder",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (slow; GPU strongly recommended)",
    )
    return parser.parse_args()


def collect_images(input_path: Path, batch: bool) -> list[Path]:
    if batch or input_path.is_dir():
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input folder not found: {input_path}")
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No images found in folder: {input_path}")
        return images

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return [input_path]


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def composite_on_background(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    canvas = Image.new("RGB", image.size, background)
    canvas.paste(image, mask=image.split()[-1])
    return canvas


def resize_for_model(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_size:
        # SDXL prefers multiples of 8
        new_w = max(8, (width // 8) * 8)
        new_h = max(8, (height // 8) * 8)
        if (new_w, new_h) != (width, height):
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    scale = max_size / long_edge
    new_w = max(8, int(width * scale) // 8 * 8)
    new_h = max(8, int(height * scale) // 8 * 8)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def canny_control_image(image: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return Image.fromarray(edges).convert("RGB")


def restore_alpha(
    enhanced_rgb: Image.Image,
    original_rgba: Image.Image,
    background: tuple[int, int, int],
) -> Image.Image:
    """Keep a clean solid background while preserving the garment silhouette."""
    alpha = original_rgba.split()[-1].resize(enhanced_rgb.size, Image.Resampling.LANCZOS)
    # Slightly firm up soft alpha edges for a catalog cutout look
    alpha_np = np.array(alpha).astype(np.float32)
    alpha_np = np.clip((alpha_np - 20.0) * (255.0 / 215.0), 0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha_np, mode="L")

    canvas = Image.new("RGB", enhanced_rgb.size, background)
    canvas.paste(enhanced_rgb, mask=alpha)
    return canvas


def output_path_for(input_path: Path, output: Path, mode: str, batch: bool) -> Path:
    if batch or output.is_dir() or output.suffix == "":
        folder = output if output.suffix == "" or output.is_dir() or batch else output.parent
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{input_path.stem}-{mode}-enhanced.png"

    if mode != "both" and output.suffix:
        # Single mode with explicit file path
        return output.with_suffix(".png") if output.suffix.lower() != ".png" else output

    folder = output.parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{output.stem}-{mode}.png"


class Enhancer:
    def __init__(self, backend: str, use_cpu: bool = False) -> None:
        self.backend = backend
        self.device = "cpu" if use_cpu else self._pick_device()
        self.pipe = None

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required. Install AI deps with:\n"
                '  pip install -r requirements-ai.txt'
            ) from exc
        return "cpu"

    def load(self) -> None:
        if self.backend == "flux":
            self._load_flux()
        else:
            self._load_sdxl()

    def _load_sdxl(self) -> None:
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

        print("Loading SDXL + ControlNet (Canny)... first run downloads ~7 GB")
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=dtype,
        )
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self.pipe = pipe

    def _load_flux(self) -> None:
        import torch
        from diffusers import FluxImg2ImgPipeline

        print("Loading FLUX.1-dev... first run downloads ~24 GB (HF token may be required)")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        )
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        self.pipe = pipe

    def enhance(
        self,
        image_rgb: Image.Image,
        preset: ModePreset,
        seed: int,
        strength_override: float | None = None,
    ) -> Image.Image:
        import torch

        strength = strength_override if strength_override is not None else preset.strength
        strength = float(np.clip(strength, 0.15, 0.7))
        generator = torch.Generator(device="cpu").manual_seed(seed)

        if self.backend == "flux":
            result = self.pipe(
                prompt=preset.prompt,
                image=image_rgb,
                strength=strength,
                guidance_scale=preset.guidance,
                num_inference_steps=preset.steps,
                generator=generator,
            ).images[0]
        else:
            control = canny_control_image(image_rgb)
            result = self.pipe(
                prompt=preset.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image_rgb,
                control_image=control,
                strength=strength,
                guidance_scale=preset.guidance,
                controlnet_conditioning_scale=preset.controlnet_scale,
                num_inference_steps=preset.steps,
                generator=generator,
            ).images[0]
        return result.convert("RGB")


def enhance_file(
    enhancer: Enhancer,
    input_path: Path,
    output_path: Path,
    preset: ModePreset,
    seed: int,
    max_size: int,
    strength_override: float | None,
) -> Path:
    original = load_rgba(input_path)
    prepared = composite_on_background(original, preset.background)
    prepared = resize_for_model(prepared, max_size)

    print(
        f"  [{preset.name}] {prepared.size[0]}x{prepared.size[1]} "
        f"strength={strength_override if strength_override is not None else preset.strength}"
    )
    enhanced = enhancer.enhance(prepared, preset, seed, strength_override)

    # Upscale back toward original resolution for print-friendly output
    if enhanced.size != original.size:
        enhanced = enhanced.resize(original.size, Image.Resampling.LANCZOS)

    final = restore_alpha(enhanced, original, preset.background)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path, format="PNG")
    print(f"  Saved: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()

    try:
        images = collect_images(args.input, args.batch)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    modes = ["pro", "realistic"] if args.mode == "both" else [args.mode]
    batch_mode = args.batch or args.input.is_dir()

    try:
        enhancer = Enhancer(backend=args.backend, use_cpu=args.cpu)
        enhancer.load()
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        print(
            "\nInstall AI dependencies (GPU recommended):\n"
            "  pip install -r requirements-ai.txt\n"
            "For FLUX.1-dev you also need a Hugging Face token with model access:\n"
            "  huggingface-cli login",
            file=sys.stderr,
        )
        return 1

    for image_path in images:
        print(f"Enhancing: {image_path.name}")
        for mode in modes:
            preset = PRESETS[mode]
            if batch_mode or args.mode == "both":
                out = output_path_for(image_path, args.output, mode, batch=True)
            else:
                out = args.output
                if out.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    out = out / f"{image_path.stem}-{mode}-enhanced.png"
                elif out.suffix.lower() != ".png":
                    out = out.with_suffix(".png")

            enhance_file(
                enhancer,
                image_path,
                out,
                preset,
                args.seed,
                args.max_size,
                args.strength,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
