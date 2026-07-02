# Tee mockup generator

Image processing utilities for studio assets. Background removal runs locally with [rembg](https://github.com/danielgatis/rembg) and BiRefNet models - free, offline, and suitable for portraits with fine hair edges.

| Raw Tee | Raw Artwork | Mockup Result |
|:-------:|:-----------:|:-------------:|
| <img src="https://github.com/dtquocbao/mockup-generator/blob/main/Assets/extracted-tee/MCC001_nobg.png?raw=true" width="250"> | <img src="https://github.com/dtquocbao/mockup-generator/blob/main/Assets/extracted-artworks/Loop%20Embroidery_extracted.png?raw=true" width="250"> | <img src="https://github.com/dtquocbao/mockup-generator/blob/main/Assets/mockups/MCC001_nobg/Loop%20Embroidery.png?raw=true" width="250"> |

## Requirements

- Python 3.11+
- See `requirements.txt`

## Setup

```powershell
pip install -r requirements.txt
```

On first run, the selected model is downloaded once to `~/.u2net/` (~1 GB per model).

## Extract embroidery artwork

Remove fabric backgrounds from embroidered product photos so the artwork can be used in mockups. Uses **bria-rmbg** (BRIA RMBG 2.0) by default, trained on e-commerce/product imagery and strong on textured backgrounds.

```powershell
python scripts/extract_artwork.py --batch --trim
```

Process a single file:

```powershell
python scripts/extract_artwork.py -i "Assets/attachment/绣花/绣花/3D Embroidery.png" -o "extracted/3D Embroidery.png"
```

Batch a folder:

```powershell
python scripts/extract_artwork.py -i "./embroidery_photos" -o "./extracted" --batch --trim
```

Finer thread/bead edges (slower):

```powershell
python scripts/extract_artwork.py -i "photo.png" -o "artwork.png" -a --trim
```

### Extract options

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input image or folder |
| `-o`, `--output` | Output PNG path or folder |
| `-m`, `--model` | `bria-rmbg` (default), `birefnet-general`, `birefnet-dis`, or `isnet-general-use` |
| `--batch` | Process every image in the input folder |
| `-a`, `--alpha-matting` | Refine edges (use for frayed thread or bead outlines) |
| `--trim` | Crop transparent padding around the artwork |
| `--padding` | Padding pixels when trimming (default: 8) |

**Tips:** Low-contrast cases (e.g. white embroidery on white fabric) may need `-a` or try `-m birefnet-dis`. Outputs go to `Assets/attachment/绣花/绣花/extracted/` by default.

## Remove background (portraits)

Default input/output (MCC sample portrait):

```powershell
python scripts/remove_background.py
```

Custom paths:

```powershell
python scripts/remove_background.py -i "path/to/photo.jpg" -o "path/to/output.png"
```

Batch process a folder:

```powershell
python scripts/remove_background.py -i "./input_folder" -o "./output_folder" --batch
```

Sharper hair edges (slower):

```powershell
python scripts/remove_background.py -a
```

### Options

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input image or folder |
| `-o`, `--output` | Output PNG path or folder |
| `-m`, `--model` | `birefnet-portrait` (default), `birefnet-general`, or `u2net_human_seg` |
| `--batch` | Process all images in the input folder |
| `-a`, `--alpha-matting` | Refine edges with alpha matting |

Output is always PNG with a transparent background.

## Batch generate mockups

Generate every artwork × tee combination automatically (AI placement, tee cached per shirt):

```powershell
python scripts/batch_mockups.py
```

Defaults:
- Artworks: `Assets/attachment/绣花/绣花/extracted`
- Tees: `Assets/OneDrive_1_7-1-2026/MCC-edited/no-models`
- Output: `Assets/mockups/{tee_name}/{artwork_name}.png` (transparent PNG by default)

Resume a partial run:

```powershell
python scripts/batch_mockups.py --skip-existing
```

Test with a small sample:

```powershell
python scripts/batch_mockups.py --limit 4
```

Opaque JPEG with auto contrasting background:

```powershell
python scripts/batch_mockups.py --opaque
```

## Fabric swatch mockups

Cropped square of tee fabric with centered embroidery — like the close-up samples in `Assets/samples/` (not the full garment).

```powershell
python scripts/generate_fabric_swatch.py -t "tee.png" -a "artwork.png" -o "swatch.png"
```

Batch all artworks × all tees (600×600 px default):

```powershell
python scripts/generate_fabric_swatch.py --batch
```

Output: `Assets/fabric-swatches/{tee_name}/{artwork_name}.png`

## Generate t-shirt mockup

Composite artwork onto a blank tee photo with displacement mapping, fabric texture, and luminance-aware blending (standard print-mockup workflow).

```powershell
python scripts/generate_mockup.py -t "blank_tee.jpg" -a "artwork.png" -o "mockup.jpg"
```

Default tee template (`MCC001-2.JPG`):

```powershell
python scripts/generate_mockup.py -a "artwork.png"
```

Tune placement and realism:

```powershell
python scripts/generate_mockup.py -a "artwork.png" --placement 0.5 0.40 0.26 0.30 --displacement 12 --texture 0.22 --blend auto
```

### Mockup options

| Flag | Description |
|------|-------------|
| `-t`, `--tee` | Blank t-shirt photo |
| `-a`, `--artwork` | Artwork image (PNG with transparency recommended) |
| `-o`, `--output` | Output image path |
| `--placement` | Print area: center-x, center-y, width, height (0–1 fractions) |
| `--displacement` | Wrinkle displacement strength in pixels (default: 12) |
| `--texture` | Fabric texture amount 0–1 (default: 0.22) |
| `--opacity` | Artwork opacity 0–1 (default: 0.94) |
| `--blend` | `auto`, `multiply`, `overlay`, `soft_light`, or `normal` |
| `--auto-placement` | Detect tee with AI and center artwork (default: on) |
| `--no-auto-placement` | Use manual `--placement` values |
| `--background` | `auto` (contrasting), `transparent`, or `R,G,B` for processing |
| `--opaque` | Save opaque JPEG/PNG instead of transparent PNG |
| `--max-dimension` | Resize long edge for speed (0 = full resolution, default: 2500) |

**Tips:** Mockups save as **transparent PNG** by default so dark tees don't blend into the background. Use `--opaque` for JPEG with an auto-selected contrasting background. Override with `--background 240,240,240`.

## Project layout

```
Assets/                 # Source images and outputs
scripts/
  remove_background.py  # Portrait background removal
  extract_artwork.py    # Embroidery/product artwork extraction
  generate_mockup.py    # Single t-shirt mockup generator
  batch_mockups.py      # Batch all artworks × all tees
  generate_fabric_swatch.py  # Cropped fabric swatch mockups
requirements.txt
```
