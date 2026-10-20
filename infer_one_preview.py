
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from util.util import save_image, tensor2im


WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    preview_parser = argparse.ArgumentParser(add_help=False)
    preview_parser.add_argument("--input", required=True, type=str,
                                help="input .svs/.tif WSI or ordinary RGB image")
    preview_parser.add_argument("--output_dir", required=True, type=str,
                                help="directory to save preview images")
    preview_parser.add_argument("--patch_size", default=512, type=int,
                                help="patch size cropped from WSI")
    preview_parser.add_argument("--x", default=None, type=int,
                                help="level-0 WSI x coordinate. If omitted, auto-select a tissue patch")
    preview_parser.add_argument("--y", default=None, type=int,
                                help="level-0 WSI y coordinate. If omitted, auto-select a tissue patch")
    preview_parser.add_argument("--level", default=0, type=int,
                                help="OpenSlide level to read from; default 0")
    preview_parser.add_argument("--thumb_max", default=2048, type=int,
                                help="max thumbnail side used for auto tissue search")
    preview_parser.add_argument("--min_tissue_ratio", default=0.35, type=float,
                                help="minimum tissue ratio for auto-selected patch")
    preview_parser.add_argument("--max_tries", default=2000, type=int,
                                help="max candidates for auto tissue search")
    preview_parser.add_argument("--seed", default=1234, type=int,
                                help="seed for deterministic candidate order")

    preview_opt, remaining = preview_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    opt = TestOptions().parse()
    return preview_opt, opt


def _require_openslide():
    try:
        import openslide
        return openslide
    except Exception as exc:
        raise RuntimeError(
            "Reading WSI requires openslide-python and system OpenSlide. "
            "Install e.g. `conda install -c conda-forge openslide-python openslide` "
            "in the remote environment."
        ) from exc


def tissue_ratio_rgb(arr: np.ndarray) -> float:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = maxc - minc
    brightness = rgb.mean(axis=2)
    tissue = (brightness < 0.88) & (saturation > 0.05)
    return float(tissue.mean())


def auto_select_wsi_patch(
    slide,
    patch_size: int,
    level: int,
    thumb_max: int,
    min_tissue_ratio: float,
    max_tries: int,
    seed: int,
) -> Tuple[int, int, float]:
    w0, h0 = slide.dimensions
    if level < 0 or level >= len(slide.level_dimensions):
        raise ValueError(f"Invalid level {level}; slide has {len(slide.level_dimensions)} levels")
    down = float(slide.level_downsamples[level])
    read_w0 = int(round(patch_size * down))
    read_h0 = int(round(patch_size * down))
    if w0 < read_w0 or h0 < read_h0:
        raise ValueError(f"Patch size {patch_size} at level {level} exceeds slide size {slide.dimensions}")

    scale = min(thumb_max / max(w0, h0), 1.0)
    tw, th = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    thumb_np = np.asarray(thumb)

    step = max(8, int(round(patch_size * scale)))
    xs = list(range(0, max(1, tw - step), max(1, step // 2)))
    ys = list(range(0, max(1, th - step), max(1, step // 2)))
    candidates = [(x, y) for y in ys for x in xs]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    candidates = candidates[:max_tries]

    best = (0, 0, -1.0)
    for tx, ty in candidates:
        crop = thumb_np[ty:min(ty + step, th), tx:min(tx + step, tw), :]
        ratio = tissue_ratio_rgb(crop)
        if ratio > best[2]:
            best = (tx, ty, ratio)
        if ratio >= min_tissue_ratio:
            x0 = int(round(tx / scale))
            y0 = int(round(ty / scale))
            x0 = min(max(0, x0), max(0, w0 - read_w0))
            y0 = min(max(0, y0), max(0, h0 - read_h0))
            return x0, y0, ratio

    tx, ty, ratio = best
    x0 = int(round(tx / scale))
    y0 = int(round(ty / scale))
    x0 = min(max(0, x0), max(0, w0 - read_w0))
    y0 = min(max(0, y0), max(0, h0 - read_h0))
    return x0, y0, ratio


def load_patch(preview_opt) -> Tuple[Image.Image, Optional[Tuple[int, int]], Optional[float]]:
    input_path = Path(preview_opt.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    suffix = input_path.suffix.lower()

    if suffix in WSI_EXTS:
        openslide = _require_openslide()
        slide = openslide.OpenSlide(str(input_path))
        level = int(preview_opt.level)
        if preview_opt.x is not None and preview_opt.y is not None:
            x0, y0 = int(preview_opt.x), int(preview_opt.y)
            ratio = None
        else:
            x0, y0, ratio = auto_select_wsi_patch(
                slide,
                patch_size=preview_opt.patch_size,
                level=level,
                thumb_max=preview_opt.thumb_max,
                min_tissue_ratio=preview_opt.min_tissue_ratio,
                max_tries=preview_opt.max_tries,
                seed=preview_opt.seed,
            )
        patch = slide.read_region(
            (x0, y0),
            level,
            (preview_opt.patch_size, preview_opt.patch_size),
        ).convert("RGB")
        slide.close()
        return patch, (x0, y0), ratio

    if suffix in IMAGE_EXTS:
        img = Image.open(input_path).convert("RGB")
        if img.size != (preview_opt.patch_size, preview_opt.patch_size):
            img = img.resize((preview_opt.patch_size, preview_opt.patch_size), Image.BICUBIC)
        return img, None, None

    raise ValueError(f"Unsupported input extension: {suffix}")


def make_compare(input_patch: Image.Image, fake_patch: Image.Image, text: str) -> Image.Image:
    w, h = input_patch.size
    canvas = Image.new("RGB", (w * 2, h + 36), "white")
    canvas.paste(input_patch, (0, 36))
    canvas.paste(fake_patch, (w, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Input frozen patch", fill=(0, 0, 0))
    draw.text((w + 10, 10), "Generated FFPE-like patch", fill=(0, 0, 0))
    if text:
        draw.text((10, h + 18), text, fill=(0, 0, 0))
    return canvas


def main():
    preview_opt, opt = parse_args()
    output_dir = Path(preview_opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False

    input_patch, coord, ratio = load_patch(preview_opt)
    input_patch_path = output_dir / "input_patch.png"
    input_patch.save(input_patch_path)

    transform = get_transform(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    tensor = transform(input_patch).unsqueeze(0).to(model.device)
    model.real_A = tensor
    with torch.no_grad():
        model.inference()

    fake_np = tensor2im(model.fake_B)
    fake_path = output_dir / "fake_ffpe.png"
    save_image(fake_np, str(fake_path), aspect_ratio=1.0)
    fake_patch = Image.fromarray(fake_np).convert("RGB")

    info_parts = []
    if coord is not None:
        info_parts.append(f"x={coord[0]}, y={coord[1]}")
    if ratio is not None:
        info_parts.append(f"tissue_ratio={ratio:.3f}")
    compare = make_compare(input_patch, fake_patch, " | ".join(info_parts))
    compare_path = output_dir / "compare.png"
    compare.save(compare_path)

    print("Saved:")
    print(f"  input : {input_patch_path}")
    print(f"  fake  : {fake_path}")
    print(f"  compare: {compare_path}")
    if coord is not None:
        print(f"WSI coordinate level-0: x={coord[0]}, y={coord[1]}")
    if ratio is not None:
        print(f"Auto tissue ratio: {ratio:.3f}")


if __name__ == "__main__":
    main()
