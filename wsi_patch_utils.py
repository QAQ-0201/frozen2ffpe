
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from util.util import tensor2im

WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

@dataclass
class TissueMask:
    mask: np.ndarray
    downsample_x: float
    downsample_y: float
    image_size: Tuple[int, int]
    requested_mag: float
    effective_mag: float
    segmenter: str


def require_openslide():
    try:
        import openslide
        return openslide
    except Exception as exc:
        raise RuntimeError(
            "OpenSlide is required. Install on the remote server with: "
            "conda install -c conda-forge openslide-python openslide"
        ) from exc


def get_source_mag(slide, user_source_mag: float) -> float:
    if user_source_mag and user_source_mag > 0:
        return float(user_source_mag)
    props = slide.properties
    for key in (
        "openslide.objective-power",
        "aperio.AppMag",
        "hamamatsu.SourceLens",
        "tiff.ImageDescription",
    ):
        val = props.get(key)
        if val is None:
            continue
        if key == "tiff.ImageDescription":

            import re
            m = re.search(r"AppMag\s*=\s*([0-9.]+)", val)
            if m:
                return float(m.group(1))
            continue
        try:
            return float(str(val).strip())
        except ValueError:
            pass
    raise RuntimeError(
        "Cannot infer source magnification from WSI metadata. "
        "Please pass --source_mag 20 or --source_mag 40."
    )


def read_segmentation_image(slide, base_mag: float, requested_mag: float, max_dim: int) -> Tuple[Image.Image, float, float, float]:
    requested_downsample = max(float(base_mag) / float(requested_mag), 1.0)
    level = slide.get_best_level_for_downsample(requested_downsample)
    level_downsample = float(slide.level_downsamples[level])
    level_w, level_h = slide.level_dimensions[level]

    if max(level_w, level_h) > max_dim:
        scale = float(max_dim) / float(max(level_w, level_h))
        out_w = max(1, int(round(level_w * scale)))
        out_h = max(1, int(round(level_h * scale)))
    else:
        out_w, out_h = int(level_w), int(level_h)


    image = slide.get_thumbnail((out_w, out_h)).convert("RGB")
    downsample_x = slide.dimensions[0] / image.size[0]
    downsample_y = slide.dimensions[1] / image.size[1]
    effective_mag = float(base_mag) / ((downsample_x + downsample_y) / 2.0)
    return image, downsample_x, downsample_y, effective_mag


def segment_otsu(image: Image.Image, min_component_area: int = 256) -> np.ndarray:
    import cv2

    arr = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturation_blur = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, sat_mask = cv2.threshold(saturation_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_background = value > 245
    mask = (sat_mask > 0) & (~bright_background)

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            keep[labels == label] = 1
    return keep.astype(bool)


class HESTSegmenter:
    checkpoint_name = "deeplabv3_seg_v4.ckpt"
    repo_id = "MahmoodLab/hest-tissue-seg"

    def __init__(self, weights_path: Optional[Path], allow_download: bool, device: str, confidence_thresh: float) -> None:
        self.confidence_thresh = float(confidence_thresh)
        self.device = self._resolve_device(device)
        self.model, self.transform = self._build(weights_path, allow_download)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build(self, weights_path: Optional[Path], allow_download: bool):
        from torch import nn
        from torchvision import transforms
        from torchvision.models.segmentation import deeplabv3_resnet50

        if weights_path is None:
            if not allow_download:
                raise FileNotFoundError(
                    "HEST weights were not provided. Pass --hest_weights PATH, "
                    "or add --allow_download to fetch MahmoodLab/hest-tissue-seg."
                )
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id=self.repo_id,
                repo_type="model",
                allow_patterns=[self.checkpoint_name],
            )
            weights_path = Path(checkpoint_dir) / self.checkpoint_name

        if not weights_path.exists():
            raise FileNotFoundError(f"HEST checkpoint not found: {weights_path}")

        model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        raw_state = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.replace("model.", ""): value
            for key, value in raw_state.items()
            if "aux" not in key
        }
        model.load_state_dict(state_dict, strict=True)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return model, transform

    def predict_mask(
        self,
        image: Image.Image,
        batch_size: int,
        tile_size: int = 512,
        overlap: int = 64,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        import torch.nn.functional as F

        image = image.convert("RGB")
        width, height = image.size
        step = tile_size - overlap
        pad_w = max(tile_size, math.ceil(max(width - overlap, 1) / step) * step + overlap)
        pad_h = max(tile_size, math.ceil(max(height - overlap, 1) / step) * step + overlap)

        canvas = Image.new("RGB", (pad_w, pad_h), "white")
        canvas.paste(image, (0, 0))
        coords = [(x, y) for y in range(0, pad_h - tile_size + 1, step) for x in range(0, pad_w - tile_size + 1, step)]
        prob_sum = np.zeros((pad_h, pad_w), dtype=np.float32)
        count = np.zeros((pad_h, pad_w), dtype=np.uint16)

        done = 0
        with torch.inference_mode():
            for idx in range(0, len(coords), batch_size):
                batch_coords = coords[idx: idx + batch_size]
                tensors = []
                for x, y in batch_coords:
                    tile = canvas.crop((x, y, x + tile_size, y + tile_size))
                    tensors.append(self.transform(tile))
                batch = torch.stack(tensors).to(self.device)
                logits = self.model(batch)["out"]
                probs = F.softmax(logits, dim=1)[:, 1, :, :].detach().cpu().numpy()
                for (x, y), prob in zip(batch_coords, probs):
                    prob_sum[y: y + tile_size, x: x + tile_size] += prob
                    count[y: y + tile_size, x: x + tile_size] += 1
                done += len(batch_coords)
                if progress:
                    progress(done, len(coords))

        count[count == 0] = 1
        prob = prob_sum / count
        return (prob[:height, :width] > self.confidence_thresh)


def build_tissue_mask(
    slide,
    base_mag: float,
    segmenter_name: str,
    mask_mag: float,
    mask_max_dim: int,
    hest_weights: Optional[Path],
    allow_download: bool,
    device: str,
    confidence_thresh: float,
    batch_size: int,
    hest_segmenter: Optional["HESTSegmenter"] = None,
) -> TissueMask:
    image, downsample_x, downsample_y, effective_mag = read_segmentation_image(
        slide=slide,
        base_mag=base_mag,
        requested_mag=mask_mag,
        max_dim=mask_max_dim,
    )
    print(
        f"segmentation image: {image.size[0]}x{image.size[1]}, "
        f"requested {mask_mag:g}x, effective {effective_mag:.2f}x"
    )

    if segmenter_name == "hest":
        segmenter = hest_segmenter
        if segmenter is None:
            segmenter = HESTSegmenter(
                weights_path=hest_weights,
                allow_download=allow_download,
                device=device,
                confidence_thresh=confidence_thresh,
            )
        print(f"HEST device: {segmenter.device}")
        mask = segmenter.predict_mask(
            image=image,
            batch_size=batch_size,
            progress=lambda done, total: print(f"HEST tiles {done}/{total}") if done == total or done % 25 == 0 else None,
        )
    elif segmenter_name == "otsu":
        mask = segment_otsu(image)
    else:
        raise ValueError(f"Unknown segmenter: {segmenter_name}")

    return TissueMask(
        mask=mask,
        downsample_x=downsample_x,
        downsample_y=downsample_y,
        image_size=image.size,
        requested_mag=mask_mag,
        effective_mag=effective_mag,
        segmenter=segmenter_name,
    )


def tissue_fraction_from_mask(mask: TissueMask, x: int, y: int, span: int) -> float:
    raw_x0 = int(np.floor(x / mask.downsample_x))
    raw_y0 = int(np.floor(y / mask.downsample_y))
    raw_x1 = int(np.ceil((x + span) / mask.downsample_x))
    raw_y1 = int(np.ceil((y + span) / mask.downsample_y))
    full_area = max(1, raw_x1 - raw_x0) * max(1, raw_y1 - raw_y0)

    x0 = max(0, raw_x0)
    y0 = max(0, raw_y0)
    x1 = min(mask.mask.shape[1], raw_x1)
    y1 = min(mask.mask.shape[0], raw_y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask.mask[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0


    return float(np.count_nonzero(roi) / full_area)


def save_green_mask_overlay(slide, tissue_mask: TissueMask, overlay_path: Path, qc_max_dim: int) -> None:
    slide_w, slide_h = slide.dimensions
    scale = min(float(qc_max_dim) / slide_w, float(qc_max_dim) / slide_h, 1.0)
    qc_w = max(1, int(round(slide_w * scale)))
    qc_h = max(1, int(round(slide_h * scale)))
    image = slide.get_thumbnail((qc_w, qc_h)).convert("RGB")
    qc_w, qc_h = image.size

    mask = Image.fromarray((tissue_mask.mask.astype(np.uint8) * 255), mode="L")
    mask = mask.resize((qc_w, qc_h), Image.Resampling.NEAREST)
    green = Image.new("RGBA", (qc_w, qc_h), (0, 220, 80, 0))
    green.putalpha(mask.point(lambda value: 100 if value else 0))
    overlay = Image.alpha_composite(image.convert("RGBA"), green)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(overlay_path, "PNG", compress_level=1)


def save_selected_tiles_overlay(
    slide,
    rows: List[Dict[str, object]],
    overlay_path: Path,
    source_span: int,
    qc_max_dim: int,
    skipped_background_total: Optional[int] = None,
) -> None:
    from PIL import ImageDraw

    slide_w, slide_h = slide.dimensions
    scale = min(float(qc_max_dim) / slide_w, float(qc_max_dim) / slide_h, 1.0)
    qc_w = max(1, int(round(slide_w * scale)))
    qc_h = max(1, int(round(slide_h * scale)))
    image = slide.get_thumbnail((qc_w, qc_h)).convert("RGBA")
    qc_w, qc_h = image.size
    scale_x = qc_w / slide_w
    scale_y = qc_h / slide_h

    overlay = Image.new("RGBA", (qc_w, qc_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    selected = 0
    skipped = 0
    for row in rows:
        is_skipped = str(row.get("skipped_background", "False")).lower() == "true"
        if is_skipped:
            skipped += 1
            continue
        if "level0_x" not in row or "level0_y" not in row:
            continue
        x = int(row["level0_x"])
        y = int(row["level0_y"])
        x0 = int(round(x * scale_x))
        y0 = int(round(y * scale_y))
        x1 = int(round((x + source_span) * scale_x))
        y1 = int(round((y + source_span) * scale_y))
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 180), fill=(0, 255, 0, 35), width=1)
        selected += 1

    out = Image.alpha_composite(image, overlay)
    draw_out = ImageDraw.Draw(out)
    skipped_display = skipped if skipped_background_total is None else int(skipped_background_total)
    draw_out.text(
        (10, 10),
        f"selected tiles: {selected}; skipped background: {skipped_display}",
        fill=(255, 0, 0, 255),
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(overlay_path, "PNG", compress_level=1)


def tissue_ratio_rgb(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = maxc - minc
    brightness = arr.mean(axis=2)
    tissue = (brightness < 0.88) & (saturation > 0.05)
    return float(tissue.mean())


def read_target_mag_patch(
    slide,
    out_x: int,
    out_y: int,
    tile_size: int,
    source_to_target_downsample: float,
) -> Image.Image:
    src_x = int(round(out_x * source_to_target_downsample))
    src_y = int(round(out_y * source_to_target_downsample))
    src_w = int(round(tile_size * source_to_target_downsample))
    src_h = int(round(tile_size * source_to_target_downsample))
    src_w = max(1, src_w)
    src_h = max(1, src_h)

    rgba = slide.read_region((src_x, src_y), 0, (src_w, src_h)).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    bg.alpha_composite(rgba)
    patch = bg.convert("RGB")
    if patch.size != (tile_size, tile_size):
        patch = patch.resize((tile_size, tile_size), Image.BICUBIC)
    return patch


def save_tile(img: Image.Image, path: Path, tile_format: str, jpeg_quality: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tile_format == "jpg":
        img.save(path, quality=jpeg_quality, subsampling=0)
    else:
        img.save(path)


def progress_iter(items: Iterable, total: int):
    try:
        from tqdm import tqdm
        return tqdm(items, total=total, desc="Translating x20 tiles")
    except Exception:
        return items


def translate_batch(
    model,
    transform,
    patches: List[Image.Image],
    device,
    sample_seeds: Optional[List[int]] = None,
) -> List[Image.Image]:
    tensors = [transform(p).unsqueeze(0) for p in patches]
    batch = torch.cat(tensors, dim=0).to(device)
    model.real_A = batch
    with torch.no_grad():
        if sample_seeds is None:
            model.inference()
        else:
            model.inference(sample_seeds=sample_seeds)
    out = []
    fake = model.fake_B.detach()
    for i in range(fake.size(0)):
        out.append(Image.fromarray(tensor2im(fake[i:i + 1])).convert("RGB"))
    return out
