
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from wsi_patch_utils import (
    HESTSegmenter,
    WSI_EXTS,
    build_tissue_mask,
    require_openslide,
    save_green_mask_overlay,
    save_selected_tiles_overlay,
    tissue_fraction_from_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HEST-filtered x20/512 lossless PNG extraction for FID/KID"
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", default=0, type=int, help="0 processes every WSI")
    parser.add_argument("--target_mag", default=20.0, type=float)
    parser.add_argument("--tile_size", default=512, type=int)
    parser.add_argument("--overlap", default=0, type=int)
    parser.add_argument(
        "--source_mag",
        default=0.0,
        type=float,
        help="0 infers from MPP first; positive value forces every WSI to that magnification",
    )
    parser.add_argument("--hest_weights", required=True, type=Path)
    parser.add_argument("--hest_confidence", default=0.5, type=float)
    parser.add_argument("--hest_batch_size", default=8, type=int)
    parser.add_argument("--segment_device", default="auto", type=str)
    parser.add_argument("--mask_mag", default=10.0, type=float)
    parser.add_argument("--mask_max_dim", default=20000, type=int)
    parser.add_argument(
        "--tissue_threshold",
        default=0.20,
        type=float,
        help="patch is retained only when HEST tissue fraction is strictly greater than this value",
    )
    parser.add_argument("--qc_max_dim", default=4096, type=int)
    parser.add_argument("--png_compress", default=1, type=int, choices=range(10))
    parser.add_argument(
        "--save_workers",
        default=1,
        type=int,
        help="parallel WSI readers used to save selected PNG patches; 1 preserves serial behavior",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def float_property(properties, keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = properties.get(key)
        if value is None:
            continue
        try:
            parsed = float(str(value).strip().strip("'\""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def infer_source_mag(slide, forced_mag: float) -> Tuple[float, str, Optional[float], Optional[float]]:
    if forced_mag > 0:
        return float(forced_mag), "forced", None, None

    properties = slide.properties
    mpp_x = float_property(
        properties,
        ("openslide.mpp-x", "aperio.MPP", "hamamatsu.XResolution"),
    )
    mpp_y = float_property(
        properties,
        ("openslide.mpp-y", "aperio.MPP", "hamamatsu.YResolution"),
    )
    mpp_values = [value for value in (mpp_x, mpp_y) if value is not None]
    if mpp_values:
        mean_mpp = sum(mpp_values) / len(mpp_values)
        if 0.15 <= mean_mpp <= 0.80:

            return 10.0 / mean_mpp, "mpp", mpp_x, mpp_y

    objective = float_property(
        properties,
        ("openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"),
    )
    if objective is not None:
        return objective, "objective", mpp_x, mpp_y

    description = str(properties.get("tiff.ImageDescription", ""))
    match = re.search(r"AppMag\s*=\s*([0-9.]+)", description)
    if match:
        return float(match.group(1)), "description", mpp_x, mpp_y
    raise RuntimeError("Cannot infer source magnification from MPP or objective metadata")


def collect_slides(input_dir: Path, recursive: bool, limit: int) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    slides = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in WSI_EXTS)
    if limit > 0:
        slides = slides[:limit]
    return slides


def read_patch(slide, x: int, y: int, source_span: int, tile_size: int) -> Image.Image:
    rgba = slide.read_region((x, y), 0, (source_span, source_span)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    patch = white.convert("RGB")
    if patch.size != (tile_size, tile_size):
        patch = patch.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return patch


def valid_png(path: Path, tile_size: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            if image.size != (tile_size, tile_size) or image.format != "PNG":
                return False
            image.verify()
        return True
    except Exception:
        return False


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def completed_slide_metadata(slide_path: Path, args: argparse.Namespace):
    slide_stem = slide_path.stem
    metadata_path = args.output_dir / "metadata" / f"{slide_stem}.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = int(metadata["selected_tiles"])
        expected_rule = f"HEST tissue fraction > {float(args.tissue_threshold):g}"
        required_matches = (
            math.isclose(float(metadata["target_mag"]), float(args.target_mag)),
            int(metadata["tile_size"]) == int(args.tile_size),
            int(metadata["overlap"]) == int(args.overlap),
            metadata.get("selection_rule") == expected_rule,
            math.isclose(float(metadata["hest_confidence"]), float(args.hest_confidence)),
            math.isclose(float(metadata["mask_mag"]), float(args.mask_mag)),
            int(metadata["mask_max_dim"]) == int(args.mask_max_dim),
        )
        if not all(required_matches):
            return None

        patch_dir = args.output_dir / "patches"
        target_mag = float(args.target_mag)
        pattern = f"{slide_stem}__x*_y*__{target_mag:g}x_{int(args.tile_size)}.png"
        if sum(1 for _ in patch_dir.glob(pattern)) != selected:
            return None

        coords_path = args.output_dir / "coords" / f"{slide_stem}_patches.csv"
        green_mask = args.output_dir / "qc" / f"{slide_stem}_green_mask.png"
        selected_overlay = args.output_dir / "qc" / f"{slide_stem}_selected_tiles.png"
        if not (coords_path.is_file() and green_mask.is_file() and selected_overlay.is_file()):
            return None
        with coords_path.open("r", newline="", encoding="utf-8") as stream:
            if sum(1 for _ in stream) != selected + 1:
                return None
        return metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_patch_jobs(
    jobs,
    slide_path: Path,
    primary_slide,
    source_span: int,
    tile_size: int,
    png_compress: int,
    workers: int,
) -> None:
    if not jobs:
        return

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    def save_with_slide(worker_slide, job) -> None:
        patch_path, x, y = job
        patch = read_patch(worker_slide, x, y, source_span, tile_size)
        temporary = patch_path.with_name(patch_path.name + ".tmp")
        try:
            patch.save(temporary, "PNG", compress_level=png_compress)
            temporary.replace(patch_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    if workers == 1:
        iterator = tqdm(jobs, desc="Saving selected PNG patches") if tqdm else jobs
        for job in iterator:
            save_with_slide(primary_slide, job)
        return

    openslide = require_openslide()
    thread_state = threading.local()
    worker_slides = []
    worker_slides_lock = threading.Lock()

    def save_one(job) -> None:
        worker_slide = getattr(thread_state, "slide", None)
        if worker_slide is None:
            worker_slide = openslide.OpenSlide(str(slide_path))
            thread_state.slide = worker_slide
            with worker_slides_lock:
                worker_slides.append(worker_slide)
        save_with_slide(worker_slide, job)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(save_one, job) for job in jobs]
            iterator = as_completed(futures)
            if tqdm:
                iterator = tqdm(iterator, total=len(futures), desc=f"Saving PNG ({workers} workers)")
            for future in iterator:
                future.result()
    finally:
        for worker_slide in worker_slides:
            worker_slide.close()


def process_slide(slide_path: Path, args: argparse.Namespace, hest: HESTSegmenter) -> Dict[str, object]:
    if args.resume and not args.overwrite:
        completed = completed_slide_metadata(slide_path, args)
        if completed is not None:
            print(
                f"\n=== {slide_path.name} ===\n"
                f"resume: complete slide skipped ({int(completed['selected_tiles'])} patches)",
                flush=True,
            )
            return completed

    openslide = require_openslide()
    slide = openslide.OpenSlide(str(slide_path))
    try:
        slide_stem = slide_path.stem
        tile_size = int(args.tile_size)
        base_mag, mag_source, mpp_x, mpp_y = infer_source_mag(slide, float(args.source_mag))
        target_mag = float(args.target_mag)
        source_to_target = base_mag / target_mag
        source_span = max(1, int(round(tile_size * source_to_target)))
        source_step = max(1, int(round((tile_size - int(args.overlap)) * source_to_target)))
        if int(args.overlap) < 0 or int(args.overlap) >= tile_size:
            raise ValueError("--overlap must be in [0, tile_size)")

        width, height = slide.dimensions
        xs = range(0, max(0, width - source_span + 1), source_step)
        ys = range(0, max(0, height - source_span + 1), source_step)
        total_grid = len(xs) * len(ys)
        if total_grid == 0:
            raise RuntimeError(f"WSI is smaller than one target patch: {width}x{height}")

        print(
            f"\n=== {slide_path.name} ===\n"
            f"level0={width}x{height}; source={base_mag:.3f}x ({mag_source}); "
            f"target={target_mag:g}x; level0_span={source_span}; grid={len(xs)}x{len(ys)}",
            flush=True,
        )
        if mpp_x is not None or mpp_y is not None:
            print(f"MPP: x={mpp_x}, y={mpp_y}", flush=True)

        tissue_mask = build_tissue_mask(
            slide=slide,
            base_mag=base_mag,
            segmenter_name="hest",
            mask_mag=float(args.mask_mag),
            mask_max_dim=int(args.mask_max_dim),
            hest_weights=args.hest_weights,
            allow_download=False,
            device=str(args.segment_device),
            confidence_thresh=float(args.hest_confidence),
            batch_size=int(args.hest_batch_size),
            hest_segmenter=hest,
        )

        patch_dir = args.output_dir / "patches"
        qc_dir = args.output_dir / "qc"
        coord_dir = args.output_dir / "coords"
        metadata_dir = args.output_dir / "metadata"
        for directory in (patch_dir, qc_dir, coord_dir, metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if args.overwrite:
            for old_path in patch_dir.glob(f"{slide_stem}__x*_y*__{target_mag:g}x_{tile_size}.png"):
                old_path.unlink()

        green_mask = qc_dir / f"{slide_stem}_green_mask.png"
        selected_overlay = qc_dir / f"{slide_stem}_selected_tiles.png"
        save_green_mask_overlay(slide, tissue_mask, green_mask, int(args.qc_max_dim))

        rows: List[Dict[str, object]] = []
        save_jobs = []
        skipped = 0
        processed = 0
        try:
            from tqdm import tqdm

            iterator = tqdm(((x, y) for y in ys for x in xs), total=total_grid, desc=slide_stem)
        except Exception:
            iterator = ((x, y) for y in ys for x in xs)

        for x, y in iterator:
            tissue_fraction = tissue_fraction_from_mask(tissue_mask, x, y, source_span)

            if tissue_fraction <= float(args.tissue_threshold):
                skipped += 1
                continue
            patch_name = (
                f"{slide_stem}__x{x}_y{y}__{target_mag:g}x_{tile_size}.png"
            )
            patch_path = patch_dir / patch_name
            if not (args.resume and valid_png(patch_path, tile_size)):
                save_jobs.append((patch_path, x, y))
            rows.append(
                {
                    "slide": slide_path.name,
                    "patch_name": patch_name,
                    "level0_x": x,
                    "level0_y": y,
                    "level0_span": source_span,
                    "target_mag": target_mag,
                    "tile_size": tile_size,
                    "tissue_fraction": tissue_fraction,
                    "patch_path": str(patch_path),
                    "skipped_background": False,
                }
            )
            processed += 1

        print(
            f"selected={processed}; already_valid={processed - len(save_jobs)}; "
            f"to_save={len(save_jobs)}; save_workers={int(args.save_workers)}",
            flush=True,
        )
        save_patch_jobs(
            jobs=save_jobs,
            slide_path=slide_path,
            primary_slide=slide,
            source_span=source_span,
            tile_size=tile_size,
            png_compress=int(args.png_compress),
            workers=int(args.save_workers),
        )

        save_selected_tiles_overlay(
            slide=slide,
            rows=rows,
            overlay_path=selected_overlay,
            source_span=source_span,
            qc_max_dim=int(args.qc_max_dim),
            skipped_background_total=skipped,
        )

        csv_path = coord_dir / f"{slide_stem}_patches.csv"
        fieldnames = [
            "slide", "patch_name", "level0_x", "level0_y", "level0_span",
            "target_mag", "tile_size", "tissue_fraction", "patch_path",
            "skipped_background",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        metadata = {
            "slide": str(slide_path),
            "source_level0_size": [width, height],
            "source_mag": base_mag,
            "source_mag_detection": mag_source,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "target_mag": target_mag,
            "tile_size": tile_size,
            "overlap": int(args.overlap),
            "source_span": source_span,
            "total_grid_tiles": total_grid,
            "selected_tiles": len(rows),
            "selection_rule": f"HEST tissue fraction > {float(args.tissue_threshold):g}",
            "tissue_threshold": float(args.tissue_threshold),
            "hest_confidence": float(args.hest_confidence),
            "mask_mag": float(args.mask_mag),
            "effective_mask_mag": float(tissue_mask.effective_mag),
            "mask_max_dim": int(args.mask_max_dim),
            "png_lossless": True,
            "green_mask_png": str(green_mask),
            "selected_tiles_png": str(selected_overlay),
            "coords_csv": str(csv_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_json(metadata_dir / f"{slide_stem}.json", metadata)
        print(
            f"selected={len(rows)}; skipped={skipped}; patches={patch_dir}; "
            f"QC={selected_overlay}",
            flush=True,
        )
        return metadata
    finally:
        slide.close()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.hest_weights.is_file():
        raise FileNotFoundError(f"HEST checkpoint not found: {args.hest_weights}")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if not 0.0 <= args.tissue_threshold <= 1.0:
        raise ValueError("--tissue_threshold must be in [0, 1]")
    if args.tile_size <= 0 or args.target_mag <= 0 or args.mask_max_dim <= 0:
        raise ValueError("tile_size, target_mag, and mask_max_dim must be positive")

    slides = collect_slides(args.input_dir, args.recursive, args.limit)
    if not slides:
        raise RuntimeError(f"No supported WSI files found in {args.input_dir}")
    stems = [slide.stem for slide in slides]
    if len(set(stems)) != len(stems):
        raise RuntimeError("Duplicate WSI stems would cause flat patch-name collisions")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HEST once for all slides...", flush=True)
    hest = HESTSegmenter(
        weights_path=args.hest_weights,
        allow_download=False,
        device=args.segment_device,
        confidence_thresh=float(args.hest_confidence),
    )
    print(f"Slides={len(slides)}; HEST device={hest.device}", flush=True)

    summary: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for index, slide_path in enumerate(slides, 1):
        print(f"\n[{index}/{len(slides)}] {slide_path}", flush=True)
        try:
            summary.append(process_slide(slide_path, args, hest))
        except Exception as exc:
            failures.append({"slide": str(slide_path), "error": repr(exc)})
            print(f"FAILED: {slide_path}: {exc}", file=sys.stderr, flush=True)

    atomic_json(
        args.output_dir / "extraction_summary.json",
        {
            "slides": summary,
            "failures": failures,
            "total_selected_patches": sum(int(item["selected_tiles"]) for item in summary),
        },
    )
    print(
        f"\nFinished: success={len(summary)}/{len(slides)}, failures={len(failures)}, "
        f"patches={sum(int(item['selected_tiles']) for item in summary)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
