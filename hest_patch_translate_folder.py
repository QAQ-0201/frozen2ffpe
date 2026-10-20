
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform

from wsi_patch_utils import (
    HESTSegmenter,
    WSI_EXTS,
    build_tissue_mask,
    get_source_mag,
    progress_iter,
    read_target_mag_patch,
    require_openslide,
    save_green_mask_overlay,
    save_selected_tiles_overlay,
    save_tile,
    tissue_fraction_from_mask,
    translate_batch,
)


def parse_args():


    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--input_dir", required=True, type=str, help="folder containing input frozen WSIs")
    p.add_argument("--output_dir", required=True, type=str, help="patch-only workflow output directory")
    p.add_argument("--recursive", action="store_true", help="search input_dir recursively")
    p.add_argument("--limit", default=0, type=int, help="process at most N slides; 0 means all")
    p.add_argument("--overwrite", action="store_true", help="overwrite per-slide outputs if they exist")
    p.add_argument("--skip_existing", action="store_true", help="skip a slide if its metadata json already exists")
    p.add_argument(
        "--trust_legacy_existing",
        action="store_true",
        help="adopt pre-existing patch files made before run_config.json was introduced",
    )

    p.add_argument("--source_mag", default=40.0, type=float, help="source level-0 magnification; 0 = infer metadata")
    p.add_argument(
        "--allow_mag_mismatch",
        action="store_true",
        help="continue when --source_mag disagrees with readable WSI objective-power metadata",
    )
    p.add_argument("--target_mag", default=20.0, type=float, help="target patch magnification")
    p.add_argument("--tile_size", default=512, type=int, help="patch size at target magnification")
    p.add_argument("--model_batch_size", default=2, type=int, help="generator inference batch size")
    p.add_argument("--max_tiles", default=0, type=int, help="debug: process at most N selected tissue tiles per slide")

    p.add_argument("--segmenter", default="hest", choices=["hest", "otsu"], help="tissue segmentation backend")
    p.add_argument("--hest_weights", default="", type=str, help="path to HEST deeplabv3_seg_v4.ckpt")
    p.add_argument("--allow_download", action="store_true", help="allow HEST weight download if no local weight is passed")
    p.add_argument("--hest_confidence", default=0.5, type=float, help="HEST foreground confidence threshold")
    p.add_argument("--hest_batch_size", default=4, type=int, help="HEST segmentation batch size")
    p.add_argument("--segment_device", default="auto", type=str, help="HEST device: auto/cuda/cpu")
    p.add_argument("--mask_mag", default=10.0, type=float, help="magnification used to build tissue mask")
    p.add_argument("--mask_max_dim", default=20000, type=int, help="max side length of mask image")
    p.add_argument("--tissue_threshold", default=0.10, type=float, help="minimum tissue fraction for selecting a tile")
    p.add_argument("--qc_max_dim", default=4096, type=int, help="max side length of QC thumbnails")

    p.add_argument("--tile_format", default="png", choices=["png", "jpg"], help="patch image format")
    p.add_argument("--jpeg_quality", default=95, type=int, help="JPEG quality if --tile_format jpg")
    p.add_argument("--no_save_input_patches", action="store_true",
                   help="do not save original frozen valA patches; fake_B and CSV are still saved")
    p.add_argument("--seed", default=1234, type=int, help="seed for deterministic stochastic bridge sequence")

    workflow_opt, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    model_opt = TestOptions().parse()
    return workflow_opt, model_opt


def collect_slides(input_dir: Path, recursive: bool, limit: int) -> List[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    slides = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in WSI_EXTS)
    if limit > 0:
        slides = slides[:limit]
    return slides


def validate_workflow_args(args) -> None:
    positive_ints = {
        "tile_size": args.tile_size,
        "model_batch_size": args.model_batch_size,
        "hest_batch_size": args.hest_batch_size,
        "mask_max_dim": args.mask_max_dim,
        "qc_max_dim": args.qc_max_dim,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name} must be > 0, got {value}")
    if float(args.target_mag) <= 0:
        raise ValueError(f"--target_mag must be > 0, got {args.target_mag}")
    if float(args.mask_mag) <= 0:
        raise ValueError(f"--mask_mag must be > 0, got {args.mask_mag}")
    if float(args.source_mag) < 0:
        raise ValueError(f"--source_mag must be 0 (infer) or > 0, got {args.source_mag}")
    if int(args.limit) < 0 or int(args.max_tiles) < 0:
        raise ValueError("--limit and --max_tiles must be >= 0")
    if not 0.0 <= float(args.tissue_threshold) <= 1.0:
        raise ValueError("--tissue_threshold must be in [0, 1]")
    if not 0.0 <= float(args.hest_confidence) <= 1.0:
        raise ValueError("--hest_confidence must be in [0, 1]")
    if args.overwrite and args.skip_existing:
        raise ValueError("--overwrite and --skip_existing are mutually exclusive")


def assert_unique_slide_stems(slides: List[Path]) -> None:
    groups: Dict[str, List[Path]] = {}
    for slide in slides:
        groups.setdefault(safe_slide_stem(slide), []).append(slide)
    duplicates = {stem: paths for stem, paths in groups.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{stem}: {[str(path) for path in paths]}"
            for stem, paths in sorted(duplicates.items())
        )
        raise RuntimeError(
            "Multiple input slides would share the same output subfolder. "
            f"Rename them or separate the runs. Collisions: {details}"
        )


def safe_slide_stem(slide: Path) -> str:


    return slide.stem.replace("/", "_").replace("\\", "_").replace(":", "_")


def stable_patch_seed(base_seed: int, slide_identity: str, patch_identity: str) -> int:
    payload = f"{int(base_seed)}\0{slide_identity}\0{patch_identity}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % ((1 << 63) - 1)


def valid_patch_image(path: Path, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            if image.size != (expected_size, expected_size):
                return False
            image.verify()
        return True
    except Exception:
        return False


def file_fingerprint(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve(strict=False)
    result: Dict[str, object] = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        result.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    else:
        result["missing"] = True
    return result


def write_json_atomic(path: Path, value) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def patch_name(
    slide_stem: str,
    ix: int,
    iy: int,
    x20_x: int,
    x20_y: int,
    level0_x: int,
    level0_y: int,
    target_mag: float,
    tile_size: int,
    ext: str,
) -> str:
    mag = f"{target_mag:g}x"
    return (
        f"{slide_stem}__tile_y{iy:06d}_x{ix:06d}"
        f"__x20_x{x20_x}_y{x20_y}"
        f"__level0_x{level0_x}_y{level0_y}"
        f"__{mag}_{tile_size}{ext}"
    )


def reset_slide_outputs(paths: List[Path]) -> None:
    for path in paths:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def build_run_config(
    slide_path: Path,
    slide,
    args,
    opt,
    source_mag: float,
    metadata_mag: Optional[float],
) -> Dict[str, object]:
    checkpoint_path = (
        Path(opt.checkpoints_dir) / str(opt.name) / f"{opt.epoch}_net_G.pth"
    )
    config: Dict[str, object] = {
        "schema": "hest_patch_translate_v2",
        "input": file_fingerprint(slide_path),
        "source_level0_size": list(slide.dimensions),
        "source_mag": float(source_mag),
        "metadata_mag": float(metadata_mag) if metadata_mag is not None else None,
        "target_mag": float(args.target_mag),
        "tile_size": int(args.tile_size),
        "max_tiles": int(args.max_tiles),
        "segmenter": str(args.segmenter),
        "mask_mag": float(args.mask_mag),
        "mask_max_dim": int(args.mask_max_dim),
        "hest_confidence": float(args.hest_confidence),
        "tissue_threshold": float(args.tissue_threshold),
        "qc_max_dim": int(args.qc_max_dim),
        "tile_format": str(args.tile_format),
        "jpeg_quality": int(args.jpeg_quality) if str(args.tile_format) == "jpg" else None,
        "save_input_patches": not bool(args.no_save_input_patches),
        "seed": int(args.seed),
        "model": {
            "name": str(opt.name),
            "epoch": str(opt.epoch),
            "checkpoint": file_fingerprint(checkpoint_path),
            "netG": str(opt.netG),
            "ngf": int(opt.ngf),
            "n_blocks": int(opt.n_blocks),
            "num_timesteps": int(opt.num_timesteps),
            "tau": float(opt.tau),
            "preprocess": str(opt.preprocess),
            "use_sab": bool(getattr(opt, "use_sab", False)),
        },
    }
    if str(args.segmenter) == "hest" and args.hest_weights:
        config["hest_checkpoint"] = file_fingerprint(Path(args.hest_weights))
    return config


def prepare_run_config(
    run_config_path: Path,
    expected_config: Dict[str, object],
    valA_dir: Path,
    fakeB_dir: Path,
    trust_legacy_existing: bool,
) -> None:
    if run_config_path.exists():
        try:
            with open(run_config_path, "r", encoding="utf-8") as handle:
                existing_config = json.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read existing run config {run_config_path}: {exc}. "
                "Use --overwrite or a new output directory."
            ) from exc
        if existing_config != expected_config:
            raise RuntimeError(
                f"Existing outputs use a different input/model/config: {run_config_path}. "
                "Use --overwrite or a new output directory."
            )
        return

    has_legacy_outputs = any(valA_dir.glob("*")) or any(fakeB_dir.glob("*"))
    if has_legacy_outputs and not trust_legacy_existing:
        raise RuntimeError(
            "Existing patch files have no run_config.json, so their model/config cannot be verified. "
            "Use --overwrite, a new output directory, or --trust_legacy_existing to adopt them."
        )
    if has_legacy_outputs:
        print(f"[warning] adopting legacy existing outputs under {run_config_path.parent}")
    write_json_atomic(run_config_path, expected_config)


def completed_slide_outputs_are_valid(
    metadata_path: Path,
    valA_dir: Path,
    fakeB_dir: Path,
    tile_ext: str,
    tile_size: int,
    save_input: bool,
) -> bool:
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        expected = int(metadata["selected_tiles"])
    except Exception:
        return False

    def collect(directory: Path) -> List[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == tile_ext
        )

    fake_files = collect(fakeB_dir)
    if len(fake_files) != expected:
        return False
    if not all(valid_patch_image(path, tile_size) for path in fake_files):
        return False
    if save_input:
        input_files = collect(valA_dir)
        if len(input_files) != expected:
            return False
        if not all(valid_patch_image(path, tile_size) for path in input_files):
            return False
    return True


def process_slide(
    slide_path: Path,
    args,
    opt,
    model,
    transform,
    hest_segmenter=None,
) -> Dict[str, object]:
    openslide = require_openslide()
    slide = openslide.OpenSlide(str(slide_path))
    try:
        return _process_open_slide(
            slide_path,
            slide,
            args,
            opt,
            model,
            transform,
            hest_segmenter,
        )
    finally:
        slide.close()


def _process_open_slide(
    slide_path: Path,
    slide,
    args,
    opt,
    model,
    transform,
    hest_segmenter=None,
) -> Dict[str, object]:

    slide_stem = safe_slide_stem(slide_path)
    output_root = Path(args.output_dir)
    slide_dir = output_root / slide_stem
    valA_dir = slide_dir / "valA"
    fakeB_dir = slide_dir / "fake_B"
    qc_dir = slide_dir / "qc"
    coord_dir = slide_dir
    meta_dir = slide_dir
    log_marker = meta_dir / "metadata.json"
    run_config_path = meta_dir / "run_config.json"

    if args.overwrite:
        reset_slide_outputs([
            valA_dir,
            fakeB_dir,
            coord_dir / "tiles.csv",
            qc_dir / "green_mask.png",
            qc_dir / "selected_tiles.png",
            log_marker,
            run_config_path,
        ])

    valA_dir.mkdir(parents=True, exist_ok=True)
    fakeB_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    coord_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    source_mag = get_source_mag(slide, float(args.source_mag))
    metadata_mag: Optional[float] = None
    if float(args.source_mag) > 0:
        try:
            metadata_mag = get_source_mag(slide, 0.0)
        except RuntimeError:
            metadata_mag = None
        if metadata_mag is not None and not math.isclose(
            source_mag,
            metadata_mag,
            rel_tol=0.05,
            abs_tol=0.5,
        ):
            message = (
                f"--source_mag={source_mag:g} conflicts with WSI metadata x{metadata_mag:g} "
                f"for {slide_path.name}"
            )
            if not args.allow_mag_mismatch:
                raise RuntimeError(message + "; correct --source_mag or add --allow_mag_mismatch")
            print(f"[warning] {message}; using explicit x{source_mag:g}")
    target_mag = float(args.target_mag)
    source_to_target_downsample = source_mag / target_mag
    if source_to_target_downsample <= 0:
        raise ValueError(f"Invalid magnification ratio: source={source_mag}, target={target_mag}")

    tile_size = int(args.tile_size)
    source_span = int(round(tile_size * source_to_target_downsample))
    w0, h0 = slide.dimensions
    out_w = int(math.ceil(w0 / source_to_target_downsample))
    out_h = int(math.ceil(h0 / source_to_target_downsample))
    nx = int(math.ceil(out_w / tile_size))
    ny = int(math.ceil(out_h / tile_size))
    total_grid_tiles = nx * ny
    tile_ext = ".jpg" if args.tile_format == "jpg" else ".png"
    save_input = not bool(args.no_save_input_patches)

    run_config = build_run_config(slide_path, slide, args, opt, source_mag, metadata_mag)
    prepare_run_config(
        run_config_path=run_config_path,
        expected_config=run_config,
        valA_dir=valA_dir,
        fakeB_dir=fakeB_dir,
        trust_legacy_existing=bool(args.trust_legacy_existing),
    )

    if args.skip_existing and log_marker.exists():
        if completed_slide_outputs_are_valid(
            metadata_path=log_marker,
            valA_dir=valA_dir,
            fakeB_dir=fakeB_dir,
            tile_ext=tile_ext,
            tile_size=tile_size,
            save_input=save_input,
        ):
            print(f"[skip verified existing] {slide_path.name}")
            with open(log_marker, "r", encoding="utf-8") as handle:
                result = json.load(handle)
            result["skipped_existing"] = True
            return result
        print(f"[resume incomplete existing] {slide_path.name}")

    print(f"\n=== {slide_path.name} ===")
    print(f"level0: {w0} x {h0}; source x{source_mag:g} -> target x{target_mag:g}; grid {nx} x {ny}")
    print(f"slide output dir: {slide_dir}")
    print(f"output fake_B dir: {fakeB_dir}")

    hest_weights = Path(args.hest_weights) if args.hest_weights else None
    tissue_mask = build_tissue_mask(
        slide=slide,
        base_mag=source_mag,
        segmenter_name=str(args.segmenter),
        mask_mag=float(args.mask_mag),
        mask_max_dim=int(args.mask_max_dim),
        hest_weights=hest_weights,
        allow_download=bool(args.allow_download),
        device=str(args.segment_device),
        confidence_thresh=float(args.hest_confidence),
        batch_size=int(args.hest_batch_size),
        hest_segmenter=hest_segmenter,
    )

    green_mask_path = qc_dir / "green_mask.png"
    save_green_mask_overlay(slide, tissue_mask, green_mask_path, qc_max_dim=int(args.qc_max_dim))
    print(f"mask QC: {green_mask_path}")

    rows: List[Dict[str, object]] = []
    pending_patches: List[Image.Image] = []
    pending_rows: List[Dict[str, object]] = []
    pending_seeds: List[int] = []
    selected = 0
    skipped_background = 0

    def flush_pending():
        nonlocal pending_patches, pending_rows, pending_seeds
        if not pending_patches:
            return
        translated = translate_batch(
            model,
            transform,
            pending_patches,
            model.device,
            sample_seeds=pending_seeds,
        )
        for out_img, row in zip(translated, pending_rows):
            fake_path = Path(str(row["fake_B_path"]))
            save_tile(out_img, fake_path, args.tile_format, int(args.jpeg_quality))
            rows.append(row)
        pending_patches = []
        pending_rows = []
        pending_seeds = []

    for idx in progress_iter(range(total_grid_tiles), total=total_grid_tiles):
        iy = idx // nx
        ix = idx % nx
        x20_x = ix * tile_size
        x20_y = iy * tile_size
        level0_x = int(round(x20_x * source_to_target_downsample))
        level0_y = int(round(x20_y * source_to_target_downsample))

        ratio = tissue_fraction_from_mask(tissue_mask, level0_x, level0_y, source_span)
        if ratio < float(args.tissue_threshold):
            skipped_background += 1
            continue

        name = patch_name(
            slide_stem=slide_stem,
            ix=ix,
            iy=iy,
            x20_x=x20_x,
            x20_y=x20_y,
            level0_x=level0_x,
            level0_y=level0_y,
            target_mag=target_mag,
            tile_size=tile_size,
            ext=tile_ext,
        )
        valA_path: Optional[Path] = valA_dir / name if save_input else None
        fakeB_path = fakeB_dir / name
        inference_seed = stable_patch_seed(int(args.seed), slide_path.name, name)
        row = {
            "index": selected,
            "slide": slide_path.name,
            "patch_name": name,
            "ix": ix,
            "iy": iy,
            "x20_x": x20_x,
            "x20_y": x20_y,
            "level0_x": level0_x,
            "level0_y": level0_y,
            "level0_span": source_span,
            "tile_size": tile_size,
            "source_mag": source_mag,
            "target_mag": target_mag,
            "tissue_fraction": ratio,
            "valA_path": str(valA_path) if valA_path is not None else "",
            "fake_B_path": str(fakeB_path),
            "inference_seed": inference_seed,
            "skipped_background": False,
        }

        if not args.overwrite and valid_patch_image(fakeB_path, tile_size):


            if valA_path is not None and not valid_patch_image(valA_path, tile_size):
                patch = read_target_mag_patch(
                    slide,
                    out_x=x20_x,
                    out_y=x20_y,
                    tile_size=tile_size,
                    source_to_target_downsample=source_to_target_downsample,
                )
                save_tile(patch, valA_path, args.tile_format, int(args.jpeg_quality))
            rows.append(row)
            selected += 1
            if args.max_tiles > 0 and selected >= int(args.max_tiles):
                break
            continue

        patch = read_target_mag_patch(
            slide,
            out_x=x20_x,
            out_y=x20_y,
            tile_size=tile_size,
            source_to_target_downsample=source_to_target_downsample,
        )
        if valA_path is not None and not valid_patch_image(valA_path, tile_size):
            save_tile(patch, valA_path, args.tile_format, int(args.jpeg_quality))

        pending_patches.append(patch)
        pending_rows.append(row)
        pending_seeds.append(inference_seed)
        selected += 1
        if len(pending_patches) >= int(args.model_batch_size):
            flush_pending()
        if args.max_tiles > 0 and selected >= int(args.max_tiles):
            break
    flush_pending()

    selected_overlay_path = qc_dir / "selected_tiles.png"
    save_selected_tiles_overlay(
        slide=slide,
        rows=rows,
        overlay_path=selected_overlay_path,
        source_span=source_span,
        qc_max_dim=int(args.qc_max_dim),
        skipped_background_total=skipped_background,
    )

    csv_path = coord_dir / "tiles.csv"
    fieldnames = [
        "index", "slide", "patch_name", "ix", "iy", "x20_x", "x20_y",
        "level0_x", "level0_y", "level0_span", "tile_size",
        "source_mag", "target_mag", "tissue_fraction",
        "valA_path", "fake_B_path", "inference_seed", "skipped_background",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(r["index"])):
            writer.writerow(row)

    meta = {
        "input": str(slide_path),
        "source_level0_size": [w0, h0],
        "source_mag": source_mag,
        "metadata_mag": metadata_mag,
        "target_mag": target_mag,
        "source_to_target_downsample": source_to_target_downsample,
        "tile_size": tile_size,
        "level0_span": source_span,
        "x20_output_size": [out_w, out_h],
        "tiles_x": nx,
        "tiles_y": ny,
        "total_grid_tiles": total_grid_tiles,
        "selected_tiles": len(rows),
        "skipped_background": skipped_background,
        "segmenter": str(args.segmenter),
        "mask_mag": float(args.mask_mag),
        "effective_mask_mag": float(tissue_mask.effective_mag),
        "mask_max_dim": int(args.mask_max_dim),
        "hest_confidence": float(args.hest_confidence),
        "tissue_threshold": float(args.tissue_threshold),
        "valA_dir": str(valA_dir) if save_input else None,
        "fake_B_dir": str(fakeB_dir),
        "slide_output_dir": str(slide_dir),
        "coords_csv": str(csv_path),
        "green_mask_png": str(green_mask_path),
        "selected_tiles_png": str(selected_overlay_path),
        "model_name": opt.name,
        "epoch": opt.epoch,
        "checkpoints_dir": opt.checkpoints_dir,
        "run_config": str(run_config_path),
        "deterministic_per_patch_noise": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(log_marker, meta)

    print(f"selected tiles: {len(rows)}; skipped background grid tiles: {skipped_background}")
    print(f"coords CSV: {csv_path}")
    print(f"selected tile QC: {selected_overlay_path}")
    return meta


def main() -> None:
    args, opt = parse_args()
    validate_workflow_args(args)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    slides = collect_slides(input_dir, bool(args.recursive), int(args.limit))
    if not slides:
        raise RuntimeError(f"No WSI files found in {input_dir}; supported: {sorted(WSI_EXTS)}")
    assert_unique_slide_stems(slides)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    opt.num_threads = 0
    opt.batch_size = int(args.model_batch_size)
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False

    transform = get_transform(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    hest_segmenter = None
    if str(args.segmenter) == "hest":
        hest_weights = Path(args.hest_weights) if args.hest_weights else None
        print("Loading HEST segmenter once for all slides...")
        hest_segmenter = HESTSegmenter(
            weights_path=hest_weights,
            allow_download=bool(args.allow_download),
            device=str(args.segment_device),
            confidence_thresh=float(args.hest_confidence),
        )

    print(f"Slides: {len(slides)}")
    print(f"Input : {input_dir}")
    print(f"Output: {output_root}")

    summary = []
    for i, slide in enumerate(slides, 1):
        print(f"\n[{i}/{len(slides)}] {slide}")
        try:
            summary.append(
                process_slide(
                    slide,
                    args,
                    opt,
                    model,
                    transform,
                    hest_segmenter=hest_segmenter,
                )
            )
        except Exception as exc:
            print(f"FAILED: {slide} -> {exc}", file=sys.stderr)
            summary.append({"slide": str(slide), "failed": True, "error": repr(exc)})

    summary_path = output_root / "workflow_summary.json"
    write_json_atomic(summary_path, summary)
    n_fail = sum(1 for item in summary if item.get("failed"))
    print("\nAll done.")
    print(f"summary: {summary_path}")
    print(f"failed slides: {n_fail}/{len(summary)}")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
