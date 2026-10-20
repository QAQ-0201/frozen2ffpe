
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image
import torch

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from util.util import save_image, tensor2im


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():


    deploy_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    deploy_parser.add_argument("--input_dir", required=True, type=str, help="folder containing frozen/input images")
    deploy_parser.add_argument("--output_dir", required=True, type=str, help="folder to save translated images")
    deploy_parser.add_argument("--recursive", action="store_true", help="process images recursively and preserve relative subfolders")
    deploy_parser.add_argument("--max_images", default=0, type=int, help="process at most N images; 0 means all")
    deploy_parser.add_argument("--model_batch_size", default=1, type=int,
                               help="number of images translated together on one GPU")
    deploy_parser.add_argument("--skip_existing", action="store_true",
                               help="skip outputs that already exist and are non-empty")

    deploy_opt, remaining = deploy_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    opt = TestOptions().parse()
    return deploy_opt, opt


def collect_images(input_dir: Path, recursive: bool):
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    deploy_opt, opt = parse_args()
    input_dir = Path(deploy_opt.input_dir)
    output_dir = Path(deploy_opt.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    opt.num_threads = 0
    if deploy_opt.model_batch_size < 1:
        raise ValueError("--model_batch_size must be at least 1")
    opt.batch_size = deploy_opt.model_batch_size
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False

    transform = get_transform(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    files = collect_images(input_dir, deploy_opt.recursive)
    if deploy_opt.max_images > 0:
        files = files[: deploy_opt.max_images]
    if deploy_opt.skip_existing:
        pending_files = []
        for path in files:
            rel_path = path.relative_to(input_dir) if deploy_opt.recursive else Path(path.name)
            out_path = output_dir / rel_path
            if not out_path.is_file() or out_path.stat().st_size == 0:
                pending_files.append(path)
        skipped = len(files) - len(pending_files)
        files = pending_files
        print(f"Skipping {skipped} existing non-empty outputs")
    if not files:
        print("No pending images; all requested outputs already exist")
        return

    print(f"Processing {len(files)} images")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    batch_size = deploy_opt.model_batch_size
    completed = 0
    for start in range(0, len(files), batch_size):
        batch_paths = files[start:start + batch_size]
        tensors = []
        output_paths = []
        for path in batch_paths:
            rel_path = path.relative_to(input_dir) if deploy_opt.recursive else Path(path.name)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as img:
                tensors.append(transform(img.convert("RGB")))
            output_paths.append(out_path)

        model.real_A = torch.stack(tensors, dim=0).to(model.device)
        with torch.no_grad():
            model.inference()
        for batch_idx, (path, out_path) in enumerate(zip(batch_paths, output_paths)):
            fake = tensor2im(model.fake_B[batch_idx:batch_idx + 1])
            save_image(fake, str(out_path), aspect_ratio=1.0)

        completed += len(batch_paths)
        if completed == len(batch_paths) or completed % 100 < len(batch_paths) or completed == len(files):
            print(f"[{completed}/{len(files)}] {batch_paths[-1].name} -> {output_paths[-1]}", flush=True)


if __name__ == "__main__":
    main()
