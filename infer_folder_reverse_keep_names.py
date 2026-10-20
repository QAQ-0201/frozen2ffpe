from __future__ import annotations

import argparse
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
    deploy = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    deploy.add_argument("--input_dir", required=True)
    deploy.add_argument("--output_dir", required=True)
    deploy.add_argument("--recursive", action="store_true")
    deploy.add_argument("--max_images", type=int, default=0)
    deploy.add_argument("--model_batch_size", type=int, default=1)
    deploy.add_argument("--skip_existing", action="store_true")
    deploy.add_argument("--reverse_timestep", type=int, default=-1,
                        help="k condition for d=1; -1 means num_timesteps-1")
    deploy.add_argument("--seed", type=int, default=None)
    dopt, remaining = deploy.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    opt = TestOptions().parse()
    return dopt, opt


def main():
    dopt, opt = parse_args()
    input_dir = Path(dopt.input_dir)
    output_dir = Path(dopt.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if dopt.model_batch_size < 1:
        raise ValueError("--model_batch_size must be >= 1")

    opt.num_threads = 0
    opt.batch_size = dopt.model_batch_size
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False
    if dopt.seed is not None:
        torch.manual_seed(dopt.seed)

    transform = get_transform(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    net = model.netG.module if isinstance(model.netG, torch.nn.DataParallel) else model.netG
    net.eval()

    files = sorted(p for p in (input_dir.rglob("*") if dopt.recursive else input_dir.glob("*"))
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if dopt.max_images > 0:
        files = files[:dopt.max_images]

    pending = []
    for path in files:
        rel = path.relative_to(input_dir) if dopt.recursive else Path(path.name)
        out = output_dir / rel
        if not dopt.skip_existing or not out.is_file() or out.stat().st_size == 0:
            pending.append((path, out))
    print(f"Processing {len(pending)} images (reverse d=1)")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    k = opt.num_timesteps - 1 if dopt.reverse_timestep < 0 else dopt.reverse_timestep
    if not 0 <= k < opt.num_timesteps:
        raise ValueError(f"reverse_timestep must be in [0, {opt.num_timesteps - 1}]")
    print(f"Reverse condition: k={k}, num_timesteps={opt.num_timesteps}, direction=1")

    for start in range(0, len(pending), dopt.model_batch_size):
        batch = pending[start:start + dopt.model_batch_size]
        xs, outs = [], []
        for path, out in batch:
            out.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                xs.append(transform(image.convert("RGB")))
            outs.append(out)

        x = torch.stack(xs, dim=0).to(model.device)
        t = torch.full((x.shape[0],), k, dtype=torch.long, device=model.device)
        z = torch.randn((x.shape[0], 4 * opt.ngf), device=model.device)
        with torch.no_grad():
            y = net(x, t, z, direction=1)

        for i, out in enumerate(outs):
            save_image(tensor2im(y[i:i + 1]), str(out), aspect_ratio=1.0)
        done = min(start + len(batch), len(pending))
        print(f"[{done}/{len(pending)}] {batch[-1][0].name} -> {outs[-1]}", flush=True)


if __name__ == "__main__":
    main()
