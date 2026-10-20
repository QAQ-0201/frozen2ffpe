
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
import torch


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():

    deploy = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    deploy.add_argument("--input_dir", required=True)
    deploy.add_argument("--output_dir", required=True)
    deploy.add_argument("--recursive", action="store_true")
    deploy.add_argument("--max_images", type=int, default=100,
                        help="first N sorted images; 0 means all")
    deploy.add_argument("--model_batch_size", type=int, default=1)
    deploy.add_argument("--seed", type=int, default=1234)
    dopt, remaining = deploy.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    from options.test_options import TestOptions

    return dopt, TestOptions().parse()


def collect_images(input_dir, recursive, max_images):
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = sorted(p for p in iterator
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if max_images > 0:
        files = files[:max_images]
    seen = {}
    for path in files:
        rel = path.relative_to(input_dir).with_suffix(".png")

        key = rel.as_posix().casefold()
        if key in seen:
            raise RuntimeError(f"Duplicate output name: {seen[key]} and {path}")
        seen[key] = path
    return files


def capture_bridge_states(model):
    states = {}
    total_steps = model.opt.num_timesteps

    def observe_input(_module, args):
        xt, time_idx = args[:2]
        idx = time_idx.detach().reshape(-1)
        if idx.numel() == 0 or not torch.all(idx == idx[0]).item():
            raise RuntimeError("Expected a single bridge stage for the batch")
        step = int(idx[0].item())
        if step not in range(total_steps) or step in states:
            raise RuntimeError(f"Unexpected or repeated generator stage: {step}")
        states[step] = xt.detach().cpu().clone()

    handle = model.netG.register_forward_pre_hook(observe_input)
    try:
        with torch.no_grad():
            model.inference()
    finally:
        handle.remove()

    if set(states) != set(range(total_steps)):
        raise RuntimeError(f"Incomplete bridge states: {sorted(states)}")
    return states, model.fake_B.detach().cpu().clone()


def save_png(batch_tensor, batch_index, output_path):
    rgb = batch_tensor[batch_index].detach().cpu().float().clamp(-1.0, 1.0)
    rgb = ((rgb.permute(1, 2, 0) + 1.0) / 2.0 * 255.0).to(torch.uint8).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path, format="PNG")


def main():
    dopt, opt = parse_args()
    input_dir = Path(dopt.input_dir).resolve()
    output_dir = Path(dopt.output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("--output_dir must not be inside --input_dir")
    if dopt.model_batch_size < 1 or dopt.max_images < 0:
        raise ValueError("Batch size must be >= 1 and max_images must be >= 0")
    if opt.model != "sb" or opt.num_timesteps < 2:
        raise ValueError("This exporter requires --model sb and num_timesteps >= 2")

    files = collect_images(input_dir, dopt.recursive, dopt.max_images)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    from data.base_dataset import get_transform
    from models import create_model

    opt.num_threads = 0
    opt.batch_size = dopt.model_batch_size
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False
    transform = get_transform(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    torch.manual_seed(dopt.seed)

    labels = [f"x{i}" for i in range(opt.num_timesteps)] + ["fake_B"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    print(f"Input: {input_dir}\nOutput: {output_dir}", flush=True)
    print(f"Images: {len(files)}; export: {', '.join(labels)}", flush=True)
    print("x0 is the original model input; x1... are bridge states, not G predictions.",
          flush=True)

    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source"] + labels)
        writer.writeheader()
        for start in range(0, len(files), dopt.model_batch_size):
            paths = files[start:start + dopt.model_batch_size]
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            if len({tuple(t.shape) for t in tensors}) != 1:
                raise RuntimeError("Images in a batch must have the same size; "
                                   "use --model_batch_size 1 for mixed sizes")

            model.real_A = torch.stack(tensors).to(model.device)
            states, final_prediction = capture_bridge_states(model)
            for index, path in enumerate(paths):
                rel = path.relative_to(input_dir).with_suffix(".png")
                row = {"source": str(path)}
                for step in range(opt.num_timesteps):
                    label = f"x{step}"
                    out = output_dir / label / rel
                    save_png(states[step], index, out)
                    row[label] = str(out)
                out = output_dir / "fake_B" / rel
                save_png(final_prediction, index, out)
                row["fake_B"] = str(out)
                writer.writerow(row)
            stream.flush()
            completed = start + len(paths)
            print(f"[{completed}/{len(files)}] {paths[-1].name}", flush=True)

    print(f"Done: {len(files) * len(labels)} PNG images; manifest: {manifest}",
          flush=True)


if __name__ == "__main__":
    main()
