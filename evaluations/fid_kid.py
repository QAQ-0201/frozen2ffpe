
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INCEPTION_PATH = Path(__file__).resolve().parent / "inception.py"
if not INCEPTION_PATH.exists():
    raise FileNotFoundError(f"Cannot find bundled Inception implementation: {INCEPTION_PATH}")
spec = importlib.util.spec_from_file_location("bundled_inception", INCEPTION_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load bundled Inception implementation: {INCEPTION_PATH}")
inception_module = importlib.util.module_from_spec(spec)
sys.modules["bundled_inception"] = inception_module
spec.loader.exec_module(inception_module)
InceptionV3 = inception_module.InceptionV3


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute FID and KID between real and generated image folders.",
    )
    parser.add_argument("--real_dir", required=True, type=str, help="Directory containing real target-domain images.")
    parser.add_argument("--fake_dir", required=True, type=str, help="Directory containing generated images.")
    parser.add_argument("--real_pattern", default="*", type=str, help="Glob pattern for real images.")
    parser.add_argument("--fake_pattern", default="*", type=str, help="Glob pattern for fake images.")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for Inception feature extraction.")
    parser.add_argument("--dims", default=2048, type=int, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help="Inception feature dimension.")
    parser.add_argument("--image_size", default=299, type=int,
                        help="Resize images to this size before Inception. Use 299 for standard FID features.")
    parser.add_argument("--gpu", default="", type=str,
                        help="GPU id visible to this process, e.g. 0. Empty string means CPU.")
    parser.add_argument("--max_images", default=0, type=int,
                        help="If >0, use at most this many images from each folder after shuffling.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed for subsampling and KID subsets.")
    parser.add_argument("--kid_subsets", default=100, type=int, help="Number of random subsets for KID.")
    parser.add_argument("--kid_subset_size", default=100, type=int,
                        help="Subset size for KID. Automatically clipped to available image count.")
    parser.add_argument(
        "--tissue_filter",
        default="none",
        choices=("none", "hest"),
        help="Also report a predeclared HEST tissue-only FID/KID; FULL is always reported.",
    )
    parser.add_argument(
        "--real_tissue_reference_dir",
        default="",
        type=str,
        help="Images used only for real-patch tissue QC; empty uses real_dir images.",
    )
    parser.add_argument(
        "--fake_tissue_reference_dir",
        default="",
        type=str,
        help="Source valA images with names matching fake_B; do not filter on generated pixels.",
    )
    parser.add_argument("--hest_weights", default="", type=str, help="Path to deeplabv3_seg_v4.ckpt.")
    parser.add_argument("--hest_confidence", default=0.5, type=float)
    parser.add_argument("--tissue_threshold", default=0.10, type=float)
    parser.add_argument("--hest_batch_size", default=4, type=int)
    parser.add_argument(
        "--tissue_qc_csv",
        default="",
        type=str,
        help="Required with HEST; records every patch tissue fraction and inclusion decision.",
    )
    parser.add_argument("--result_json", default="", type=str, help="Optional structured result output.")
    parser.add_argument(
        "--slide_report_csv",
        default="",
        type=str,
        help="Optional fake-slide influence ranking CSV; does not alter FULL FID.",
    )
    parser.add_argument(
        "--slide_id_regex",
        default="",
        type=str,
        help="Optional regex whose first capture group is the slide ID.",
    )
    parser.add_argument(
        "--slide_report_top",
        default=20,
        type=int,
        help="Number of worst slide influences printed to the console; 0 prints none.",
    )
    parser.add_argument(
        "--patch_report_csv",
        default="",
        type=str,
        help=(
            "Optional fake-patch influence ranking CSV. This is a diagnostic "
            "leave-one-out diagonal-FID approximation and does not alter FULL FID."
        ),
    )
    parser.add_argument(
        "--patch_report_limit",
        default=5000,
        type=int,
        help="Number of worst fake patches saved to patch_report_csv; 0 saves all.",
    )
    parser.add_argument(
        "--patch_report_print_top",
        default=20,
        type=int,
        help="Number of worst patch influences printed to the console; 0 prints none.",
    )
    return parser.parse_args()


def collect_images(root: str | Path, pattern: str, recursive: bool) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    iterator: Iterable[Path]
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return files


def maybe_subsample(files: Sequence[Path], max_images: int, seed: int) -> List[Path]:
    files = list(files)
    if max_images <= 0 or len(files) <= max_images:
        return files
    rng = random.Random(seed)
    rng.shuffle(files)
    return sorted(files[:max_images])


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if image_size > 0 and (tensor.shape[-2] != image_size or tensor.shape[-1] != image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def match_tissue_reference_files(
    evaluation_files: Sequence[Path],
    reference_dir: str,
    domain: str,
) -> List[Path]:
    if not reference_dir:
        return list(evaluation_files)
    candidates = collect_images(reference_dir, "*", recursive=True)
    by_name = {}
    duplicates = set()
    for path in candidates:
        if path.name in by_name:
            duplicates.add(path.name)
        else:
            by_name[path.name] = path
    if duplicates:
        first = sorted(duplicates)[:5]
        raise RuntimeError(
            f"{domain} tissue reference contains duplicate filenames; first: {first}"
        )
    missing = [path.name for path in evaluation_files if path.name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"{domain} tissue reference is missing {len(missing)} matching files; "
            f"first: {missing[:5]}"
        )
    return [by_name[path.name] for path in evaluation_files]


@torch.no_grad()
def get_hest_tissue_fractions(
    files: Sequence[Path],
    weights_path: Path,
    confidence: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not weights_path.is_file():
        raise FileNotFoundError(f"HEST checkpoint not found: {weights_path}")

    from wsi_patch_utils import HESTSegmenter

    segmenter = HESTSegmenter(
        weights_path=weights_path,
        allow_download=False,
        device=str(device),
        confidence_thresh=confidence,
    )
    fractions = np.empty(len(files), dtype=np.float32)
    for start in tqdm(range(0, len(files), batch_size), desc="HEST tissue QC"):
        end = min(start + batch_size, len(files))
        tensors = []
        for path in files[start:end]:
            with Image.open(path) as image:
                tensors.append(segmenter.transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        logits = segmenter.model(batch)["out"]
        foreground = F.softmax(logits, dim=1)[:, 1] > confidence
        fractions[start:end] = (
            foreground.float().mean(dim=(1, 2)).detach().cpu().numpy()
        )
    del segmenter
    return fractions


def write_tissue_qc_csv(
    output_path: Path,
    real_files: Sequence[Path],
    real_reference_files: Sequence[Path],
    real_fractions: np.ndarray,
    fake_files: Sequence[Path],
    fake_reference_files: Sequence[Path],
    fake_fractions: np.ndarray,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "image",
                "tissue_reference",
                "tissue_fraction",
                "included_tissue_only",
            ),
        )
        writer.writeheader()
        for domain, files, references, fractions in (
            ("real", real_files, real_reference_files, real_fractions),
            ("fake", fake_files, fake_reference_files, fake_fractions),
        ):
            for image_path, reference_path, fraction in zip(files, references, fractions):
                writer.writerow(
                    {
                        "domain": domain,
                        "image": str(image_path),
                        "tissue_reference": str(reference_path),
                        "tissue_fraction": f"{float(fraction):.8f}",
                        "included_tissue_only": int(float(fraction) >= threshold),
                    }
                )
    os.replace(temporary, output_path)


DEFAULT_SLIDE_PATTERNS = (
    re.compile(r"^(.*?)(?=__tile_y\d+_x\d+)", re.IGNORECASE),
    re.compile(r"^(.*?)(?=__x20_x\d+_y\d+)", re.IGNORECASE),
    re.compile(r"^(.*?)(?=__x\d+_y\d+)", re.IGNORECASE),
)


def infer_slide_id(path: Path, root: Path, custom_regex: str) -> str:
    stem = path.stem
    if custom_regex:
        match = re.search(custom_regex, stem)
        if match is None or match.lastindex is None:
            raise ValueError(
                f"--slide_id_regex did not capture a slide ID from: {path.name}"
            )
        slide_id = match.group(1).strip()
        if not slide_id:
            raise ValueError(f"empty slide ID captured from: {path.name}")
        return slide_id
    for pattern in DEFAULT_SLIDE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(1)
    try:
        relative_parent = path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        relative_parent = Path(".")
    if str(relative_parent) not in ("", "."):
        return str(relative_parent).replace(os.sep, "/")
    raise ValueError(
        "cannot infer slide ID from flat filename "
        f"{path.name!r}; pass --slide_id_regex with the slide ID in capture group 1"
    )


def _sample_variance_from_sums(
    sample_sum: np.ndarray,
    sample_sum_squares: np.ndarray,
    count: int,
) -> np.ndarray:
    if count < 2:
        raise ValueError("at least 2 samples are required to estimate variance")
    variance = (
        sample_sum_squares - (sample_sum * sample_sum) / float(count)
    ) / float(count - 1)
    return np.maximum(variance, 0.0)


def _feature_sums(
    features: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_sum = np.zeros(features.shape[1], dtype=np.float64)
    feature_sum_squares = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, features.shape[0], chunk_size):
        batch = features[start: start + chunk_size]
        feature_sum += batch.sum(axis=0, dtype=np.float64)
        feature_sum_squares += np.einsum(
            "ij,ij->j",
            batch,
            batch,
            dtype=np.float64,
            optimize=True,
        )
    return feature_sum, feature_sum_squares


def _diagonal_fid(
    mean_real: np.ndarray,
    variance_real: np.ndarray,
    mean_fake: np.ndarray,
    variance_fake: np.ndarray,
) -> float:
    mean_term = np.square(mean_real - mean_fake).sum()
    covariance_term = (
        variance_real
        + variance_fake
        - 2.0 * np.sqrt(np.maximum(variance_real * variance_fake, 0.0))
    ).sum()
    return float(mean_term + covariance_term)


def write_slide_influence_report(
    output_path: Path,
    fake_root: Path,
    fake_files: Sequence[Path],
    real_features: np.ndarray,
    fake_features: np.ndarray,
    custom_regex: str,
    print_top: int,
) -> List[dict]:
    if len(fake_files) != fake_features.shape[0]:
        raise ValueError("fake file/feature count mismatch")
    groups: dict[str, List[int]] = {}
    for index, path in enumerate(fake_files):
        slide_id = infer_slide_id(path, fake_root, custom_regex)
        groups.setdefault(slide_id, []).append(index)
    if len(groups) < 2:
        raise RuntimeError(
            f"slide report found only {len(groups)} group; check filename parsing"
        )

    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    real_sum, real_sum_squares = _feature_sums(real_features)
    fake_sum, fake_sum_squares = _feature_sums(fake_features)
    real_mean = real_sum / float(n_real)
    real_variance = _sample_variance_from_sums(
        real_sum,
        real_sum_squares,
        n_real,
    )
    fake_mean = fake_sum / float(n_fake)
    fake_variance = _sample_variance_from_sums(
        fake_sum,
        fake_sum_squares,
        n_fake,
    )
    full_mean_term = float(np.square(real_mean - fake_mean).sum())
    full_diag_fid = _diagonal_fid(
        real_mean,
        real_variance,
        fake_mean,
        fake_variance,
    )

    rows: List[dict] = []
    for slide_id, indices in groups.items():
        index_array = np.asarray(indices, dtype=np.int64)
        group = fake_features[index_array]
        group_count = group.shape[0]
        remaining_count = n_fake - group_count
        if remaining_count < 2:
            continue
        group_sum, group_sum_squares = _feature_sums(group)
        group_mean = group_sum / float(group_count)
        if group_count >= 2:
            group_variance = _sample_variance_from_sums(
                group_sum,
                group_sum_squares,
                group_count,
            )
            group_diag_fid = _diagonal_fid(
                real_mean,
                real_variance,
                group_mean,
                group_variance,
            )
        else:
            group_diag_fid = None

        remaining_sum = fake_sum - group_sum
        remaining_sum_squares = fake_sum_squares - group_sum_squares
        remaining_mean = remaining_sum / float(remaining_count)
        remaining_variance = _sample_variance_from_sums(
            remaining_sum,
            remaining_sum_squares,
            remaining_count,
        )
        remaining_mean_term = float(np.square(real_mean - remaining_mean).sum())
        remaining_diag_fid = _diagonal_fid(
            real_mean,
            real_variance,
            remaining_mean,
            remaining_variance,
        )
        rows.append(
            {
                "slide_id": slide_id,
                "patch_count": group_count,
                "patch_fraction": group_count / float(n_fake),
                "group_mean_distance_to_real": float(
                    np.square(group_mean - real_mean).sum()
                ),
                "group_diagonal_fid_to_real": group_diag_fid,
                "delta_global_mean_term_if_removed": (
                    full_mean_term - remaining_mean_term
                ),
                "delta_global_diag_fid_if_removed": (
                    full_diag_fid - remaining_diag_fid
                ),
                "global_mean_term": full_mean_term,
                "global_diagonal_fid": full_diag_fid,
            }
        )

    rows.sort(
        key=lambda row: row["delta_global_diag_fid_if_removed"],
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank_worst_first"] = rank

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    fieldnames = (
        "rank_worst_first",
        "slide_id",
        "patch_count",
        "patch_fraction",
        "group_mean_distance_to_real",
        "group_diagonal_fid_to_real",
        "delta_global_mean_term_if_removed",
        "delta_global_diag_fid_if_removed",
        "global_mean_term",
        "global_diagonal_fid",
    )
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)

    if print_top > 0:
        print(f"\n===== WORST {min(print_top, len(rows))} FAKE SLIDES =====")
        print("Positive delta means the slide worsens the global distribution match.")
        for row in rows[:print_top]:
            print(
                f"{row['rank_worst_first']:>3}. {row['slide_id']}  "
                f"n={row['patch_count']}  "
                f"delta_diag={row['delta_global_diag_fid_if_removed']:.6f}  "
                f"delta_mean={row['delta_global_mean_term_if_removed']:.6f}"
            )
    return rows


def write_patch_influence_report(
    output_path: Path,
    fake_root: Path,
    fake_files: Sequence[Path],
    real_features: np.ndarray,
    fake_features: np.ndarray,
    report_limit: int,
    print_top: int,
    chunk_size: int = 256,
) -> List[dict]:
    if len(fake_files) != fake_features.shape[0]:
        raise ValueError("fake file/feature count mismatch")
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    if n_real < 2 or n_fake < 3:
        raise ValueError("patch influence report requires >=2 real and >=3 fake images")

    real_sum, real_sum_squares = _feature_sums(real_features)
    fake_sum, fake_sum_squares = _feature_sums(fake_features)
    real_mean = real_sum / float(n_real)
    real_variance = _sample_variance_from_sums(
        real_sum,
        real_sum_squares,
        n_real,
    )
    fake_mean = fake_sum / float(n_fake)
    fake_variance = _sample_variance_from_sums(
        fake_sum,
        fake_sum_squares,
        n_fake,
    )
    full_mean_term = float(np.square(real_mean - fake_mean).sum())
    full_diag_fid = _diagonal_fid(
        real_mean,
        real_variance,
        fake_mean,
        fake_variance,
    )

    remaining_count = n_fake - 1
    rows: List[dict] = []
    root_resolved = fake_root.expanduser().resolve()
    for start in tqdm(
        range(0, n_fake, chunk_size),
        desc="Ranking fake patch influence",
    ):
        end = min(start + chunk_size, n_fake)
        batch = np.asarray(fake_features[start:end], dtype=np.float64)
        remaining_sum = fake_sum[None, :] - batch
        remaining_sum_squares = fake_sum_squares[None, :] - np.square(batch)
        remaining_mean = remaining_sum / float(remaining_count)
        remaining_variance = (
            remaining_sum_squares
            - np.square(remaining_sum) / float(remaining_count)
        ) / float(remaining_count - 1)
        remaining_variance = np.maximum(remaining_variance, 0.0)

        remaining_mean_terms = np.square(
            remaining_mean - real_mean[None, :]
        ).sum(axis=1)
        remaining_covariance_terms = (
            real_variance[None, :]
            + remaining_variance
            - 2.0
            * np.sqrt(
                np.maximum(real_variance[None, :] * remaining_variance, 0.0)
            )
        ).sum(axis=1)
        remaining_diag_fids = remaining_mean_terms + remaining_covariance_terms
        distances_to_real_mean = np.square(
            batch - real_mean[None, :]
        ).sum(axis=1)

        for local_index, global_index in enumerate(range(start, end)):
            path = fake_files[global_index]
            try:
                relative_path = str(path.resolve().relative_to(root_resolved))
            except ValueError:
                relative_path = path.name
            rows.append(
                {
                    "fake_path": str(path.resolve()),
                    "fake_relative_path": relative_path,
                    "fake_filename": path.name,
                    "delta_global_diag_fid_if_removed": float(
                        full_diag_fid - remaining_diag_fids[local_index]
                    ),
                    "delta_global_mean_term_if_removed": float(
                        full_mean_term - remaining_mean_terms[local_index]
                    ),
                    "squared_feature_distance_to_real_mean": float(
                        distances_to_real_mean[local_index]
                    ),
                    "global_diagonal_fid": full_diag_fid,
                    "global_mean_term": full_mean_term,
                }
            )

    rows.sort(
        key=lambda row: row["delta_global_diag_fid_if_removed"],
        reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank_worst_first"] = rank

    selected_rows = rows if report_limit == 0 else rows[:report_limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    fieldnames = (
        "rank_worst_first",
        "fake_path",
        "fake_relative_path",
        "fake_filename",
        "delta_global_diag_fid_if_removed",
        "delta_global_mean_term_if_removed",
        "squared_feature_distance_to_real_mean",
        "global_diagonal_fid",
        "global_mean_term",
    )
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    os.replace(temporary, output_path)

    if print_top > 0:
        print(f"\n===== WORST {min(print_top, len(selected_rows))} FAKE PATCHES =====")
        print("Positive delta means omitting one patch lowers diagonal-FID approximation.")
        for row in selected_rows[:print_top]:
            print(
                f"{row['rank_worst_first']:>5}. "
                f"delta_diag={row['delta_global_diag_fid_if_removed']:.9f}  "
                f"{row['fake_relative_path']}"
            )
    return selected_rows


@torch.no_grad()
def get_activations(
    files: Sequence[Path],
    model: InceptionV3,
    batch_size: int,
    dims: int,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    if len(files) == 0:
        raise RuntimeError("No images found for feature extraction.")

    model.eval()
    pred_arr = np.empty((len(files), dims), dtype=np.float64)

    for start in tqdm(range(0, len(files), batch_size), desc="Extracting Inception features"):
        end = min(start + batch_size, len(files))
        batch = torch.stack([load_image_tensor(p, image_size) for p in files[start:end]], dim=0)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)

        pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.detach().cpu().numpy().reshape(pred.size(0), -1)

    return pred_arr


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariances have different shapes: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance sqrt has non-negligible imaginary part: {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def calculate_fid_components(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> Tuple[float, float, float]:
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise RuntimeError("FID needs at least 2 real and 2 fake images to estimate covariance.")
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    mean_term = float((mu_real - mu_fake).dot(mu_real - mu_fake))
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    covariance_term = float(fid - mean_term)
    return fid, mean_term, covariance_term


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    return calculate_fid_components(real_features, fake_features)[0]


def polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dim = x.shape[1]
    return (x.dot(y.T) / dim + 1.0) ** 3


def unbiased_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        raise RuntimeError("KID needs at least 2 real and 2 fake images.")

    k_xx = polynomial_kernel(x, x)
    k_yy = polynomial_kernel(y, y)
    k_xy = polynomial_kernel(x, y)

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def calculate_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subset_size: int,
    n_subsets: int,
    seed: int,
) -> Tuple[float, float, int]:
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    subset_size = min(subset_size, n_real, n_fake)
    if subset_size < 2:
        raise RuntimeError("KID needs at least 2 real and 2 fake images.")

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        real_idx = rng.choice(n_real, subset_size, replace=False)
        fake_idx = rng.choice(n_fake, subset_size, replace=False)
        values.append(unbiased_mmd2(real_features[real_idx], fake_features[fake_idx]))

    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0), subset_size


def calculate_metric_bundle(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    kid_subset_size: int,
    kid_subsets: int,
    seed: int,
) -> dict:
    fid, fid_mean_term, fid_covariance_term = calculate_fid_components(
        real_features,
        fake_features,
    )
    kid_mean, kid_std, actual_subset_size = calculate_kid(
        real_features,
        fake_features,
        subset_size=kid_subset_size,
        n_subsets=kid_subsets,
        seed=seed,
    )
    return {
        "real_images": int(real_features.shape[0]),
        "fake_images": int(fake_features.shape[0]),
        "fid": fid,
        "fid_mean_term": fid_mean_term,
        "fid_covariance_term": fid_covariance_term,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "kid_subset_size": actual_subset_size,
        "kid_subsets": kid_subsets,
    }


def print_metric_bundle(title: str, result: dict) -> None:
    print(f"\n===== {title} =====")
    print(f"Images: real={result['real_images']}, fake={result['fake_images']}")
    print(f"FID: {result['fid']:.6f}")
    print(f"  mean term: {result['fid_mean_term']:.6f}")
    print(f"  covariance term: {result['fid_covariance_term']:.6f}")
    print(
        f"KID: {result['kid_mean']:.8f} +/- {result['kid_std']:.8f}  "
        f"(subset_size={result['kid_subset_size']}, subsets={result['kid_subsets']})"
    )
    print(
        f"KID x1000: {result['kid_mean'] * 1000.0:.6f} +/- "
        f"{result['kid_std'] * 1000.0:.6f}"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.hest_batch_size <= 0:
        raise ValueError("batch sizes must be > 0")
    if args.slide_report_top < 0:
        raise ValueError("--slide_report_top must be >= 0")
    if args.patch_report_limit < 0:
        raise ValueError("--patch_report_limit must be >= 0")
    if args.patch_report_print_top < 0:
        raise ValueError("--patch_report_print_top must be >= 0")
    if not 0.0 <= args.hest_confidence <= 1.0:
        raise ValueError("--hest_confidence must be in [0, 1]")
    if not 0.0 <= args.tissue_threshold <= 1.0:
        raise ValueError("--tissue_threshold must be in [0, 1]")
    if args.tissue_filter == "hest":
        if not args.hest_weights:
            raise ValueError("--hest_weights is required with --tissue_filter hest")
        if not args.fake_tissue_reference_dir:
            raise ValueError(
                "--fake_tissue_reference_dir must point to source valA when "
                "using HEST; filtering generated fake_B pixels would bias selection"
            )
        if not args.tissue_qc_csv:
            raise ValueError("--tissue_qc_csv is required with --tissue_filter hest")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    real_files = collect_images(args.real_dir, args.real_pattern, args.recursive)
    fake_files = collect_images(args.fake_dir, args.fake_pattern, args.recursive)
    real_files = maybe_subsample(real_files, args.max_images, args.seed)
    fake_files = maybe_subsample(fake_files, args.max_images, args.seed + 1)

    print(f"Real images: {len(real_files)} from {args.real_dir} pattern={args.real_pattern}")
    print(f"Fake images: {len(fake_files)} from {args.fake_dir} pattern={args.fake_pattern}")
    if len(real_files) == 0 or len(fake_files) == 0:
        raise RuntimeError("No images found. Check --real_dir/--fake_dir and patterns.")
    if min(len(real_files), len(fake_files)) < 50:
        print("WARNING: fewer than 50 images. FID/KID should be treated as rough monitoring signals only.")

    device = torch.device("cuda:0" if args.gpu != "" and torch.cuda.is_available() else "cpu")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx], resize_input=False, normalize_input=True).to(device)

    real_features = get_activations(real_files, model, args.batch_size, args.dims, device, args.image_size)
    fake_features = get_activations(fake_files, model, args.batch_size, args.dims, device, args.image_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "protocol": {
            "real_dir": str(Path(args.real_dir).resolve()),
            "fake_dir": str(Path(args.fake_dir).resolve()),
            "dims": args.dims,
            "image_size": args.image_size,
            "max_images": args.max_images,
            "seed": args.seed,
        },
        "full": calculate_metric_bundle(
            real_features,
            fake_features,
            kid_subset_size=args.kid_subset_size,
            kid_subsets=args.kid_subsets,
            seed=args.seed,
        ),
    }
    print_metric_bundle("FULL FID / KID (PRIMARY)", results["full"])

    if args.slide_report_csv:
        slide_report_path = Path(args.slide_report_csv).expanduser().resolve()
        slide_rows = write_slide_influence_report(
            output_path=slide_report_path,
            fake_root=Path(args.fake_dir),
            fake_files=fake_files,
            real_features=real_features,
            fake_features=fake_features,
            custom_regex=args.slide_id_regex,
            print_top=args.slide_report_top,
        )
        results["slide_influence_report"] = {
            "path": str(slide_report_path),
            "slides": len(slide_rows),
            "ranking_metric": "delta_global_diag_fid_if_removed",
            "positive_delta_interpretation": (
                "removing this slide lowers the global diagonal-FID approximation"
            ),
            "top_worst": slide_rows[: min(args.slide_report_top, len(slide_rows))],
        }
        print(f"Slide influence CSV: {slide_report_path}")

    if args.patch_report_csv:
        patch_report_path = Path(args.patch_report_csv).expanduser().resolve()
        patch_rows = write_patch_influence_report(
            output_path=patch_report_path,
            fake_root=Path(args.fake_dir),
            fake_files=fake_files,
            real_features=real_features,
            fake_features=fake_features,
            report_limit=args.patch_report_limit,
            print_top=args.patch_report_print_top,
        )
        results["patch_influence_report"] = {
            "path": str(patch_report_path),
            "rows_saved": len(patch_rows),
            "ranking_metric": "delta_global_diag_fid_if_removed",
            "interpretation": (
                "positive leave-one-out delta means omitting this single patch "
                "lowers the global diagonal-FID approximation; deltas are not additive"
            ),
            "top_worst": patch_rows[: min(args.patch_report_print_top, len(patch_rows))],
        }
        print(f"Patch influence CSV: {patch_report_path}")

    if args.tissue_filter == "hest":
        print("\nPreparing HEST tissue-only sensitivity analysis...")
        real_reference_files = match_tissue_reference_files(
            real_files,
            args.real_tissue_reference_dir,
            "real",
        )
        fake_reference_files = match_tissue_reference_files(
            fake_files,
            args.fake_tissue_reference_dir,
            "fake",
        )
        weights_path = Path(args.hest_weights).expanduser().resolve()
        real_fractions = get_hest_tissue_fractions(
            real_reference_files,
            weights_path,
            args.hest_confidence,
            args.hest_batch_size,
            device,
        )
        fake_fractions = get_hest_tissue_fractions(
            fake_reference_files,
            weights_path,
            args.hest_confidence,
            args.hest_batch_size,
            device,
        )
        real_keep = real_fractions >= args.tissue_threshold
        fake_keep = fake_fractions >= args.tissue_threshold
        if int(real_keep.sum()) < 2 or int(fake_keep.sum()) < 2:
            raise RuntimeError(
                "HEST tissue threshold retained fewer than 2 images in a domain: "
                f"real={int(real_keep.sum())}, fake={int(fake_keep.sum())}"
            )

        qc_path = Path(args.tissue_qc_csv).expanduser().resolve()
        write_tissue_qc_csv(
            qc_path,
            real_files,
            real_reference_files,
            real_fractions,
            fake_files,
            fake_reference_files,
            fake_fractions,
            args.tissue_threshold,
        )
        results["tissue_only"] = calculate_metric_bundle(
            real_features[real_keep],
            fake_features[fake_keep],
            kid_subset_size=args.kid_subset_size,
            kid_subsets=args.kid_subsets,
            seed=args.seed,
        )
        results["tissue_only"].update(
            {
                "filter": "hest",
                "hest_weights": str(weights_path),
                "hest_confidence": args.hest_confidence,
                "minimum_tissue_fraction": args.tissue_threshold,
                "real_excluded": int((~real_keep).sum()),
                "fake_excluded": int((~fake_keep).sum()),
                "real_tissue_reference_dir": args.real_tissue_reference_dir or args.real_dir,
                "fake_tissue_reference_dir": args.fake_tissue_reference_dir,
                "qc_csv": str(qc_path),
            }
        )
        print_metric_bundle(
            "HEST TISSUE-ONLY FID / KID (SECONDARY)",
            results["tissue_only"],
        )
        print(
            f"Excluded by fixed tissue rule: real={int((~real_keep).sum())}, "
            f"fake={int((~fake_keep).sum())}"
        )
        print(f"Tissue QC CSV: {qc_path}")

    if args.result_json:
        result_path = Path(args.result_json).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_name(result_path.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, result_path)
        print(f"Result JSON: {result_path}")


if __name__ == "__main__":
    main()
