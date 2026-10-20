# Whole-Slide Virtual Paraffinization

Unpaired frozen-section to FFPE image translation with a neural Schrödinger bridge.

Frozen sections are indispensable for intraoperative diagnosis, but freezing introduces tissue distortion, cellular artifact and staining differences that degrade both visual assessment and the performance of computational pathology models trained on FFPE tissue. This repository provides the **patch-level** training and inference code for a virtual paraffinization framework that converts frozen histology patches into FFPE-like representations, so that FFPE-trained models can be applied to frozen specimens without additional adaptation. Patches can be extracted from whole-slide images with the bundled utilities, but translation itself operates on patches and returns patches.

## Method

The generator is a single direction-conditioned network that learns both `A → B` (frozen to FFPE) and `B → A` (FFPE to frozen) mappings from unpaired data. The forward direction is the inference pathway; the reverse direction supplies bidirectional constraints during training.

Training combines:

- a multi-stage stochastic transport trajectory defined by a neural Schrödinger bridge, with five non-uniformly spaced bridge stages;
- a shared ResNet generator conditioned on the bridge stage, a stochastic latent code and an explicit direction variable, with direction-specific cross-scale skip projections;
- least-squares PatchGAN discriminators for both domains, conditioned on the bridge stage;
- a CONCH v1.5 feature-level adversarial constraint on the forward direction, with a frozen ViT-L/16 backbone and trainable multi-level heads;
- bidirectional cycle consistency (L1 + LPIPS + SSIM) and identity constraints;
- a bridge energy contrastive term with a learnable, spectrally normalized energy network.

## Repository layout

```
train.py                     training entry point
test.py                      patch-level test entry point
models/                      generator, discriminators, energy network, losses
options/                     command line options
data/                        dataset loaders
util/                        shared utilities
evaluations/                 FID / KID metrics used during validation
vision_aided_loss/           vision-aided adversarial losses and the CONCH backbone
open_clip_custom/            OpenCLIP variant required by the CONCH backbone
tests/                       CPU test for the inference path

wsi_patch_utils.py           tissue segmentation, patch sampling and batch translation utilities
hest_extract_wsi_patches.py  extract coordinate-preserving frozen patches from WSIs
hest_patch_translate_folder.py  WSI folder to translated patch folder
infer_folder_keep_names.py   folder-level patch inference, filenames preserved
infer_folder_reverse_keep_names.py  reverse direction patch inference
infer_one_preview.py         single-image preview
infer_bridge_states.py       dump intermediate bridge states
```

## Installation

Python 3.9+ and PyTorch are required.

```bash
pip install torch torchvision numpy scipy pillow tqdm dominate
pip install lpips timm transformers huggingface_hub
pip install antialiased-cnns gdown
```

Whole-slide I/O additionally requires OpenSlide and pyvips:

```bash
conda install -c conda-forge openslide-python openslide pyvips
```

### Pretrained weights

Two external checkpoints are downloaded automatically on first run and cached under `~/.cache/torch/hub/checkpoints/`:

- `vgg16-397923af.pth`, used by LPIPS;
- `pt_inception-2015-12-05-6726825d.pth`, used by FID/KID during validation.

The CONCH v1.5 backbone is **not** downloaded automatically and must be placed at `./checkpoints/conchv1_5.bin` relative to the repository root. It is required whenever `--lambda_GAN_aux > 0`, which is the configuration used in the paper. Request access to the weights from the CONCH model page, then copy the file:

```bash
mkdir -p checkpoints
cp /path/to/conchv1_5.bin checkpoints/conchv1_5.bin
```

If the file is missing, training stops at start-up with `FileNotFoundError: ./checkpoints/conchv1_5.bin`. To train without the feature-level adversarial term, pass `--lambda_GAN_aux 0`.

## Data

Training data is organized as unpaired domains:

```
datasets/frozen2ffpe/
├── trainA/     frozen patches
├── trainB/     FFPE patches
├── valA/
└── valB/
```

No spatial correspondence between `trainA` and `trainB` is required.

## Training

```bash
python train.py \
  --dataroot ./datasets/frozen2ffpe \
  --name frozen2ffpe_sb \
  --model sb --mode sb \
  --dataset_mode unaligned \
  --crop_size 512 --load_size 512 --preprocess none \
  --batch_size 2 --n_epochs 2 --n_epochs_decay 0 \
  --ngf 64 --n_blocks 9 \
  --lambda_NCE 0 --lambda_SB 1.0 \
  --lambda_cycle 1.0 --lambda_cycle_lpips 8 --lambda_cycle_ssim 2.0 \
  --lambda_GAN 1.0 --lambda_GAN_aux 0.5 \
  --gpu_ids 0
```

Checkpoints, validation images and the FID/KID log are written to `checkpoints/<name>/`. The best generator is selected by validation FID.

## Inference

Patch-folder inference. Input and output are plain image files, filenames are preserved:

```bash
python infer_folder_keep_names.py \
  --input_dir  /data/patches/frozen \
  --output_dir /data/patches/ffpe_like \
  --name frozen2ffpe_sb --checkpoints_dir ./checkpoints \
  --model sb --mode sb --epoch best --eval \
  --crop_size 512 --load_size 512 --preprocess none \
  --ngf 64 --n_blocks 9 \
  --lambda_NCE 0 --lambda_SB 1.0 \
  --lambda_cycle 1.0 --lambda_cycle_lpips 8 --lambda_cycle_ssim 2.0 \
  --lambda_GAN 1.0 --lambda_GAN_aux 0.5 \
  --gpu_ids 0
```

Add `--recursive` to keep subdirectory structure, or `--max_images N` to process a subset. Use `infer_folder_reverse_keep_names.py` for the FFPE-to-frozen direction.

Extract patches from WSIs first, and optionally translate them in the same pass:

```bash
python hest_extract_wsi_patches.py \
  --input_dir /data/wsi/frozen --output_dir /data/patches/frozen \
  --source_mag 40 --target_mag 20 --tile_size 512 \
  --segmenter hest --hest_weights /path/to/deeplabv3_seg_v4.ckpt

python hest_patch_translate_folder.py \
  --input_dir /data/wsi/frozen --output_dir /data/patches/ffpe_like \
  --source_mag 40 --target_mag 20 --tile_size 512 \
  --segmenter hest --hest_weights /path/to/deeplabv3_seg_v4.ckpt \
  --name frozen2ffpe_sb --checkpoints_dir ./checkpoints \
  --model sb --mode sb --epoch best --eval \
  --crop_size 512 --load_size 512 --preprocess none \
  --ngf 64 --n_blocks 9 \
  --lambda_NCE 0 --lambda_SB 1.0 \
  --lambda_cycle 1.0 --lambda_cycle_lpips 8 --lambda_cycle_ssim 2.0 \
  --lambda_GAN 1.0 --lambda_GAN_aux 0.5 \
  --gpu_ids 0
```

Single image preview:

```bash
python infer_one_preview.py --help
```

## Tests

```bash
python -m unittest discover -s tests
```

## Acknowledgements

This work builds on the official implementation of [Unpaired Image-to-Image Translation via Neural Schrödinger Bridge](https://arxiv.org/abs/2305.15086) (ICLR 2024), and uses [CONCH](https://huggingface.co/MahmoodLab/CONCH) as the pathology feature backbone. The anti-aliased downsampling blocks in `vision_aided_loss/blurpool.py` are from [BlurPool](https://github.com/adobe/antialiased-cnns) and are distributed under CC BY-NC-SA 4.0.

## License

See `LICENSE`. Third-party components retain their original licenses.
