# GroundingDINO + C-REX

A detection-based REC baseline built directly on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), trained with standard counting losses (`loss_label`, `loss_point`) plus our C-REX contrastive objective (`loss_contrast_img_hm`) added alongside.

This corresponds to the "GrDINO" rows in Table 1 (REC-8K) and Table 2 (FSC-147) of the paper.

## Setup

```bash
conda create -n crex-gdino python=3.9 -y
conda activate crex-gdino
pip install -r requirements.txt
python setup.py build_ext --inplace  # compiles the deformable-attention CUDA ops
```

Download the pretrained GroundingDINO Swin-T / Swin-B checkpoints from the [official release](https://github.com/IDEA-Research/GroundingDINO/releases) (`groundingdino_swint_ogc.pth`, `groundingdino_swinb_cogcoor.pth`) — these are the initialization weights before REC fine-tuning.

## Data

Place data so the paths below resolve relative to this directory (or point `--config`'s `anno_path` / the env vars below at your own layout):

- **REC-8K**: obtain images and annotations from the official [REC-8K release](https://github.com/sydai/referring-expression-counting) (Dai et al., CVPR 2024). Expected layout:
  - `anno/annotations.json`, `anno/splits.json` — REC-8K's released annotation/split files
  - Image root: set `REC8K_IMAGE_ROOT=/path/to/rec-8k/images` (defaults to `data/rec-8k`)
- **FSC-147**: obtain from the [official FSC-147 release](https://github.com/cvlab-stonybrook/LearningToCountEverything).
  - `anno/fsc_coco.json`, `anno/fsc_splits.json` — converted to this repo's annotation format (same schema as REC-8K: `{image_id: {caption: {points: [...], class, attribute, type}}}`)
  - Image root: set `FSC147_IMAGE_ROOT=/path/to/images_384_VarV2` (defaults to `data/FSC147_384_V2/images_384_VarV2`)

## Configs

| Config | Dataset | Losses | Paper reference |
|---|---|---|---|
| `configs/main/rec8k_baseline_swinT.py` | REC-8K | labels + points | Table 1, "GrDINO" (Swin-T) |
| `configs/main/rec8k_baseline_swinB.py` | REC-8K | labels + points | Table 1, "GrDINO" (Swin-B) |
| `configs/main/rec8k_crex_swinT.py` | REC-8K | + C-REX | Table 1, "GrDINO + C-REX" (Swin-T) |
| `configs/main/rec8k_crex_swinB.py` | REC-8K | + C-REX | Table 1, "GrDINO + C-REX" (Swin-B) |
| `configs/main/fsc147_baseline_swinT.py` | FSC-147 | labels + points | Table 2, "GroundingDino" |
| `configs/main/fsc147_crex_swinT.py` | FSC-147 | + C-REX | Table 2, "GroundingDino + C-REX" |

## Training

```bash
python train.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
    --checkpoint groundingdino_swint_ogc.pth --output-dir runs/rec8k_crex_swinT

python train.py --dataset fsc147 --config configs/main/fsc147_crex_swinT.py \
    --checkpoint groundingdino_swint_ogc.pth --output-dir runs/fsc147_crex_swinT --batch-size 2
```

Checkpoints (best val F1 / MAE / RMSE, plus the corresponding test-set snapshots) and a `stats.txt` training log are written to `--output-dir`.

## Inference

```bash
python inference.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
    --checkpoint runs/rec8k_crex_swinT/best_f1_model.pth --split test --output predictions.json
```

Uses thresholds `t1=0.25` (CLS token) / `t2=0.35` (remaining tokens) following Dai et al. [19]; override with `--t1` / `--t2`.

## Checkpoints

Pretrained C-REX checkpoints for this baseline (Swin-T and Swin-B, REC-8K and FSC-147): **[link TBD — Google Drive]**
