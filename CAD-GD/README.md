# CAD-GD + C-REX

[CAD-GD](https://arxiv.org/abs/2503.12460) (Wang et al., CVPR 2025, "Exploring Contextual Attribute Density in Referring Expression Counting") combines density prediction with localization for REC. Here we replace its image-text contrastive objective with our C-REX objective.

This corresponds to the "CAD-GD" / "CAD-GD†" rows in Table 1 (REC-8K) and CAD-GD row in Table 2 (FSC-147) of the paper.

Unlike the `GroundingDINO/` and `GroundingREC/` baselines, this codebase does not use a `losses`/`weight_dict` config scheme — baseline vs. C-REX is simply a different training script with a different criterion class, and Swin-T vs. Swin-B is a `--config` CLI argument (the two backbone configs are otherwise identical).

## Setup

```bash
conda create -n crex-cadgd python=3.9 -y
conda activate crex-cadgd
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Download the pretrained GroundingDINO Swin-T / Swin-B checkpoints from the [official release](https://github.com/IDEA-Research/GroundingDINO/releases) — used as `--pretrain_model` initialization before REC fine-tuning.

## Data

Same REC-8K / FSC-147 sources as the other baselines (see `../GroundingDINO/README.md`). CAD-GD additionally requires precomputed density maps — see `utils/create_density_maps.py` (update the `anno_path`/`image_root`/`output_root` constants at the top of the file for your layout, following [Lempitsky & Zisserman, 2010]).

Expected layout (relative to this directory):
- `datasets/rec-8k/annotations.json`, `datasets/rec-8k/splits.json`
- `datasets/FSC147/annotation_FSC147_384.json`, `datasets/FSC147/fsc_splits.json`, `datasets/FSC147/ImageClasses_FSC147.txt`, `datasets/FSC147/{train,val,test}.txt`
- Image roots via `REC8K_IMAGE_ROOT` and `FSC147_IMAGE_ROOT` env vars (same convention as the other baselines)

## Configs

| Config | Backbone |
|---|---|
| `configs/GroundingDINO_SwinT_density_guide.py` | Swin-T |
| `configs/GroundingDINO_SwinB_density_guide.py` | Swin-B |
| `configs/FSC_GroundingDINO_SwinT_density_guide.py` | Swin-T (FSC-147) |

## Training

REC-8K baseline (no C-REX):
```bash
python train_rec8k.py --config configs/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model groundingdino_swint_ogc.pth --stats_dir exp/rec8k_baseline_swinT
```

REC-8K + C-REX:
```bash
python train_rec8k_crex.py --config configs/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model groundingdino_swint_ogc.pth --stats_dir exp/rec8k_crex_swinT
```

Pass `--config configs/GroundingDINO_SwinB_density_guide.py` for the Swin-B variants used in Table 1.

FSC-147 (baseline / C-REX):
```bash
python train_fsc.py --config configs/FSC_GroundingDINO_SwinT_density_guide.py \
    --pretrain_model groundingdino_swint_ogc.pth --stats_dir exp/fsc147_baseline_swinT

python train_fsc_crex.py --config configs/FSC_GroundingDINO_SwinT_density_guide.py \
    --pretrain_model groundingdino_swint_ogc.pth --stats_dir exp/fsc147_crex_swinT
```

## Evaluation: CAD-GD vs. CAD-GD†

Both use the **same trained checkpoint** — the difference is purely at inference time, in `utils/tester_rec.py`:
- **CAD-GD**: `threshold()` — a fixed probability threshold per token (`--threshold1`/`--threshold2`, default 0.25/0.35)
- **CAD-GD†**: `threshold_dynamic()` — selects the top-N most confident boxes, where N comes from the model's own predicted density-map count, instead of a fixed threshold

```bash
python test_rec.py --config configs/GroundingDINO_SwinT_density_guide.py \
    --pretrain_model exp/rec8k_crex_swinT/model.pth --stats_dir exp/eval_rec8k_crex_swinT
```

```bash
python test_fsc.py --config configs/FSC_GroundingDINO_SwinT_density_guide.py \
    --pretrain_model exp/fsc147_crex_swinT/model.pth --stats_dir exp/eval_fsc147_crex_swinT
```

`inference_rec8k_once.py` / `inference_fsc_once.py` / `inference_fsc_once_crex.py` dump per-(image, RE) predictions to JSON, following the same pattern. `inference_rec8k_paraphrased.py` reproduces the paraphrase-robustness evaluation (Table 5a).

## Checkpoints

Pretrained C-REX checkpoints for this baseline (Swin-T and Swin-B, REC-8K and FSC-147): **[link TBD — Google Drive]**
