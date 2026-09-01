# GroundingREC + C-REX

[GroundingREC](https://github.com/sydai/referring-expression-counting) (Dai et al., CVPR 2024) is a GroundingDINO-based REC model trained with standard counting losses plus an image-text contrastive loss (contrasting image tokens against positive/negative referring-expression text embeddings). Here we replace that image-text contrastive loss with our C-REX objective (`loss_contrast_img_hm`).

This corresponds to the "GrREC" rows in Table 1 (REC-8K) and Table 2 (FSC-147) of the paper.

**Architecture note**: GroundingREC uses `two_stage_type="standard"` (vs. `"default"` for our own GroundingDINO baseline in `../GroundingDINO/`) — a different query-initialization code path in the transformer decoder. This is intentional; it is not interchangeable with the GroundingDINO baseline's configs.

## Setup

Same as `../GroundingDINO/README.md`:

```bash
conda create -n crex-grec python=3.9 -y
conda activate crex-grec
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Data

Same REC-8K / FSC-147 layout as `../GroundingDINO/README.md`.

## Checkpoints and baselines

- **REC-8K baseline ("GrREC", no C-REX)**: per the paper, we use the **publicly released GroundingREC checkpoint** directly (no local training) — obtain it from the [official GroundingREC repo](https://github.com/sydai/referring-expression-counting). There is no local baseline-only REC-8K config here.
- **REC-8K + C-REX**: fine-tuned locally starting from the pretrained GroundingDINO Swin-T/Swin-B weights (same initialization as the GroundingDINO baseline), using `configs/main/rec8k_crex_swinT.py` / `swinB.py`.
- **FSC-147**: no official GroundingREC-on-FSC checkpoint exists, so both the baseline and C-REX variant are trained locally here.

## Configs

| Config | Dataset | Losses | Paper reference |
|---|---|---|---|
| `configs/main/rec8k_crex_swinT.py` | REC-8K | labels + points + C-REX | Table 1, "GrREC + C-REX" (Swin-T) |
| `configs/main/rec8k_crex_swinB.py` | REC-8K | labels + points + C-REX | Table 1, "GrREC + C-REX" (Swin-B) |
| `configs/main/fsc147_baseline_swinT.py` | FSC-147 | labels + points + image-text CL (`loss_contrast`) | Table 2, "GroundingREC" |
| `configs/main/fsc147_crex_swinT.py` | FSC-147 | labels + points + C-REX | Table 2, "GroundingREC + C-REX" |

## Training

```bash
python train.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
    --checkpoint groundingdino_swint_ogc.pth --output-dir runs/rec8k_crex_swinT

python train.py --dataset fsc147 --config configs/main/fsc147_baseline_swinT.py \
    --checkpoint groundingdino_swint_ogc.pth --output-dir runs/fsc147_baseline_swinT --batch-size 2

python train.py --dataset fsc147 --config configs/main/fsc147_crex_swinT.py \
    --checkpoint groundingdino_swint_ogc.pth --output-dir runs/fsc147_crex_swinT --batch-size 2
```

## Inference

```bash
python inference.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
    --checkpoint runs/rec8k_crex_swinT/best_f1_model.pth --split test --output predictions.json
```

To evaluate the external GroundingREC baseline checkpoint on REC-8K, point `--checkpoint` at the officially released weights and use a matching (non-C-REX) config from the [official GroundingREC repo](https://github.com/sydai/referring-expression-counting) — no config for this is included here since it was never retrained locally.

## Checkpoints

Pretrained C-REX checkpoints for this baseline (Swin-T and Swin-B, REC-8K; Swin-T, FSC-147): **[link TBD — Google Drive]**
