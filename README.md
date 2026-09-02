# What is the Right Embedding Space for Contrastive Learning in Referring Expression Counting? [ECCV 2026]

**Kostas Triaridis\*, Panagiotis Kaliosis\*, E-Ro Nguyen, Jingyi Xu, Dimitris Samaras, Hieu Le**
(\*equal contribution)

[[arXiv](https://arxiv.org/abs/2505.22850)] [[Project Page](https://cvlab-stonybrook.github.io/C-REX/)]

## Overview

Referring Expression Counting (REC) requires distinguishing visually similar objects described by fine-grained text cues. Prior methods tackle this with image-text contrastive learning, which is limited by poor image-text alignment and a scarcity of negative referring expressions. **C-REX** instead performs contrastive learning entirely in the visual embedding space: for each image-RE pair, predicted tokens are matched to ground-truth points via Bipartite Matching, and the matched tokens are pulled together while all other tokens serve as negatives. This removes the dependence on text alignment and gives access to a much larger pool of negatives (>100 image tokens vs. <5 REs).

C-REX is a drop-in replacement for the contrastive loss in any REC architecture. We apply it to three baselines and get consistent state-of-the-art improvements (up to 28% MAE, 24.5% RMSE), and show it also transfers to class-agnostic counting.

## Repository structure

Each baseline is a **self-contained** codebase with its own environment, configs, training/inference scripts, and README:

- [`src/GroundingDINO_C-REX/`](src/GroundingDINO_C-REX/README.md) — our own detection-based baseline built on GroundingDINO, with C-REX added alongside the standard counting losses
- [`src/GroundingREC_C-REX/`](src/GroundingREC_C-REX/README.md) — [Dai et al., CVPR 2024](https://github.com/sydai/referring-expression-counting), with its image-text contrastive loss replaced by C-REX
- [`src/CAD-GD_C-REX/`](src/CAD-GD_C-REX/README.md) — [Wang et al., CVPR 2025](https://arxiv.org/abs/2503.12460), with its image-text contrastive loss replaced by C-REX

See each folder's README for setup, data preparation, training, and evaluation instructions specific to that baseline.

## Datasets

- **REC-8K**: [official release](https://github.com/sydai/referring-expression-counting) (Dai et al., CVPR 2024)
- **FSC-147**: [official release](https://github.com/cvlab-stonybrook/LearningToCountEverything) (Ranjan et al., CVPR 2021)

We do not redistribute dataset images or annotations here — see the per-baseline READMEs for the expected directory layout once you've obtained them from the sources above.

## Checkpoints

Pretrained C-REX checkpoints for all three baselines (both backbones, both datasets where applicable) are hosted on Google Drive: **[link TBD]**

## Results

Highlights on REC-8K (Swin-B backbone; see the paper for full results, FSC-147, and ablations):

| Method | Val MAE↓ | Val RMSE↓ | Val F1↑ | Test MAE↓ | Test RMSE↓ | Test F1↑ |
|---|---|---|---|---|---|---|
| GroundingREC | 5.66 | 15.24 | 0.71 | 5.42 | 18.47 | 0.70 |
| + C-REX | 4.57 | 14.22 | 0.72 | 4.58 | 18.07 | 0.73 |
| CAD-GD | 5.31 | 13.68 | 0.72 | 5.36 | 16.14 | 0.73 |
| + C-REX | 5.06 | 12.73 | 0.72 | 5.02 | 15.09 | 0.73 |
| GroundingDINO | 4.58 | 14.64 | 0.73 | 4.64 | 17.58 | 0.74 |
| + C-REX | **4.00** | 14.49 | **0.77** | **4.20** | 16.76 | **0.78** |

## Citation

```bibtex
@inproceedings{triaridis2026crex,
  title     = {What is the Right Embedding Space for Contrastive Learning in Referring Expression Counting?},
  author    = {Triaridis, Kostas and Kaliosis, Panagiotis and Nguyen, E-Ro and Xu, Jingyi and Samaras, Dimitris and Le, Hieu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

This research was partially supported by NSF grants IIS-2123920 and IIS-2212046, and by the CCI startup fund at UNC Charlotte.
