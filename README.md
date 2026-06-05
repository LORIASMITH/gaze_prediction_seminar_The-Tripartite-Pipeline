# Tripartite Synthesis for Cross-Domain Gaze Estimation

Progressive ablation integrating **FSCI**, **GFAL**, and **AGG** on normalized MPIIFaceGaze (GazeHub format), cross-person protocol.

| Stage | Configuration | MAE | Δ vs. Prev. |
|-------|--------------|-----|-------------|
| 0 | Baseline (ResNet-18 + FC) | 5.18° | — |
| 1 | + FSCI Causal Architecture | 5.04° | −0.14° |
| 2 | + FSCI + GFAL Integration | 4.63° | −0.41° |
| 3 | + FSCI + GFAL + AGG Pipeline | 4.82° | +0.19° |

## Methods

| Module | Paper | Role |
|--------|-------|------|
| **FSCI** | De-confounded Gaze Estimation (ECCV 2024) | Causal intervention via transformer token decoder + EMA confounder bank |
| **GFAL** | Gaze from Origin (AAAI 2024) | Frontalization auxiliary loss for rotation consistency |
| **AGG** | From Feature to Gaze (CVPR 2024) | Post-hoc geodesic projection replacing FC layer |

## Requirements

```bash
pip install -r requirements.txt
```

CUDA is recommended. Tested on Python 3.11, PyTorch 2.2, 2× RTX 3090.

## Dataset

Download **MPIIFaceGaze** in GazeHub normalized format from [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/) and place under:

```
data/MPII/
  Image/
    p00/ p01/ p02/ p03/ p04/ ...
  Label/
    p00.label p01.label ...
```

## Reproducing the Ablation

```bash
# Run all 4 stages (trains Stage 0-2, then applies AGG calibration)
python scripts/ablation_study.py \
  --src_dataset MPII \
  --src_root data/MPII \
  --tgt_root data/MPII \
  --src_subjects p00,p01,p02,p03 \
  --tgt_subjects p04 \
  --epochs 30

# Smoke-test with synthetic data (no dataset needed)
python scripts/ablation_study.py --synthetic
```

Results are saved to `results/ablation_results.json`.

## AGG Hyperparameter Sensitivity

To reproduce the ISOMap neighborhood sweep (Table I in the paper):

```bash
# After stage2.pth is saved, vary k:
python scripts/ablation_study.py --stages 3 --agg_n_neighbors 30  ...
python scripts/ablation_study.py --stages 3 --agg_n_neighbors 300 ...
```

## Project Structure

```
├── scripts/ablation_study.py        # Main training + evaluation
├── models/unified/gaze_net.py       # UnifiedGazeNet (all stages)
├── datasets/gaze_dataset.py         # GazeDataset loader
├── losses/gaze_losses.py            # Angular loss + GFAL loss
├── utils/
│   ├── metrics.py                   # Angular error (degrees)
│   └── gpm_utils.py                 # AGG / GPM calibration wrapper
├── configs/ablation_config.yaml     # Default hyperparameters
├── results/                         # Saved JSON results
└── Analytical-Gaze-Generalization-framework/  # Official AGG code (CVPR 2024)
```

## Citation

If you use this code, please also cite the original papers:

```
@inproceedings{liang2024fsci,
  title={De-confounded Gaze Estimation},
  author={Liang, Zijie and Bao, Yiwei and Lu, Feng},
  booktitle={ECCV},
  year={2024}
}
@inproceedings{xu2024gfal,
  title={Gaze from Origin: Learning for Generalized Gaze Estimation by Embedding the Gaze Frontalization Process},
  author={Xu, Mingjie and Lu, Feng},
  booktitle={AAAI},
  year={2024}
}
@inproceedings{bao2024agg,
  title={From Feature to Gaze: A Generalizable Replacement of Linear Layer for Gaze Estimation},
  author={Bao, Yiwei and Lu, Feng},
  booktitle={CVPR},
  year={2024}
}
```
