# [CVPR2024 Highlight] From Feature to Gaze: A Generalizable Replacement of Linear Layer for Gaze Estimation

This repository hosts the official implementation of the **Analytical Gaze Generalization (AGG)** framework, a novel approach to improve the cross-domain generalization of deep-learning-based gaze estimation models.

## Installation

Versions of key dependences are listed in `requirements.txt`. For data processing of gaze datasets, please refer to our survey paper [Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).

## Inference

Inference codes are provided at `./AGG_Inference.py`. Before running the inference code, please download models [here](https://drive.google.com/drive/folders/1m6Ym773pyLfrwTQRLI6sgO_ZY1esF03i?usp=drive_link) and put them under `./ckpt` directory, for example: `./ckpt/Baseline_ETH.pt`.

For example. to run inference of the baseline model from `Gaze360` to `MPIIFaceGaze` dataset, run the following command:

```bash
python AGG_Inference.py --source=Gaze360 --target=MPII --phase=Baseline
```

To run inference of the Geodesic Projection Module (GPM) of AGG from `ETH-XGaze` to `EyeDiap` dataset, run the following command:

``` 
python AGG_Inference.py --source=ETH --target=EyeDiapAll --phase=GPM
```

To run inference of the Sphere Oriented Training (SOT) of AGG from `ETH-XGaze` to `MPIIFaceGaze` dataset, run the following command:

```
python AGG_Inference.py --source=ETH --target=MPII --phase=SOT
```

## Training of AGG

To run the complete training and testing process of AGG, run:

```bash
bash run_AGG.sh
```

## Citation

If this work aids your research, please consider to cite our paper:

```
@inproceedings{bao2024feature,
  title={From feature to gaze: A generalizable replacement of linear layer for gaze estimation},
  author={Bao, Yiwei and Lu, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1409--1418},
  year={2024}
}
```

