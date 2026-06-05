"""
AGG neighborhood-size (k) sweep for Table I.

Reuses the committed pipeline components:
  - stage-2 checkpoint (checkpoints/ablation/stage2.pth)
  - datasets.gaze_dataset.build_loader  (same loaders as ablation_study.py)
  - utils.gpm_utils.GPMCalibrator       (the real ISOMap + GeodesicProjection)

Features are extracted ONCE; only the ISOMap+projection is recomputed per k,
so the numbers are identical to running ablation_study.py --stages 3 per k,
but much faster on CPU.

Run:
  .venv_gaze/Scripts/python.exe scripts/agg_k_sweep.py
"""
import os, sys, json, time
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.gaze_dataset import build_loader
from models.unified.gaze_net import UnifiedGazeNet
from utils.gpm_utils import GPMCalibrator, collect_features
from utils.metrics import angular_error_np, gazeto3d_np

DEVICE = 'cpu'
SEED = 42
N = 3000
K_LIST = [30, 50, 100, 200, 300]
TRANSDUCTIVE = (os.environ.get('AGG_MODE', 'transductive') == 'transductive')

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # ---- same loaders as ablation_study.py (MPII cross-person p00-p03 -> p04) ----
    train_loader = build_loader('MPII', 'data/MPII', split='full', n_samples=N,
                                batch_size=64, num_workers=0, image_size=224,
                                seed=SEED, subjects=['p00', 'p01', 'p02', 'p03'])
    val_loader = build_loader('MPII', 'data/MPII', split='full', n_samples=N,
                              batch_size=64, num_workers=0, image_size=224,
                              seed=SEED + 1, subjects=['p04'])

    # ---- load stage-2 model (no pretrain download; weights come from ckpt) ----
    model = UnifiedGazeNet(stage=2, pretrained=False).to(DEVICE)
    ckpt = os.path.join('checkpoints', 'ablation', 'stage2.pth')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    print(f'[ksweep] loaded {ckpt}')

    # ---- extract features ONCE ----
    t0 = time.time()
    print('[ksweep] collecting source features ...')
    src_feats, src_labels = collect_features(model, train_loader, DEVICE)
    print('[ksweep] collecting target features ...')
    tgt_feats, tgt_labels = collect_features(model, val_loader, DEVICE)
    print(f'[ksweep] features: src={src_feats.shape} tgt={tgt_feats.shape}  '
          f'({time.time()-t0:.0f}s)')

    tgt_labels_3d = gazeto3d_np(tgt_labels)

    results = {}
    for k in K_LIST:
        tk = time.time()
        cal = GPMCalibrator(n_neighbors=k, iso_dim=3)
        if TRANSDUCTIVE:
            cal.fit(src_feats, src_labels, tgt_features=tgt_feats)   # transductive
        else:
            cal.fit(src_feats, src_labels)                          # inductive (paper Table I)
        pred_3d = cal.predict(tgt_feats)
        err = float(angular_error_np(pred_3d, tgt_labels_3d).mean())
        results[f'k={k}'] = round(err, 4)
        print(f'[ksweep] k={k:<4d}  target MAE = {err:.2f}°   ({time.time()-tk:.0f}s)')

    os.makedirs('results', exist_ok=True)
    mode = 'transductive' if TRANSDUCTIVE else 'inductive'
    out = os.path.join('results', f'agg_k_sweep_{mode}.json')
    with open(out, 'w') as f:
        json.dump({'config': {'mode': mode, 'iso_dim': 3,
                              'n_source': N, 'n_target': N,
                              'src': 'p00-p03', 'tgt': 'p04'},
                   'results': results}, f, indent=2)
    print(f'\n[ksweep] saved -> {out}')
    print('[ksweep] summary:', results)

if __name__ == '__main__':
    main()
