"""Eval stage2 on its TRAINING subjects (p00-p03). If high -> on-disk data != training data."""
import os, sys, numpy as np, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from datasets.gaze_dataset import build_loader
from models.unified.gaze_net import UnifiedGazeNet
from utils.metrics import angular_error_deg

DEV='cpu'
m = UnifiedGazeNet(stage=2, pretrained=False).to(DEV)
m.load_state_dict(torch.load('checkpoints/ablation/stage2.pth', map_location=DEV)); m.eval()

for tag, subs, seed in [('TRAIN p00-p03', ['p00','p01','p02','p03'], 42),
                        ('TEST  p04',     ['p04'], 43)]:
    ld = build_loader('MPII','data/MPII',split='full',n_samples=3000,batch_size=64,
                      num_workers=0,image_size=224,seed=seed,subjects=subs)
    errs=[]
    with torch.no_grad():
        for imgs,gazes,hp in ld:
            gp,_,_ = m(imgs.to(DEV), hp.to(DEV), mode='eval')
            errs.append(angular_error_deg(gp, gazes.to(DEV)).cpu())
    print(f'stage2 on {tag}: {float(torch.cat(errs).mean()):.2f} deg  (n={len(ld.dataset)})')

# data sanity: pixel stats of a few images
import cv2
import glob
fs = sorted(glob.glob('data/MPII/Image/p00/*.jpg'))[:3] + sorted(glob.glob('data/MPII/Image/p04/*.jpg'))[:3]
for f in fs:
    im = cv2.imread(f)
    print(os.path.relpath(f), 'shape', None if im is None else im.shape,
          'mean', None if im is None else round(float(im.mean()),1),
          'std', None if im is None else round(float(im.std()),1))
