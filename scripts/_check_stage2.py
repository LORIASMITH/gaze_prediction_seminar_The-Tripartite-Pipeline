"""Sanity check: does the loaded stage-2 model reproduce json's 4.63 on p04?"""
import os, sys
import numpy as np, torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from datasets.gaze_dataset import build_loader
from models.unified.gaze_net import UnifiedGazeNet
from utils.metrics import angular_error_deg

DEV='cpu'
val = build_loader('MPII','data/MPII',split='full',n_samples=3000,batch_size=64,
                   num_workers=0,image_size=224,seed=43,subjects=['p04'])
m = UnifiedGazeNet(stage=2, pretrained=False).to(DEV)
m.load_state_dict(torch.load('checkpoints/ablation/stage2.pth', map_location=DEV))
m.eval()
errs=[]
with torch.no_grad():
    for imgs,gazes,hp in val:
        gp,_,_ = m(imgs.to(DEV), hp.to(DEV), mode='eval')
        errs.append(angular_error_deg(gp, gazes.to(DEV)).cpu())
print('stage2 own-head MAE on p04 =', float(torch.cat(errs).mean()), 'deg')
