"""Evaluate stage0/1/2 checkpoints (own head) on p04 vs json claims."""
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
json_claim = {0:5.1841, 1:5.043, 2:4.6318}
for st in (0,1,2):
    m = UnifiedGazeNet(stage=st, pretrained=False).to(DEV)
    m.load_state_dict(torch.load(f'checkpoints/ablation/stage{st}.pth', map_location=DEV))
    m.eval()
    errs=[]
    with torch.no_grad():
        for imgs,gazes,hp in val:
            gp,_,_ = m(imgs.to(DEV), hp.to(DEV), mode='eval')
            errs.append(angular_error_deg(gp, gazes.to(DEV)).cpu())
    real=float(torch.cat(errs).mean())
    print(f'stage{st}: real={real:6.2f}deg   json_claim={json_claim[st]:.2f}deg   diff={real-json_claim[st]:+.2f}')
