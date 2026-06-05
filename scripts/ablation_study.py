"""
Cross-Domain Gaze Estimation — Full Ablation Study
====================================================

Trains and evaluates four configurations on ETH-XGaze → MPIIGaze:

  Stage 0 : Baseline         ResNet18 + FC
  Stage 1 : +FSCI            + causal token decoder + EMA intervention
  Stage 2 : +FSCI+GFAL       + frontalization auxiliary loss
  Stage 3 : +FSCI+GFAL+AGG   + post-training geodesic projection calibration

Usage
-----
# With real datasets (GazeHub format):
  python scripts/ablation_study.py \
    --eth_root  /path/to/xgaze_224 \
    --mpii_root /path/to/MPIIFaceGaze \
    --n_source  3000 --n_target 3000

# Quick smoke-test with synthetic data (no real datasets needed):
  python scripts/ablation_study.py --synthetic

All checkpoints are saved under checkpoints/ablation/.
Results are printed as a table and saved to results/ablation_results.json.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets.gaze_dataset import build_loader
from losses.gaze_losses import compute_loss, angular_loss
from models.unified.gaze_net import UnifiedGazeNet
from utils.metrics import mean_angular_error_deg, angular_error_deg
from utils.gpm_utils import GPMCalibrator, collect_features


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser('Cross-domain gaze ablation study')
    p.add_argument('--src_dataset', default='MPII',
                   choices=['Gaze360', 'ETH', 'MPII'],
                   help='Source domain dataset (default: MPII)')
    p.add_argument('--src_root',   default='data/MPII',
                   help='Source domain root directory (GazeHub format)')
    p.add_argument('--tgt_root',   default='data/MPII',
                   help='Target domain root (GazeHub format); same dir as src for cross-person')
    p.add_argument('--src_subjects', default='p00,p01,p02,p03',
                   help='Comma-separated source subjects (e.g. p00,p01,p02,p03)')
    p.add_argument('--tgt_subjects', default='p04',
                   help='Comma-separated target subjects (e.g. p04)')
    # Legacy aliases kept for backward compatibility
    p.add_argument('--eth_root',   default=None, help='[legacy] alias for --src_root when ETH')
    p.add_argument('--mpii_root',  default=None, help='[legacy] alias for --tgt_root')
    p.add_argument('--n_source',   type=int, default=3000, help='ETH training samples')
    p.add_argument('--n_target',   type=int, default=3000, help='MPII evaluation samples')
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--lambda_gfal',  type=float, default=0.5)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--image_size',   type=int, default=224)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--synthetic', action='store_true',
                   help='Use synthetic random data (for smoke-testing)')
    p.add_argument('--stages', default='0,1,2,3',
                   help='Comma-separated stages to run (e.g. 0,1,2,3)')
    p.add_argument('--ckpt_dir',    default='checkpoints/ablation')
    p.add_argument('--result_dir',  default='results')
    p.add_argument('--log_dir',     default='logs/ablation')
    p.add_argument('--no_pretrain', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--agg_n_neighbors', type=int, default=None,
                   help='ISOMap k-neighbors for AGG (default: auto = min(300, 40%% of N))')
    p.add_argument('--agg_iso_dim', type=int, default=3,
                   help='ISOMap output dimension for AGG (default: 3 — required by SphereAlignment)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_model(stage: int, pretrained: bool, device: str) -> nn.Module:
    model = UnifiedGazeNet(stage=stage, pretrained=pretrained)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model.to(device)


def get_inner_model(model: nn.Module) -> UnifiedGazeNet:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    errs = []
    for imgs, gazes, head_poses in loader:
        imgs       = imgs.to(device)
        gazes      = gazes.to(device)
        head_poses = head_poses.to(device)
        gaze_pred, _, _ = model(imgs, head_poses, mode='eval')
        errs.append(angular_error_deg(gaze_pred, gazes).cpu())
    return torch.cat(errs).mean().item()


def train_one_epoch(model, loader, optimizer, stage, lambda_gfal, device):
    model.train()
    get_inner_model(model).reset_ez()
    total_loss = 0.0
    for imgs, gazes, head_poses in loader:
        imgs       = imgs.to(device)
        gazes      = gazes.to(device)
        head_poses = head_poses.to(device)

        gaze_pred, _, g0_pred = model(imgs, head_poses, mode='train')
        loss, _ = compute_loss(gaze_pred, gazes, g0_pred, head_poses,
                                stage=stage, lambda_gfal=lambda_gfal)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_stage(
    stage: int,
    args,
    train_loader,
    val_loader,
    writer: SummaryWriter,
) -> tuple:
    """Train one ablation stage. Returns (model, best_val_error)."""
    label = ['Baseline', '+FSCI', '+FSCI+GFAL'][stage]
    print(f'\n{"="*60}')
    print(f'  Stage {stage}: {label}')
    print(f'{"="*60}')

    model = build_model(stage, pretrained=not args.no_pretrain, device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f'stage{stage}.pth')
    best_err  = float('inf')
    t0        = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, stage, args.lambda_gfal, args.device)
        scheduler.step()
        val_err = evaluate(model, val_loader, args.device)

        if val_err < best_err:
            best_err = val_err
            torch.save(get_inner_model(model).state_dict(), ckpt_path)

        elapsed = time.time() - t0
        eta     = elapsed / epoch * (args.epochs - epoch)
        print(f'  Epoch {epoch:3d}/{args.epochs}  '
              f'train_loss={train_loss:.4f}  '
              f'val_err={val_err:.2f}°  '
              f'best={best_err:.2f}°  '
              f'eta={eta/60:.1f}min')

        tag = f'stage{stage}'
        writer.add_scalar(f'{tag}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{tag}/val_error',  val_err,    epoch)

    print(f'  ✓  Stage {stage} done  best={best_err:.2f}°  '
          f'total_time={( time.time()-t0)/60:.1f}min')

    # Reload best checkpoint
    get_inner_model(model).load_state_dict(torch.load(ckpt_path, map_location=args.device))
    return model, best_err


def run_agg_calibration(model, train_loader, val_loader, args) -> float:
    """
    Stage 3: AGG — fit GPM on source-domain features, evaluate on target.
    Returns angular error in degrees.
    """
    print(f'\n{"="*60}')
    print('  Stage 3: +FSCI+GFAL+AGG (post-training GPM calibration)')
    print(f'{"="*60}')

    device = args.device
    print('  Collecting source-domain features ...')
    src_feats, src_labels = collect_features(model, train_loader, device)

    print('  Collecting target-domain features ...')
    tgt_feats, tgt_labels = collect_features(model, val_loader, device)

    calibrator = GPMCalibrator(n_neighbors=args.agg_n_neighbors, iso_dim=args.agg_iso_dim)
    # Transductive mode: pass target features so ISOMap covers both domains
    calibrator.fit(src_feats, src_labels, tgt_features=tgt_feats)

    gaze_pred_3d = calibrator.predict(tgt_feats)         # (N, 3) unit vecs

    from utils.metrics import angular_error_np, gazeto3d_np
    tgt_labels_3d = gazeto3d_np(tgt_labels)              # (N, 3)
    errs = angular_error_np(gaze_pred_3d, tgt_labels_3d) # (N,) in degrees
    err  = float(errs.mean())
    print(f'  ✓  Stage 3 AGG  val_error={err:.2f}°')

    # Save calibrator
    os.makedirs(args.ckpt_dir, exist_ok=True)
    calibrator.save(os.path.join(args.ckpt_dir, 'gpm_calibrator.pkl'))
    return err


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    seed_everything(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir,    exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    stages_to_run = list(map(int, args.stages.split(',')))

    # Resolve legacy aliases
    if args.eth_root and args.src_dataset == 'ETH':
        args.src_root = args.eth_root
    if args.mpii_root:
        args.tgt_root = args.mpii_root

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.synthetic:
        print('[!] Running in SYNTHETIC mode (random data — for smoke-testing)')
        train_loader = build_loader(
            args.src_dataset, args.src_root, split='train', n_samples=args.n_source,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, seed=args.seed, synthetic=True)
        val_loader = build_loader(
            'MPII', args.tgt_root, split='full', n_samples=args.n_target,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, seed=args.seed + 1, synthetic=True)
    else:
        # Verify data paths
        for name, root in [(args.src_dataset, args.src_root), ('MPII', args.tgt_root)]:
            img_dir = os.path.join(root, 'Image')
            lbl_dir = os.path.join(root, 'Label')
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                print(f'\n[ERROR] {name} dataset not found at: {root}')
                print('  Expected structure:')
                print(f'    {root}/Image/{{subject}}/{{img}}.jpg')
                print(f'    {root}/Label/{{subject}}.label')
                print('\n  Run:  python scripts/prepare_data.py --help')
                print('  Or:   python scripts/ablation_study.py --synthetic\n')
                sys.exit(1)

        src_subs = [s.strip() for s in args.src_subjects.split(',') if s.strip()]
        tgt_subs = [s.strip() for s in args.tgt_subjects.split(',') if s.strip()]
        print(f'[Data] {args.src_dataset} (source) : {args.src_root}  subjects={src_subs}  → sample {args.n_source}')
        print(f'[Data] MPII           (target) : {args.tgt_root}  subjects={tgt_subs}  → sample {args.n_target}')

        train_loader = build_loader(
            args.src_dataset, args.src_root, split='full', n_samples=args.n_source,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, seed=args.seed,
            subjects=src_subs if src_subs else None)
        val_loader = build_loader(
            'MPII', args.tgt_root, split='full', n_samples=args.n_target,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, seed=args.seed + 1,
            subjects=tgt_subs if tgt_subs else None)

    print(f'[Data] source batches={len(train_loader)}  target batches={len(val_loader)}')
    print(f'[GPU ] {args.device}  ({torch.cuda.device_count()} GPUs)')
    print(f'[Cfg ] epochs={args.epochs}  lr={args.lr}  bs={args.batch_size}')

    # ── Run ablation ──────────────────────────────────────────────────────────
    results   = {}
    stage_names = {
        0: 'Baseline',
        1: '+FSCI',
        2: '+FSCI+GFAL',
        3: '+FSCI+GFAL+AGG',
    }

    last_model = None

    for stage in stages_to_run:
        if stage <= 2:
            model, err = train_stage(stage, args, train_loader, val_loader, writer)
            results[stage_names[stage]] = round(err, 4)
            if stage == 2:
                last_model = model
        elif stage == 3:
            if last_model is None:
                ckpt2 = os.path.join(args.ckpt_dir, 'stage2.pth')
                if os.path.isfile(ckpt2):
                    print(f'[AGG] Loading saved stage-2 checkpoint: {ckpt2}')
                    last_model = build_model(2, pretrained=not args.no_pretrain, device=args.device)
                    get_inner_model(last_model).load_state_dict(
                        torch.load(ckpt2, map_location=args.device))
                else:
                    print('[AGG] No stage-2 checkpoint found; training stage 2 first...')
                    last_model, _ = train_stage(2, args, train_loader, val_loader, writer)
            err = run_agg_calibration(last_model, train_loader, val_loader, args)
            results[stage_names[3]] = round(err, 4)

    writer.close()

    # ── Results table ─────────────────────────────────────────────────────────
    print('\n' + '=' * 50)
    src_label = f'{args.src_dataset}({args.src_subjects})' if args.src_subjects else args.src_dataset
    tgt_label = f'MPII({args.tgt_subjects})' if args.tgt_subjects else 'MPII'
    print(f'  Ablation Results  ({src_label} → {tgt_label})')
    print('  Angular Error [degrees]  ↓ lower is better')
    print('=' * 50)
    for name, err in results.items():
        bar = '█' * int(err)
        print(f'  {name:<22}  {err:6.2f}°  {bar}')
    print('=' * 50)

    out_path = os.path.join(args.result_dir, 'ablation_results.json')
    # Merge with existing results so partial re-runs don't erase other stages
    if os.path.isfile(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            existing.get('results', {}).update(results)
            results = existing['results']
        except (json.JSONDecodeError, KeyError):
            pass
    with open(out_path, 'w') as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f'\n  Results saved → {out_path}')


if __name__ == '__main__':
    main()
