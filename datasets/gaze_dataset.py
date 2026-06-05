"""
GazeHub normalized-format dataset loader.

Supported datasets
------------------
ETH      : ETH-XGaze (source domain option)
           Image/subject0001/…  Label/subject0001.label
           Head pose stored as (pitch, yaw) → swapped to (yaw, pitch).

Gaze360  : MIT Gaze360 (source domain option — 360° panorama, wide head-pose range)
           Image/train/…  Image/test/…
           Label/train.label  Label/test.label
           Gaze column indices in GazeHub label: 1=gaze2d, 2=head2d  (same as others)
           Head pose can exceed [-π/2, π/2] → clamped with arcsin(sin(·)).
           Frames with invalid head pose [100.0, 100.0] are skipped.

MPII     : MPIIFaceGaze (target domain)
           Image/p00/…  Label/p00.label
           Head pose stored as (yaw, pitch) → no swap needed.

All label files: first line = header, remaining lines = data.
Data line (space-separated):  img_path  gaze2d  head2d  ...
  where gaze2d = "yaw,pitch" and head2d = "yaw,pitch" (both in radians).
"""

import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration table
# ─────────────────────────────────────────────────────────────────────────────

_CFG = {
    # name : (train_subjects_fn, test_subjects_fn, swap_head_pose, wrap_head_pose, skip_invalid_hp)
    'ETH': dict(
        train_fn=lambda subs: subs[:75],
        test_fn =lambda subs: subs[75:],
        full_fn =lambda subs: subs,
        swap_hp=True,   # stored as (pitch, yaw) → swap to (yaw, pitch)
        wrap_hp=False,
        skip_invalid=False,
        gaze_col=1,
        head_col=2,
    ),
    'Gaze360': dict(
        # GazeHub stores Gaze360 as train/test top-level folders
        train_fn=lambda subs: [s for s in subs if s == 'train'],
        test_fn =lambda subs: [s for s in subs if s == 'test'],
        full_fn =lambda subs: [s for s in subs if s in ('train', 'test')],
        swap_hp=False,
        wrap_hp=True,   # arcsin(sin(·)) clips to [-π/2, π/2]
        skip_invalid=True,  # skip head_pose == [100.0, 100.0]
        gaze_col=1,
        head_col=2,
    ),
    'MPII': dict(
        train_fn=lambda subs: subs[:14],
        test_fn =lambda subs: subs[14:],
        full_fn =lambda subs: subs,
        swap_hp=False,
        wrap_hp=False,
        skip_invalid=False,
        gaze_col=1,
        head_col=2,
    ),
}

_INVALID_HP = [100.0, 100.0]


# ─────────────────────────────────────────────────────────────────────────────
# Main dataset class
# ─────────────────────────────────────────────────────────────────────────────

class GazeDataset(Dataset):
    """
    GazeHub normalized-format loader.  Supports ETH-XGaze, Gaze360, MPIIFaceGaze.
    """

    def __init__(
        self,
        dataset_name: str,          # 'ETH' | 'Gaze360' | 'MPII'
        data_root: str,
        split: str = 'train',       # 'train' | 'test' | 'full'
        n_samples: int = None,      # random down-sample; None = keep all
        image_size: int = 224,
        seed: int = 42,
        subjects: list = None,      # explicit subject list, overrides split logic
    ):
        assert dataset_name in _CFG, \
            f"Unknown dataset '{dataset_name}'. Choose from {list(_CFG)}"

        self.dataset_name = dataset_name
        self.data_root    = data_root
        self.image_size   = image_size
        cfg = _CFG[dataset_name]

        # ── Enumerate subject folders ────────────────────────────────────────
        img_dir  = os.path.join(data_root, 'Image')
        lbl_dir  = os.path.join(data_root, 'Label')
        all_subs = sorted(os.listdir(img_dir))

        if subjects is not None:
            subjects = [s for s in subjects if s in all_subs]
        else:
            split_fn = {'train': cfg['train_fn'], 'test': cfg['test_fn'],
                        'full':  cfg['full_fn']}.get(split, cfg['full_fn'])
            subjects = split_fn(all_subs)

        # ── Read label files ─────────────────────────────────────────────────
        gc = cfg['gaze_col']
        hc = cfg['head_col']

        self.lines = []
        for subject in subjects:
            lbl_path = os.path.join(lbl_dir, subject + '.label')
            if not os.path.exists(lbl_path):
                continue
            with open(lbl_path, 'r') as f:
                rows = f.readlines()[1:]          # skip header line

            for row in rows:
                parts = row.strip().split(' ')
                if len(parts) <= max(gc, hc):
                    continue

                img_rel = parts[0]
                try:
                    gaze2d = list(map(float, parts[gc].split(',')))
                    head2d = list(map(float, parts[hc].split(',')))
                except ValueError:
                    continue

                if len(gaze2d) < 2 or len(head2d) < 2:
                    continue

                # Skip invalid frames (Gaze360 marks bad frames with head=[100,100])
                if cfg['skip_invalid'] and head2d[:2] == _INVALID_HP:
                    continue

                # ETH: swap (pitch, yaw) → (yaw, pitch)
                if cfg['swap_hp']:
                    head2d = [head2d[1], head2d[0]]

                # Gaze360: wrap head-pose angles into [-π/2, π/2]
                if cfg['wrap_hp']:
                    head2d = [math.asin(math.sin(head2d[0])),
                              math.asin(math.sin(head2d[1]))]

                self.lines.append((img_rel, gaze2d[:2], head2d[:2]))

        # ── Random sub-sampling ──────────────────────────────────────────────
        if n_samples is not None and n_samples < len(self.lines):
            rng = random.Random(seed)
            self.lines = rng.sample(self.lines, n_samples)

        if len(self.lines) == 0:
            raise RuntimeError(
                f'[GazeDataset] No samples found for {dataset_name}/{split} '
                f'at {data_root}. Check the dataset path and split name.'
            )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_rel, gaze2d, head2d = self.lines[idx]
        img_path = os.path.join(self.data_root, 'Image', img_rel)

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)              # HWC → CHW

        return (
            torch.from_numpy(img),
            torch.tensor(gaze2d, dtype=torch.float32),
            torch.tensor(head2d, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset  (smoke-test without real data)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticGazeDataset(Dataset):
    """Random images + plausible gaze/head-pose labels for smoke-testing."""

    def __init__(self, n_samples: int = 3000, image_size: int = 224, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n          = n_samples
        self.image_size = image_size
        yaw   = rng.uniform(-math.radians(40), math.radians(40),  n_samples)
        pitch = rng.uniform(-math.radians(30), math.radians(30),  n_samples)
        self.gazes      = np.stack([yaw, pitch], axis=1).astype(np.float32)
        hp_y  = rng.uniform(-math.radians(50), math.radians(50),  n_samples)
        hp_p  = rng.uniform(-math.radians(40), math.radians(40),  n_samples)
        self.head_poses = np.stack([hp_y, hp_p], axis=1).astype(np.float32)
        self.img_seeds  = rng.integers(0, 2**31, n_samples)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(self.img_seeds[idx]))
        img = rng.integers(0, 256, (self.image_size, self.image_size, 3), np.uint8)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return (
            torch.from_numpy(img),
            torch.from_numpy(self.gazes[idx]),
            torch.from_numpy(self.head_poses[idx]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(
    dataset_name: str,
    data_root: str,
    split: str = 'train',
    n_samples: int = None,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    seed: int = 42,
    synthetic: bool = False,
    subjects: list = None,
) -> DataLoader:
    if synthetic:
        ds = SyntheticGazeDataset(n_samples or 3000, image_size, seed)
    else:
        ds = GazeDataset(dataset_name, data_root, split, n_samples, image_size, seed,
                         subjects=subjects)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
