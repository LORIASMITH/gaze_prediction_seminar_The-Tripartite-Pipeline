"""
Data preparation script for cross-domain gaze estimation.

Downloads and converts Gaze360 (source) and MPIIFaceGaze (target) into
GazeHub normalized format:

    data/
      Gaze360/
        Image/train/subject0001/img.jpg
        Image/test/subject0001/img.jpg
        Label/train.label
        Label/test.label
      MPII/
        Image/p00/img.jpg
        Label/p00.label

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Download sources
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gaze360 (MIT CSAIL, free, no registration required for the image subset)
  Option A – Direct from MIT:
    http://gaze360.csail.mit.edu/files/gaze360_data.zip   (~22 GB full)
  Option B – GazeHub pre-processed 224×224 face crops:
    Contact phi-ai@buaa.edu.cn or follow their preprocessing scripts at
    https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/

MPIIFaceGaze (MPI-INF, free for research)
  Option A – Kaggle (face images only, needs conversion):
    kaggle datasets download -d vimal704/mpiifacegaze
  Option B – Official MPI:
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/
    research/gaze-based-human-computer-interaction/its-written-all-over-your-face
  Option C – GazeHub pre-processed (recommended):
    Follow scripts at https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Download and convert MPIIFaceGaze from Kaggle:
  python scripts/prepare_data.py --mpii-from-kaggle

# Convert already-downloaded raw Gaze360 data to GazeHub format:
  python scripts/prepare_data.py --gaze360-raw /path/to/gaze360_data

# Convert already-downloaded raw MPIIFaceGaze to GazeHub format:
  python scripts/prepare_data.py --mpii-raw /path/to/MPIIFaceGaze

# Verify that both datasets are ready:
  python scripts/prepare_data.py --verify
"""

import argparse
import math
import os
import shutil
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_DIR      = os.path.join(_ROOT, 'data')
GAZE360_DIR   = os.path.join(DATA_DIR, 'Gaze360')
MPII_DIR      = os.path.join(DATA_DIR, 'MPII')


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def _count_samples(dataset_dir, subjects):
    """Count total label lines across all given subject label files."""
    total = 0
    lbl_dir = os.path.join(dataset_dir, 'Label')
    for sub in subjects:
        lbl = os.path.join(lbl_dir, sub + '.label')
        if os.path.isfile(lbl):
            with open(lbl) as f:
                total += max(0, len(f.readlines()) - 1)
    return total


def verify():
    """Check dataset directories and report sample counts."""
    ok = True
    print('\n── Dataset verification ───────────────────────────────')

    # Gaze360
    g360_img = os.path.join(GAZE360_DIR, 'Image')
    g360_lbl = os.path.join(GAZE360_DIR, 'Label')
    if os.path.isdir(g360_img) and os.path.isdir(g360_lbl):
        splits = [d for d in os.listdir(g360_img) if d in ('train', 'test')]
        n = _count_samples(GAZE360_DIR, splits)
        print(f'  Gaze360   ✓  splits={splits}  samples={n:,}')
    else:
        print(f'  Gaze360   ✗  not found at {GAZE360_DIR}')
        ok = False

    # MPII
    mpii_img = os.path.join(MPII_DIR, 'Image')
    mpii_lbl = os.path.join(MPII_DIR, 'Label')
    if os.path.isdir(mpii_img) and os.path.isdir(mpii_lbl):
        subs = sorted(os.listdir(mpii_img))
        n = _count_samples(MPII_DIR, subs)
        print(f'  MPIIFaceGaze ✓  subjects={len(subs)}  samples={n:,}')
    else:
        print(f'  MPIIFaceGaze ✗  not found at {MPII_DIR}')
        ok = False

    if ok:
        print('\n  Both datasets ready.  Run:')
        print('    python scripts/ablation_study.py \\')
        print('      --src_dataset Gaze360 --src_root data/Gaze360 \\')
        print('      --tgt_root data/MPII --n_source 3000 --n_target 3000')
    else:
        print('\n  Missing datasets.  See the instructions at the top of this file.')
    print()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle MPIIFaceGaze download + conversion
# ─────────────────────────────────────────────────────────────────────────────

def download_mpii_kaggle():
    """
    Download vimal704/mpiifacegaze from Kaggle and convert to GazeHub format.

    The Kaggle dataset contains p00–p14 face images with annotation .txt files.
    Each annotation line: idx leye_x leye_y reye_x reye_y mouth_x mouth_y ...
    plus a '2d_gaze_target' entry giving the 2D screen coordinates.

    NOTE: The Kaggle MPII version does NOT include 3D gaze angles directly.
    We approximate 2D screen gaze → (yaw, pitch) using a pinhole model with
    the standard MPIIFaceGaze calibration (focal ≈ 960 px, principal ≈ centre).

    For best results, use the official MPI normalized version which provides
    3D gaze angles directly.
    """
    try:
        import kaggle  # noqa
    except ImportError:
        print('[ERROR] kaggle package not installed.  Run:  pip install kaggle')
        print('Also set up your Kaggle API token at ~/.kaggle/kaggle.json')
        sys.exit(1)

    dl_dir = os.path.join(DATA_DIR, '_kaggle_mpii')
    os.makedirs(dl_dir, exist_ok=True)
    print(f'[Kaggle] Downloading vimal704/mpiifacegaze → {dl_dir}')
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'vimal704/mpiifacegaze',
         '-p', dl_dir, '--unzip'],
        check=True
    )

    # Try to find the downloaded structure
    raw_root = None
    for d in os.listdir(dl_dir):
        candidate = os.path.join(dl_dir, d)
        if os.path.isdir(candidate) and any(
            s.startswith('p0') for s in os.listdir(candidate)
        ):
            raw_root = candidate
            break
    if raw_root is None:
        raw_root = dl_dir

    print(f'[Kaggle] Converting {raw_root} → GazeHub format at {MPII_DIR}')
    convert_mpii_kaggle(raw_root, MPII_DIR)
    print('[Kaggle] Done.')


def convert_mpii_kaggle(raw_root: str, out_root: str):
    """
    Convert Kaggle vimal704/mpiifacegaze raw structure to GazeHub format.

    Raw structure:
        raw_root/p00/day01/frame0001.jpg
                 p00/day01/p00_annotations.txt
    GazeHub output:
        out_root/Image/p00/frame0001.jpg
        out_root/Label/p00.label
    """
    # Standard MPIIFaceGaze camera intrinsics (approximate)
    fx, fy   = 960.0, 960.0
    cx, cy   = 640.0, 480.0    # assuming 1280×960 frames

    os.makedirs(os.path.join(out_root, 'Image'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'Label'), exist_ok=True)

    subjects = sorted(d for d in os.listdir(raw_root) if d.startswith('p'))
    for subj in subjects:
        subj_raw = os.path.join(raw_root, subj)
        subj_img = os.path.join(out_root, 'Image', subj)
        os.makedirs(subj_img, exist_ok=True)

        label_lines = ['face gaze2d head2d\n']

        for day in sorted(os.listdir(subj_raw)):
            day_dir = os.path.join(subj_raw, day)
            if not os.path.isdir(day_dir):
                continue

            # Find annotation file
            ann_files = [f for f in os.listdir(day_dir) if f.endswith('_annotations.txt')]
            if not ann_files:
                continue
            ann_path = os.path.join(day_dir, ann_files[0])

            try:
                ann = np.loadtxt(ann_path)
            except Exception:
                continue

            for row in ann:
                if len(row) < 9:
                    continue
                frame_idx    = int(row[0])
                gaze_x, gaze_y = row[-2], row[-1]    # screen gaze position

                # Convert screen coords to angular gaze (rough approx)
                dx = gaze_x - cx
                dy = gaze_y - cy
                yaw   = math.atan2(-dx, fx)
                pitch = math.atan2(-dy, math.sqrt(dx**2 + fx**2))

                img_name = f'{subj}_day{day[-2:]}_frame{frame_idx:04d}.jpg'
                img_src  = os.path.join(day_dir, f'{frame_idx:04d}.jpg')
                img_dst  = os.path.join(subj_img, img_name)

                if not os.path.isfile(img_src):
                    img_src = os.path.join(day_dir, f'frame{frame_idx:04d}.jpg')
                if not os.path.isfile(img_src):
                    continue

                shutil.copy2(img_src, img_dst)

                # head2d placeholder (0,0) — Kaggle version lacks head pose
                hp_yaw, hp_pitch = 0.0, 0.0
                rel_path  = os.path.join(subj, img_name)
                label_lines.append(
                    f'{rel_path} {yaw:.6f},{pitch:.6f} '
                    f'{hp_yaw:.6f},{hp_pitch:.6f}\n'
                )

        lbl_path = os.path.join(out_root, 'Label', subj + '.label')
        with open(lbl_path, 'w') as f:
            f.writelines(label_lines)
        print(f'  {subj}: {len(label_lines)-1} samples')


# ─────────────────────────────────────────────────────────────────────────────
# Gaze360 raw → GazeHub format conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_gaze360_raw(raw_root: str, out_root: str):
    """
    Convert the official MIT Gaze360 dataset to GazeHub format.

    Official Gaze360 structure (after extraction):
        raw_root/
          imgs/                     ← 360° panoramic frames (JPEG)
          metadata.txt              ← per-frame: split, subject, gaze, head, ...
          [or train.txt / test.txt]

    GazeHub output:
        out_root/Image/train/{subj}/frame.jpg
        out_root/Image/test/{subj}/frame.jpg
        out_root/Label/train.label
        out_root/Label/test.label

    NOTE: Gaze360 provides gaze as a 3D unit vector in world coordinates.
    We convert to (yaw, pitch) using the standard gaze convention:
        g = [-cos(p)sin(y),  -sin(p),  -cos(p)cos(y)]
        yaw   = arctan2(-g[0], -g[2])
        pitch = arcsin(-g[1])

    Face normalization (cropping face from 360° panorama) requires additional
    steps.  If you have the GazeHub pre-processed version (xgaze_224-style),
    place it directly at out_root — no conversion needed.
    """
    import csv

    # Look for metadata file
    meta_candidates = ['metadata.txt', 'train.txt', 'test.txt', 'labels.txt']
    meta_path = None
    for c in meta_candidates:
        p = os.path.join(raw_root, c)
        if os.path.isfile(p):
            meta_path = p
            break

    if meta_path is None:
        print(f'[ERROR] Cannot find metadata file in {raw_root}')
        print('  Expected: metadata.txt or train/test split files')
        print('  If you have the GazeHub pre-processed version, place it directly at:')
        print(f'    {out_root}')
        sys.exit(1)

    print(f'[Gaze360] Reading metadata: {meta_path}')
    os.makedirs(os.path.join(out_root, 'Image', 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'Image', 'test'),  exist_ok=True)
    os.makedirs(os.path.join(out_root, 'Label'),            exist_ok=True)

    split_lines = {'train': ['face gaze2d head2d\n'],
                   'test':  ['face gaze2d head2d\n']}
    img_dir  = os.path.join(raw_root, 'imgs')
    counts   = {'train': 0, 'test': 0}

    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # Expected columns: split frame_idx gx gy gz hx hy hz [optional extra]
            split    = parts[0] if parts[0] in ('train', 'test') else 'train'
            frame    = parts[1] if len(parts) > 1 else parts[0]
            try:
                gx, gy, gz = float(parts[2]), float(parts[3]), float(parts[4])
                hx, hy, hz = float(parts[5]), float(parts[6]), float(parts[7])
            except (ValueError, IndexError):
                continue

            # Convert 3D gaze vector to (yaw, pitch)
            gaze_yaw   = math.atan2(-gx, -gz)
            gaze_pitch = math.asin(max(-1.0, min(1.0, -gy)))

            # Head pose (yaw, pitch) — clamp to valid range
            head_yaw   = math.atan2(-hx, -hz)
            head_pitch = math.asin(max(-1.0, min(1.0, -hy)))
            head_pitch = math.asin(math.sin(head_pitch))   # wrap

            # Copy image
            src_img = os.path.join(img_dir, f'{frame}.jpg')
            if not os.path.isfile(src_img):
                src_img = os.path.join(img_dir, frame)
            if not os.path.isfile(src_img):
                continue

            rel_path = os.path.join(split, f'{frame}.jpg')
            dst_img  = os.path.join(out_root, 'Image', rel_path)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(src_img, dst_img)

            split_lines[split].append(
                f'{rel_path} {gaze_yaw:.6f},{gaze_pitch:.6f} '
                f'{head_yaw:.6f},{head_pitch:.6f}\n'
            )
            counts[split] += 1

    for split, lines in split_lines.items():
        lbl_path = os.path.join(out_root, 'Label', split + '.label')
        with open(lbl_path, 'w') as f:
            f.writelines(lines)
        print(f'  {split}: {counts[split]} samples written to {lbl_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Official MPIIFaceGaze conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_mpii_official(raw_root: str, out_root: str):
    """
    Convert official MPIIFaceGaze (normalized version with .mat files) to GazeHub format.

    Official structure:
        raw_root/p00/day01/p00_day01.mat    ← contains face images + labels
    or:
        raw_root/p00/p00.mat                ← per-subject mat file

    This uses scipy.io to read .mat files.
    """
    try:
        import scipy.io as sio
        import cv2
    except ImportError:
        print('[ERROR] scipy is required.  Run: pip install scipy')
        sys.exit(1)

    os.makedirs(os.path.join(out_root, 'Image'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'Label'), exist_ok=True)

    subjects = sorted(d for d in os.listdir(raw_root) if d.startswith('p'))
    for subj in subjects:
        subj_raw = os.path.join(raw_root, subj)
        subj_out = os.path.join(out_root, 'Image', subj)
        os.makedirs(subj_out, exist_ok=True)

        label_lines = ['face gaze2d head2d\n']

        # Walk mat files
        for root, _, files in os.walk(subj_raw):
            for fname in sorted(files):
                if not fname.endswith('.mat'):
                    continue
                mat_path = os.path.join(root, fname)
                try:
                    data = sio.loadmat(mat_path)
                except Exception:
                    continue

                # Mat typically contains: 'data', 'label', etc.
                # label: [gaze_x_pixel, gaze_y_pixel] or [yaw, pitch]
                if 'data' not in data:
                    continue
                frames = data['data']
                labels = data.get('label', data.get('gaze', None))
                if labels is None:
                    continue

                for i in range(frames.shape[0] if frames.ndim > 2 else 1):
                    try:
                        img = frames[i] if frames.ndim > 2 else frames
                        lbl = labels[i] if labels.ndim > 1 else labels
                    except IndexError:
                        continue

                    img_name = f'{fname[:-4]}_{i:04d}.jpg'
                    img_path = os.path.join(subj_out, img_name)
                    cv2.imwrite(img_path, img)

                    # Assume label is [yaw, pitch] in radians if small values
                    if abs(lbl[0]) < math.pi and abs(lbl[1]) < math.pi:
                        yaw, pitch = float(lbl[0]), float(lbl[1])
                    else:
                        # Pixel coordinates → convert assuming 30° FOV
                        yaw   = math.atan2(float(lbl[0]) - 320, 960)
                        pitch = math.atan2(float(lbl[1]) - 240, 960)

                    rel = os.path.join(subj, img_name)
                    label_lines.append(f'{rel} {yaw:.6f},{pitch:.6f} 0.0,0.0\n')

        lbl_path = os.path.join(out_root, 'Label', subj + '.label')
        with open(lbl_path, 'w') as f:
            f.writelines(label_lines)
        print(f'  {subj}: {len(label_lines)-1} samples')


# ─────────────────────────────────────────────────────────────────────────────
# GazeHub pre-processed: just check and symlink
# ─────────────────────────────────────────────────────────────────────────────

def link_gazehub(src: str, dst: str):
    """Create a symlink (or copy) from a GazeHub-format directory."""
    if not os.path.isdir(src):
        print(f'[ERROR] Source directory not found: {src}')
        sys.exit(1)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f'[INFO] {dst} already exists — skipping link.')
        return
    os.symlink(os.path.abspath(src), dst)
    print(f'[Link] {src} → {dst}')


# ─────────────────────────────────────────────────────────────────────────────
# Download summary / instructions
# ─────────────────────────────────────────────────────────────────────────────

def print_download_guide():
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║          Dataset Download Guide (Gaze360 + MPIIFaceGaze)        ║
╚══════════════════════════════════════════════════════════════════╝

─── Gaze360 (source domain) ────────────────────────────────────────

  Method 1 – GazeHub pre-processed (RECOMMENDED, no conversion needed)
    ① Download GazeHub normalization code for Gaze360:
       https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/
       → Gaze360 → face-based processing code
    ② Download original Gaze360 (no registration required):
       http://gaze360.csail.mit.edu/files/gaze360_data.zip  (~22 GB)
    ③ Run GazeHub normalization to get Image/ + Label/ structure
    ④ Place result at:  data/Gaze360/

  Method 2 – Convert raw Gaze360 yourself
    python scripts/prepare_data.py --gaze360-raw /path/to/gaze360_data

─── MPIIFaceGaze (target domain) ───────────────────────────────────

  Method 1 – GazeHub pre-processed (RECOMMENDED)
    ① Download MPIIFaceGaze from:
       https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/
       research/gaze-based-human-computer-interaction/its-written-all-over-your-face
    ② Apply GazeHub normalization script (face-based MPIIFaceGaze)
    ③ Place result at:  data/MPII/

  Method 2 – From Kaggle (face images, approximate gaze angles)
    ① Install kaggle:   pip install kaggle
    ② Set up token:     https://www.kaggle.com/settings → API → Create Token
                        → place kaggle.json at ~/.kaggle/kaggle.json
    ③ Run:
       python scripts/prepare_data.py --mpii-from-kaggle

  Method 3 – Convert downloaded official MPIIFaceGaze (.mat files)
    python scripts/prepare_data.py --mpii-raw /path/to/MPIIFaceGaze

─── After datasets are ready ────────────────────────────────────────

  Verify:
    python scripts/prepare_data.py --verify

  Run ablation (3000 samples per domain, 30 epochs):
    python scripts/ablation_study.py \\
      --src_dataset Gaze360 --src_root data/Gaze360 \\
      --tgt_root data/MPII \\
      --n_source 3000 --n_target 3000 --epochs 30

  Quick smoke-test (no real data needed):
    python scripts/ablation_study.py --synthetic --epochs 5
"""
    print(guide)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Prepare Gaze360 + MPIIFaceGaze for GazeHub format',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--verify',        action='store_true',
                   help='Check dataset readiness and print sample counts')
    p.add_argument('--guide',         action='store_true',
                   help='Print download instructions')
    p.add_argument('--mpii-from-kaggle', action='store_true',
                   help='Download MPIIFaceGaze from Kaggle + convert')
    p.add_argument('--gaze360-raw',   default=None, metavar='DIR',
                   help='Convert raw MIT Gaze360 directory to GazeHub format')
    p.add_argument('--mpii-raw',      default=None, metavar='DIR',
                   help='Convert raw MPIIFaceGaze (.mat) directory to GazeHub format')
    p.add_argument('--gaze360-link',  default=None, metavar='DIR',
                   help='Link existing GazeHub-format Gaze360 directory')
    p.add_argument('--mpii-link',     default=None, metavar='DIR',
                   help='Link existing GazeHub-format MPII directory')
    p.add_argument('--out-gaze360',   default=GAZE360_DIR)
    p.add_argument('--out-mpii',      default=MPII_DIR)
    args = p.parse_args()

    if len(sys.argv) == 1 or args.guide:
        print_download_guide()
        return

    if args.verify:
        verify()
        return

    if args.mpii_from_kaggle:
        download_mpii_kaggle()

    if args.gaze360_raw:
        print(f'[Gaze360] Converting raw data: {args.gaze360_raw} → {args.out_gaze360}')
        convert_gaze360_raw(args.gaze360_raw, args.out_gaze360)

    if args.mpii_raw:
        print(f'[MPII] Converting raw data: {args.mpii_raw} → {args.out_mpii}')
        convert_mpii_official(args.mpii_raw, args.out_mpii)

    if args.gaze360_link:
        link_gazehub(args.gaze360_link, args.out_gaze360)

    if args.mpii_link:
        link_gazehub(args.mpii_link, args.out_mpii)

    verify()


if __name__ == '__main__':
    main()
