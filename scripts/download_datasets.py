"""
One-command dataset downloader for cross-domain gaze estimation.

MPIIFaceGaze (target domain)
  Source : https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip
           (21 GB total, no login required)
  Trick  : ZIP64 range-request → download ONLY p00.mat (~1.4 GB = 3000 samples)
  Output : data/MPII/Image/p00/<frame>.jpg  +  data/MPII/Label/p00.label

Gaze360 (source domain)
  Source : http://gaze360.csail.mit.edu/files/gaze360_data.zip
           OR direct torrent / mirror (see --gaze360-url)
  Needs  : GazeHub normalization code (auto-cloned from X-Shi repo)
  Output : data/Gaze360/Image/{train,test}/…  +  data/Gaze360/Label/{train,test}.label

Usage
-----
# Download + convert MPIIFaceGaze (1.4 GB, no registration):
  python scripts/download_datasets.py --mpii

# Download + normalize Gaze360 (requires raw zip, ~22 GB):
  python scripts/download_datasets.py --gaze360 --gaze360-url <URL>

# Both at once:
  python scripts/download_datasets.py --mpii --gaze360 --gaze360-url <URL>

# Just verify existing data:
  python scripts/download_datasets.py --verify
"""

import argparse
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zlib

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, 'data')
MPII_DIR    = os.path.join(DATA_DIR, 'MPII')
G360_DIR    = os.path.join(DATA_DIR, 'Gaze360')

MPII_ZIP_URL = ('https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/'
                'MPIIFaceGaze_normalized.zip')

# ─────────────────────────────────────────────────────────────────────────────
# Progress bar helper
# ─────────────────────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self, total: int, desc: str = ''):
        self.total = total
        self.done  = 0
        self.desc  = desc

    def update(self, n: int):
        self.done += n
        pct = self.done / max(self.total, 1) * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        mb  = self.done / 1024**2
        tot = self.total / 1024**2
        print(f'\r  {self.desc} [{bar}] {pct:5.1f}%  {mb:.0f}/{tot:.0f} MB',
              end='', flush=True)

    def close(self):
        print()


# ─────────────────────────────────────────────────────────────────────────────
# ZIP64 partial-download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url, headers={'Range': f'bytes={start}-{end}'}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _parse_zip64_central_dir(url: str, total_size: int):
    """
    Download only the ZIP64 central directory and parse file entries.
    Returns dict: filename → {local_offset, comp_size, uncomp_size, compress_method}
    """
    # Step 1: tail to find EOCD / ZIP64 EOCD locator
    tail = _http_range(url, total_size - 65536, total_size - 1)

    Z64_LOC = b'PK\x06\x07'
    pos = tail.rfind(Z64_LOC)
    if pos < 0:
        raise RuntimeError('ZIP64 EOCD Locator not found')

    z64_eocd_off = struct.unpack_from('<Q', tail, pos + 8)[0]

    # Step 2: download ZIP64 EOCD record (56 bytes)
    z64_eocd = _http_range(url, z64_eocd_off, z64_eocd_off + 55)
    cd_size   = struct.unpack_from('<Q', z64_eocd, 40)[0]
    cd_offset = struct.unpack_from('<Q', z64_eocd, 48)[0]

    # Step 3: download central directory
    cd_data = _http_range(url, cd_offset, cd_offset + cd_size - 1)

    # Step 4: parse entries
    entries = {}
    i = 0
    while i < len(cd_data):
        if cd_data[i:i+4] != b'PK\x01\x02':
            break
        comp_method  = struct.unpack_from('<H', cd_data, i+10)[0]
        comp_size    = struct.unpack_from('<I', cd_data, i+20)[0]
        uncomp_size  = struct.unpack_from('<I', cd_data, i+24)[0]
        fname_len    = struct.unpack_from('<H', cd_data, i+28)[0]
        extra_len    = struct.unpack_from('<H', cd_data, i+30)[0]
        comment_len  = struct.unpack_from('<H', cd_data, i+32)[0]
        local_offset = struct.unpack_from('<I', cd_data, i+42)[0]
        fname        = cd_data[i+46:i+46+fname_len].decode('utf-8', errors='replace')

        # Resolve ZIP64 extended info
        ex_i = i + 46 + fname_len
        ex_end = ex_i + extra_len
        j = ex_i
        while j < ex_end - 3:
            tag  = struct.unpack_from('<H', cd_data, j)[0]
            esz  = struct.unpack_from('<H', cd_data, j+2)[0]
            if tag == 0x0001:
                vals = [struct.unpack_from('<Q', cd_data, j+4+k*8)[0]
                        for k in range(esz // 8)]
                if uncomp_size  == 0xFFFFFFFF and vals: uncomp_size  = vals.pop(0)
                if comp_size    == 0xFFFFFFFF and vals: comp_size    = vals.pop(0)
                if local_offset == 0xFFFFFFFF and vals: local_offset = vals.pop(0)
            j += 4 + esz

        entries[fname] = dict(local_offset=local_offset, comp_size=comp_size,
                              uncomp_size=uncomp_size, method=comp_method)
        i += 46 + fname_len + extra_len + comment_len

    return entries


def _download_zip_entry(url: str, entry: dict, progress_desc: str) -> bytes:
    """
    Download a single file from a remote ZIP via HTTP range requests.
    Returns the raw (decompressed) bytes.
    """
    # Read local file header to get actual data start
    lhdr = _http_range(url, entry['local_offset'], entry['local_offset'] + 29)
    if lhdr[:4] != b'PK\x03\x04':
        raise RuntimeError('Invalid local file header signature')
    fname_len = struct.unpack_from('<H', lhdr, 26)[0]
    extra_len = struct.unpack_from('<H', lhdr, 28)[0]
    data_start = entry['local_offset'] + 30 + fname_len + extra_len

    comp_size = entry['comp_size']
    prog = _Progress(comp_size, progress_desc)

    # Stream download in 4 MB chunks
    chunk = 4 * 1024 * 1024
    compressed = bytearray()
    pos = data_start
    while pos < data_start + comp_size:
        end = min(pos + chunk - 1, data_start + comp_size - 1)
        data = _http_range(url, pos, end)
        compressed.extend(data)
        prog.update(len(data))
        pos += len(data)
    prog.close()

    if entry['method'] == 0:          # STORE
        return bytes(compressed)
    elif entry['method'] == 8:        # DEFLATE
        return zlib.decompress(bytes(compressed), -15)
    else:
        raise RuntimeError(f"Unsupported compression method: {entry['method']}")


# ─────────────────────────────────────────────────────────────────────────────
# MPIIFaceGaze: download p00.mat → convert to GazeHub format
# ─────────────────────────────────────────────────────────────────────────────

def _load_mat_arrays(mat_path: str):
    """
    Load images + labels from a MPIIFaceGaze_normalized .mat file.
    Supports MATLAB v5 (scipy) and v7.3 HDF5 (h5py).

    MPIIFaceGaze_normalized structure (v7.3 HDF5):
      Data/data   : (N, 3, H, W)  float32, values 0-255
      Data/label  : (N, 16)       float32
                    col 0 = gaze_yaw   (rad)
                    col 1 = gaze_pitch (rad)
                    col 2 = head_yaw   (rad)
                    col 3 = head_pitch (rad)
                    col 4-15 = screen / landmark pixel coords (unused here)

    Returns: images (N, 3, H, W) uint8, labels (N, 4) float32
    """
    try:
        import h5py
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'h5py', '-q'], check=True)
        import h5py

    # Try h5py first (covers v7.3; most common for MPIIFaceGaze_normalized.zip)
    try:
        with h5py.File(mat_path, 'r') as f:
            if 'Data' in f:
                images_raw = f['Data']['data'][:]   # (N, 3, H, W) float32
                labels_raw = f['Data']['label'][:]  # (N, 16) float32
            elif 'data' in f and 'label' in f:
                images_raw = f['data'][:]
                labels_raw = f['label'][:]
            else:
                raise ValueError(f'h5py: unknown keys {list(f.keys())}')
        images = np.clip(images_raw, 0, 255).astype(np.uint8)  # float32→uint8
        labels = labels_raw[:, :4].astype(np.float32)          # keep cols 0-3 only
        return images, labels
    except Exception as e_h5:
        pass

    # Fall back to scipy (v5 format)
    try:
        import scipy.io as sio
        mat = sio.loadmat(mat_path)
        if 'Data' in mat:
            ds = mat['Data'][0, 0]
            images_raw = ds['data']
            labels_raw = ds['label']
        elif 'data' in mat and 'label' in mat:
            images_raw = mat['data']
            labels_raw = mat['label']
        else:
            raise ValueError(f'scipy: unknown keys {list(mat.keys())}')
        images = np.clip(images_raw, 0, 255).astype(np.uint8)
        labels = labels_raw[:, :4].astype(np.float32)
        return images, labels
    except NotImplementedError:
        raise RuntimeError(
            f'Cannot load {mat_path}: requires h5py for MATLAB v7.3 format')


def _mat_to_gazehub(mat_path: str, subject: str, out_root: str):
    """Convert a MPIIFaceGaze .mat file to GazeHub format (JPEG + .label)."""
    images, labels = _load_mat_arrays(mat_path)
    # images: (N, 3, H, W) uint8   labels: (N, 4) [gaze_yaw, gaze_pitch, head_yaw, head_pitch]

    n = images.shape[0]
    img_dir = os.path.join(out_root, 'Image', subject)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'Label'), exist_ok=True)

    lines = ['face gaze2d head2d\n']
    for i in range(n):
        img = images[i]                       # (3, H, W)
        img = img.transpose(1, 2, 0)          # CHW → HWC
        img_bgr = img[:, :, ::-1]             # RGB → BGR for cv2

        fname = f'{i:04d}.jpg'
        cv2.imwrite(os.path.join(img_dir, fname), img_bgr)

        gaze_yaw, gaze_pitch = float(labels[i, 0]), float(labels[i, 1])
        head_yaw,  head_pitch = float(labels[i, 2]), float(labels[i, 3])

        rel = os.path.join(subject, fname)
        lines.append(f'{rel} {gaze_yaw:.6f},{gaze_pitch:.6f} '
                     f'{head_yaw:.6f},{head_pitch:.6f}\n')

    lbl_path = os.path.join(out_root, 'Label', subject + '.label')
    with open(lbl_path, 'w') as fh:
        fh.writelines(lines)

    print(f'  {subject}: {n} samples → {img_dir}')
    return n


def download_mpii(subjects=None, out_root=None):
    """
    Download selected MPIIFaceGaze subjects from MPI ZIP (range request, no registration).

    subjects : list of 'p00'…'p14'; default = ['p00'] (3000 samples)
    out_root : output directory; default = data/MPII/
    """
    subjects = subjects or ['p00']
    out_root = out_root or MPII_DIR

    url = MPII_ZIP_URL
    print(f'[MPIIFaceGaze] Fetching ZIP index from MPI ...')
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers['Content-Length'])
    print(f'  Total ZIP size: {total/1024**3:.2f} GB')

    entries = _parse_zip64_central_dir(url, total)
    print(f'  ZIP entries: {len(entries)}')

    total_samples = 0
    for subj in subjects:
        key = f'MPIIFaceGaze_normalizad/{subj}.mat'   # note: typo in original zip
        if key not in entries:
            # Try alternative spellings
            alt = f'MPIIFaceGaze_normalized/{subj}.mat'
            if alt in entries:
                key = alt
            else:
                avail = [k for k in entries if subj in k]
                print(f'  [WARN] {subj}.mat not found. Available: {avail}')
                continue

        e = entries[key]

        # Cache .mat file on disk so we can re-run without re-downloading
        cache_dir = os.path.join(out_root, '_mat_cache')
        os.makedirs(cache_dir, exist_ok=True)
        mat_path = os.path.join(cache_dir, f'{subj}.mat')

        if os.path.isfile(mat_path):
            print(f'\n[MPIIFaceGaze] Using cached {mat_path}')
        else:
            print(f'\n[MPIIFaceGaze] Downloading {key}  ({e["uncomp_size"]/1024**3:.2f} GB) ...')
            mat_bytes = _download_zip_entry(url, e, subj)
            with open(mat_path, 'wb') as fh:
                fh.write(mat_bytes)
            del mat_bytes  # free memory immediately

        print(f'  Converting {subj}.mat → GazeHub format ...')
        n = _mat_to_gazehub(mat_path, subj, out_root)
        total_samples += n

    print(f'\n[MPIIFaceGaze] Done.  {total_samples} samples in {out_root}')


# ─────────────────────────────────────────────────────────────────────────────
# Gaze360: download raw + normalize with X-Shi pipeline
# ─────────────────────────────────────────────────────────────────────────────

def download_gaze360(raw_url: str, out_root: str = None):
    """
    Download raw Gaze360 and normalize to GazeHub format using the
    X-Shi Data-Normalization-Gaze-Estimation pipeline.

    raw_url  : direct URL to gaze360_data.zip (e.g. from MIT mirror)
    out_root : output directory; default = data/Gaze360/
    """
    out_root = out_root or G360_DIR
    norm_repo = os.path.join(DATA_DIR, '_normalization_code')
    raw_dir   = os.path.join(DATA_DIR, '_gaze360_raw')
    os.makedirs(raw_dir, exist_ok=True)

    # Step 1: clone normalization code
    if not os.path.isdir(norm_repo):
        print('[Gaze360] Cloning X-Shi normalization code ...')
        subprocess.run(
            ['git', 'clone', '--depth=1',
             'https://github.com/X-Shi/Data-Normalization-Gaze-Estimation.git',
             norm_repo],
            check=True
        )
    else:
        print('[Gaze360] Normalization code already present.')

    # Step 2: download raw zip
    raw_zip = os.path.join(raw_dir, 'gaze360_data.zip')
    if not os.path.isfile(raw_zip):
        print(f'[Gaze360] Downloading raw data from: {raw_url}')
        with urllib.request.urlopen(raw_url, timeout=60) as r:
            total = int(r.headers.get('Content-Length', 0))
            prog  = _Progress(total, 'Gaze360 raw')
            with open(raw_zip, 'wb') as f:
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    prog.update(len(chunk))
            prog.close()
    else:
        print(f'[Gaze360] Raw zip already downloaded: {raw_zip}')

    # Step 3: extract
    print('[Gaze360] Extracting ...')
    subprocess.run(['unzip', '-q', '-n', raw_zip, '-d', raw_dir], check=True)

    # Step 4: find the normalization script for Gaze360 face-based 224×224
    gaze360_norm = None
    for root, dirs, files in os.walk(norm_repo):
        for f in files:
            if 'gaze360' in f.lower() and f.endswith('.py'):
                gaze360_norm = os.path.join(root, f)
                break

    if gaze360_norm is None:
        print('[Gaze360] Normalization script not found in repo.')
        print('  Please run the normalization manually:')
        print(f'    cd {norm_repo}')
        print('  Follow the README for Gaze360 face-based 224×224 normalization.')
        return

    # Step 5: run normalization
    raw_imgs = os.path.join(raw_dir, 'imgs')
    os.makedirs(out_root, exist_ok=True)
    print(f'[Gaze360] Running normalization: {gaze360_norm}')
    subprocess.run(
        [sys.executable, gaze360_norm,
         '--input',  raw_imgs,
         '--output', out_root],
        check=True
    )
    print(f'[Gaze360] Done.  Output at {out_root}')


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify():
    ok = True
    print('\n── Dataset verification ───────────────────────────────────')
    for name, d, split_key in [
        ('Gaze360',      G360_DIR,  'train'),
        ('MPIIFaceGaze', MPII_DIR,  'p00'),
    ]:
        img_dir = os.path.join(d, 'Image')
        lbl_dir = os.path.join(d, 'Label')
        lbl_f   = os.path.join(lbl_dir, split_key + '.label')
        if os.path.isdir(img_dir) and os.path.isfile(lbl_f):
            with open(lbl_f) as f:
                n = max(0, len(f.readlines()) - 1)
            print(f'  {name:<15} ✓  {lbl_f}  ({n} samples in {split_key})')
        else:
            print(f'  {name:<15} ✗  not found at {d}')
            ok = False
    print()
    if ok:
        print('  Ready. Run:')
        print('    python scripts/ablation_study.py \\')
        print('      --src_dataset Gaze360 --src_root data/Gaze360 \\')
        print('      --tgt_root data/MPII --n_source 3000 --n_target 3000')
    else:
        print('  Use  python scripts/download_datasets.py --help  for options.')
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Auto-download Gaze360 + MPIIFaceGaze in GazeHub format',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--mpii',          action='store_true',
                   help='Download MPIIFaceGaze p00 (~1.4 GB range-request, no login)')
    p.add_argument('--mpii-subjects', default='p00',
                   help='Comma-separated subjects to download (default: p00 = 3000 samples)')
    p.add_argument('--mpii-out',      default=MPII_DIR)
    p.add_argument('--gaze360',       action='store_true',
                   help='Download + normalize Gaze360 (requires --gaze360-url)')
    p.add_argument('--gaze360-url',   default=None,
                   help='Direct URL to gaze360_data.zip')
    p.add_argument('--gaze360-out',   default=G360_DIR)
    p.add_argument('--verify',        action='store_true',
                   help='Check dataset readiness')
    args = p.parse_args()

    if len(sys.argv) == 1:
        p.print_help()
        return

    if args.verify:
        verify()
        return

    if args.mpii:
        subjects = [s.strip() for s in args.mpii_subjects.split(',')]
        download_mpii(subjects=subjects, out_root=args.mpii_out)

    if args.gaze360:
        if not args.gaze360_url:
            print('[ERROR] --gaze360-url is required for --gaze360.')
            print('  Gaze360 has no pre-processed mirror.  Options:')
            print('  1. Register at https://ait.ethz.ch/xgaze to get ETH-XGaze')
            print('     (already in GazeHub format, no conversion needed)')
            print('  2. Download raw Gaze360 from http://gaze360.csail.mit.edu')
            print('     then pass the URL to --gaze360-url')
            sys.exit(1)
        download_gaze360(args.gaze360_url, out_root=args.gaze360_out)

    verify()


if __name__ == '__main__':
    main()
