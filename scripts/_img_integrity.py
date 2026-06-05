"""Scan each MPII subject: how many JPEGs are readable vs corrupt."""
import os, glob, cv2
for sub in ['p00','p01','p02','p03','p04']:
    fs = sorted(glob.glob(f'data/MPII/Image/{sub}/*.jpg'))
    bad = 0; sizes = []
    for f in fs:
        sizes.append(os.path.getsize(f))
        if cv2.imread(f) is None:
            bad += 1
    n = len(fs)
    import statistics
    print(f'{sub}: {n} jpgs, {bad} unreadable ({100*bad/max(n,1):.0f}%), '
          f'size min={min(sizes)}B max={max(sizes)}B median={int(statistics.median(sizes))}B')
