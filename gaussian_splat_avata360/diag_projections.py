#!/usr/bin/env python3
"""Diagnose ohne matplotlib: rastert Top-Down (XY) und Seite (XZ) beider Wolken,
Hoehe-kodiert (turbo-aehnlich), gleicher Metermassstab. Speichert PNG via PIL."""
import os, numpy as np, open3d as o3d
from PIL import Image

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
PXM   = 4   # Pixel pro Meter

def read_pcd_xyz(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"POINTS"): n = int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f, dtype=np.float32, count=n*3).reshape(n, 3).astype(float)

def turbo(v):  # v in [0,1] -> RGB uint8, grobe turbo-Approx
    v = np.clip(v, 0, 1)
    r = np.clip(1.5 - abs(4*v - 3), 0, 1)
    g = np.clip(1.5 - abs(4*v - 2), 0, 1)
    b = np.clip(1.5 - abs(4*v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)

def raster(P, ai, bi, ci, bounds=None):
    """Projektion auf Achsen ai,bi; Farbe nach Achse ci. bounds=(amin,amax,bmin,bmax)."""
    a, b, c = P[:, ai], P[:, bi], P[:, ci]
    if bounds is None:
        amin, amax = np.percentile(a, [0.5, 99.5]); bmin, bmax = np.percentile(b, [0.5, 99.5])
    else:
        amin, amax, bmin, bmax = bounds
    W = max(1, int((amax - amin) * PXM)); H = max(1, int((bmax - bmin) * PXM))
    ia = ((a - amin) / (amax - amin) * (W - 1)).astype(int)
    ib = ((b - bmin) / (bmax - bmin) * (H - 1)).astype(int)
    ok = (ia >= 0) & (ia < W) & (ib >= 0) & (ib < H)
    ia, ib, c = ia[ok], ib[ok], c[ok]
    cn = (c - np.percentile(c, 1)) / (np.percentile(c, 99) - np.percentile(c, 1) + 1e-9)
    col = turbo(cn)
    img = np.zeros((H, W, 3), np.uint8)
    img[H - 1 - ib, ia] = col      # b nach oben
    return img, (amin, amax, bmin, bmax)

def label(img, txt):
    from PIL import ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8*len(txt)+8, 18], fill=(0,0,0)); d.text((4, 3), txt, fill=(255,255,255))
    return np.asarray(im)

L = read_pcd_xyz(LIDAR)
S = np.asarray(o3d.io.read_point_cloud(SPLAT).points)
lo, hi = np.percentile(S, [1, 99], axis=0)
S = S[np.all((S >= lo) & (S <= hi), axis=1)]

tiles = []
for name, P in [("LiDAR XY (Farbe=Z)", L), ("Splat XY (Farbe=Z)", S)]:
    img, _ = raster(P, 0, 1, 2); tiles.append(label(img, name))
for name, P in [("LiDAR XZ (Farbe=Y)", L), ("Splat XZ (Farbe=Y)", S)]:
    img, _ = raster(P, 0, 2, 1); tiles.append(label(img, name))

# auf gemeinsame Breite bringen, untereinander stapeln
W = max(t.shape[1] for t in tiles)
def pad(t):
    out = np.zeros((t.shape[0]+10, W, 3), np.uint8)
    out[:t.shape[0], :t.shape[1]] = t; return out
canvas = np.concatenate([pad(t) for t in tiles], 0)
out = os.path.join(ROOT, "output", "diag_projections.png")
Image.fromarray(canvas).save(out)
print("->", out, canvas.shape)
print("LiDAR  bounds X", np.round([L[:,0].min(), L[:,0].max()],1),
      "Y", np.round([L[:,1].min(), L[:,1].max()],1), "Z", np.round([L[:,2].min(), L[:,2].max()],1))
print("Splat  bounds X", np.round([S[:,0].min(), S[:,0].max()],1),
      "Y", np.round([S[:,1].min(), S[:,1].max()],1), "Z", np.round([S[:,2].min(), S[:,2].max()],1))
