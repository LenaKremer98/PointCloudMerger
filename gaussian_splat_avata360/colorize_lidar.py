#!/usr/bin/env python3
"""Schritt 3: Coloriert die LiDAR-Wolke mit den Splat-Farben.

Nimmt den besten verfuegbaren Auto-Transform (Yaw-ICP vs. FPFH, nach geometrischer
Abdeckung gewaehlt), transformiert das farbige Splat in den LiDAR-Frame und
uebertraegt je LiDAR-Punkt die Farbe des naechsten Splat-Punkts (innerhalb THR).
Punkte ohne Nachbarn -> dunkelgrau (unkoloriert). Schreibt farbiges binaeres PLY.

ACHTUNG: Auto-Ausrichtung ist nur grob, Farben teils fehlregistriert.
"""
import os, json, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUT   = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ_colored.ply")
THR   = 1.0   # max. Abstand LiDAR<->Splat fuer Farbuebernahme [m]

def read_pcd_xyz(path):
    with open(path,"rb") as f:
        while True:
            line=f.readline()
            if line.startswith(b"POINTS"): n=int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f,dtype=np.float32,count=n*3).reshape(n,3).astype(float)

# LiDAR (voll) + Splat (voll, farbig)
L = read_pcd_xyz(LIDAR)
sp = o3d.io.read_point_cloud(SPLAT)
S  = np.asarray(sp.points); SC = np.asarray(sp.colors)
print(f"LiDAR {len(L)} Punkte | Splat {len(S)} Punkte")

# Transform-Kandidaten laden
cands = {}
for fn, key in [("splat_to_lidar.json","T_splat_to_lidar"),
                ("fpfh_result.json","T_splat_to_lidar")]:
    p=os.path.join(ROOT,"output",fn)
    if os.path.exists(p):
        cands[fn]=np.array(json.load(open(p))[key])

# besten Transform nach geometrischer Abdeckung waehlen (LiDAR-Punkte mit Splat<THR)
Ltree = cKDTree(L)
def coverage(T):
    St = (T[:3,:3] @ S.T).T + T[:3,3]
    d,_ = cKDTree(St).query(L, distance_upper_bound=THR, workers=-1)
    return np.mean(np.isfinite(d)), St
best=None
for name,T in cands.items():
    cov,_ = coverage(T)
    print(f"  {name}: Abdeckung @ {THR} m = {cov*100:.1f} %")
    if best is None or cov>best[0]: best=(cov,name,T)
cov,name,T = best
print(f"Gewaehlt: {name}  (Abdeckung {cov*100:.1f} %)")

# Splat in LiDAR-Frame, NN-Farbuebertragung
St = (T[:3,:3] @ S.T).T + T[:3,3]
tree = cKDTree(St)
d, idx = tree.query(L, distance_upper_bound=THR, workers=-1)
hit = np.isfinite(d)
rgb = np.full((len(L),3), 0.12)        # unkoloriert = dunkelgrau
rgb[hit] = SC[idx[hit]]
print(f"Koloriert: {hit.sum()}/{len(L)} ({100*hit.mean():.1f} %), Rest dunkelgrau")

# binaeres PLY schreiben
rgb8=(np.clip(rgb,0,1)*255).round().astype(np.uint8); N=len(L)
with open(OUT,"wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {N}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode())
    rec=np.zeros(N,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                          ("red","u1"),("green","u1"),("blue","u1")])
    rec["x"],rec["y"],rec["z"]=L[:,0],L[:,1],L[:,2]
    rec["red"],rec["green"],rec["blue"]=rgb8[:,0],rgb8[:,1],rgb8[:,2]
    rec.tofile(f)
print(f"-> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
