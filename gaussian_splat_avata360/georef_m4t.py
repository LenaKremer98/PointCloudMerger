#!/usr/bin/env python3
"""M4T-Rekonstruktion per RTK-GPS nach ENU (metrisch, z-oben) und als eigenes
Datenverzeichnis m4t_geo/ (sparse/0 + images/) ablegen."""
import os, json, glob, numpy as np, pycolmap
from geo_utils import to_enu, umeyama, mat2quat

ROOT = os.path.dirname(os.path.abspath(__file__))
M_SP = os.path.join(ROOT, "m4t_work", "sparse", "0")
M_IMG= os.path.join(ROOT, "m4t_work", "images")
GPS  = json.load(open(os.path.join(ROOT, "m4t_work", "m4t_gps.json")))
OUT  = os.path.join(ROOT, "m4t_geo")
os.makedirs(os.path.join(OUT, "sparse", "0"), exist_ok=True)
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)

lat0 = np.mean([v[0] for v in GPS.values()]); lon0 = np.mean([v[1] for v in GPS.values()])
rec = pycolmap.Reconstruction(M_SP)
C, G = [], []
for im in rec.images.values():
    g = GPS.get(im.name)
    if g is None: continue
    M = np.asarray(im.cam_from_world().matrix(), float)
    C.append(-M[:,:3].T @ M[:,3]); G.append(to_enu(g[0], g[1], g[2], lat0, lon0))
C, G = np.stack(C), np.stack(G)
s, R, t = umeyama(C, G); q = mat2quat(R)
rec.transform(pycolmap.Sim3d(float(s), pycolmap.Rotation3d(np.array([q[1],q[2],q[3],q[0]])), t.astype(float)))
C2 = np.stack([-(np.asarray(im.cam_from_world().matrix(),float))[:,:3].T @
                np.asarray(im.cam_from_world().matrix(),float)[:,3] for im in rec.images.values()])
print(f"Georeferenziert: s={s:.3f}, Residuum {np.linalg.norm(C2-G,axis=1).mean():.2f} m (RTK)")
rec.write(os.path.join(OUT, "sparse", "0"))
n = 0
for src in glob.glob(M_IMG+"/*.JPG")+glob.glob(M_IMG+"/*.jpg"):
    dst = os.path.join(OUT, "images", os.path.basename(src))
    if not os.path.exists(dst): os.symlink(os.path.abspath(src), dst); n += 1
print(f"{rec.num_images()} Bilder, {rec.num_points3D()} Punkte -> {OUT}; {n} Bilder verlinkt")
