#!/usr/bin/env python3
"""venv: exportiert die M4T-COLMAP-3D-Punkte (xyz) nach NPZ (fuer ICP-Verfeinerung)."""
import os, numpy as np, pycolmap
ROOT=os.path.dirname(os.path.abspath(__file__))
rec=pycolmap.Reconstruction(os.path.join(ROOT,"m4t_work","sparse","0"))
pts=list(rec.points3D.values())
xyz=np.array([p.xyz for p in pts])
rgb=np.array([p.color for p in pts])   # uint8 RGB
np.savez(os.path.join(ROOT,"output","m4t_points3d.npz"), xyz=xyz, rgb=rgb)
print(f"{len(xyz)} COLMAP-Punkte exportiert (mit Farbe).")
