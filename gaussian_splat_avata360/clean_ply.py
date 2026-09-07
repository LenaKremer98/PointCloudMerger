#!/usr/bin/env python3
"""Entfernt Ausreisser-/Floater-Gaussians, damit Viewer korrekt auf die Szene zentrieren.
Filter: (1) innerhalb Radius R um das Kamerazentrum, (2) Opazitaet > thr, (3) Skala < smax."""
import os, numpy as np
from plyfile import PlyData, PlyElement
import pycolmap

ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("DATA_DIR", ROOT)
OUTD=os.environ.get("OUT_DIR", os.path.join(ROOT,"output"))
IN =os.path.join(OUTD,"avata360_splat.ply")
OUT=os.path.join(OUTD,"avata360_splat_clean.ply")

# Kamerazentrum aus COLMAP
rec=pycolmap.Reconstruction(os.path.join(DATA,"sparse","0"))
C=[]
for im in rec.images.values():
    M=np.asarray(im.cam_from_world().matrix(),np.float32); R=M[:,:3]; t=M[:,3]
    C.append(-R.T@t)
C=np.stack(C); ctr=np.median(C,0)
cam_radius=np.percentile(np.linalg.norm(C-ctr,axis=1), 90)   # robust gg. Ausreisser-Kameras
print(f"Kamerazentrum {ctr.round(2)}, robuster Kamera-Radius {cam_radius:.2f}")

ply=PlyData.read(IN); v=ply["vertex"]; data=v.data; N=len(data)
xyz=np.stack([v["x"],v["y"],v["z"]],1)
op=1/(1+np.exp(-v["opacity"]))
sc=np.exp(np.stack([v["scale_0"],v["scale_1"],v["scale_2"]],1)).max(1)
d=np.linalg.norm(xyz-ctr,axis=1)

# Szenenradius: deckt sichtbare Umgebung, kappt Sky/Escapees
R=max(cam_radius*12, 70.0)
SMAX=3.0          # Riesen-Blobs raus
THR=0.08          # Fast-Transparente raus
keep=(d<R)&(op>THR)&(sc<SMAX)
print(f"Radius-Schwelle R={R:.1f}, smax={SMAX:.1f}, opac>{THR}")
print(f"behalten: {keep.sum()}/{N} ({100*keep.mean():.1f}%)")
kept=data[keep]
PlyData([PlyElement.describe(kept,"vertex")],
        text=False,byte_order="<").write(OUT)
xk=xyz[keep]
print("neue Ausdehnung:", (xk.max(0)-xk.min(0)).round(2))
print(f"-> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
