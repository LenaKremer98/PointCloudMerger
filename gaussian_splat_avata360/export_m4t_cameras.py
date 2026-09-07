#!/usr/bin/env python3
"""venv: exportiert M4T-COLMAP-Kameras (Pose + Intrinsics + Zentrum) nach NPZ."""
import os, numpy as np, pycolmap
ROOT=os.path.dirname(os.path.abspath(__file__))
rec=pycolmap.Reconstruction(os.path.join(ROOT,"m4t_work","sparse","0"))
names=[]; Rcw=[]; tcw=[]; C=[]; intr=[]
for im in rec.images.values():
    M=np.asarray(im.cam_from_world().matrix(),float)   # 3x4 world->cam
    R=M[:,:3]; t=M[:,3]
    cam=rec.cameras[im.camera_id]
    p=list(cam.params)   # SIMPLE_RADIAL: f, cx, cy, k
    names.append(im.name); Rcw.append(R); tcw.append(t); C.append(-R.T@t)
    intr.append([cam.width, cam.height]+p)
np.savez(os.path.join(ROOT,"output","m4t_cameras.npz"),
         names=np.array(names), Rcw=np.array(Rcw), tcw=np.array(tcw),
         C=np.array(C), intr=np.array(intr))
print(f"{len(names)} Kameras exportiert. intr[0]={intr[0]}")
