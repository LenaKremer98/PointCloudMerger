#!/usr/bin/env python3
"""Coloriert die LiDAR-Wolke mit dem in manual_align.json gespeicherten Transform.
Optional ICP-Feinschliff (Standard an).  --no-icp schaltet ihn ab."""
import os, sys, json, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUT   = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ_colored.ply")
OUTD  = os.path.join(ROOT, "output")
USE_ICP = "--no-icp" not in sys.argv
THR=1.0
for a in sys.argv[1:]:
    if a.startswith("--thr="): THR=float(a.split("=")[1])

def read_pcd_xyz(path):
    with open(path,"rb") as f:
        while True:
            line=f.readline()
            if line.startswith(b"POINTS"): n=int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f,dtype=np.float32,count=n*3).reshape(n,3).astype(float)

T=np.array(json.load(open(os.path.join(OUTD,"manual_align.json")))["T_splat_to_lidar"])
L=read_pcd_xyz(LIDAR)
sp=o3d.io.read_point_cloud(SPLAT); S=np.asarray(sp.points); SC=np.asarray(sp.colors)
lo,hi=np.percentile(S,[1,99],axis=0); keep=np.all((S>=lo)&(S<=hi),1)

if USE_ICP:
    src=o3d.geometry.PointCloud(); src.points=o3d.utility.Vector3dVector(S[keep])
    tgt=o3d.geometry.PointCloud(); tgt.points=o3d.utility.Vector3dVector(L)
    src=src.voxel_down_sample(0.5); tgt=tgt.voxel_down_sample(0.5)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,max_nn=30))
    icp=o3d.pipelines.registration.registration_icp(
        src,tgt,1.5,T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    print(f"ICP: fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.3f}")
    T=np.asarray(icp.transformation)

Sk=S[keep]; SCk=SC[keep]          # Floater raus -> saubere Farben
St=(T[:3,:3]@Sk.T).T+T[:3,3]
d,idx=cKDTree(St).query(L,distance_upper_bound=THR,workers=-1)
hit=np.isfinite(d)
rgb=np.full((len(L),3),0.42); rgb[hit]=SCk[idx[hit]]   # unkoloriert = sichtbares Grau
print(f"Koloriert: {hit.sum()}/{len(L)} ({100*hit.mean():.1f} %)")

rgb8=(np.clip(rgb,0,1)*255).round().astype(np.uint8); N=len(L)
with open(OUT,"wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {N}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode())
    rec=np.zeros(N,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
    rec["x"],rec["y"],rec["z"]=L[:,0],L[:,1],L[:,2]
    rec["red"],rec["green"],rec["blue"]=rgb8[:,0],rgb8[:,1],rgb8[:,2]
    rec.tofile(f)
print(f"-> {OUT}")
