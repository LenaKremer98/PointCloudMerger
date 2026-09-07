#!/usr/bin/env python3
"""Rechnet aus den Picks (picks_lidar.json/picks_splat.json) den Transform
Splat->LiDAR (Umeyama 7-DOF inkl. Skalierung), verfeinert per ICP und coloriert
die LiDAR-Wolke neu. Schreibt manual_align.json, merged_FinlaDRZ_colored.ply,
colored_topdown.png.

  python3 compute_manual.py            # Pick+ICP (Standard)
  python3 compute_manual.py --no-icp   # nur Picks, ohne ICP-Feinschliff
"""
import os, sys, json, numpy as np, open3d as o3d
from scipy.spatial import cKDTree

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUTD  = os.path.join(ROOT, "output")
OUT   = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ_colored.ply")
THR   = 1.0
USE_ICP = "--no-icp" not in sys.argv

def read_pcd_xyz(path):
    with open(path,"rb") as f:
        while True:
            line=f.readline()
            if line.startswith(b"POINTS"): n=int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f,dtype=np.float32,count=n*3).reshape(n,3).astype(float)

def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Ss, Dd = src-mu_s, dst-mu_d
    C = Dd.T @ Ss / len(src)
    U,D,Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: S[2,2] = -1
    R = U @ S @ Vt
    var = (Ss**2).sum()/len(src)
    s = np.trace(np.diag(D)@S)/var
    t = mu_d - s*R@mu_s
    T = np.eye(4); T[:3,:3] = s*R; T[:3,3] = t
    return T, s

PL = np.array(json.load(open(os.path.join(OUTD,"picks_lidar.json"))))   # dst
PS = np.array(json.load(open(os.path.join(OUTD,"picks_splat.json"))))   # src
assert len(PL)==len(PS) and len(PL)>=3, f"Picks passen nicht: {len(PL)} vs {len(PS)}"

T, s = umeyama(PS, PL)
res = np.linalg.norm((T[:3,:3]@PS.T).T + T[:3,3] - PL, axis=1)
print(f"Picks: {len(PL)} | Skalenkorrektur s={s:.4f}")
print(f"Pick-Residuum: mean {res.mean():.2f} m, max {res.max():.2f} m")

# --- Daten laden ---
L = read_pcd_xyz(LIDAR)
sp = o3d.io.read_point_cloud(SPLAT)
S = np.asarray(sp.points); SC = np.asarray(sp.colors)
lo,hi = np.percentile(S,[1,99],axis=0)
keep = np.all((S>=lo)&(S<=hi),1)

T_pick = T.copy()
if USE_ICP:
    src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(S[keep])
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(L)
    src = src.voxel_down_sample(0.5); tgt = tgt.voxel_down_sample(0.5)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,max_nn=30))
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, 1.5, T_pick,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    print(f"ICP-Feinschliff: fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.3f}")
    T = np.asarray(icp.transformation)

json.dump({"T_splat_to_lidar":T.tolist(),"T_pick_only":T_pick.tolist(),
           "scale":float(s),"pick_res_mean_m":float(res.mean()),"use_icp":USE_ICP},
          open(os.path.join(OUTD,"manual_align.json"),"w"), indent=2)

# --- colorieren ---
St = (T[:3,:3]@S.T).T + T[:3,3]
d, idx = cKDTree(St).query(L, distance_upper_bound=THR, workers=-1)
hit = np.isfinite(d)
rgb = np.full((len(L),3), 0.12); rgb[hit] = SC[idx[hit]]
print(f"Koloriert: {hit.sum()}/{len(L)} ({100*hit.mean():.1f} %)")

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
print(f"-> {OUT}")
print(f"-> {os.path.join(OUTD,'manual_align.json')}")
