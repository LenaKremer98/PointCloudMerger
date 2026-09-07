#!/usr/bin/env python3
"""Verfeinert die M4T->LiDAR-Ausrichtung per ICP.

Bringt die M4T-COLMAP-3D-Punkte ueber die aktuelle Kette (Umeyama->ENU, dann manuelle
Ausrichtung) in den LiDAR-Frame und richtet sie per ICP exakt an der LiDAR-Wolke aus.
Schreibt manual_align_refined.json = T_refine @ T_manual (enu->lidar verfeinert)."""
import os, json, numpy as np, open3d as o3d
ROOT=os.path.dirname(os.path.abspath(__file__)); R_EARTH=6378137.0
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)
def to_enu(lat,lon,up,lat0,lon0):
    e=np.radians(lon-lon0)*np.cos(np.radians(lat0))*R_EARTH; n=np.radians(lat-lat0)*R_EARTH
    return np.array([e,n,up])
def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=np.trace(np.diag(D)@S)/((Ss**2).sum()/len(src)); t=mu_d-s*R@mu_s
    return s,R,t

cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
names=cam["names"]; Ccol=cam["C"]
gps=json.load(open(os.path.join(ROOT,"m4t_work","m4t_gps.json")))
geo=json.load(open(os.path.join(ROOT,"output","splat_georef.json"))); lat0,lon0=geo["lat0"],geo["lon0"]
G=np.array([to_enu(gps[n][0],gps[n][1],gps[n][2],lat0,lon0) for n in names])
s_m,R_m,t_m=umeyama(Ccol,G)                       # COLMAP->ENU
Tman=np.array(json.load(open(os.path.join(ROOT,"output","manual_align.json")))["T_splat_to_lidar"])

# M4T-Punkte -> ENU -> LiDAR (Startlage)
Pc=np.load(os.path.join(ROOT,"output","m4t_points3d.npz"))["xyz"]
Penu=(s_m*(R_m@Pc.T).T)+t_m
Plid=(Tman[:3,:3]@Penu.T).T+Tman[:3,3]

src=o3d.geometry.PointCloud(); src.points=o3d.utility.Vector3dVector(Plid)
src,_=src.remove_statistical_outlier(20,2.0)
L=read_pcd(LIDAR)
tgt=o3d.geometry.PointCloud(); tgt.points=o3d.utility.Vector3dVector(L)
tgt=tgt.voxel_down_sample(0.3)
tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,max_nn=30))

ev=o3d.pipelines.registration.evaluate_registration(src,tgt,1.0,np.eye(4))
print(f"VOR ICP:  fitness={ev.fitness:.3f} rmse={ev.inlier_rmse:.3f}")
T_ref=np.eye(4)
for thr,it in [(5.0,60),(2.0,60),(1.0,80),(0.5,80)]:
    r=o3d.pipelines.registration.registration_icp(
        src,tgt,thr,T_ref,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
    T_ref=np.asarray(r.transformation)
    print(f"  ICP thr={thr}: fitness={r.fitness:.3f} rmse={r.inlier_rmse:.3f}")

Tref_man=T_ref@Tman
json.dump({"T_splat_to_lidar":Tref_man.tolist(),"T_refine":T_ref.tolist(),
           "note":"manual_align verfeinert per ICP (M4T-Punkte<->LiDAR)"},
          open(os.path.join(ROOT,"output","manual_align_refined.json"),"w"),indent=2)
print("-> output/manual_align_refined.json")
