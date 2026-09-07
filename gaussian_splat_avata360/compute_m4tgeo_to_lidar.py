#!/usr/bin/env python3
"""venv: Similarity m4t_geo-COLMAP -> LiDAR-Frame.
m4t_geo-Kamerazentren <-> dieselben Kameras im LiDAR-Frame (m4t_work-Zentren via m4t_affine).
Umeyama -> s,R,t. -> output/m4tgeo_to_lidar.json"""
import os, json, numpy as np, pycolmap
ROOT=os.path.dirname(os.path.abspath(__file__))
def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=np.trace(np.diag(D)@S)/((Ss**2).sum()/len(src)); return s,R,mu_d-s*R@mu_s

# m4t_geo Kamerazentren
rec=pycolmap.Reconstruction(os.path.join(ROOT,"m4t_geo","sparse","0"))
Cg={}
for im in rec.images.values():
    M=np.asarray(im.cam_from_world().matrix(),float); Cg[im.name]=-M[:,:3].T@M[:,3]
# m4t_work Zentren -> LiDAR via m4t_affine
cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
aff=json.load(open(os.path.join(ROOT,"output","m4t_affine.json"))); A=np.array(aff["A"]); b=np.array(aff["b"])
Cw={n:(A@c+b) for n,c in zip(cam["names"],cam["C"])}
names=[n for n in Cg if n in Cw]
src=np.array([Cg[n] for n in names]); dst=np.array([Cw[n] for n in names])
s,R,t=umeyama(src,dst)
res=np.linalg.norm((s*(R@src.T).T+t)-dst,axis=1)
print(f"m4t_geo->LiDAR: s={s:.4f}, {len(names)} Kameras, Residuum mean {res.mean():.3f} m max {res.max():.3f} m")
T=np.eye(4); T[:3,:3]=s*R; T[:3,3]=t
json.dump({"s":float(s),"R":R.tolist(),"t":t.tolist(),"T":T.tolist()},
          open(os.path.join(ROOT,"output","m4tgeo_to_lidar.json"),"w"),indent=2)
print("-> output/m4tgeo_to_lidar.json")
