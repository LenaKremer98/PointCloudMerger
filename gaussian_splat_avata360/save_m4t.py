#!/usr/bin/env python3
"""Liest die aktuelle TF map->model (Greifer), rekonstruiert den direkten
colmap->lidar Affin (A,b) wie im Node und schreibt m4t_affine.json."""
import os, json, time, numpy as np, rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as Rot
ROOT=os.path.dirname(os.path.abspath(__file__)); R_EARTH=6378137.0; OUTD=os.path.join(ROOT,"output")
def to_enu(lat,lon,up,lat0,lon0):
    return np.array([np.radians(lon-lon0)*np.cos(np.radians(lat0))*R_EARTH,np.radians(lat-lat0)*R_EARTH,up])
def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=np.trace(np.diag(D)@S)/((Ss**2).sum()/len(src)); return s,R,mu_d-s*R@mu_s

cam=np.load(os.path.join(OUTD,"m4t_cameras.npz"),allow_pickle=True); names=cam["names"]; Ccol=cam["C"]
gps=json.load(open(os.path.join(ROOT,"m4t_work","m4t_gps.json")))
geo=json.load(open(os.path.join(OUTD,"splat_georef.json"))); lat0,lon0=geo["lat0"],geo["lon0"]
G=np.array([to_enu(gps[n][0],gps[n][1],gps[n][2],lat0,lon0) for n in names])
s_m,R_m,t_m=umeyama(Ccol,G)
Tman=np.array(json.load(open(os.path.join(OUTD,"manual_align.json")))["T_splat_to_lidar"])
A0=Tman[:3,:3]@(s_m*R_m); b0=Tman[:3,:3]@t_m+Tman[:3,3]
Pc=np.load(os.path.join(OUTD,"m4t_points3d.npz"))["xyz"]
centroid=((A0@Pc.T).T+b0).mean(0)

rclpy.init(); n=Node("save_m4t"); buf=Buffer(); TransformListener(buf,n)
tf=None; t0=time.time()
while time.time()-t0<5:
    rclpy.spin_once(n,timeout_sec=0.1)
    try: tf=buf.lookup_transform("map","model",rclpy.time.Time()); break
    except Exception: pass
assert tf is not None,"keine TF map->model"
q=tf.transform.rotation; tr=tf.transform.translation
Rmk=Rot.from_quat([q.x,q.y,q.z,q.w]).as_matrix(); tmk=np.array([tr.x,tr.y,tr.z])
A=Rmk@A0; b=Rmk@(b0-centroid)+tmk
json.dump({"A":A.tolist(),"b":b.tolist()},open(os.path.join(OUTD,"m4t_affine.json"),"w"),indent=2)
print(f"Greifer t={tmk.round(2)} -> m4t_affine.json gespeichert")
rclpy.shutdown()
