#!/usr/bin/env python3
"""Multiview-Farbkonsistenz als Ausrichtungs-Guete: fuer Stichprobe von LiDAR-Punkten
die Farbstreuung ueber ALLE sehenden Kameras. Kleiner = besser ausgerichtet.
Aufruf: python3 eval_consistency.py --align=<json>"""
import os, sys, json, numpy as np
from PIL import Image
ROOT=os.path.dirname(os.path.abspath(__file__)); R_EARTH=6378137.0
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")
IMGDIR=os.path.join(ROOT,"m4t_work","images")
ALIGN=os.path.join(ROOT,"output","manual_align.json")
for a in sys.argv[1:]:
    if a.startswith("--align="): ALIGN=a.split("=",1)[1]

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)
def to_enu(lat,lon,up,lat0,lon0):
    return np.array([np.radians(lon-lon0)*np.cos(np.radians(lat0))*R_EARTH,np.radians(lat-lat0)*R_EARTH,up])
def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=np.trace(np.diag(D)@S)/((Ss**2).sum()/len(src)); return s,R,mu_d-s*R@mu_s

cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
names=cam["names"]; Rcw=cam["Rcw"]; tcw=cam["tcw"]; Ccol=cam["C"]; intr=cam["intr"]
gps=json.load(open(os.path.join(ROOT,"m4t_work","m4t_gps.json")))
geo=json.load(open(os.path.join(ROOT,"output","splat_georef.json"))); lat0,lon0=geo["lat0"],geo["lon0"]
G=np.array([to_enu(gps[n][0],gps[n][1],gps[n][2],lat0,lon0) for n in names])
s_m,R_m,t_m=umeyama(Ccol,G)
T=np.array(json.load(open(ALIGN))["T_splat_to_lidar"]); M=T[:3,:3]; tt=T[:3,3]; Minv=np.linalg.inv(M)

L=read_pcd(LIDAR)
rng=np.random.default_rng(0); idx=rng.choice(len(L),50000,replace=False); L=L[idx]
P_enu=(Minv@(L-tt).T).T; P_col=((1/s_m)*(R_m.T@(P_enu-t_m).T)).T; PT=P_col.T
N=len(L)
ssum=np.zeros((N,3)); ssq=np.zeros((N,3)); cnt=np.zeros(N)
for i,n in enumerate(names):
    W,H,f,cx,cy,k=intr[i]
    pc=(Rcw[i]@PT).T+tcw[i]; Z=pc[:,2]; fr=Z>1e-6; Zs=np.where(fr,Z,1.0)
    x=pc[:,0]/Zs; y=pc[:,1]/Zs; r2=x*x+y*y; d=1+k*r2
    u=f*x*d+cx; v=f*y*d+cy
    val=fr&(u>=0)&(u<W)&(v>=0)&(v<H)
    if val.any():
        img=np.asarray(Image.open(os.path.join(IMGDIR,str(n))).convert("RGB"))/255.0
        ui=np.clip(u[val].astype(int),0,int(W)-1); vi=np.clip(v[val].astype(int),0,int(H)-1)
        c=img[vi,ui]; ssum[val]+=c; ssq[val]+=c*c; cnt[val]+=1
m=cnt>=2
var=ssq[m]/cnt[m,None]-(ssum[m]/cnt[m,None])**2
std=np.sqrt(np.clip(var,0,None)).mean(1)   # mittlere RGB-Streuung je Punkt
print(f"[{os.path.basename(ALIGN)}] Punkte mit >=2 Sichten: {m.sum()}/{N} "
      f"({100*m.mean():.0f}%), mittl. Farbstreuung = {std.mean():.4f}  (kleiner=besser)")
