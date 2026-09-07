#!/usr/bin/env python3
"""Faerbt die LiDAR-Wolke per Reprojektion in die 255 M4T-Nadir-Fotos.

Kette: LiDAR -> (inv manuelle Ausrichtung) -> avata-ENU -> (inv Umeyama) -> M4T-COLMAP-Welt
       -> pro Kamera projizieren (SIMPLE_RADIAL) -> beste Kamera (naechst am Bildzentrum) -> RGB.

System-Python (numpy, PIL).  Optional: --refine  (ICP M4T-Punkte<->LiDAR vor Projektion, braucht open3d)
"""
import os, sys, json, numpy as np
from PIL import Image

ROOT=os.path.dirname(os.path.abspath(__file__))
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")
IMGDIR=os.path.join(ROOT,"m4t_work","images")
OUT=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ_m4t_colored.ply")
for a in sys.argv[1:]:
    if a.startswith("--out="): OUT=a.split("=",1)[1]
R_EARTH=6378137.0

def read_pcd_xyz(path):
    with open(path,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)

def to_enu(lat,lon,up,lat0,lon0):
    east=np.radians(lon-lon0)*np.cos(np.radians(lat0))*R_EARTH
    north=np.radians(lat-lat0)*R_EARTH
    return np.stack([east,north,np.full_like(east,0)+up],-1) if np.ndim(up) else np.array([east,north,up])

def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; var=(Ss**2).sum()/len(src); s=np.trace(np.diag(D)@S)/var
    t=mu_d-s*R@mu_s; return s,R,t

# --- Kameras + GPS ---
cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
names=cam["names"]; Rcw=cam["Rcw"]; tcw=cam["tcw"]; Ccol=cam["C"]; intr=cam["intr"]
gps=json.load(open(os.path.join(ROOT,"m4t_work","m4t_gps.json")))
geo=json.load(open(os.path.join(ROOT,"output","splat_georef.json")))
lat0,lon0=geo["lat0"],geo["lon0"]

G=np.array([to_enu(gps[n][0],gps[n][1],gps[n][2],lat0,lon0) for n in names])
s_m,R_m,t_m=umeyama(Ccol,G)               # COLMAP -> avata-ENU
res=np.linalg.norm((s_m*(R_m@Ccol.T).T)+t_m-G,axis=1)
print(f"M4T COLMAP->ENU: s={s_m:.4f}, GPS-Residuum mean {res.mean():.2f} m max {res.max():.2f} m")

ALIGN=os.path.join(ROOT,"output","manual_align.json"); AFFINE=None
for a in sys.argv[1:]:
    if a.startswith("--align="): ALIGN=a.split("=",1)[1]
    if a.startswith("--affine="): AFFINE=a.split("=",1)[1]

L=read_pcd_xyz(LIDAR)
if AFFINE:   # direkter colmap->lidar Affin (A,b): p_col = inv(A)(L-b)
    d=json.load(open(AFFINE)); A=np.array(d["A"]); b=np.array(d["b"])
    print("Ausrichtung: affine", os.path.basename(AFFINE))
    P_col=(np.linalg.inv(A)@(L-b).T).T
else:
    print("Ausrichtung:", os.path.basename(ALIGN))
    T=np.array(json.load(open(ALIGN))["T_splat_to_lidar"])
    M=T[:3,:3]; tt=T[:3,3]; Minv=np.linalg.inv(M)
    P_enu=(Minv@(L-tt).T).T
    P_col=((1.0/s_m)*(R_m.T@(P_enu-t_m).T)).T
print(f"LiDAR {len(L)} Punkte -> COLMAP-Welt")

# --- Projektion: beste Kamera je Punkt ---
N=len(L)
best_r=np.full(N,np.inf)
col=np.full((N,3),0.42)
PcolT=P_col.T   # 3xN
for i,n in enumerate(names):
    W,H,f,cx,cy,k=intr[i]
    pc=(Rcw[i]@PcolT).T+tcw[i]
    Z=pc[:,2]; front=Z>1e-6
    Zs=np.where(front,Z,1.0)
    x=pc[:,0]/Zs; y=pc[:,1]/Zs
    r2=x*x+y*y; d=1+k*r2
    u=f*x*d+cx; v=f*y*d+cy
    valid=front&(u>=0)&(u<W)&(v>=0)&(v<H)
    rad=np.sqrt(r2)
    sel=valid&(rad<best_r)
    if sel.any():
        img=np.asarray(Image.open(os.path.join(IMGDIR,str(n))).convert("RGB"))
        ui=np.clip(u[sel].astype(int),0,int(W)-1); vi=np.clip(v[sel].astype(int),0,int(H)-1)
        col[sel]=img[vi,ui]/255.0
        best_r[sel]=rad[sel]
    if i%50==0: print(f"  Kamera {i}/{len(names)} ...")

hit=np.isfinite(best_r)
print(f"Koloriert: {hit.sum()}/{N} ({100*hit.mean():.1f} %)")

rgb8=(np.clip(col,0,1)*255).round().astype(np.uint8)
with open(OUT,"wb") as fo:
    fo.write(("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {N}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode())
    rec=np.zeros(N,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
    rec["x"],rec["y"],rec["z"]=L[:,0],L[:,1],L[:,2]
    rec["red"],rec["green"],rec["blue"]=rgb8[:,0],rgb8[:,1],rgb8[:,2]
    rec.tofile(fo)
print(f"-> {OUT}")
