#!/usr/bin/env python3
"""Faerbt die LiDAR-Wolke mit den M4T-THERMAL-Bildern (_T.JPG).

Gleiche Reprojektion wie RGB, aber: RGB-COLMAP-Posen (Thermal co-montiert, gleiche Pose)
+ THERMAL-Intrinsics (1280x1024, FOV_h 38.2deg -> f=1848px) + Sampling aus dem per
Sequenznummer gepaarten _T.JPG. Nutzt die feinjustierte Ausrichtung m4t_affine.json.
"""
import os, sys, json, glob, numpy as np
from PIL import Image
ROOT=os.path.dirname(os.path.abspath(__file__)); R_EARTH=6378137.0
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")
TDIR=glob.glob(os.path.join(ROOT,"..","m4t","DCIM","*"+os.sep))[0]
OUT=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ_thermal.ply")
AFFINE=os.path.join(ROOT,"output","m4t_affine.json"); PALETTE="none"
for a in sys.argv[1:]:
    if a.startswith("--affine="): AFFINE=a.split("=",1)[1]
    if a.startswith("--out="): OUT=a.split("=",1)[1]
    if a.startswith("--palette="): PALETTE=a.split("=",1)[1]

def apply_palette(x, name):   # x in [0,1] -> (n,3)
    if name=="jet":
        xp=[0,.125,.375,.625,.875,1]
        r=[0,0,0,1,1,.5]; g=[0,0,1,1,0,0]; b=[.5,1,1,0,0,0]
    else:  # iron (FLIR-aehnlich)
        xp=[0,.15,.35,.5,.65,.8,.9,1]
        r=[0,.25,.55,.85,1,1,1,1]; g=[0,0,0,.2,.45,.7,.9,1]; b=[0,.45,.55,.35,.1,0,.35,1]
    return np.stack([np.interp(x,xp,r),np.interp(x,xp,g),np.interp(x,xp,b)],1)

# Thermal-Intrinsics (alle Bilder gleich)
TW,TH=1280,1024; FOVH=38.2
TF=(TW/2)/np.tan(np.radians(FOVH/2)); TCX,TCY=TW/2,TH/2
print(f"Thermal-Intrinsics: f={TF:.1f}px cx={TCX} cy={TCY}")

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)

cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
names=cam["names"]; Rcw=cam["Rcw"]; tcw=cam["tcw"]
def seq(s): return os.path.basename(str(s)).split("_")[2]
tmap={seq(p):p for p in glob.glob(TDIR+"*_T.JPG")}

d=json.load(open(AFFINE)); A=np.array(d["A"]); b=np.array(d["b"])
L=read_pcd(LIDAR)
P_col=(np.linalg.inv(A)@(L-b).T).T
PT=P_col.T; N=len(L)
print(f"LiDAR {N} Punkte; Ausrichtung affine {os.path.basename(AFFINE)}")

best_r=np.full(N,np.inf); col=np.full((N,3),0.12)
for i,n in enumerate(names):
    tp=tmap.get(seq(n))
    if tp is None: continue
    pc=(Rcw[i]@PT).T+tcw[i]; Z=pc[:,2]; fr=Z>1e-6; Zs=np.where(fr,Z,1.0)
    x=pc[:,0]/Zs; y=pc[:,1]/Zs
    u=TF*x+TCX; v=TF*y+TCY
    val=fr&(u>=0)&(u<TW)&(v>=0)&(v<TH)
    rad=np.sqrt(x*x+y*y); sel=val&(rad<best_r)
    if sel.any():
        img=np.asarray(Image.open(tp).convert("RGB"))
        ui=np.clip(u[sel].astype(int),0,TW-1); vi=np.clip(v[sel].astype(int),0,TH-1)
        col[sel]=img[vi,ui]/255.0; best_r[sel]=rad[sel]
    if i%50==0: print(f"  {i}/{len(names)} ...")

hit=np.isfinite(best_r)
print(f"Koloriert (thermal): {hit.sum()}/{N} ({100*hit.mean():.1f} %)")
if PALETTE!="none":
    lum=col.mean(1)                       # White-Hot-Intensitaet = Temperatur-Proxy
    hl=lum[hit]; lo,hi=np.percentile(hl,[2,98])
    nrm=np.clip((lum-lo)/(hi-lo+1e-9),0,1)
    col[hit]=apply_palette(nrm[hit], PALETTE)
    print(f"Palette: {PALETTE}")
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
