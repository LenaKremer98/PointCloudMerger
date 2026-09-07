#!/usr/bin/env python3
"""Echte Temperatur pro LiDAR-Punkt via DJI Thermal SDK (dji_irp).

1) Konvertiert alle _T.JPG zu Temperatur-Arrays (640x512 float32 °C), gecacht.
2) Projiziert LiDAR-Punkte (Ausrichtung m4t_affine.json) in die Thermal-Kameras,
   nimmt die Temperatur der nadir-naechsten Kamera.
3) Faerbt per Iron-Palette und speichert output/temp_cloud.npz (xyz, temp[°C], rgb).
"""
import os, sys, json, glob, subprocess, numpy as np
from PIL import Image
ROOT=os.path.dirname(os.path.abspath(__file__))
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")
TDIR=glob.glob(os.path.join(ROOT,"..","m4t","DCIM","*"+os.sep))[0]
SDK="/tmp/djisdk/DJI_thermal_SDK-main"
BIN=os.path.join(SDK,"utility","bin","linux","release_x64","dji_irp")
LIBS=os.path.join(SDK,"tsdk-core","lib","linux","release_x64")
CACHE="/tmp/m4t_temps"; os.makedirs(CACHE,exist_ok=True)
DIST="25.0"   # Flughoehe ~50 m, SDK-Max 25 m (Atmosphaeren-Korrektur)
TW,TH=640,512
TF=(TW/2)/np.tan(np.radians(38.2/2)); TCX,TCY=TW/2,TH/2

def seq(p): return os.path.basename(str(p)).split("_")[2]

def temp_array(tjpg):
    """dji_irp measure -> 640x512 float32 °C (gecacht)."""
    out=os.path.join(CACHE, seq(tjpg)+".f32")
    if not os.path.exists(out) or os.path.getsize(out)!=TW*TH*4:
        env=dict(os.environ, LD_LIBRARY_PATH=LIBS+":"+os.path.dirname(BIN))
        r=subprocess.run([BIN,"-s",tjpg,"-a","measure","-o",out,
                          "--measurefmt","float32","--distance",DIST],
                         env=env,capture_output=True)
        if r.returncode!=0 or not os.path.exists(out):
            print("FEHLER dji_irp:",tjpg,r.stderr.decode()[:200]); return None
    return np.fromfile(out,dtype="<f4").reshape(TH,TW)

def apply_palette(x):  # iron
    xp=[0,.15,.35,.5,.65,.8,.9,1]
    r=[0,.25,.55,.85,1,1,1,1]; g=[0,0,0,.2,.45,.7,.9,1]; b=[0,.45,.55,.35,.1,0,.35,1]
    return np.stack([np.interp(x,xp,r),np.interp(x,xp,g),np.interp(x,xp,b)],1)

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)

cam=np.load(os.path.join(ROOT,"output","m4t_cameras.npz"),allow_pickle=True)
names=cam["names"]; Rcw=cam["Rcw"]; tcw=cam["tcw"]
tmap={seq(p):p for p in glob.glob(TDIR+"*_T.JPG")}
d=json.load(open(os.path.join(ROOT,"output","m4t_affine.json"))); A=np.array(d["A"]); b=np.array(d["b"])
L=read_pcd(LIDAR); P_col=(np.linalg.inv(A)@(L-b).T).T; PT=P_col.T; N=len(L)
print(f"LiDAR {N} Punkte; konvertiere/lese Temperaturbilder ...")

best_r=np.full(N,np.inf); temp=np.full(N,np.nan)
for i,n in enumerate(names):
    tp=tmap.get(seq(n));
    if tp is None: continue
    Tm=temp_array(tp)
    if Tm is None: continue
    pc=(Rcw[i]@PT).T+tcw[i]; Z=pc[:,2]; fr=Z>1e-6; Zs=np.where(fr,Z,1.0)
    x=pc[:,0]/Zs; y=pc[:,1]/Zs; u=TF*x+TCX; v=TF*y+TCY
    val=fr&(u>=0)&(u<TW)&(v>=0)&(v<TH); rad=np.sqrt(x*x+y*y); sel=val&(rad<best_r)
    if sel.any():
        ui=np.clip(u[sel].astype(int),0,TW-1); vi=np.clip(v[sel].astype(int),0,TH-1)
        temp[sel]=Tm[vi,ui]; best_r[sel]=rad[sel]
    if i%50==0: print(f"  {i}/{len(names)} ...")

hit=np.isfinite(temp)
tv=temp[hit]
print(f"Temperatur: {hit.sum()}/{N} ({100*hit.mean():.1f} %) | "
      f"min {tv.min():.1f} med {np.median(tv):.1f} max {tv.max():.1f} °C")
# Iron-Farbe ueber robusten Bereich
lo,hi=np.percentile(tv,[2,98]); print(f"Farbskala (2-98%): {lo:.1f} .. {hi:.1f} °C")
nrm=np.clip((temp-lo)/(hi-lo+1e-9),0,1)
rgb=np.full((N,3),0.12); rgb[hit]=apply_palette(nrm[hit])
np.savez(os.path.join(ROOT,"output","temp_cloud.npz"),
         xyz=L.astype(np.float32), temp=temp.astype(np.float32),
         rgb=rgb.astype(np.float32), scale_lo=lo, scale_hi=hi)
print("-> output/temp_cloud.npz")
