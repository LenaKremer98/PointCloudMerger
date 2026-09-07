#!/usr/bin/env python3
"""Montage v2 (durchgehender 45°-Orbit, LiDAR-Frame, 1 Umdrehung = 240 Frames):
 A  0   -240 : ECHTES 3DGS-Splat (vorgerendert)            [1 Umdr.]
 -  240 -310 : Crossfade Splat -> LiDAR (hoehen-gefaerbt, Turbo/Z)
 B  310 -550 : Hoehen-LiDAR voll drehen                    [1 Umdr. vor Absenken]
 C  550 -660 : Schnittebene sinkt auf 4.3 m
    660 -790 : geclippt drehen                             [Clip-Phase ~1 Umdr.]
    790 -900 : Schnittebene wieder hoch (voller Hoehe)     [umgekehrter Effekt]
 -  900 -970 : Crossfade Hoehe -> Farbe (RGB), volle Hoehe
 D  970 -1210: RGB voll drehen                              [1 Umdr.]
 - 1210 -1280: Crossfade RGB -> Thermal (+ Colorbar)
 E 1280 -1520: Thermal drehen (+ Colorbar + Hotspot)        [1 Umdr.]
"""
import os, sys, numpy as np, json, open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

ROOT=os.path.dirname(os.path.abspath(__file__)); PCD="../PCD-DRZ_20-05-26"
W,H=1920,1080; FPS=30; REV=900.0       # 12 s/Umdrehung (50% langsamer)
# ~5 min = 9000 Frames; jede Hauptphase mehrere Umdrehungen
A,XAB,B,DESC,LOW,RISE,XCOL,D,XTH,E=1440,1620,3060,3300,4140,4380,4560,6360,6540,9000
ZCLIP=4.3; CENTER=np.array([-99.4,-6.2,1.5]); RADIUS=110.0; ELEV=45.0; FOV=55.0
FRAMES=os.path.join(ROOT,"renders","montage"); os.makedirs(FRAMES,exist_ok=True)
SPLATDIR=os.path.join(ROOT,"renders","splat_frames")
TEST=len(sys.argv)>1 and sys.argv[1]=="test"
FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",24)
FBig=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",28)
FS=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)
def turbo(v):
    v=np.clip(v,0,1)
    return np.stack([np.clip(1.5-abs(4*v-3),0,1),np.clip(1.5-abs(4*v-2),0,1),np.clip(1.5-abs(4*v-1),0,1)],1)
def iron(x):
    xp=[0,.15,.35,.5,.65,.8,.9,1]; r=[0,.25,.55,.85,1,1,1,1]; g=[0,0,0,.2,.45,.7,.9,1]; b=[0,.45,.55,.35,.1,0,.35,1]
    return np.stack([np.interp(x,xp,r),np.interp(x,xp,g),np.interp(x,xp,b)],1)

# ---------- Daten ----------
L=read_pcd(os.path.join(ROOT,PCD,"merged_FinlaDRZ.pcd")); z=L[:,2]
tmp=np.load(os.path.join(ROOT,"output","temp_cloud.npz"))
iron_rgb=np.clip(tmp["rgb"],0,1); temp=tmp["temp"].astype(np.float64); LO=float(tmp["scale_lo"]); HI=float(tmp["scale_hi"])
m4t=o3d.io.read_point_cloud(os.path.join(ROOT,PCD,"merged_FinlaDRZ_m4t_manual2.ply")); m4t_rgb=np.asarray(m4t.colors)
_p=o3d.geometry.PointCloud(); _p.points=o3d.utility.Vector3dVector(L)
_,k1=_p.remove_statistical_outlier(24,2.0); k1=np.array(k1)
_,k2=_p.select_by_index(k1).remove_radius_outlier(12,1.2)
keepL=np.zeros(len(L),bool); keepL[k1[np.array(k2)]]=True
zlo,zhi=np.percentile(z[keepL],[2,98]); height_rgb=turbo((z-zlo)/(zhi-zlo))
ZTOP=float(np.percentile(z[keepL],99.9))
print(f"LiDAR {len(L)} keep {int(keepL.sum())} | z-Farbe {zlo:.1f}..{zhi:.1f} m | ZTOP {ZTOP:.1f}")

def mk():
    r=rendering.OffscreenRenderer(W,H); r.scene.set_background([0.05,0.05,0.06,1.0])
    r.scene.scene.enable_sun_light(False); r.scene.view.set_post_processing(False); return r
r1,r2=mk(),mk()
MAT=rendering.MaterialRecord(); MAT.shader="defaultUnlit"; MAT.point_size=2.0
tag={"r1":None,"r2":None}
def set_geom(r,key,name,xyz,rgb):
    if tag[key]==name: return
    r.scene.clear_geometry()
    pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(xyz); pc.colors=o3d.utility.Vector3dVector(rgb)
    r.scene.add_geometry("pc",pc,MAT); tag[key]=name
def cam(r,i):
    th=2*np.pi*(i/REV); el=np.radians(ELEV)
    eye=CENTER+np.array([RADIUS*np.cos(el)*np.cos(th),RADIUS*np.cos(el)*np.sin(th),RADIUS*np.sin(el)])
    r.setup_camera(FOV,CENTER.tolist(),eye.tolist(),[0,0,1])
def render(r,i): cam(r,i); return np.asarray(r.render_to_image()).astype(np.float32)
def splat_img(i): return np.asarray(Image.open(os.path.join(SPLATDIR,f"f{i:04d}.png")).convert("RGB")).astype(np.float32)

def colorbar():
    bL,bR,bT,bB,H0=int(W*0.06),int(W*0.94),38,66,86
    cb=Image.new("RGBA",(W,H0),(18,18,18,170)); dr=ImageDraw.Draw(cb)
    grad=np.repeat((iron(np.linspace(0,1,bR-bL))*255).astype(np.uint8)[None],bB-bT,0)
    cb.paste(Image.fromarray(grad),(bL,bT)); dr.rectangle([bL,bT,bR,bB],outline=(255,255,255,255),width=2)
    dr.text((W/2,6),"Temperatur (°C)",font=FB,fill=(255,255,255,255),anchor="mt")
    dr.text((bL,bB+3),f"{LO:.0f}",font=FS,fill=(255,255,255,255),anchor="lt")
    dr.text(((bL+bR)//2,bB+3),f"{(LO+HI)/2:.0f}",font=FS,fill=(255,255,255,255),anchor="mt")
    dr.text((bR,bB+3),f"{HI:.0f}",font=FS,fill=(255,255,255,255),anchor="rt")
    return cb
CB=colorbar()
BX0,BY0,BX1,BY1=int(0.53*W),int(0.54*H),int(0.66*W),int(0.65*H)
def overlay_thermal(arr,alpha):
    fr=Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    cb=CB.copy()
    if alpha<1.0: cb.putalpha(cb.split()[3].point(lambda p:int(p*alpha)))
    fr.alpha_composite(cb,(0,0)); dr=ImageDraw.Draw(fr)
    if alpha>0.5:
        dr.rectangle([BX0,BY0,BX1,BY1],outline=(255,255,255,255),width=2)
        V=np.asarray(r1.scene.camera.get_view_matrix()); P=np.asarray(r1.scene.camera.get_projection_matrix())
        xyz=L[keepL]; tk=temp[keepL]
        hp=P@V@np.c_[xyz,np.ones(len(xyz))].T; w=hp[3]; fn=w>1e-6; ndc=hp[:3]/np.where(fn,w,1)
        sx=(ndc[0]*.5+.5)*W; sy=(1-(ndc[1]*.5+.5))*H
        ib=fn&(sx>=BX0)&(sx<=BX1)&(sy>=BY0)&(sy<=BY1)&np.isfinite(tk)
        if ib.any():
            ci=np.where(ib)[0]; hot=ci[np.argmax(tk[ci])]; hx,hy=int(sx[hot]),int(sy[hot]); t=tk[hot]
            dr.ellipse([hx-8,hy-8,hx+8,hy+8],fill=(255,30,30,255),outline=(255,255,255,255),width=2)
            lab=f"{t:.1f} °C"; bb=dr.textbbox((hx+14,hy-16),lab,font=FBig)
            dr.rectangle([bb[0]-5,bb[1]-3,bb[2]+5,bb[3]+3],fill=(0,0,0,180)); dr.text((hx+14,hy-16),lab,font=FBig,fill=(255,80,80,255))
    return np.asarray(fr.convert("RGB"))

def plane(i):
    if i<DESC: a=(i-B)/(DESC-B); return ZTOP+(ZCLIP-ZTOP)*a
    if i<LOW:  return ZCLIP
    if i<RISE: a=(i-LOW)/(RISE-LOW); return ZCLIP+(ZTOP-ZCLIP)*a
    return ZTOP

def frame(i):
    if i<A:                                    # A: Splat
        return splat_img(i)
    if i<XAB:                                  # Crossfade Splat -> Hoehe
        a=(i-A)/(XAB-A); set_geom(r2,"r2","height",L[keepL],height_rgb[keepL])
        return splat_img(i)*(1-a)+render(r2,i)*a
    if i<B:                                    # Hoehe voll
        set_geom(r1,"r1","height",L[keepL],height_rgb[keepL]); return render(r1,i)
    if i<RISE:                                 # Absenken / geclippt / Anheben (Hoehe)
        p=plane(i)
        if p>=ZTOP: set_geom(r1,"r1","height",L[keepL],height_rgb[keepL])
        else:
            idx=keepL&(z<=p); set_geom(r1,"r1",f"clip{i}",L[idx],height_rgb[idx])
        return render(r1,i)
    if i<XCOL:                                 # Crossfade Hoehe(voll) -> RGB(voll)
        a=(i-RISE)/(XCOL-RISE)
        set_geom(r1,"r1","height",L[keepL],height_rgb[keepL]); set_geom(r2,"r2","rgb",L[keepL],m4t_rgb[keepL])
        return render(r1,i)*(1-a)+render(r2,i)*a
    if i<D:                                     # RGB voll
        set_geom(r1,"r1","rgb",L[keepL],m4t_rgb[keepL]); return render(r1,i)
    if i<XTH:                                   # Crossfade RGB -> Thermal
        a=(i-D)/(XTH-D)
        set_geom(r1,"r1","rgb",L[keepL],m4t_rgb[keepL]); set_geom(r2,"r2","therm",L[keepL],iron_rgb[keepL])
        out=render(r1,i)*(1-a)+render(r2,i)*a
        return overlay_thermal(out,a)
    # E: Thermal
    set_geom(r1,"r1","therm",L[keepL],iron_rgb[keepL]); out=render(r1,i)
    return overlay_thermal(out,1.0)

if TEST:
    for i in [720,1500,2300,3180,3700,4250,4470,5400,6450,8000]:
        Image.fromarray(frame(i).astype(np.uint8)).save(os.path.join(FRAMES,f"test_{i:05d}.png")); print("test",i)
else:
    import subprocess
    OUT=os.path.join(ROOT,PCD,"montage_splat_lidar_color_thermal.mp4")
    ff=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}",
                         "-r",str(FPS),"-i","-","-c:v","libx264","-pix_fmt","yuv420p",
                         "-crf","20","-preset","medium",OUT],
                        stdin=subprocess.PIPE)
    for i in range(E):
        ff.stdin.write(frame(i).astype(np.uint8).tobytes())
        if i%120==0: print(f"  {i}/{E}",flush=True)
    ff.stdin.close(); ff.wait()
    print("fertig ->",OUT)
