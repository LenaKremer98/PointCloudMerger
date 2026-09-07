#!/usr/bin/env python3
"""Orbit-Video der Temperatur-Wolke (Iron) mit horizontaler Farbtemperaturskala oben
(links min, rechts max). Open3D OffscreenRenderer -> Frames (+Colorbar via PIL)."""
import os, numpy as np, open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

ROOT=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(ROOT,"output","temp_cloud.npz"))
xyz=d["xyz"].astype(np.float64); rgb=np.clip(d["rgb"],0,1)
LO=float(d["scale_lo"]); HI=float(d["scale_hi"])
FRAMES=os.path.join(ROOT,"renders","thermal_frames"); os.makedirs(FRAMES,exist_ok=True)

CENTER=np.array([-99.4,-6.2,1.5]); RADIUS=110.0; ELEV=45.0; FOV=55.0
W,H=2560,1440; DUR=15.0; FPS=30; N=int(DUR*FPS)

def iron(x):
    xp=[0,.15,.35,.5,.65,.8,.9,1]
    r=[0,.25,.55,.85,1,1,1,1]; g=[0,0,0,.2,.45,.7,.9,1]; b=[0,.45,.55,.35,.1,0,.35,1]
    return np.stack([np.interp(x,xp,r),np.interp(x,xp,g),np.interp(x,xp,b)],1)

# --- Colorbar (einmal bauen, auf jeden Frame legen) ---
def make_colorbar(W):
    barL,barR=int(W*0.06),int(W*0.94); barTop,barBot=46,80; H0=104
    cb=Image.new("RGBA",(W,H0),(18,18,18,180)); dr=ImageDraw.Draw(cb)
    fb=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",30)
    fs=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",26)
    xs=np.linspace(0,1,barR-barL)
    cols=(iron(xs)*255).astype(np.uint8)
    grad=np.repeat(cols[None,:,:],barBot-barTop,axis=0)
    cb.paste(Image.fromarray(grad),(barL,barTop))
    dr.rectangle([barL,barTop,barR,barBot],outline=(255,255,255,255),width=2)
    dr.text((W/2,8),"Temperatur (°C)",font=fb,fill=(255,255,255,255),anchor="mt")
    # Ticks min/mitte/max
    mid=(LO+HI)/2
    dr.text((barL,barBot+4),f"{LO:.0f}",font=fs,fill=(255,255,255,255),anchor="lt")
    dr.text(((barL+barR)//2,barBot+4),f"{mid:.0f}",font=fs,fill=(255,255,255,255),anchor="mt")
    dr.text((barR,barBot+4),f"{HI:.0f}",font=fs,fill=(255,255,255,255),anchor="rt")
    return cb
CB=make_colorbar(W)

pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(xyz); pcd.colors=o3d.utility.Vector3dVector(rgb)
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.0)
pcd,_=pcd.remove_radius_outlier(nb_points=12,radius=1.2)
print(f"{len(pcd.points)} Punkte nach Filter")

rend=rendering.OffscreenRenderer(W,H)
rend.scene.set_background([0.06,0.06,0.07,1.0])
mat=rendering.MaterialRecord(); mat.shader="defaultUnlit"; mat.point_size=2.0
rend.scene.add_geometry("pc",pcd,mat)
rend.scene.scene.enable_sun_light(False)
rend.scene.view.set_post_processing(False)

el=np.radians(ELEV)
for i in range(N):
    th=2*np.pi*i/N
    eye=CENTER+np.array([RADIUS*np.cos(el)*np.cos(th),RADIUS*np.cos(el)*np.sin(th),RADIUS*np.sin(el)])
    rend.setup_camera(FOV,CENTER.tolist(),eye.tolist(),[0,0,1])
    img=np.asarray(rend.render_to_image())
    fr=Image.fromarray(img).convert("RGBA"); fr.alpha_composite(CB,(0,0))
    fr.convert("RGB").save(os.path.join(FRAMES,f"f{i:04d}.png"))
    if i%50==0: print(f"  Frame {i}/{N}")
print("Frames+Colorbar fertig ->",FRAMES)
