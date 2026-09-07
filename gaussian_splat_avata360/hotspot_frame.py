#!/usr/bin/env python3
"""EIN Testframe: Temperatur-Orbit + Colorbar + dünnes Rechteck (weißer Rand) etwas
unterhalb-rechts der Mitte; darin heißester Punkt rot markiert + Temperatur daneben."""
import os, numpy as np, open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

ROOT=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(ROOT,"output","temp_cloud.npz"))
xyz=d["xyz"].astype(np.float64); temp=d["temp"].astype(np.float64); rgb=np.clip(d["rgb"],0,1)
LO=float(d["scale_lo"]); HI=float(d["scale_hi"])
CENTER=np.array([-99.4,-6.2,1.5]); RADIUS=110.0; ELEV=45.0; FOV=55.0
W,H=2560,1440
FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",30)
FBig=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",34)
FS=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",26)

def iron(x):
    xp=[0,.15,.35,.5,.65,.8,.9,1]; r=[0,.25,.55,.85,1,1,1,1]; g=[0,0,0,.2,.45,.7,.9,1]; b=[0,.45,.55,.35,.1,0,.35,1]
    return np.stack([np.interp(x,xp,r),np.interp(x,xp,g),np.interp(x,xp,b)],1)
def colorbar(W):
    bL,bR,bT,bB,H0=int(W*0.06),int(W*0.94),46,80,104
    cb=Image.new("RGBA",(W,H0),(18,18,18,180)); dr=ImageDraw.Draw(cb)
    grad=np.repeat((iron(np.linspace(0,1,bR-bL))*255).astype(np.uint8)[None],bB-bT,0)
    cb.paste(Image.fromarray(grad),(bL,bT)); dr.rectangle([bL,bT,bR,bB],outline=(255,255,255,255),width=2)
    dr.text((W/2,8),"Temperatur (°C)",font=FB,fill=(255,255,255,255),anchor="mt")
    dr.text((bL,bB+4),f"{LO:.0f}",font=FS,fill=(255,255,255,255),anchor="lt")
    dr.text(((bL+bR)//2,bB+4),f"{(LO+HI)/2:.0f}",font=FS,fill=(255,255,255,255),anchor="mt")
    dr.text((bR,bB+4),f"{HI:.0f}",font=FS,fill=(255,255,255,255),anchor="rt")
    return cb

pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(xyz); pcd.colors=o3d.utility.Vector3dVector(rgb)
pcd,keep=pcd.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.0)
pcd,keep2=pcd.remove_radius_outlier(nb_points=12,radius=1.2)
# Indizes der behaltenen Punkte auf Original abbilden (fuer temp)
idx_keep=np.array(keep)[keep2]
xyz_k=xyz[idx_keep]; temp_k=temp[idx_keep]

rend=rendering.OffscreenRenderer(W,H); rend.scene.set_background([0.06,0.06,0.07,1.0])
mat=rendering.MaterialRecord(); mat.shader="defaultUnlit"; mat.point_size=2.0
rend.scene.add_geometry("pc",pcd,mat); rend.scene.scene.enable_sun_light(False); rend.scene.view.set_post_processing(False)
el=np.radians(ELEV)
eye=CENTER+np.array([RADIUS*np.cos(el),0,RADIUS*np.sin(el)])   # frame 0 (th=0)
rend.setup_camera(FOV,CENTER.tolist(),eye.tolist(),[0,0,1])
img=np.asarray(rend.render_to_image())

# --- Punkte mit Render-Kamera projizieren ---
V=np.asarray(rend.scene.camera.get_view_matrix()); P=np.asarray(rend.scene.camera.get_projection_matrix())
hp=(P@V@np.c_[xyz_k,np.ones(len(xyz_k))].T)         # 4xN
w=hp[3]; front=w>1e-6; ndc=hp[:3]/np.where(front,w,1.0)
sx=(ndc[0]*0.5+0.5)*W; sy=(1.0-(ndc[1]*0.5+0.5))*H
print("CENTER projiziert ~", None)
# Box: etwas unterhalb-rechts der Mitte, duenn
bx0,by0,bx1,by1=int(0.53*W),int(0.54*H),int(0.66*W),int(0.65*H)
inbox=front&(sx>=bx0)&(sx<=bx1)&(sy>=by0)&(sy<=by1)&np.isfinite(temp_k)
print("Punkte in Box:",int(inbox.sum()))
fr=Image.fromarray(img).convert("RGBA"); fr.alpha_composite(colorbar(W),(0,0)); dr=ImageDraw.Draw(fr)
dr.rectangle([bx0,by0,bx1,by1],outline=(255,255,255,255),width=2)
if inbox.any():
    ci=np.where(inbox)[0]; hot=ci[np.argmax(temp_k[ci])]
    hx,hy=int(sx[hot]),int(sy[hot]); t=temp_k[hot]
    r=10; dr.ellipse([hx-r,hy-r,hx+r,hy+r],fill=(255,30,30,255),outline=(255,255,255,255),width=2)
    label=f"{t:.1f} °C"
    tx,ty=hx+16,hy-18
    bb=dr.textbbox((tx,ty),label,font=FBig)
    dr.rectangle([bb[0]-6,bb[1]-4,bb[2]+6,bb[3]+4],fill=(0,0,0,180))
    dr.text((tx,ty),label,font=FBig,fill=(255,80,80,255))
    print(f"Heißester Punkt in Box: {t:.1f} °C @ Pixel ({hx},{hy})")
out=os.path.join(ROOT,"output","hotspot_test.png")
fr.convert("RGB").save(out); print("->",out)
