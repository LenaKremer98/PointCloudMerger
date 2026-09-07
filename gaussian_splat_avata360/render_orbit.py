#!/usr/bin/env python3
"""Orbit-Video um den georeferenzierten m4t_geo-Splat: 360deg-Umrundung,
Kamera 45deg von oben auf das Szenenzentrum blickend, 15 s @ 30 fps."""
import os, numpy as np, torch
from plyfile import PlyData
from PIL import Image
from gsplat.rendering import rasterization

ROOT = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"
PLY  = os.path.join(ROOT, "m4t_geo", "output", "avata360_splat_clean.ply")
OUTDIR = os.path.join(ROOT, "renders"); os.makedirs(OUTDIR, exist_ok=True)
FRAMEDIR = os.path.join(OUTDIR, "_orbit_frames"); os.makedirs(FRAMEDIR, exist_ok=True)

# --- Splat laden (volles SH-Grad 3) ---
v = PlyData.read(PLY)["vertex"]; N = len(v["x"])
means  = torch.tensor(np.stack([v["x"],v["y"],v["z"]],1), dtype=torch.float32, device=DEV)
sh0    = torch.tensor(np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1), dtype=torch.float32, device=DEV)[:,None,:]
frest  = np.stack([v[f"f_rest_{i}"] for i in range(45)], 1).reshape(N,3,15)
shN    = torch.tensor(np.transpose(frest,(0,2,1)), dtype=torch.float32, device=DEV)
colors = torch.cat([sh0, shN], 1)
scales = torch.tensor(np.stack([v["scale_0"],v["scale_1"],v["scale_2"]],1), dtype=torch.float32, device=DEV).exp()
quats  = torch.tensor(np.stack([v["rot_0"],v["rot_1"],v["rot_2"],v["rot_3"]],1), dtype=torch.float32, device=DEV)
opac   = torch.tensor(v["opacity"], dtype=torch.float32, device=DEV).sigmoid()
print(f"{N} Gaussians geladen")

# --- robustes Szenenzentrum + Orbit-Radius ---
P = means.cpu().numpy()
center = np.median(P, 0).astype(np.float32)
rh = np.sqrt(((P[:,:2]-center[:2])**2).sum(1))
scene_r = float(np.percentile(rh, 95))            # ~115 m
Rh = scene_r * 1.30                               # horizontaler Abstand
# POI: vom ersten Frame (az=0) aus nach rechts (+Y) auf die blauen Container
target = np.array([1.5, 76.6, 2.9], np.float32)
print(f"center={center}  target={target}  scene_r(p95)={scene_r:.1f}  Rh={Rh:.1f}")

def lookat(eye, target, up=np.array([0,0,1.])):
    f = target-eye; f/=np.linalg.norm(f)
    r = np.cross(f,up); r/=np.linalg.norm(r); u = np.cross(r,f)
    R = np.stack([r,-u,f]); t = -R@eye
    M = np.eye(4); M[:3,:3]=R; M[:3,3]=t
    return torch.tensor(M, dtype=torch.float32, device=DEV)

W,H,foc = 1920,1080,1100
K = torch.tensor([[foc,0,W/2],[0,foc,H/2],[0,0,1]], dtype=torch.float32, device=DEV)
FPS = 30
SPEED = 0.70                                       # 30% langsamere Drehung
NF = int(round(FPS*15/SPEED))                      # volle 360deg -> ~643 frames (~21.4 s)
ELEV = np.deg2rad(45.0)
print(f"{NF} frames @ {FPS}fps = {NF/FPS:.1f} s")

for i in range(NF):
    az = 2*np.pi*i/NF
    eye = target + Rh*np.array([np.cos(ELEV)*np.cos(az),
                                np.cos(ELEV)*np.sin(az),
                                np.sin(ELEV)], np.float32)
    vm = lookat(eye, target)
    with torch.no_grad():
        r,_,_ = rasterization(means=means,quats=quats,scales=scales,opacities=opac,
                              colors=colors,viewmats=vm[None],Ks=K[None],width=W,height=H,
                              sh_degree=3,packed=False,render_mode="RGB")
    img = (r[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(FRAMEDIR, f"f{i:04d}.png"))
    if i % 30 == 0: print(f"frame {i}/{NF}")
print("Frames fertig.")
