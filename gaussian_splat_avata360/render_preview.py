#!/usr/bin/env python3
"""Rendert den trainierten Splat aus PLY von einigen COLMAP-Kameras und legt
Vergleichsbilder (Original | Render) ab. Render mit DC-Farbe (sh_degree=0),
unabhaengig von f_rest-Reihenfolge -> robuste visuelle Validierung."""
import os, numpy as np, torch
from plyfile import PlyData
from PIL import Image
import pycolmap
from gsplat.rendering import rasterization

ROOT = os.path.dirname(os.path.abspath(__file__))
PLY  = os.path.join(ROOT, "output", "avata360_splat.ply")
DEV  = "cuda"

v = PlyData.read(PLY)["vertex"]
N = len(v["x"])
means = torch.tensor(np.stack([v["x"],v["y"],v["z"]],1), dtype=torch.float32, device=DEV)
sh0   = torch.tensor(np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1), dtype=torch.float32, device=DEV)[:,None,:]
# f_rest channel-major (N,3,15) -> (N,15,3); volles SH-Grad 3
frest = np.stack([v[f"f_rest_{i}"] for i in range(45)], 1).reshape(N,3,15)
shN = torch.tensor(np.transpose(frest,(0,2,1)), dtype=torch.float32, device=DEV)
colors = torch.cat([sh0, shN], dim=1)            # (N,16,3)
scales= torch.tensor(np.stack([v["scale_0"],v["scale_1"],v["scale_2"]],1), dtype=torch.float32, device=DEV).exp()
quats = torch.tensor(np.stack([v["rot_0"],v["rot_1"],v["rot_2"],v["rot_3"]],1), dtype=torch.float32, device=DEV)
opac  = torch.tensor(v["opacity"], dtype=torch.float32, device=DEV).sigmoid()
print(f"{means.shape[0]} Gaussians geladen (volles SH-Grad 3)")

rec = pycolmap.Reconstruction(os.path.join(ROOT,"sparse","0"))
views = list(rec.images.values())
picks = [views[0], views[len(views)//2], views[-1]]

for n,img in enumerate(picks):
    cam = rec.cameras[img.camera_id]; p=list(cam.params)
    K = torch.tensor([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]],dtype=torch.float32,device=DEV)
    M = np.asarray(img.cam_from_world().matrix(),np.float32)
    vm = torch.eye(4,device=DEV); vm[:3,:4]=torch.from_numpy(M).to(DEV)
    W,H = int(cam.width),int(cam.height)
    with torch.no_grad():
        r,_,_ = rasterization(means=means,quats=quats,scales=scales,opacities=opac,
                              colors=colors,viewmats=vm[None],Ks=K[None],width=W,height=H,
                              sh_degree=3,packed=False,render_mode="RGB")
    pred=(r[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    gt=np.asarray(Image.open(os.path.join(ROOT,"images",img.name)).convert("RGB"))
    combo=np.concatenate([gt,np.full((H,8,3),255,np.uint8),pred],1)
    out=os.path.join(ROOT,"output",f"preview_{n}_{img.name}")
    Image.fromarray(combo).save(out)
    print("gespeichert:",os.path.basename(out))
