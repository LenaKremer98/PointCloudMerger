#!/usr/bin/env python3
"""Render aus gezielt guten Trainingskameras + eine freie Orbit-Ansicht von aussen."""
import os, sys, numpy as np, torch, math
from plyfile import PlyData
from PIL import Image
import pycolmap
from gsplat.rendering import rasterization
ROOT=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
PLY=os.path.join(ROOT,"output","avata360_splat_clean.ply")
v=PlyData.read(PLY)["vertex"]; N=len(v["x"])
means=torch.tensor(np.stack([v["x"],v["y"],v["z"]],1),dtype=torch.float32,device=DEV)
sh0=torch.tensor(np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1),dtype=torch.float32,device=DEV)[:,None,:]
frest=np.stack([v[f"f_rest_{i}"] for i in range(45)],1).reshape(N,3,15)
shN=torch.tensor(np.transpose(frest,(0,2,1)),dtype=torch.float32,device=DEV)
colors=torch.cat([sh0,shN],1)
scales=torch.tensor(np.stack([v["scale_0"],v["scale_1"],v["scale_2"]],1),dtype=torch.float32,device=DEV).exp()
quats=torch.tensor(np.stack([v["rot_0"],v["rot_1"],v["rot_2"],v["rot_3"]],1),dtype=torch.float32,device=DEV)
opac=torch.tensor(v["opacity"],dtype=torch.float32,device=DEV).sigmoid()
print(f"{N} Gaussians")

rec=pycolmap.Reconstruction(os.path.join(ROOT,"sparse","0"))
by_name={im.name:im for im in rec.images.values()}

def render(vm,K,W,H):
    with torch.no_grad():
        r,_,_=rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=colors,
            viewmats=vm[None],Ks=K[None],width=W,height=H,sh_degree=3,packed=False,render_mode="RGB")
    return (r[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)

# gezielte gute Trainingskameras (zeigen Gebaeude/Boden mit wenig Schwarz)
wanted=[n for n in ["f020_l1_y090_p55.jpg","f015_l1_y180_p55.jpg","f025_l1_y000_p35.jpg","f010_l1_y270_p55.jpg"] if n in by_name][:3]
for k,nm in enumerate(wanted):
    im=by_name[nm]; cam=rec.cameras[im.camera_id]; p=list(cam.params)
    K=torch.tensor([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]],dtype=torch.float32,device=DEV)
    M=np.asarray(im.cam_from_world().matrix(),np.float32)
    vm=torch.eye(4,device=DEV); vm[:3,:4]=torch.from_numpy(M).to(DEV)
    gt=np.asarray(Image.open(os.path.join(ROOT,"images",nm)).convert("RGB"))
    combo=np.concatenate([gt,np.full((cam.height,8,3),255,np.uint8),render(vm,K,cam.width,cam.height)],1)
    Image.fromarray(combo).save(os.path.join(ROOT,"output",f"check_{k}_{nm}")); print("ok",nm)
print("fertig")
