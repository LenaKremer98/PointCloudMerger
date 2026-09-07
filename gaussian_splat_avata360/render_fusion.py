#!/usr/bin/env python3
"""Render des fusionierten Splats: Avata-Schraegansicht, M4T-Nadir (je GT|Render),
und eine freie Luftuebersicht des gesamten georeferenzierten Gelaendes."""
import os, numpy as np, torch, math
from plyfile import PlyData
from PIL import Image
import pycolmap
from gsplat.rendering import rasterization
ROOT=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
FUS=os.path.join(ROOT,"fusion")
v=PlyData.read(os.path.join(FUS,"output","avata360_splat_clean.ply"))["vertex"]; N=len(v["x"])
means=torch.tensor(np.stack([v["x"],v["y"],v["z"]],1),dtype=torch.float32,device=DEV)
sh0=torch.tensor(np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1),dtype=torch.float32,device=DEV)[:,None,:]
frest=np.stack([v[f"f_rest_{i}"] for i in range(45)],1).reshape(N,3,15)
shN=torch.tensor(np.transpose(frest,(0,2,1)),dtype=torch.float32,device=DEV)
colors=torch.cat([sh0,shN],1)
scales=torch.tensor(np.stack([v["scale_0"],v["scale_1"],v["scale_2"]],1),dtype=torch.float32,device=DEV).exp()
quats=torch.tensor(np.stack([v["rot_0"],v["rot_1"],v["rot_2"],v["rot_3"]],1),dtype=torch.float32,device=DEV)
opac=torch.tensor(v["opacity"],dtype=torch.float32,device=DEV).sigmoid()
print(f"{N} Gaussians")
rec=pycolmap.Reconstruction(os.path.join(FUS,"sparse","0")); by_name={im.name:im for im in rec.images.values()}

def render(vm,K,W,H):
    with torch.no_grad():
        r,_,_=rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=colors,
            viewmats=vm[None],Ks=K[None],width=W,height=H,sh_degree=3,packed=False,render_mode="RGB")
    return (r[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)

def from_image(nm,outname):
    im=by_name[nm]; cam=rec.cameras[im.camera_id]; p=list(cam.params)
    K=torch.tensor([[p[0],0,p[1]],[0,p[0],p[2]],[0,0,1]],dtype=torch.float32,device=DEV)
    M=np.asarray(im.cam_from_world().matrix(),np.float32)
    vm=torch.eye(4,device=DEV); vm[:3,:4]=torch.from_numpy(M).to(DEV)
    gt=np.asarray(Image.open(os.path.join(FUS,"images",nm)).convert("RGB"))
    Image.fromarray(np.concatenate([gt,np.full((cam.height,8,3),255,np.uint8),render(vm,K,cam.width,cam.height)],1)).save(
        os.path.join(FUS,"output",outname)); print("ok",outname)

# Avata-Schraegansicht + M4T-Nadir
from_image("f020_l1_y090_p55.jpg","fus_avata.jpg")
m4t_name=next(n for n in by_name if n.endswith("_V.JPG"))
from_image(m4t_name,"fus_m4t.jpg")

# --- freie Luftuebersicht (look-at) ---
def lookat(eye,target,up=np.array([0,0,1.])):
    f=target-eye; f/=np.linalg.norm(f)
    r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
    R=np.stack([r,-u,f])           # world->cam (z=forward)
    t=-R@eye
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=t
    return torch.tensor(M,dtype=torch.float32,device=DEV)
W,H=1600,1000; foc=900
K=torch.tensor([[foc,0,W/2],[0,foc,H/2],[0,0,1]],dtype=torch.float32,device=DEV)
eye=np.array([10.,-90.,75.]); target=np.array([0.,30.,2.])
img=render(lookat(eye,target),K,W,H)
Image.fromarray(img).save(os.path.join(FUS,"output","fus_overview.jpg")); print("ok fus_overview.jpg")
