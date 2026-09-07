#!/usr/bin/env python3
"""venv (gsplat+CUDA): rendert das ECHTE 3DGS-Splat (m4t_geo) im LiDAR-Frame
auf der Montage-Orbitbahn -> renders/splat_frames/f####.png (Frames 0..NMAX).
Transform m4t_geo->LiDAR aus output/m4tgeo_to_lidar.json (rigide, s=1)."""
import os, sys, json, numpy as np, torch
from gsplat import rasterization
from PIL import Image
ROOT=os.path.dirname(os.path.abspath(__file__))
PLY=os.path.join(ROOT,"m4t_geo","output","avata360_splat.ply")
OUTD=os.path.join(ROOT,"renders","splat_frames"); os.makedirs(OUTD,exist_ok=True)
W,H=1920,1080; FOV=55.0; CENTER=np.array([-99.4,-6.2,1.5]); RADIUS=110.0; ELEV=45.0
REV=900.0   # Frames pro Umdrehung (muss mit build_montage uebereinstimmen)
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 320
dev="cuda"; C0=0.28209479177387814

def load_ply(path):
    with open(path,"rb") as f:
        hdr=b''
        while b'end_header' not in hdr: hdr+=f.readline()
        names=[l.split()[-1] for l in hdr.decode().splitlines() if l.startswith('property float')]
        n=[int(l.split()[-1]) for l in hdr.decode().splitlines() if l.startswith('element vertex')][0]
        d=np.fromfile(f,np.float32,n*len(names)).reshape(n,len(names))
    c={nm:i for i,nm in enumerate(names)}
    xyz=d[:,[c['x'],c['y'],c['z']]]
    fdc=d[:,[c['f_dc_0'],c['f_dc_1'],c['f_dc_2']]]
    op=d[:,c['opacity']]; sc=d[:,[c['scale_0'],c['scale_1'],c['scale_2']]]
    q=d[:,[c['rot_0'],c['rot_1'],c['rot_2'],c['rot_3']]]  # w,x,y,z
    return xyz,fdc,op,sc,q

xyz,fdc,op,sc,quat=load_ply(PLY)
T=np.array(json.load(open(os.path.join(ROOT,"output","m4tgeo_to_lidar.json")))["T"])
R=T[:3,:3]; t=T[:3,3]                      # s=1 rigide
xyz=(R@xyz.T).T+t
# Quaternion-Rotation: R_global ∘ q  (w,x,y,z)
def mat2q(M):
    tr=np.trace(M)
    if tr>0:
        S=np.sqrt(tr+1)*2; w=.25*S; x=(M[2,1]-M[1,2])/S; y=(M[0,2]-M[2,0])/S; z=(M[1,0]-M[0,1])/S
    elif M[0,0]>M[1,1] and M[0,0]>M[2,2]:
        S=np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2; w=(M[2,1]-M[1,2])/S; x=.25*S; y=(M[0,1]+M[1,0])/S; z=(M[0,2]+M[2,0])/S
    elif M[1,1]>M[2,2]:
        S=np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2; w=(M[0,2]-M[2,0])/S; x=(M[0,1]+M[1,0])/S; y=.25*S; z=(M[1,2]+M[2,1])/S
    else:
        S=np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2; w=(M[1,0]-M[0,1])/S; x=(M[0,2]+M[2,0])/S; y=(M[1,2]+M[2,1])/S; z=.25*S
    return np.array([w,x,y,z])
def qmul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],1)
quat=qmul(mat2q(R),quat)

means=torch.tensor(xyz,dtype=torch.float32,device=dev)
quats=torch.tensor(quat,dtype=torch.float32,device=dev)
scales=torch.tensor(np.exp(sc),dtype=torch.float32,device=dev)
opac=torch.tensor(1/(1+np.exp(-op)),dtype=torch.float32,device=dev)
cols=torch.tensor(np.clip(0.5+C0*fdc,0,1),dtype=torch.float32,device=dev)
print(f"{len(means)} Gaussians geladen/transformiert")

fy=(H/2)/np.tan(np.radians(FOV/2)); fx=fy
K=torch.tensor([[[fx,0,W/2],[0,fy,H/2],[0,0,1]]],dtype=torch.float32,device=dev)
bg=torch.tensor([[0.05,0.05,0.06]],dtype=torch.float32,device=dev)
def viewmat(th):
    el=np.radians(ELEV)
    eye=CENTER+np.array([RADIUS*np.cos(el)*np.cos(th),RADIUS*np.cos(el)*np.sin(th),RADIUS*np.sin(el)])
    f=CENTER-eye; f/=np.linalg.norm(f)
    r=np.cross(f,[0,0,1]); r/=np.linalg.norm(r); dn=np.cross(f,r)
    Rwc=np.stack([r,dn,f],0); V=np.eye(4); V[:3,:3]=Rwc; V[:3,3]=-Rwc@eye
    return V

for i in range(NMAX):
    th=2*np.pi*(i/REV)
    V=torch.tensor(viewmat(th)[None],dtype=torch.float32,device=dev)
    with torch.no_grad():
        out,_,_=rasterization(means,quats,scales,opac,cols,V,K,W,H,render_mode="RGB")
    img=(out[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUTD,f"f{i:04d}.png"))
    if i%60==0: print(f"  {i}/{NMAX}")
print("fertig ->",OUTD)
