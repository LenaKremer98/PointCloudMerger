#!/usr/bin/env python3
"""Orbit-Video um die blauen Container, 45-Grad-Blick nach unten, 20 s.
Open3D OffscreenRenderer -> PNG-Frames -> (ffmpeg extern) mp4."""
import os, sys, numpy as np, open3d as o3d
from open3d.visualization import rendering

ROOT=os.path.dirname(os.path.abspath(__file__))
PLY=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ_m4t_manual2.ply")
FRAMES=os.path.join(ROOT,"renders","orbit_frames"); os.makedirs(FRAMES,exist_ok=True)

CENTER=np.array([-99.4,-6.2,1.5])   # blaue Container
RADIUS=float(os.environ.get("RADIUS","85"))
ELEV=float(os.environ.get("ELEV","45"))
FOV=float(os.environ.get("FOV","55"))
W,H=2560,1440
DUR=20.0; FPS=30; N=int(DUR*FPS)
TEST = len(sys.argv)>1 and sys.argv[1]=="test"

pcd=o3d.io.read_point_cloud(PLY)
print(f"{len(pcd.points)} Punkte geladen")
# fliegende Punkte / Ausreisser entfernen
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
pcd,_=pcd.remove_radius_outlier(nb_points=12, radius=1.2)
print(f"{len(pcd.points)} Punkte nach Ausreisser-Filter")

rend=rendering.OffscreenRenderer(W,H)
rend.scene.set_background([0.12,0.12,0.12,1.0])
mat=rendering.MaterialRecord(); mat.shader="defaultUnlit"; mat.point_size=2.0
rend.scene.add_geometry("pc",pcd,mat)
rend.scene.scene.enable_sun_light(False)
rend.scene.view.set_post_processing(False)   # kein Tonemapping -> Farben exakt wie RViz (RGB8)

el=np.radians(ELEV)
idxs=[0,N//4,N//2,3*N//4] if TEST else range(N)
for i in idxs:
    th=2*np.pi*i/N
    eye=CENTER+np.array([RADIUS*np.cos(el)*np.cos(th),
                         RADIUS*np.cos(el)*np.sin(th),
                         RADIUS*np.sin(el)])
    rend.setup_camera(FOV, CENTER.tolist(), eye.tolist(), [0,0,1])
    img=rend.render_to_image()
    fn=os.path.join(FRAMES, f"f{i:04d}.png")
    o3d.io.write_image(fn, img)
    if i%60==0: print(f"  Frame {i}/{N}")
print("Frames fertig ->", FRAMES)
