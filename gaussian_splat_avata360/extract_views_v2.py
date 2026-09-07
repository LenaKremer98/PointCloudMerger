#!/usr/bin/env python3
"""Korrekte Pipeline: Full-Res-OSV Dual-Fisheye -> rektilineare Perspektiv-Views.
Extrahiert N Frames aus beiden Linsen-Streams und entzerrt sie ins Horizont-Band."""
import os, glob, subprocess, cv2
from fisheye_to_perspective import perspective_from_fisheye

ROOT  = os.path.dirname(os.path.abspath(__file__))
OSV   = os.path.join(ROOT, "..", "avatar360", "DCIM", "DJI_001",
                     "DJI_20260520144939_0001_D.OSV")
IMG   = os.path.join(ROOT, "images")        # ueberschreibt die alten (falschen) Views
NF    = 45                                  # Frames (mehr Parallaxe)
DUR   = 88.2882
OUT   = 1600                                # native Fisheye-Pixeldichte (~1560px/78deg)
HFOV  = 78.0                                # 78/2=39; +pitch bleibt < 95 (190deg-Kreis) -> KEIN Schwarz
# (lens, yaw, pitch) - nur Boden-Linse (zeigt Gebaeude/Container/Boden), zwei Neigungen, voll im Kreis
VIEWS = [(1,yaw,pitch) for pitch in (35,55) for yaw in (0,90,180,270)]

# alte Views entfernen
for f in glob.glob(os.path.join(IMG, "*.jpg")): os.remove(f)
os.makedirs(IMG, exist_ok=True)
t0 = "/tmp/v2_l0"; t1 = "/tmp/v2_l1"
for d in (t0,t1):
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(d+"/*.jpg"): os.remove(f)

print(f"Extrahiere {NF} Full-Res-Frames aus beiden Linsen ...")
for stream,dst in [("0:0",t0),("0:1",t1)]:
    subprocess.run(["ffmpeg","-y","-i",OSV,"-map",stream,
                    "-vf",f"fps={NF/DUR}","-q:v","2",
                    os.path.join(dst,"f_%03d.jpg")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
l0 = sorted(glob.glob(t0+"/*.jpg")); l1 = sorted(glob.glob(t1+"/*.jpg"))
print(f"  lens0={len(l0)} lens1={len(l1)} Frames")

n=0
for fi in range(min(len(l0),len(l1))):
    imgs = {0: cv2.imread(l0[fi]), 1: cv2.imread(l1[fi])}
    for (lens,yaw,pitch) in VIEWS:
        p = perspective_from_fisheye(imgs[lens], yaw, pitch, out=OUT, hfov_deg=HFOV)
        cv2.imwrite(os.path.join(IMG,f"f{fi:03d}_l{lens}_y{yaw:03d}_p{pitch:02d}.jpg"), p,
                    [cv2.IMWRITE_JPEG_QUALITY,95]); n+=1
print(f"{n} korrekte Perspektiv-Views -> {IMG}")
