#!/usr/bin/env python3
"""Equirectangular (360) frames -> perspektivische Pinhole-Views fuer COLMAP/3DGS.

Pro Panorama werden mehrere rektilineare Ansichten gerendert (gemeinsame Intrinsik),
sodass COLMAP eine normale Pinhole-SfM rechnen kann.
"""
import os, glob, math
import numpy as np
import cv2

SRC   = "frames_equirect"
DST   = "images"                      # gsplat/COLMAP erwartet Bilder unter images/
OUT_W = OUT_H = 1024                  # quadratische Views
HFOV  = 90.0                          # Grad
# (yaw, pitch) in Grad: 6er-Ring am Horizont, leicht nach unten (Drohne schaut vorn/unten)
VIEWS = [(yaw, -20.0) for yaw in range(0, 360, 60)]

f  = (OUT_W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
cx = OUT_W / 2.0
cy = OUT_H / 2.0

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float64)
def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float64)

def build_maps(eq_w, eq_h, yaw_deg, pitch_deg):
    u, v = np.meshgrid(np.arange(OUT_W), np.arange(OUT_H))
    x = (u - cx) / f
    y = (v - cy) / f
    z = np.ones_like(x)
    d = np.stack([x, y, z], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    R = rot_y(math.radians(yaw_deg)) @ rot_x(math.radians(pitch_deg))
    d = d @ R.T
    X, Y, Z = d[..., 0], d[..., 1], d[..., 2]
    lon = np.arctan2(X, Z)            # [-pi, pi]
    lat = np.arcsin(np.clip(Y, -1, 1))  # [-pi/2, pi/2]
    map_x = (lon / (2 * math.pi) + 0.5) * eq_w
    map_y = (0.5 - lat / math.pi) * eq_h
    return map_x.astype(np.float32), map_y.astype(np.float32)

os.makedirs(DST, exist_ok=True)
frames = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
assert frames, "keine equirect-Frames gefunden"
sample = cv2.imread(frames[0]); eq_h, eq_w = sample.shape[:2]
maps = {v: build_maps(eq_w, eq_h, *v) for v in VIEWS}

n = 0
for fi, fp in enumerate(frames):
    img = cv2.imread(fp)
    for vi, view in enumerate(VIEWS):
        mx, my = maps[view]
        persp = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = os.path.join(DST, f"f{fi:03d}_v{vi}.jpg")
        cv2.imwrite(out, persp, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1

print(f"{n} perspektivische Views aus {len(frames)} Panoramen ({len(VIEWS)} Views/Pano)")
print(f"Intrinsik (gemeinsam): SIMPLE_PINHOLE f={f:.2f} cx={cx:.1f} cy={cy:.1f}  {OUT_W}x{OUT_H}")
