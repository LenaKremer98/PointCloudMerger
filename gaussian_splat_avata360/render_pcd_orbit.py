#!/usr/bin/env python3
"""Point-Cloud-Orbit, dessen Kamerabahn die Splat-Umrundung nahtlos fortsetzt.
Gleiche Kamera-Konvention/FOV/Elevation/Drehgeschwindigkeit wie render_orbit.py,
Hoehen-Regenbogen wie das Original-Video. Reiner NumPy-Z-Buffer-Splat-Renderer."""
import os, sys, numpy as np
from PIL import Image

def hsv_to_rgb(h, s, v):
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], 1)

ROOT = os.path.dirname(os.path.abspath(__file__))
PCD  = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
OUTDIR = os.path.join(ROOT, "renders", "_pcd_frames"); os.makedirs(OUTDIR, exist_ok=True)
TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

# --- PCD laden (ASCII-Header bis DATA, dann float32-Triplets) ---
def load_pcd(p):
    with open(p, "rb") as f:
        H = {}
        while True:
            ln = f.readline()
            t = ln.decode("ascii", "ignore").split()
            if t: H[t[0]] = t[1:]
            if ln.startswith(b"DATA"): break
        n = int(H["POINTS"][0]); rec = sum(map(int, H["SIZE"]))
        buf = f.read(n * rec)
    return np.frombuffer(buf[:n*rec], np.float32).reshape(n, rec // 4)[:, :3].astype(np.float32)

P = load_pcd(PCD)
print(f"{len(P)} Punkte geladen")

# --- Hoehen-Regenbogen: blau(tief) -> ... -> rot -> magenta(hoch) ---
z = P[:, 2]
lo, hi = np.percentile(z, 1), np.percentile(z, 99)
t = np.clip((z - lo) / (hi - lo), 0, 1)
hue = (0.66 - 0.83 * t) % 1.0
COL = (hsv_to_rgb(hue, np.ones_like(t), np.ones_like(t)) * 255).astype(np.uint8)   # (N,3)

# --- robustes Zentrum + Orbit-Radius (wie beim Splat) ---
center = np.median(P, 0).astype(np.float32)
rh = np.sqrt(((P[:, :2] - center[:2]) ** 2).sum(1))
scene_r = float(np.percentile(rh, 95))
Rh = scene_r * 1.30
print(f"center={center} scene_r(p95)={scene_r:.1f} Rh={Rh:.1f}")

W, H, foc = 1920, 1080, 1100
cx, cy = W / 2, H / 2
ELEV = np.deg2rad(45.0)

# Drehung identisch zum Splat (643 frames / 360deg); Start versetzt fuer Crossfade-Alignment
FPS = 30
N = 643
XF = 60                       # Crossfade-Fenster (frames) -> Startazimut zurueckgesetzt
AZ0 = 360.0 - 360.0 * XF / N  # damit Splat-Ende und PCD-Start im XF deckungsgleich rotieren

def viewmat(az_deg):
    az = np.deg2rad(az_deg)
    eye = center + Rh * np.array([np.cos(ELEV)*np.cos(az), np.cos(ELEV)*np.sin(az), np.sin(ELEV)], np.float32)
    f = center - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0,0,1.]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R = np.stack([r, -u, f]); tvec = -R @ eye
    return R.astype(np.float32), tvec.astype(np.float32)

def render(az_deg):
    R, tv = viewmat(az_deg)
    Pc = P @ R.T + tv                       # world -> cam
    zc = Pc[:, 2]
    m = zc > 0.1
    Pc = Pc[m]; zc = zc[m]; col = COL[m]
    u = (foc * Pc[:, 0] / zc + cx)
    v = (foc * Pc[:, 1] / zc + cy)
    ui = np.round(u).astype(np.int32); vi = np.round(v).astype(np.int32)
    inb = (ui >= 1) & (ui < W-1) & (vi >= 1) & (vi < H-1)
    ui, vi, zc, col = ui[inb], vi[inb], zc[inb], col[inb]
    order = np.argsort(-zc)                  # fern -> nah (nah ueberschreibt)
    ui, vi, col = ui[order], vi[order], col[order]
    img = np.zeros((H, W, 3), np.uint8)
    for dy in (-1, 0, 1):                    # 3x3-Splat fuer dichten Look
        for dx in (-1, 0, 1):
            img[vi+dy, ui+dx] = col
    return img

if TEST:
    for az in (AZ0, 90, 200):
        Image.fromarray(render(az)).save(f"/tmp/pcd_test_{int(az)}.png")
        print("test az", round(az,1))
    sys.exit(0)

for i in range(N):
    az = AZ0 + 360.0 * i / N
    Image.fromarray(render(az)).save(os.path.join(OUTDIR, f"f{i:04d}.png"))
    if i % 60 == 0: print(f"frame {i}/{N}")
print("PCD-Frames fertig.")
