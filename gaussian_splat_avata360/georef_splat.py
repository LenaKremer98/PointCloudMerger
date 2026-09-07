#!/usr/bin/env python3
"""Schritt 1: Georeferenziert das Avata-Splat auf metrischen Maßstab.

Umeyama-Similarity: COLMAP-Kamerazentren (sparse/0) -> GPS-ENU (aus SRT).
Wendet den Transform (s,R,t) auf die (bereinigte) Splat-PLY an, berechnet RGB
aus den SH-DC-Koeffizienten und schreibt eine metrische, farbige Punktwolke.
Speichert den Transform zur Weiterverwendung als JSON.

venv-Python (pycolmap), kein open3d nötig.
"""
import os, json, numpy as np, pycolmap
from geo_utils import parse_srt, to_enu, umeyama

ROOT   = os.path.dirname(os.path.abspath(__file__))
SPARSE = os.path.join(ROOT, "sparse", "0")
SRT    = os.path.join(ROOT, "..", "avatar360", "DCIM", "DJI_001", "DJI_20260520144939_0001_D.SRT")
PLY_IN = os.path.join(ROOT, "output", "avata360_splat_clean.ply")
OUT_DIR= os.path.join(ROOT, "output")
FPS    = 45/88.2882   # identisch zu georef_merge.py

C0 = 0.28209479177387814  # SH-DC -> Farbe

def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        names, n = [], 0
        for line in header.decode().splitlines():
            if line.startswith("property float"): names.append(line.split()[-1])
            if line.startswith("element vertex"): n = int(line.split()[-1])
        data = np.fromfile(f, dtype=np.float32, count=n*len(names)).reshape(n, len(names))
    return {nm: data[:, i] for i, nm in enumerate(names)}

# --- 1. Korrespondenzen Kamerazentrum <-> GPS-ENU ---
rec = pycolmap.Reconstruction(SPARSE)
st, sla, slo, sra = parse_srt(SRT)
lat0, lon0 = sla.mean(), slo.mean()   # eigener ENU-Origin (irrelevant fuer spaetere Registrierung)

C, G = [], []
for im in rec.images.values():
    M = np.asarray(im.cam_from_world().matrix(), float)
    C.append(-M[:, :3].T @ M[:, 3])
    fi = int(im.name[1:4]); j = np.argmin(np.abs(st - fi/FPS))
    G.append(to_enu(sla[j], slo[j], sra[j], lat0, lon0))
C, G = np.stack(C), np.stack(G)
s, R, t = umeyama(C, G)
C2 = (s * (R @ C.T).T) + t
res = np.linalg.norm(C2 - G, axis=1)
print(f"Umeyama: scale s={s:.4f}")
print(f"GPS-Residuum: mean {res.mean():.2f} m, median {np.median(res):.2f} m, max {res.max():.2f} m  ({len(C)} Kameras)")

# --- 2. Splat laden, transformieren, einfaerben ---
v = read_ply(PLY_IN)
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
xyz_m = (s * (R @ xyz.T).T) + t        # metrisches ENU
dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
rgb = np.clip(0.5 + C0 * dc, 0, 1)

ext = xyz_m.max(0) - xyz_m.min(0)
print(f"Splat metrisch: {len(xyz_m)} Punkte, Ausdehnung {ext.round(1)} m")
lo, hi = np.percentile(xyz_m, [1, 99], axis=0)
print(f"  robuste Ausdehnung (1-99%): {(hi-lo).round(1)} m")

# --- 3. Transform speichern ---
json.dump({"scale": float(s), "R": R.tolist(), "t": t.tolist(),
           "lat0": float(lat0), "lon0": float(lon0),
           "gps_res_mean_m": float(res.mean()), "gps_res_max_m": float(res.max())},
          open(os.path.join(OUT_DIR, "splat_georef.json"), "w"), indent=2)

# --- 4. farbige metrische Wolke als binaeres PLY ---
out = os.path.join(OUT_DIR, "avata360_splat_metric_rgb.ply")
rgb8 = (rgb * 255).round().astype(np.uint8)
N = len(xyz_m)
with open(out, "wb") as f:
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {N}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    f.write(hdr.encode())
    rec_dt = np.zeros(N, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                 ("red","u1"),("green","u1"),("blue","u1")])
    rec_dt["x"], rec_dt["y"], rec_dt["z"] = xyz_m[:,0], xyz_m[:,1], xyz_m[:,2]
    rec_dt["red"], rec_dt["green"], rec_dt["blue"] = rgb8[:,0], rgb8[:,1], rgb8[:,2]
    rec_dt.tofile(f)
print(f"-> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
print(f"-> {os.path.join(OUT_DIR,'splat_georef.json')}")
