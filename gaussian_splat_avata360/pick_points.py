#!/usr/bin/env python3
"""Korrespondenz-Picking LiDAR <-> Splat (Open3D).

Fenster 1: LiDAR (nach Hoehe eingefaerbt). Shift+Linksklick auf 3-4 markante Punkte.
Fenster 2: Splat (Originalfarben). Dieselben Punkte in GLEICHER Reihenfolge.
Steuerung: Shift+Linksklick = Punkt waehlen, Shift+Rechtsklick = letzten zuruecknehmen,
           Maus = drehen/zoomen,  Q oder Esc = Fenster schliessen & weiter.

Schreibt picks_lidar.json / picks_splat.json. Danach 'compute' (separat) rechnet
den Transform. Hier wird nach beiden Fenstern direkt gerechnet, falls Anzahl passt.
"""
import os, json, numpy as np, open3d as o3d

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUT   = os.path.join(ROOT, "output")

def read_pcd_xyz(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"POINTS"): n = int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f, dtype=np.float32, count=n*3).reshape(n, 3).astype(float)

def turbo(v):
    v = np.clip(v, 0, 1)
    return np.stack([np.clip(1.5-abs(4*v-3),0,1),
                     np.clip(1.5-abs(4*v-2),0,1),
                     np.clip(1.5-abs(4*v-1),0,1)], -1)

def pick(pc, title):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pc)
    vis.run()              # blockiert bis Fenster geschlossen
    vis.destroy_window()
    return vis.get_picked_points()

# --- LiDAR vorbereiten (hoehenfarbig, leicht ausgeduennt) ---
L = read_pcd_xyz(LIDAR)
lpc = o3d.geometry.PointCloud(); lpc.points = o3d.utility.Vector3dVector(L)
lpc = lpc.voxel_down_sample(0.15)
Lp = np.asarray(lpc.points)
z = Lp[:,2]; zn = (z-np.percentile(z,2))/(np.percentile(z,98)-np.percentile(z,2)+1e-9)
lpc.colors = o3d.utility.Vector3dVector(turbo(zn))

# --- Splat vorbereiten (Originalfarben, Floater gekappt) ---
sp = o3d.io.read_point_cloud(SPLAT)
S = np.asarray(sp.points)
lo, hi = np.percentile(S, [1, 99], axis=0)
sp = sp.select_by_index(np.where(np.all((S>=lo)&(S<=hi),1))[0])
sp = sp.voxel_down_sample(0.1)

print("\n=== FENSTER 1: LiDAR (Farbe = Hoehe) ===")
print("Shift+Linksklick auf 3-4 markante Merkmale, dann Q druecken.\n")
il = pick(lpc, "1/2 LiDAR  -  Shift+Klick auf Merkmale, dann Q")
PL = np.asarray(lpc.points)[il]
json.dump(PL.tolist(), open(os.path.join(OUT,"picks_lidar.json"),"w"))
print(f"LiDAR-Picks: {len(il)} -> {PL.round(2).tolist()}")

print("\n=== FENSTER 2: Splat (Originalfarben) ===")
print("Dieselben Merkmale in GLEICHER Reihenfolge anklicken, dann Q.\n")
isp = pick(sp, "2/2 Splat  -  gleiche Punkte, gleiche Reihenfolge, dann Q")
PS = np.asarray(sp.points)[isp]
json.dump(PS.tolist(), open(os.path.join(OUT,"picks_splat.json"),"w"))
print(f"Splat-Picks: {len(isp)} -> {PS.round(2).tolist()}")

if len(PL) != len(PS) or len(PL) < 3:
    print(f"\n!! Anzahl passt nicht ({len(PL)} vs {len(PS)}) oder <3."
          " Picks sind gespeichert; bitte erneut starten oder 'compute_manual.py' nutzen.")
    raise SystemExit(1)

print("\nPicks gespeichert. Rechne Transform mit compute_manual.py ...")
