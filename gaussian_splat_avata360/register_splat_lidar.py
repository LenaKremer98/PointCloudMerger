#!/usr/bin/env python3
"""Schritt 2: Registriert das metrische Splat (Quelle) auf die LiDAR-Wolke (Ziel).

Beide sind metrisch und Z=oben -> Grobausrichtung per Yaw-Multistart (Schwerpunkt-
ausrichtung + Yaw-Sweep), Feinausrichtung per Point-to-Plane ICP. Bestes Ergebnis
nach Fitness/RMSE. Speichert T (Splat->LiDAR) und eine Vorschau-Wolke.

System-Python (open3d).
"""
import os, json, numpy as np, open3d as o3d, copy

ROOT   = os.path.dirname(os.path.abspath(__file__))
LIDAR  = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT  = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUTDIR = os.path.join(ROOT, "output")
VOXEL  = 0.5

def read_pcd_xyz(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"POINTS"): n = int(line.split()[1])
            if line.startswith(b"DATA"): break
        d = np.fromfile(f, dtype=np.float32, count=n*3).reshape(n, 3)
    return d.astype(np.float64)

# --- Laden ---
lidar_xyz = read_pcd_xyz(LIDAR)
splat = o3d.io.read_point_cloud(SPLAT)
splat_xyz = np.asarray(splat.points)
splat_rgb = np.asarray(splat.colors)
print(f"LiDAR {len(lidar_xyz)} | Splat {len(splat_xyz)}")

lidar = o3d.geometry.PointCloud(); lidar.points = o3d.utility.Vector3dVector(lidar_xyz)

# --- Splat-Floater kappen (robuster Kasten 1-99%) ---
lo, hi = np.percentile(splat_xyz, [1, 99], axis=0)
m = np.all((splat_xyz >= lo) & (splat_xyz <= hi), axis=1)
splat = splat.select_by_index(np.where(m)[0])
print(f"Splat nach Box-Crop: {len(splat.points)}")

# --- Downsample + SOR + Normalen ---
def prep(pc):
    pc = pc.voxel_down_sample(VOXEL)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL*3, max_nn=30))
    return pc

src = prep(copy.deepcopy(splat))   # Splat (Quelle)
tgt = prep(copy.deepcopy(lidar))   # LiDAR (Ziel)
print(f"Downsampled: Splat {len(src.points)} | LiDAR {len(tgt.points)}")

src_c = src.get_center(); tgt_c = tgt.get_center()
print(f"Schwerpunkt Splat {src_c.round(1)} | LiDAR {tgt_c.round(1)}")

def yaw_T(theta, src_center, tgt_center):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    T = np.eye(4); T[:3,:3] = R
    # erst Quelle in Ursprung, drehen, dann auf Zielschwerpunkt
    T[:3,3] = tgt_center - R @ src_center
    return T

# --- Yaw-Multistart + ICP ---
best = None
for deg in range(0, 360, 30):
    Tinit = yaw_T(np.radians(deg), src_c, tgt_c)
    # zwei ICP-Stufen: grob (3 m) dann fein (1 m)
    r1 = o3d.pipelines.registration.registration_icp(
        src, tgt, 3.0, Tinit,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    r2 = o3d.pipelines.registration.registration_icp(
        src, tgt, 1.0, r1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    score = (r2.fitness, -r2.inlier_rmse)
    print(f"  yaw {deg:3d}: fitness={r2.fitness:.3f} rmse={r2.inlier_rmse:.3f}")
    if best is None or score > best[0]:
        best = (score, deg, r2)

score, deg, r = best
print(f"\nBeste Grobausrichtung: yaw {deg}, fitness={r.fitness:.3f}, rmse={r.inlier_rmse:.3f}")

# --- Feinausrichtung ---
rf = o3d.pipelines.registration.registration_icp(
    src, tgt, 0.5, r.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
print(f"Fein-ICP (0.5 m): fitness={rf.fitness:.3f}, rmse={rf.inlier_rmse:.3f}")
T = rf.transformation

json.dump({"T_splat_to_lidar": T.tolist(), "yaw_deg": deg,
           "fitness": rf.fitness, "inlier_rmse": rf.inlier_rmse, "voxel": VOXEL},
          open(os.path.join(OUTDIR, "splat_to_lidar.json"), "w"), indent=2)
print(f"-> {os.path.join(OUTDIR,'splat_to_lidar.json')}")

# --- Vorschau: farbiges (downsampled) Splat im LiDAR-Frame + graues LiDAR ---
prev_splat = copy.deepcopy(splat).voxel_down_sample(0.3).transform(T)
prev_lidar = copy.deepcopy(lidar).voxel_down_sample(0.3)
prev_lidar.paint_uniform_color([0.5, 0.5, 0.5])
prev = prev_splat + prev_lidar
out = os.path.join(OUTDIR, "preview_aligned.ply")
o3d.io.write_point_cloud(out, prev)
print(f"-> {out}  (farbiges Splat ueber grauem LiDAR, zur Sichtkontrolle)")
