#!/usr/bin/env python3
"""Render a 45-degree top-down orbit video of the merged point cloud,
colored by height using the exact RViz 'rainbow' colormap.
Headless numpy rasterizer -> PNG frames -> ffmpeg mp4."""
import sys, os, math
import numpy as np
import open3d as o3d

PCD = "merged_FinlaDRZ.pcd"
W, H = 1600, 900
FPS = 30
PER_REV = 15.3846       # seconds per full revolution (10 / 0.65 -> 35% slower)
N_REV = 3               # total revolutions (1: full, 2: cut down to Z_CUT, 3: held)
Z_CUT = 4.3             # final clip height (m, cloud z): points above are removed
ELEV_DEG = 45.0          # camera elevation above horizontal
FOV_DEG = 50.0
VOXEL = 0.12             # downsample voxel size (m); 0 disables
OUTDIR = "orbit_frames"
SPLAT = 0                # splat radius in px (0 -> single pixel, 1 -> 3x3 block)
OUTLIER_NB = 20          # statistical outlier removal: neighbors considered
OUTLIER_STD = 1.5        # ... std-dev ratio threshold (lower = more aggressive)
RADIUS_NB = 8            # radius outlier removal: min neighbors within RADIUS_R
RADIUS_R = 0.5           # ... search radius (m); kills isolated floating points

def rviz_rainbow(value):
    """RViz AxisColor/Intensity 'rainbow' (invert_rainbow=False).
    value: normalized height in [0,1]. Returns (N,3) float RGB 0..1."""
    value = np.clip(value, 0.0, 1.0)
    value = 1.0 - value                # rviz default: low -> red, high -> violet
    h = value * 5.0 + 1.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    even = (i & 1) == 0
    f = np.where(even, 1.0 - f, f)
    n = 1.0 - f
    r = np.zeros_like(value); g = np.zeros_like(value); b = np.zeros_like(value)
    m = i <= 1; r[m] = n[m]; b[m] = 1.0
    m = i == 2; g[m] = n[m]; b[m] = 1.0
    m = i == 3; g[m] = 1.0;  b[m] = n[m]
    m = i == 4; r[m] = n[m]; g[m] = 1.0
    m = i >= 5; r[m] = 1.0;  g[m] = n[m]
    return np.stack([r, g, b], axis=1)

def load():
    pc = o3d.io.read_point_cloud(PCD)
    if VOXEL > 0:
        pc = pc.voxel_down_sample(VOXEL)
    before = len(pc.points)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=OUTLIER_NB,
                                          std_ratio=OUTLIER_STD)
    mid = len(pc.points)
    pc, _ = pc.remove_radius_outlier(nb_points=RADIUS_NB, radius=RADIUS_R)
    print("outlier removal: %d -> %d (stat) -> %d (radius)" %
          (before, mid, len(pc.points)), file=sys.stderr)
    p = np.asarray(pc.points, dtype=np.float64)
    return p

def main():
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    test4 = len(sys.argv) > 1 and sys.argv[1] == "test4"
    p = load()
    print("rendering points:", len(p), file=sys.stderr)

    # height color (z), normalized by robust percentiles
    z = p[:, 2]
    zlo, zhi = np.percentile(z, [2, 98])
    norm = (z - zlo) / (zhi - zlo + 1e-9)
    colors = (rviz_rainbow(norm) * 255).astype(np.uint8)

    # framing: center on median xy, mid-z; fit bounding sphere (98th pct radius)
    center = np.array([np.median(p[:, 0]), np.median(p[:, 1]),
                       0.5 * (np.percentile(z, 1) + np.percentile(z, 99))])
    d = np.linalg.norm(p - center, axis=1)
    radius = np.percentile(d, 90)
    fov = math.radians(FOV_DEG)
    dist = radius / math.sin(fov / 2.0) * 0.78   # frame the dense core, allow slight edge clip
    f_px = (H / 2.0) / math.tan(fov / 2.0)

    os.makedirs(OUTDIR, exist_ok=True)
    per_rev = FPS * PER_REV                      # frames per revolution
    n_frames = 1 if test else int(round(per_rev * N_REV))
    z_top = z.max() + 0.01                       # start with nothing clipped
    el = math.radians(ELEV_DEG)
    world_up = np.array([0.0, 0.0, 1.0])

    plus = [(dx, dy) for dx in range(-SPLAT, SPLAT + 1)
                     for dy in range(-SPLAT, SPLAT + 1)]

    if test4:
        n_frames = 4
    testcut = len(sys.argv) > 1 and sys.argv[1] == "testcut"
    frame_ids = [0, int(1.5 * per_rev), int(1.9 * per_rev), int(2.5 * per_rev)] \
                if testcut else range(n_frames)
    for fi in frame_ids:
        az = (math.pi / 2.0) * fi if test4 else 2.0 * math.pi * fi / per_rev
        # clip plane: rev 1 full, rev 2 descends top->Z_CUT, rev 3+ held at Z_CUT
        if fi < per_rev:
            clip = z_top
        elif fi < 2 * per_rev:
            prog = (fi - per_rev) / per_rev      # 0..1 across revolution 2
            clip = z_top - prog * (z_top - Z_CUT)
        else:
            clip = Z_CUT
        dirv = np.array([math.cos(el) * math.cos(az),
                         math.cos(el) * math.sin(az),
                         math.sin(el)])
        cam = center + dist * dirv
        forward = (center - cam); forward /= np.linalg.norm(forward)
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        rel = p - cam
        zc = rel @ forward
        valid = (zc > 0.1) & (z <= clip)
        xc = rel @ right
        yc = rel @ up
        sx = (f_px * xc / zc + W / 2.0)
        sy = (H / 2.0 - f_px * yc / zc)
        px = np.round(sx).astype(np.int64)
        py = np.round(sy).astype(np.int64)
        inb = valid & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[inb]; py = py[inb]; zc_v = zc[inb]; col = colors[inb]

        img = np.zeros((H, W, 3), dtype=np.uint8)
        # expand splat, keep depth ordering across all splat pixels
        all_idx = []
        all_z = []
        all_col = []
        for dx, dy in plus:
            qx = px + dx; qy = py + dy
            ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            all_idx.append(qy[ok] * W + qx[ok])
            all_z.append(zc_v[ok])
            all_col.append(col[ok])
        idx = np.concatenate(all_idx)
        zz = np.concatenate(all_z)
        cc = np.concatenate(all_col)
        order = np.argsort(-zz)            # far first; near overwrites
        idx = idx[order]; cc = cc[order]
        flat = img.reshape(-1, 3)
        flat[idx] = cc                     # last write per pixel = nearest

        from PIL import Image
        Image.fromarray(img).save(os.path.join(OUTDIR, f"frame_{fi:04d}.png"))
        if fi % 30 == 0:
            print("frame", fi, "/", n_frames, file=sys.stderr)
    print("done frames", file=sys.stderr)

if __name__ == "__main__":
    main()
