"""Jeden Punkt der Wolke in die Luftbilder zurückprojizieren und einfärben.

Es werden nicht zwei Wolken verschmolzen. Der Punkt wird über die
Ausrichtung in den COLMAP-Rahmen zurückgerechnet, dort in jede Kamera
projiziert, und die Farbe kommt aus dem Bild, in dem er am nächsten am
Bildmittelpunkt liegt. Bei einer Nadirbefliegung ist das die Kamera, welche
am steilsten auf ihn heruntersieht, und damit die mit der geringsten
Verzerrung und der geringsten Gefahr, hinter etwas zu greifen.
"""
import os

import numpy as np
from PIL import Image


def _project(P_cam, size, params, model):
    """Kamerakoordinaten nach Pixel. Deckt die COLMAP-Modelle ab, welche
    bei einem Kartierungsflug vorkommen."""
    W, H = size
    Z = P_cam[:, 2]
    front = Z > 1e-6
    Zs = np.where(front, Z, 1.0)
    x = P_cam[:, 0] / Zs
    y = P_cam[:, 1] / Zs
    m = str(model)
    if m in ("SIMPLE_PINHOLE",):
        f, cx, cy = params[:3]
        fx = fy = f
        d = 1.0
    elif m in ("PINHOLE",):
        fx, fy, cx, cy = params[:4]
        d = 1.0
    elif m in ("SIMPLE_RADIAL",):
        f, cx, cy, k = params[:4]
        fx = fy = f
        d = 1.0 + k * (x * x + y * y)
    elif m in ("RADIAL",):
        f, cx, cy, k1, k2 = params[:5]
        fx = fy = f
        r2 = x * x + y * y
        d = 1.0 + k1 * r2 + k2 * r2 * r2
    elif m in ("OPENCV",):
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x * x + y * y
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * d + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * d + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return fx * xd + cx, fy * yd + cy, front, np.sqrt(r2)
    else:
        raise ValueError(f"Kameramodell {m} wird nicht unterstuetzt")
    r2 = x * x + y * y
    return fx * x * d + cx, fy * y * d + cy, front, np.sqrt(r2)


def colorize(points, cams, image_dir, A, b, max_radius=0.0,
             fallback=(107, 107, 107), progress=None, cancel=None):
    """Punkte einfärben. `cams` ist das NPZ aus dem COLMAP-Export.

    Gibt (rgb uint8, anteil_getroffen).
    """
    names = cams["names"]
    Rcw = cams["Rcw"]
    tcw = cams["tcw"]
    size = cams["size"]
    params = cams["params"]
    model = cams["model"].item() if cams["model"].shape == () else str(cams["model"])

    N = len(points)
    Ainv = np.linalg.inv(A)
    P_col = (Ainv @ (points - b).T).T
    PT = P_col.T

    best_r = np.full(N, np.inf)
    col = np.zeros((N, 3), np.uint8)
    col[:] = fallback

    for i, n in enumerate(names):
        if cancel and cancel():
            raise RuntimeError("abgebrochen")
        pc = (Rcw[i] @ PT).T + tcw[i]
        u, v, front, rad = _project(pc, size[i], params[i], model)
        W, H = size[i]
        valid = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if max_radius > 0:
            valid &= rad < max_radius
        sel = valid & (rad < best_r)
        if sel.any():
            img = np.asarray(Image.open(os.path.join(image_dir, str(n))).convert("RGB"))
            ih, iw = img.shape[:2]
            ui = np.clip(u[sel].astype(int), 0, iw - 1)
            vi = np.clip(v[sel].astype(int), 0, ih - 1)
            col[sel] = img[vi, ui]
            best_r[sel] = rad[sel]
        if progress and (i % 20 == 0 or i == len(names) - 1):
            progress(f"Einfaerben: Kamera {i + 1}/{len(names)}, "
                     f"{np.isfinite(best_r).mean() * 100:.1f} % getroffen")

    return col, float(np.isfinite(best_r).mean())
