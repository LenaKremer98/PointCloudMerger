"""Draufsicht für die Kontrolle der Ausrichtung.

Grau ist die Zielwolke, orange sind die Fotopunkte nach der Ausrichtung.
Passt beides übereinander, stimmt die Lage.
"""
import numpy as np
from PIL import Image

from .georef import rot_z


def _bounds(points, rand=0.06):
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    dx, dy = x1 - x0, y1 - y0
    return x0 - dx * rand, y0 - dy * rand, x1 + dx * rand, y1 + dy * rand


def render(target, model_enu, yaw, t, size=(640, 480), box=None):
    """PIL-Bild der Draufsicht. `box` erlaubt einen festen Ausschnitt,
    damit das Bild beim Schieben der Regler nicht springt."""
    W, H = size
    x0, y0, x1, y1 = box if box else _bounds(target)
    sx = W / max(x1 - x0, 1e-6)
    sy = H / max(y1 - y0, 1e-6)
    s = min(sx, sy)
    ox = (W - (x1 - x0) * s) / 2
    oy = (H - (y1 - y0) * s) / 2

    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (18, 18, 22)

    def to_px(P):
        ix = ((P[:, 0] - x0) * s + ox).astype(np.int64)
        iy = (H - ((P[:, 1] - y0) * s + oy)).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        return ix[ok], iy[ok], ok

    # Zielwolke, nach Hoehe aufgehellt
    ix, iy, ok = to_px(target)
    z = target[ok, 2]
    if len(z):
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        g = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        v = (60 + g * 170).astype(np.uint8)
        o = np.argsort(z)
        img[iy[o], ix[o]] = np.stack([v[o], v[o], v[o]], 1)

    # Fotopunkte darueber, halbdurchsichtig, sonst deckt das Orange die
    # Wolke zu und man sieht gerade das nicht mehr, worauf es ankommt
    P = (rot_z(yaw) @ model_enu.T).T + np.asarray(t, float)
    if len(P) > 60000:
        P = P[::len(P) // 60000 + 1]
    ix, iy, ok = to_px(P)
    if len(ix):
        cur = img[iy, ix].astype(np.int16)
        mix = (cur * 0.45 + np.array([255, 150, 40]) * 0.55).astype(np.uint8)
        img[iy, ix] = mix

    return Image.fromarray(img), (x0, y0, x1, y1)
