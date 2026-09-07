"""Die georeferenzierten Fotopunkte auf die LiDAR-Karte ausrichten.

Der Trick, der das automatisierbar macht: beide Rahmen sind lotrecht. Das
ENU-System ist es per Definition, und eine LiDAR-Karte aus einem
LiDAR-Inertial-SLAM ebenfalls, weil die IMU die Schwerkraft kennt. Beim
Datensatz dieses Repos ist die Restneigung zwischen beiden Rahmen exakt
null, es bleiben also nur vier Freiheitsgrade: eine Drehung um die
Hochachse und eine Verschiebung.

Zwei Stufen:

1. Grob über Draufsichten. Beide Wolken werden zu einer Höhenkarte
   gerastert, dann wird für jeden Kandidatenwinkel die beste Verschiebung
   per Kreuzkorrelation über die FFT gesucht. Das kostet eine FFT je Winkel
   und findet den Winkel zuverlässig.
2. Fein gegen das Höhenmodell des LiDAR. Die Fotopunkte einer
   Nadirbefliegung liegen auf genau der Oberfläche, welche das LiDAR von
   oben sieht. Statt Nachbarschaftssuche im Raum, die am Rauschen der
   Punktwolke hängen bleibt, wird der Höhenunterschied zur Rasterkarte
   minimiert, mit einem weichen robusten Kern.
"""
import numpy as np
from scipy.optimize import minimize

from .georef import rot_z


def robust_subset(points, camera_centers, z_pct=(1, 99), rand=40.0):
    """Die groben Ausreisser der Photogrammetrie wegwerfen.

    COLMAP setzt einzelne Punkte kilometerweit daneben. Behalten wird, was
    im Perzentilkasten liegt und nicht weiter als der beflogene Bereich
    plus Rand vom Flugmittelpunkt entfernt ist.
    """
    P = np.asarray(points, float)
    lo = np.percentile(P, z_pct[0], axis=0)
    hi = np.percentile(P, z_pct[1], axis=0)
    P = P[np.all((P >= lo) & (P <= hi), axis=1)]
    if camera_centers is not None and len(camera_centers):
        c = np.median(np.asarray(camera_centers, float)[:, :2], axis=0)
        r = np.max(np.linalg.norm(np.asarray(camera_centers, float)[:, :2] - c, axis=1)) + rand
        P = P[np.linalg.norm(P[:, :2] - c, axis=1) < r]
    return P


def _height_raster(P, cx, cy, res, n):
    """Draufsicht: je Zelle die groesste Hoehe, dann normiert."""
    ix = np.floor((P[:, 0] - cx) / res).astype(np.int64) + n // 2
    iy = np.floor((P[:, 1] - cy) / res).astype(np.int64) + n // 2
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    ix, iy, z = ix[ok], iy[ok], P[ok, 2]
    img = np.full((n, n), np.nan, np.float32)
    o = np.argsort(z)                      # aufsteigend, das Maximum bleibt stehen
    img[iy[o], ix[o]] = z[o]
    m = np.isfinite(img)
    out = np.zeros((n, n), np.float32)
    if m.any():
        v = img[m]
        out[m] = v - v.mean()
        sd = out[m].std()
        if sd > 1e-6:
            out[m] /= sd
    return out


def _dsm(P, res):
    """Hoehenmodell der Zielwolke: je Zelle die groesste Hoehe."""
    x0 = P[:, 0].min() - 2 * res
    y0 = P[:, 1].min() - 2 * res
    nx = int((P[:, 0].max() - x0) / res) + 4
    ny = int((P[:, 1].max() - y0) / res) + 4
    ix = ((P[:, 0] - x0) / res).astype(np.int64)
    iy = ((P[:, 1] - y0) / res).astype(np.int64)
    grid = np.full((ny, nx), np.nan, np.float32)
    o = np.argsort(P[:, 2])
    grid[iy[o], ix[o]] = P[o, 2]
    valid = np.isfinite(grid)
    return np.where(valid, grid, 0).astype(np.float32), valid, x0, y0, nx, ny


def coarse_yaw(model, target, res=0.5, n=512, step=2.0, progress=None):
    """Drehwinkel und Verschiebung in der Ebene über Draufsichten.

    Gibt (yaw_rad, tx, ty, kennwerte). `kennwerte` enthält die besten
    Winkel mit ihrem Wert, damit sich beurteilen lässt, wie eindeutig der
    Treffer war.
    """
    cT = np.array([target[:, 0].mean(), target[:, 1].mean()])
    cM = np.array([np.median(model[:, 0]), np.median(model[:, 1])])
    FT = np.fft.rfft2(_height_raster(target, cT[0], cT[1], res, n))
    d = model[:, :2] - cM
    Mr = np.empty_like(model)
    Mr[:, 2] = model[:, 2]

    def score(yaw_deg):
        a = np.radians(yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        Mr[:, 0] = ca * d[:, 0] - sa * d[:, 1]
        Mr[:, 1] = sa * d[:, 0] + ca * d[:, 1]
        r = np.fft.irfft2(FT * np.conj(np.fft.rfft2(_height_raster(Mr, 0.0, 0.0, res, n))), s=(n, n))
        k = int(np.argmax(r))
        py, px = divmod(k, n)
        if py > n // 2:
            py -= n
        if px > n // 2:
            px -= n
        return float(r.flat[k]), px, py

    grob = []
    for i, yd in enumerate(np.arange(0.0, 360.0, step)):
        grob.append((score(yd), yd))
        if progress and i % 20 == 0:
            progress(f"Ausrichtung grob: {yd:.0f} von 360 Grad")
    best = max(grob, key=lambda e: e[0][0])
    yaw_best = best[1]
    val, px, py = best[0]
    for yd in np.arange(yaw_best - step, yaw_best + step + 1e-9, 0.25):
        v, a, b = score(yd)
        if v > val:
            val, px, py, yaw_best = v, a, b, yd

    ranked = sorted(((e[0][0], e[1]) for e in grob), reverse=True)
    zweitbester = next((v for v, y in ranked if abs(((y - yaw_best + 180) % 360) - 180) > 10), 0.0)
    kennwerte = {"wert": val, "zweitbester": zweitbester,
                 "verhaeltnis": float(val / zweitbester) if zweitbester > 0 else float("inf"),
                 "beste_winkel": [(round(v, 2), y) for v, y in ranked[:5]]}

    yaw = np.radians(yaw_best)
    T = cT + np.array([px, py]) * res
    txy = T - rot_z(yaw)[:2, :2] @ cM
    return yaw, txy[0], txy[1], kennwerte


def fit_to_dsm(model, target, yaw, tx, ty, res=0.4, progress=None):
    """Feinschliff gegen das Höhenmodell, vier Freiheitsgrade.

    Der Kern exp(-(r/sigma)^2) zählt Punkte, die auf der Oberfläche liegen,
    und ignoriert Ausreisser, statt sie wie ein quadratischer Fehler nach
    vorne zu ziehen. Sigma wird in drei Stufen enger.
    """
    grid, valid, x0, y0, nx, ny = _dsm(target, res)

    def cost(p, sigma):
        a, ax, ay, az = p
        ca, sa = np.cos(a), np.sin(a)
        X = ca * model[:, 0] - sa * model[:, 1] + ax
        Y = sa * model[:, 0] + ca * model[:, 1] + ay
        jx = ((X - x0) / res).astype(np.int64)
        jy = ((Y - y0) / res).astype(np.int64)
        ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
        if ok.sum() < 500:
            return 0.0
        jx, jy = jx[ok], jy[ok]
        v = valid[jy, jx]
        if v.sum() < 500:
            return 0.0
        r = (model[ok, 2] + az)[v] - grid[jy[v], jx[v]]
        return -float(np.exp(-(r / sigma) ** 2).sum()) / len(model)

    Mr = (rot_z(yaw) @ model.T).T
    tz = float(np.median(target[:, 2]) - np.median(Mr[:, 2]))
    p = np.array([yaw, tx, ty, tz])
    for sigma, step in ((2.0, 1.0), (1.0, 0.4), (0.5, 0.15)):
        simplex = np.vstack([p,
                             p + [np.radians(2 * step), 0, 0, 0],
                             p + [0, step, 0, 0],
                             p + [0, 0, step, 0],
                             p + [0, 0, 0, step]])
        res_opt = minimize(cost, p, args=(sigma,), method="Nelder-Mead",
                           options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000,
                                        initial_simplex=simplex))
        p = res_opt.x
        if progress:
            progress(f"Ausrichtung fein (sigma {sigma} m): Winkel {np.degrees(p[0]):.2f} Grad, "
                     f"Verschiebung {p[1]:.2f} {p[2]:.2f} {p[3]:.2f}")
    inlier = -cost(p, 0.5)
    return float(p[0]), p[1:].astype(float), float(inlier)


def auto_align(model_enu, target, progress=None):
    """Beide Stufen nacheinander. Gibt yaw (rad), t (3,) und Kennzahlen."""
    yaw, tx, ty, k = coarse_yaw(model_enu, target, progress=progress)
    if progress:
        progress(f"Ausrichtung grob: {np.degrees(yaw):.2f} Grad, "
                 f"Eindeutigkeit {k['verhaeltnis']:.2f}")
    yaw, t, inlier = fit_to_dsm(model_enu, target, yaw, tx, ty, progress=progress)
    k["treffer"] = inlier
    return yaw, t, k


def affine(yaw, t, s, R, t_enu):
    """Die ganze Kette als ein Affin COLMAP -> Zielwolke: p_ziel = A p + b."""
    Rz = rot_z(yaw)
    A = Rz @ (s * R)
    b = Rz @ t_enu + np.asarray(t, float)
    return A, b


def quality(model_enu, target, yaw, t, res=0.4):
    """Anteil der Fotopunkte, welche nach der Ausrichtung auf dem
    Höhenmodell liegen, mit 0,5 m Toleranz. Grobe, aber ehrliche Kennzahl."""
    grid, valid, x0, y0, nx, ny = _dsm(target, res)
    P = (rot_z(yaw) @ model_enu.T).T + np.asarray(t, float)
    jx = ((P[:, 0] - x0) / res).astype(np.int64)
    jy = ((P[:, 1] - y0) / res).astype(np.int64)
    ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
    if ok.sum() == 0:
        return 0.0, 0.0
    jx, jy = jx[ok], jy[ok]
    v = valid[jy, jx]
    if v.sum() == 0:
        return 0.0, 0.0
    r = P[ok, 2][v] - grid[jy[v], jx[v]]
    return float((np.abs(r) < 0.5).mean()), float(np.median(np.abs(r)))
