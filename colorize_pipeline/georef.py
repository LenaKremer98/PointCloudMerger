"""Den COLMAP-Rahmen über die RTK-Geotags metrisch machen.

COLMAP rekonstruiert nur bis auf eine Ähnlichkeit genau, der Maßstab ist
willkürlich. Die Kamerazentren haben aber über den Geotag eine bekannte Lage
im lokalen ENU-System. Eine Umeyama-Ähnlichkeit darauf liefert Maßstab,
Drehung und Verschiebung auf einen Schlag.
"""
import numpy as np

R_EARTH = 6378137.0


def to_enu(lat, lon, up, lat0, lon0):
    """Geografisch nach lokal Ost/Nord/Hoch in Metern.

    Die Hoehe wird direkt uebernommen, bei DJI ist das die relative Hoehe
    ueber dem Startpunkt. Fuer ein Gebiet von einigen hundert Metern reicht
    die ebene Naeherung.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    east = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * R_EARTH
    north = np.radians(lat - lat0) * R_EARTH
    return np.stack([east, north, np.asarray(up, float)], -1)


def umeyama(src, dst):
    """Aehnlichkeit src->dst, also dst ~= s*R@src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Ss, Dd = src - mu_s, dst - mu_d
    C = Dd.T @ Ss / len(src)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (Ss ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var
    return s, R, mu_d - s * R @ mu_s


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def colmap_to_enu(names, centers, gps):
    """Aehnlichkeit COLMAP -> ENU aus den Geotags der registrierten Bilder.

    Gibt (s, R, t, lat0, lon0, residuum_m).
    """
    have = [i for i, n in enumerate(names) if str(n) in gps]
    if len(have) < 4:
        raise ValueError(f"Nur {len(have)} Bilder mit Geotag, das reicht nicht")
    lat0 = float(np.mean([gps[str(names[i])][0] for i in have]))
    lon0 = float(np.mean([gps[str(names[i])][1] for i in have]))
    C = np.asarray([centers[i] for i in have], float)
    G = np.asarray([to_enu(*gps[str(names[i])], lat0, lon0) for i in have], float)
    s, R, t = umeyama(C, G)
    res = np.linalg.norm((s * (R @ C.T).T) + t - G, axis=1)
    return s, R, t, lat0, lon0, float(res.mean())
