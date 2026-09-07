"""Die Schritte der Reihe nach, mit Zwischenständen im Arbeitsordner.

Jeder Schritt legt sein Ergebnis ab und wird beim nächsten Lauf
übersprungen, wenn es schon da ist. Vor allem die Rekonstruktion lohnt das,
sie dauert je nach Zahl der Bilder eine halbe Stunde und länger.
"""
import json
import os
import time

import numpy as np

from . import cloudio, colorize as colorize_mod, georef, photos, register, sfm


class Abbruch(RuntimeError):
    pass


class Pipeline:
    """Ablauf für einen Datensatz.

    Der Ablauf ist absichtlich in einzelne Methoden zerlegt, damit die
    Oberfläche zwischen Ausrichtung und Einfärben anhalten und die Lage von
    Hand nachjustieren lassen kann.
    """

    def __init__(self, cloud_path, photo_dir, work_dir, out_path,
                 bag_topic=None, voxel=0.05, max_width=1600,
                 sparse_dir=None, python_exe=None,
                 log=None, cancel=None):
        self.cloud_path = cloud_path
        self.photo_dir = photo_dir
        self.work = work_dir
        self.out_path = out_path
        self.bag_topic = bag_topic
        self.voxel = voxel
        self.max_width = max_width
        self.sparse_dir = sparse_dir
        self.python_exe = python_exe
        self._log = log or (lambda s: None)
        self._cancel = cancel or (lambda: False)

        self.points = None
        self.cams = None
        self.gps = None
        self.model_enu = None      # gefilterte Fotopunkte in ENU
        self.enu = None            # (s, R, t, lat0, lon0, residuum)
        self.yaw = None
        self.t = None
        self.kennwerte = {}

        os.makedirs(self.work, exist_ok=True)

    # ------------------------------------------------------------- Helfer
    def log(self, msg):
        self._log(msg)

    def check(self):
        if self._cancel():
            raise Abbruch("abgebrochen")

    def _p(self, *parts):
        return os.path.join(self.work, *parts)

    # ------------------------------------------------------------ Schritte
    def load_cloud(self):
        self.check()
        cache = self._p("cloud.npy")
        if os.path.exists(cache):
            self.points = np.load(cache)
            self.log(f"Punktwolke aus dem Arbeitsordner: {len(self.points)} Punkte")
            return self.points
        t0 = time.time()
        self.log(f"Punktwolke lesen: {os.path.basename(self.cloud_path)}")
        P = cloudio.read_points(self.cloud_path, bag_topic=self.bag_topic,
                               voxel=self.voxel, progress=self.log)
        P = P[np.isfinite(P).all(1)]
        np.save(cache, P)
        self.points = P
        self.log(f"{len(P)} Punkte, Ausdehnung {(P.max(0) - P.min(0)).round(1)} m "
                 f"[{time.time() - t0:.1f} s]")
        return P

    def prepare_photos(self):
        self.check()
        img_dir = self._p("images")
        gps_path = self._p("gps.json")
        files = photos.scan_photos(self.photo_dir)
        if not files:
            raise RuntimeError("Keine Bilder im Flugordner gefunden")
        self.log(f"{len(files)} RGB-Bilder gefunden")
        photos.prepare_images(files, img_dir, self.max_width, progress=self.log)
        self.check()
        if os.path.exists(gps_path):
            self.gps = json.load(open(gps_path))
        else:
            self.gps = photos.read_geotags(files, progress=self.log)
            photos.write_gps(self.gps, gps_path)
        self.log(f"Geotags fuer {len(self.gps)} von {len(files)} Bildern")
        if len(self.gps) < 4:
            raise RuntimeError("Zu wenige Bilder mit Geotag, ohne die geht der Maszstab nicht")
        return img_dir

    def reconstruct(self):
        """COLMAP laufen lassen oder ein vorhandenes Modell nehmen."""
        self.check()
        npz = self._p("cameras.npz")
        if os.path.exists(npz):
            self.cams = np.load(npz, allow_pickle=True)
            self.log(f"COLMAP-Modell aus dem Arbeitsordner: {len(self.cams['names'])} Kameras")
            return self.cams
        py = self.python_exe or sfm.find_python()
        if not py:
            raise RuntimeError(
                "Kein Interpreter mit pycolmap gefunden. Entweder einen angeben "
                "oder ein fertiges COLMAP-Modell (sparse/0) auswaehlen.")
        sparse = self.sparse_dir
        if not sparse:
            sparse = self._p("sparse", "0")
            if not os.path.exists(os.path.join(sparse, "cameras.bin")):
                self.log("Rekonstruktion startet, das dauert. Bei 255 Bildern "
                         "etwa eine halbe Stunde.")
                sfm.run_sfm(py, self._p("images"), self.work,
                            progress=self.log, cancel=self._cancel)
        self.log("Kameras und Punkte exportieren")
        sfm.export_model(py, sparse, npz, progress=self.log, cancel=self._cancel)
        self.cams = np.load(npz, allow_pickle=True)
        return self.cams

    def georeference(self):
        self.check()
        names = self.cams["names"]
        C = self.cams["C"]
        s, R, t, lat0, lon0, res = georef.colmap_to_enu(names, C, self.gps)
        self.enu = (s, R, t, lat0, lon0, res)
        self.log(f"COLMAP nach ENU: Maszstab {s:.3f}, GPS-Residuum {res:.2f} m")
        if res > 5.0:
            self.log("Achtung: das Residuum ist gross. Ohne RTK sind die Geotags "
                     "ungenau, die Ausrichtung wird entsprechend weicher.")
        xyz = self.cams["xyz"]
        Cenu = (s * (R @ C.T)).T + t
        M = (s * (R @ xyz.T)).T + t
        self.model_enu = register.robust_subset(M, Cenu)
        self.log(f"{len(self.model_enu)} von {len(M)} Fotopunkten nach dem Ausreisserfilter")
        return self.enu

    def align(self):
        self.check()
        cache = self._p("align.json")
        if os.path.exists(cache):
            d = json.load(open(cache))
            self.yaw, self.t = d["yaw"], np.array(d["t"])
            self.kennwerte = d.get("kennwerte", {})
            self.log(f"Ausrichtung aus dem Arbeitsordner: {np.degrees(self.yaw):.2f} Grad")
            return self.yaw, self.t
        t0 = time.time()
        target = self.points
        if len(target) > 400000:
            step = len(target) // 400000 + 1
            target = target[::step]
        self.yaw, self.t, self.kennwerte = register.auto_align(
            self.model_enu, target, progress=self.log)
        anteil, med = register.quality(self.model_enu, target, self.yaw, self.t)
        self.kennwerte.update({"anteil_auf_flaeche": anteil, "median_abweichung": med})
        self.log(f"Ausrichtung fertig: {np.degrees(self.yaw):.2f} Grad, "
                 f"{anteil * 100:.1f} % der Fotopunkte liegen auf der Oberflaeche "
                 f"(Median {med:.2f} m) [{time.time() - t0:.1f} s]")
        self.save_align()
        return self.yaw, self.t

    def save_align(self):
        json.dump({"yaw": float(self.yaw), "t": list(map(float, self.t)),
                   "kennwerte": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                                 for k, v in self.kennwerte.items()}},
                  open(self._p("align.json"), "w"), indent=2)

    def affine(self):
        s, R, t, *_ = self.enu
        return register.affine(self.yaw, self.t, s, R, t)

    def colorize(self):
        self.check()
        A, b = self.affine()
        json.dump({"A": A.tolist(), "b": b.tolist()},
                  open(self._p("affine.json"), "w"), indent=2)
        rgb, hit = colorize_mod.colorize(self.points, self.cams, self._p("images"),
                                         A, b, progress=self.log, cancel=self._cancel)
        cloudio.write_ply_rgb(self.out_path, self.points, rgb)
        self.log(f"Fertig: {hit * 100:.1f} % eingefaerbt -> {self.out_path}")
        return hit

    # --------------------------------------------------------- Gesamtlauf
    def run(self, stop_before_color=False):
        self.load_cloud()
        self.prepare_photos()
        self.reconstruct()
        self.georeference()
        self.align()
        if stop_before_color:
            return None
        return self.colorize()
