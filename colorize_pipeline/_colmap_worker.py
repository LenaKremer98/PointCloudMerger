#!/usr/bin/env python3
"""Läuft in einem Interpreter mit pycolmap, aufgerufen als Unterprozess.

Das System-Python hat hier open3d und ROS, aber kein pycolmap, und im venv
mit pycolmap fehlt open3d. Deshalb die Trennung. Ausgaben gehen zeilenweise
nach stdout, die Oberfläche zeigt sie im Verlauf an.

  sfm    <bilder> <arbeitsordner>   -> <arbeitsordner>/sparse/0
  export <sparse/0> <ziel.npz>     -> Posen, Intrinsics, 3D-Punkte
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap


def sfm(image_dir, work_dir):
    image_dir = Path(image_dir)
    work = Path(work_dir)
    db = work / "colmap" / "database.db"
    sparse = work / "sparse"
    db.parent.mkdir(parents=True, exist_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    n = len(list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg")))
    print(f"[1/3] Merkmale aus {n} Bildern, eine gemeinsame Kamera", flush=True)
    try:
        pycolmap.extract_features(database_path=db, image_path=image_dir,
                                  camera_mode=pycolmap.CameraMode.SINGLE)
    except TypeError:
        pycolmap.extract_features(str(db), str(image_dir),
                                  camera_mode=pycolmap.CameraMode.SINGLE)

    print("[2/3] Bilder paarweise vergleichen", flush=True)
    if n > 400:
        # bei vielen Bildern ist der vollstaendige Vergleich zu teuer
        pycolmap.match_sequential(db)
    else:
        pycolmap.match_exhaustive(db)

    print("[3/3] Rekonstruktion", flush=True)
    maps = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not maps:
        print("FEHLER: keine Rekonstruktion zustande gekommen", flush=True)
        return 2
    best = max(maps.values(), key=lambda m: m.num_reg_images())
    out0 = sparse / "0"
    out0.mkdir(exist_ok=True)
    best.write(out0)
    print(f"FERTIG {best.num_reg_images()}/{n} Bilder registriert, "
          f"{best.num_points3D()} Punkte -> {out0}", flush=True)
    return 0


def export(sparse_dir, out_npz):
    rec = pycolmap.Reconstruction(sparse_dir)
    names, Rcw, tcw, C, intr = [], [], [], [], []
    for im in rec.images.values():
        M = np.asarray(im.cam_from_world().matrix(), float)
        R, t = M[:, :3], M[:, 3]
        cam = rec.cameras[im.camera_id]
        names.append(im.name)
        Rcw.append(R)
        tcw.append(t)
        C.append(-R.T @ t)
        intr.append([cam.width, cam.height, cam.model.name] + list(cam.params))
    pts = np.asarray([p.xyz for p in rec.points3D.values()], float)
    err = np.asarray([p.error for p in rec.points3D.values()], float)
    trk = np.asarray([p.track.length() for p in rec.points3D.values()], np.int32)
    model = intr[0][2] if intr else "?"
    params = np.asarray([r[3:] for r in intr], float)
    np.savez(out_npz, names=np.array(names), Rcw=np.array(Rcw), tcw=np.array(tcw),
             C=np.array(C), size=np.array([[r[0], r[1]] for r in intr], float),
             params=params, model=np.array(model), xyz=pts, err=err, track=trk)
    print(f"FERTIG {len(names)} Kameras, {len(pts)} Punkte, Modell {model} -> {out_npz}",
          flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sfm":
        sys.exit(sfm(sys.argv[2], sys.argv[3]))
    if cmd == "export":
        sys.exit(export(sys.argv[2], sys.argv[3]))
    print(f"unbekannter Befehl {cmd}")
    sys.exit(1)
