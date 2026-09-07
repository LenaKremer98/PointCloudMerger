#!/usr/bin/env python3
"""COLMAP-SfM ueber pycolmap auf den perspektivischen Views.
Erzeugt sparse/0/{cameras,images,points3D}.bin -> Input fuer gsplat."""
import os, shutil
from pathlib import Path
import pycolmap

ROOT   = Path(__file__).parent
BASE   = Path(os.environ.get("SFM_DIR", ROOT))   # erlaubt eigenes Arbeitsverzeichnis
IMAGES = BASE / "images"
DB     = BASE / "colmap" / "database.db"
SPARSE = BASE / "sparse"          # gsplat-Parser erwartet <data>/sparse/0
DB.parent.mkdir(parents=True, exist_ok=True)
if SPARSE.exists():
    shutil.rmtree(SPARSE)
SPARSE.mkdir(parents=True, exist_ok=True)
if DB.exists():
    DB.unlink()

print("[1/3] Feature-Extraktion (SINGLE camera, gemeinsame Intrinsik) ...")
# alle Views teilen sich exakt dieselbe Pinhole-Intrinsik
try:
    pycolmap.extract_features(
        database_path=DB, image_path=IMAGES,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
except TypeError:
    pycolmap.extract_features(str(DB), str(IMAGES),
                              camera_mode=pycolmap.CameraMode.SINGLE)

print("[2/3] Exhaustive Matching ...")
pycolmap.match_exhaustive(DB)

print("[3/3] Incremental Mapping ...")
maps = pycolmap.incremental_mapping(DB, IMAGES, SPARSE)
if not maps:
    raise SystemExit("FEHLER: keine Rekonstruktion zustande gekommen.")
# groesstes Modell als 0 sicherstellen
best = max(maps.values(), key=lambda m: m.num_reg_images())
out0 = SPARSE / "0"
out0.mkdir(exist_ok=True)
best.write(out0)
print(f"FERTIG: {best.num_reg_images()}/{len(list(IMAGES.glob('*.jpg')))} Bilder registriert, "
      f"{best.num_points3D()} 3D-Punkte -> {out0}")
