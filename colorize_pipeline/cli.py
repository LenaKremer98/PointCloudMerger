"""Dieselbe Pipeline ohne Oberfläche, für einen Lauf im Hintergrund.

    python3 -m colorize_pipeline.cli \
        --wolke  PCD-DRZ_20-05-26/merged_FinlaDRZ.pcd \
        --flug   .../DJI_202605201521_001_Gebietsroute-erstellen1 \
        --ziel   ergebnis.ply
"""
import argparse
import os
import sys
import time

from . import cloudio, sfm
from .pipeline import Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="colorize_pipeline",
        description="LiDAR-Punktwolke mit den Bildern eines Mäanderfluges einfärben")
    ap.add_argument("--wolke", required=True,
                    help="PCD, PLY oder ROS-2-Bag-Ordner")
    ap.add_argument("--thema", default=None,
                    help="PointCloud2-Thema, wenn die Wolke aus einem Bag kommt")
    ap.add_argument("--flug", required=True, help="Ordner mit den Bildern des Fluges")
    ap.add_argument("--ziel", required=True, help="Ausgabedatei (PLY)")
    ap.add_argument("--arbeit", default=None, help="Arbeitsordner für Zwischenstände")
    ap.add_argument("--breite", type=int, default=1600, help="Arbeitsbreite der Bilder")
    ap.add_argument("--voxel", type=float, default=0.05, help="Ausdünnung für Bagwolken (m)")
    ap.add_argument("--sparse", default=None, help="fertiges COLMAP-Modell (Ordner sparse/0)")
    ap.add_argument("--python", default=None, help="Interpreter mit pycolmap")
    ap.add_argument("--themen-zeigen", action="store_true",
                    help="nur die PointCloud2-Themen des Bags auflisten")
    a = ap.parse_args(argv)

    if a.themen_zeigen:
        for n, t, c in cloudio.list_bag_topics(a.wolke):
            print(f"{n:38s} {c:8d}  {t}")
        return 0

    work = a.arbeit or os.path.join(os.path.dirname(os.path.abspath(a.ziel)),
                                    "einfaerben_arbeit")
    t0 = time.time()
    p = Pipeline(cloud_path=a.wolke, photo_dir=a.flug, work_dir=work, out_path=a.ziel,
                 bag_topic=a.thema, voxel=a.voxel, max_width=a.breite,
                 sparse_dir=a.sparse, python_exe=a.python or sfm.find_python(),
                 log=lambda s: print(f"[{time.time() - t0:6.1f}s] {s}", flush=True))
    p.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
