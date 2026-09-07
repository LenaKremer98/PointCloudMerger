"""Den Ordner mit den Mäanderflugdaten auswerten.

Gesucht sind die RGB-Bilder eines Kartierungsfluges samt Geotag. Bei DJI
heissen sie `*_V.JPG`, daneben liegen die Thermalbilder `*_T.JPG`. Die Höhe
steht nicht im EXIF, sondern als `RelativeAltitude` im XMP-Block.
"""
import json
import os
import re
import shutil
import subprocess
import glob

from PIL import Image

_XMP_ALT = re.compile(rb"RelativeAltitude\s*=\s*\"?([+-]?[0-9.]+)")
_XMP_LAT = re.compile(rb"GpsLatitude\s*=\s*\"?([+-]?[0-9.]+)")
_XMP_LON = re.compile(rb"GpsLon[gG]itude\s*=\s*\"?([+-]?[0-9.]+)")


def scan_photos(folder):
    """Alle RGB-Bilder des Fluges, sortiert. Bevorzugt die _V-Reihe."""
    pats = ["*_V.JPG", "*_V.jpg"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, "**", p), recursive=True)
    if not files:
        for p in ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG"):
            files += glob.glob(os.path.join(folder, "**", p), recursive=True)
        # Thermalbilder heraushalten, die haben eine eigene Optik
        files = [f for f in files if not os.path.basename(f).upper().endswith("_T.JPG")]
    return sorted(set(files))


def _geotags_exiftool(paths):
    out = subprocess.run(
        ["exiftool", "-n", "-T", "-FileName", "-GPSLatitude", "-GPSLongitude",
         "-RelativeAltitude", *paths],
        capture_output=True, text=True).stdout
    gps = {}
    for line in out.strip().splitlines():
        p = line.split("\t")
        if len(p) >= 4 and p[1] not in ("-", "") and p[2] not in ("-", "") and p[3] not in ("-", ""):
            try:
                gps[p[0]] = [float(p[1]), float(p[2]), float(p[3])]
            except ValueError:
                pass
    return gps


def _exif_gps(path):
    """Rueckfallweg ohne exiftool: EXIF fuer lat/lon, XMP fuer die Hoehe."""
    lat = lon = alt = None
    try:
        im = Image.open(path)
        ex = im.getexif()
        gi = ex.get_ifd(0x8825) if hasattr(ex, "get_ifd") else None
        if gi:
            def dms(v):
                return float(v[0]) + float(v[1]) / 60.0 + float(v[2]) / 3600.0
            if 2 in gi and 4 in gi:
                lat = dms(gi[2]) * (-1 if str(gi.get(1, "N")).upper().startswith("S") else 1)
                lon = dms(gi[4]) * (-1 if str(gi.get(3, "E")).upper().startswith("W") else 1)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            head = f.read(256 * 1024)
        m = _XMP_ALT.search(head)
        if m:
            alt = float(m.group(1))
        if lat is None:
            ml, mo = _XMP_LAT.search(head), _XMP_LON.search(head)
            if ml and mo:
                lat, lon = float(ml.group(1)), float(mo.group(1))
    except Exception:
        pass
    if lat is None or lon is None or alt is None:
        return None
    return [lat, lon, alt]


def read_geotags(paths, progress=None):
    """{Dateiname: [lat, lon, rel_alt]} fuer alle Bilder mit Geotag."""
    gps = {}
    if shutil.which("exiftool"):
        # exiftool in Haeppchen, sonst wird die Kommandozeile zu lang
        for i in range(0, len(paths), 200):
            gps.update(_geotags_exiftool(paths[i:i + 200]))
            if progress:
                progress(f"Geotags: {len(gps)}/{len(paths)}")
    if len(gps) < len(paths):
        for i, p in enumerate(paths):
            n = os.path.basename(p)
            if n in gps:
                continue
            g = _exif_gps(p)
            if g:
                gps[n] = g
            if progress and i % 25 == 0:
                progress(f"Geotags (ohne exiftool): {len(gps)}/{len(paths)}")
    return gps


def prepare_images(paths, dst, max_width=1600, progress=None):
    """Bilder auf eine Arbeitsgroesse bringen. Die COLMAP-Intrinsics gelten
    genau fuer diese Groesse, aus ihnen wird spaeter gesampelt."""
    os.makedirs(dst, exist_ok=True)
    done = 0
    for i, f in enumerate(paths):
        out = os.path.join(dst, os.path.basename(f))
        if os.path.exists(out):
            done += 1
            continue
        im = Image.open(f)
        w, h = im.size
        s = max_width / max(w, h)
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        im.convert("RGB").save(out, quality=95)
        done += 1
        if progress and i % 10 == 0:
            progress(f"Bilder verkleinern: {i + 1}/{len(paths)}")
    return done


def write_gps(gps, path):
    with open(path, "w") as f:
        json.dump(gps, f, indent=1)


def summarize(folder):
    """Kurzer Ueberblick fuer die Oberflaeche."""
    imgs = scan_photos(folder)
    thermal = glob.glob(os.path.join(folder, "**", "*_T.JPG"), recursive=True)
    return {"bilder": len(imgs), "thermal": len(thermal),
            "beispiel": os.path.basename(imgs[0]) if imgs else ""}
