#!/usr/bin/env python3
"""M4T-RGB-Bilder (_V.JPG) herunterskalieren + GPS (lat/lon/rel_alt) als JSON ablegen."""
import os, glob, json, subprocess
from PIL import Image

SRC  = "../m4t/DCIM/DJI_202605201521_001_Gebietsroute-erstellen1"
DST  = "m4t_work/images"
MAXW = 1600
os.makedirs(DST, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(SRC, "*_V.JPG")))
print(f"{len(imgs)} M4T-RGB-Bilder, skaliere auf max {MAXW}px ...")
for f in imgs:
    im = Image.open(f); w,h = im.size
    s = MAXW/max(w,h)
    im.resize((round(w*s), round(h*s)), Image.LANCZOS).save(
        os.path.join(DST, os.path.basename(f)), quality=95)

# GPS via exiftool (numerisch): lat, lon, RelativeAltitude
out = subprocess.run(["exiftool","-n","-T","-FileName","-GPSLatitude","-GPSLongitude",
                      "-RelativeAltitude", *imgs], capture_output=True, text=True).stdout
gps = {}
for line in out.strip().splitlines():
    p = line.split("\t")
    if len(p)>=4 and p[1] not in ("-",""):
        gps[p[0]] = [float(p[1]), float(p[2]), float(p[3])]
json.dump(gps, open("m4t_work/m4t_gps.json","w"))
print(f"GPS fuer {len(gps)} Bilder gespeichert -> m4t_work/m4t_gps.json")
print("Beispiel:", list(gps.items())[0])
