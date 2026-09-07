#!/usr/bin/env python3
"""Georeferenziert Avata- und M4T-Rekonstruktion per GPS in ein gemeinsames ENU-Frame
und schreibt ein zusammengefuehrtes COLMAP-Textmodell + gemeinsamen images/-Ordner."""
import os, json, glob, shutil, numpy as np, pycolmap
from geo_utils import parse_srt, to_enu, umeyama, mat2quat

ROOT   = os.path.dirname(os.path.abspath(__file__))
AV_SP  = os.path.join(ROOT, "sparse", "0")            # Avata (1600er v4)
AV_IMG = os.path.join(ROOT, "images")
M_SP   = os.path.join(ROOT, "m4t_work", "sparse", "0")
M_IMG  = os.path.join(ROOT, "m4t_work", "images")
SRT    = os.path.join(ROOT, "..", "avatar360", "DCIM", "DJI_001", "DJI_20260520144939_0001_D.SRT")
M_GPS  = json.load(open(os.path.join(ROOT, "m4t_work", "m4t_gps.json")))
FUS    = os.path.join(ROOT, "fusion")
FPS    = 45/88.2882
os.makedirs(os.path.join(FUS, "sparse", "0"), exist_ok=True)
os.makedirs(os.path.join(FUS, "images"), exist_ok=True)

# gemeinsamer ENU-Origin = Mittel der (RTK-genauen) M4T-GPS
mlat = np.array([v[0] for v in M_GPS.values()]); mlon = np.array([v[1] for v in M_GPS.values()])
lat0, lon0 = mlat.mean(), mlon.mean()
print(f"ENU-Origin: lat0={lat0:.6f} lon0={lon0:.6f}")

def georef(rec, gps_of_image, label):
    C, G = [], []
    for im in rec.images.values():
        g = gps_of_image(im.name)
        if g is None: continue
        M = np.asarray(im.cam_from_world().matrix(), float)
        C.append(-M[:,:3].T @ M[:,3]); G.append(g)
    C, G = np.stack(C), np.stack(G)
    s, R, t = umeyama(C, G)
    from geo_utils import mat2quat
    q = mat2quat(R)
    rec.transform(pycolmap.Sim3d(float(s), pycolmap.Rotation3d(np.array([q[1],q[2],q[3],q[0]])), t.astype(float)))
    C2 = np.stack([-(np.asarray(im.cam_from_world().matrix(),float))[:,:3].T @
                    np.asarray(im.cam_from_world().matrix(),float)[:,3] for im in rec.images.values()])
    res = np.linalg.norm(C2-G, axis=1)
    print(f"  [{label}] s={s:.3f}, Residuum GPS: mean {res.mean():.2f} m, max {res.max():.2f} m, {len(C)} Kameras")
    return rec

# --- Avata GPS pro Bild (Frame->SRT) ---
st, sla, slo, sra = parse_srt(SRT)
def av_gps(name):
    fi = int(name[1:4]); j = np.argmin(np.abs(st - fi/FPS))
    return to_enu(sla[j], slo[j], sra[j], lat0, lon0)
# --- M4T GPS pro Bild (EXIF json) ---
def m_gps(name):
    v = M_GPS.get(name);  return None if v is None else to_enu(v[0], v[1], v[2], lat0, lon0)

print("Georeferenziere ...")
rec_a = georef(pycolmap.Reconstruction(AV_SP), av_gps, "Avata")
rec_m = georef(pycolmap.Reconstruction(M_SP),  m_gps,  "M4T")

# --- gemeinsames Textmodell schreiben ---
def cam_line(cid, cam):
    return f"{cid} {cam.model.name} {cam.width} {cam.height} " + " ".join(f"{p:.10g}" for p in cam.params)
def img_lines(iid, im, cam_id):
    M = np.asarray(im.cam_from_world().matrix(), float)   # 3x4 world->cam
    q = mat2quat(M[:,:3]); tt = M[:,3]                    # q = wxyz
    return f"{iid} {q[0]:.10g} {q[1]:.10g} {q[2]:.10g} {q[3]:.10g} {tt[0]:.10g} {tt[1]:.10g} {tt[2]:.10g} {cam_id} {im.name}\n\n"

cams, imgs, pts = [], [], []
IID_OFF, PID_OFF, CAM_A, CAM_M = 1000000, 1000000, 1, 2
# Avata: Kamera-ID -> CAM_A
for cid, cam in rec_a.cameras.items(): cams.append(cam_line(CAM_A, cam)); break
for im in rec_a.images.values(): imgs.append(img_lines(im.image_id, im, CAM_A))
for pid, p in rec_a.points3D.items():
    c = p.color; pts.append(f"{pid} {p.xyz[0]:.6f} {p.xyz[1]:.6f} {p.xyz[2]:.6f} {c[0]} {c[1]} {c[2]} {p.error:.3f}\n")
# M4T: Kamera-ID -> CAM_M, IDs offset
for cid, cam in rec_m.cameras.items(): cams.append(cam_line(CAM_M, cam)); break
for im in rec_m.images.values(): imgs.append(img_lines(im.image_id+IID_OFF, im, CAM_M))
for pid, p in rec_m.points3D.items():
    c = p.color; pts.append(f"{pid+PID_OFF} {p.xyz[0]:.6f} {p.xyz[1]:.6f} {p.xyz[2]:.6f} {c[0]} {c[1]} {c[2]} {p.error:.3f}\n")

sp0 = os.path.join(FUS, "sparse", "0")
open(os.path.join(sp0,"cameras.txt"),"w").write("\n".join(cams)+"\n")
open(os.path.join(sp0,"images.txt"),"w").write("".join(imgs))
open(os.path.join(sp0,"points3D.txt"),"w").write("".join(pts))

# --- images/ verlinken ---
n=0
for src in glob.glob(AV_IMG+"/*.jpg")+glob.glob(M_IMG+"/*.JPG")+glob.glob(M_IMG+"/*.jpg"):
    dst=os.path.join(FUS,"images",os.path.basename(src))
    if not os.path.exists(dst): os.symlink(os.path.abspath(src),dst); n+=1
print(f"Merged: {rec_a.num_images()}+{rec_m.num_images()} Bilder, "
      f"{rec_a.num_points3D()+rec_m.num_points3D()} Punkte -> {sp0}")
print(f"{n} Bilder verlinkt -> {FUS}/images")
# Kontrolle: zurueckladen
r=pycolmap.Reconstruction(sp0)
print(f"Verifikation: {r.num_cameras()} Kameras, {r.num_images()} Bilder, {r.num_points3D()} Punkte gelesen")
