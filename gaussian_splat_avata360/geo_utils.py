#!/usr/bin/env python3
"""Hilfsfunktionen: SRT-GPS-Parsing, ENU-Konversion, Umeyama-Similarity, Rot<->Quat."""
import re, numpy as np
R_EARTH = 6378137.0

def to_enu(lat, lon, up, lat0, lon0):
    """lat/lon (Grad) + up(=rel_alt, m) -> lokales ENU (m). up wird direkt uebernommen."""
    east  = np.radians(np.asarray(lon)-lon0)*np.cos(np.radians(lat0))*R_EARTH
    north = np.radians(np.asarray(lat)-lat0)*R_EARTH
    return np.stack([east, north, np.asarray(up,float)], -1)

def parse_srt(path):
    """-> dict: cue-Zeit(s) -> (lat,lon,rel_alt), als sortierte Arrays (t,lat,lon,ralt)."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    t,la,lo,ra = [],[],[],[]
    for b in re.split(r"\n\s*\n", txt):
        mt  = re.search(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->", b)
        mla = re.search(r"latitude:\s*([-\d.]+)", b)
        mlo = re.search(r"longitude:\s*([-\d.]+)", b)
        mra = re.search(r"rel_alt:\s*([-\d.]+)", b)
        if mt and mla and mlo and mra:
            h,m,s,ms = map(int, mt.groups())
            t.append(h*3600+m*60+s+ms/1000.0)
            la.append(float(mla.group(1))); lo.append(float(mlo.group(1))); ra.append(float(mra.group(1)))
    o = np.argsort(t)
    return np.array(t)[o], np.array(la)[o], np.array(lo)[o], np.array(ra)[o]

def umeyama(src, dst):
    """Similarity src->dst: dst ~= s*R@src + t. src,dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Ss, Dd = src-mu_s, dst-mu_d
    C = Dd.T @ Ss / len(src)
    U,D,Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: S[2,2] = -1
    R = U @ S @ Vt
    var = (Ss**2).sum()/len(src)
    s = np.trace(np.diag(D)@S)/var
    t = mu_d - s*R@mu_s
    return s, R, t

def mat2quat(R):
    """3x3 Rotationsmatrix -> Quaternion (qw,qx,qy,qz), COLMAP-Konvention."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2; qw=0.25*S
        qx=(R[2,1]-R[1,2])/S; qy=(R[0,2]-R[2,0])/S; qz=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; qw=(R[2,1]-R[1,2])/S
        qx=0.25*S; qy=(R[0,1]+R[1,0])/S; qz=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; qw=(R[0,2]-R[2,0])/S
        qx=(R[0,1]+R[1,0])/S; qy=0.25*S; qz=(R[1,2]+R[2,1])/S
    else:
        S=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; qw=(R[1,0]-R[0,1])/S
        qx=(R[0,2]+R[2,0])/S; qy=(R[1,2]+R[2,1])/S; qz=0.25*S
    q=np.array([qw,qx,qy,qz]); return q/np.linalg.norm(q)
