#!/usr/bin/env python3
"""Korrekte Entzerrung: zirkuläres Fisheye -> rektilineare Perspektive (Equidistant-Modell)."""
import cv2, numpy as np, math

CX = CY = 1920.0       # Kreismitte (3840x3840)
R_CIRC = 1920.0        # Kreisradius
FOV_FISH = math.radians(190.0)   # Fisheye-Bildwinkel
F_FISH = R_CIRC / (FOV_FISH/2)   # equidistant: r = F_FISH * theta

def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def perspective_from_fisheye(fish, yaw, pitch, out=1280, hfov_deg=90.0):
    """pitch = Polarwinkel von der Linsenachse (+Z), yaw = Azimut UM die Achse.
    Rotation um Z haelt den Polarwinkel konstant -> alle Yaws gleich weit im Kreis."""
    fp = (out/2)/math.tan(math.radians(hfov_deg)/2)
    u,v = np.meshgrid(np.arange(out), np.arange(out))
    d = np.stack([(u-out/2)/fp, (v-out/2)/fp, np.ones_like(u,float)], -1)
    d /= np.linalg.norm(d,axis=-1,keepdims=True)
    Rm = rot_z(math.radians(yaw)) @ rot_x(math.radians(pitch))
    d = d @ Rm.T
    X,Y,Z = d[...,0],d[...,1],d[...,2]
    theta = np.arccos(np.clip(Z,-1,1))            # Winkel zur Achse
    phi   = np.arctan2(Y,X)
    r = F_FISH*theta
    mx = (CX + r*np.cos(phi)).astype(np.float32)
    my = (CY + r*np.sin(phi)).astype(np.float32)
    persp = cv2.remap(fish, mx, my, cv2.INTER_LINEAR, borderValue=(0,0,0))
    persp[theta > FOV_FISH/2] = 0                 # ausserhalb des Bildkreises
    return persp

if __name__ == "__main__":
    # Test: ein paar Ansichten erzeugen
    l0 = cv2.imread("/tmp/osv_lens0.jpg")   # Himmel-Linse
    l1 = cv2.imread("/tmp/osv_lens1.jpg")   # Boden-Linse
    cv2.imwrite("/tmp/persp_l1_axis.jpg", perspective_from_fisheye(l1, 0, 0))     # gerade runter
    cv2.imwrite("/tmp/persp_l1_horiz.jpg", perspective_from_fisheye(l1, 0, 60))   # Richtung Horizont
    cv2.imwrite("/tmp/persp_l0_horiz.jpg", perspective_from_fisheye(l0, 0, 65))   # Horizont von oben
    print("Testansichten erzeugt")
