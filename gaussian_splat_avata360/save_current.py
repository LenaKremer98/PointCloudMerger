#!/usr/bin/env python3
"""Liest die aktuelle TF map->splat (Greifer-Pose), kombiniert mit Skalierung (argv[1])
und schreibt manual_align.json mit T_splat_to_lidar (identisch zur Node-Logik):
  P_world = (s*R) P + (p - s*R*centroid)
centroid = Schwerpunkt des box-gecroppten metrischen Splats (wie im Node)."""
import os, sys, json, time, numpy as np, open3d as o3d, rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as Rot

ROOT=os.path.dirname(os.path.abspath(__file__))
SPLAT=os.path.join(ROOT,"output","avata360_splat_metric_rgb.ply")
OUTD=os.path.join(ROOT,"output")
scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0

# centroid wie im Node
S=np.asarray(o3d.io.read_point_cloud(SPLAT).points)
lo,hi=np.percentile(S,[1,99],axis=0)
centroid=S[np.all((S>=lo)&(S<=hi),1)].mean(0)

rclpy.init()
n=Node("save_current")
buf=Buffer(); TransformListener(buf,n)
tf=None
t0=time.time()
while time.time()-t0<5.0:
    rclpy.spin_once(n,timeout_sec=0.1)
    try:
        tf=buf.lookup_transform("map","splat",rclpy.time.Time())
        break
    except Exception:
        pass
if tf is None:
    print("FEHLER: keine TF map->splat"); sys.exit(1)
q=tf.transform.rotation; tr=tf.transform.translation
R=Rot.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
p=np.array([tr.x,tr.y,tr.z])
M=scale*R; t=p-M@centroid
T=np.eye(4); T[:3,:3]=M; T[:3,3]=t
json.dump({"T_splat_to_lidar":T.tolist(),"scale":scale,
           "pose":[p[0],p[1],p[2],q.x,q.y,q.z,q.w]},
          open(os.path.join(OUTD,"manual_align.json"),"w"),indent=2)
print(f"Pose: t={p.round(2)}  scale={scale}")
print(f"-> manual_align.json gespeichert")
rclpy.shutdown()
