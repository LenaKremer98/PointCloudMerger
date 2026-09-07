#!/usr/bin/env python3
"""Feinjustage in RViz: farbige M4T-COLMAP-Punkte (beweglich) ueber grauem LiDAR (fix).

Startlage = aktuelle manuelle Ausrichtung (Kette colmap->enu->lidar). 6-DOF-Greifer
verschiebt/dreht das M4T-Modell; Rechtsklick->Speichern schreibt m4t_affine.json
(direkter colmap->lidar Affin A,b) und kann sofort neu colorieren.
"""
import os, json, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers import InteractiveMarkerServer, MenuHandler
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as Rot

ROOT=os.path.dirname(os.path.abspath(__file__)); R_EARTH=6378137.0
LIDAR=os.path.join(ROOT,"..","PCD-DRZ_20-05-26","merged_FinlaDRZ.pcd")
OUTD=os.path.join(ROOT,"output")

def read_pcd(p):
    with open(p,"rb") as f:
        while True:
            l=f.readline()
            if l.startswith(b"POINTS"): n=int(l.split()[1])
            if l.startswith(b"DATA"): break
        return np.fromfile(f,np.float32,n*3).reshape(n,3).astype(np.float64)
def to_enu(lat,lon,up,lat0,lon0):
    return np.array([np.radians(lon-lon0)*np.cos(np.radians(lat0))*R_EARTH,np.radians(lat-lat0)*R_EARTH,up])
def umeyama(src,dst):
    mu_s,mu_d=src.mean(0),dst.mean(0); Ss,Dd=src-mu_s,dst-mu_d
    C=Dd.T@Ss/len(src); U,D,Vt=np.linalg.svd(C); S=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: S[2,2]=-1
    R=U@S@Vt; s=np.trace(np.diag(D)@S)/((Ss**2).sum()/len(src)); return s,R,mu_d-s*R@mu_s

def cloud_msg(frame, xyz, rgb_u32, stamp):
    n=len(xyz); arr=np.zeros(n,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4")])
    arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]; arr["rgb"]=rgb_u32.view(np.float32)
    m=PointCloud2(); m.header=Header(frame_id=frame); m.header.stamp=stamp; m.height=1; m.width=n
    m.fields=[PointField(name=nm,offset=o,datatype=PointField.FLOAT32,count=1) for nm,o in (("x",0),("y",4),("z",8),("rgb",12))]
    m.is_bigendian=False; m.point_step=16; m.row_step=16*n; m.is_dense=True; m.data=arr.tobytes(); return m

class Align(Node):
    def __init__(self):
        super().__init__("m4t_align")
        # M4T-Punkte + Farbe
        d=np.load(os.path.join(OUTD,"m4t_points3d.npz")); Pc=d["xyz"]; rgb=d["rgb"].astype(np.uint32)
        # Kette colmap->enu->lidar (Startlage = aktuelle manuelle Ausrichtung)
        cam=np.load(os.path.join(OUTD,"m4t_cameras.npz"),allow_pickle=True)
        names=cam["names"]; Ccol=cam["C"]
        gps=json.load(open(os.path.join(ROOT,"m4t_work","m4t_gps.json")))
        geo=json.load(open(os.path.join(OUTD,"splat_georef.json"))); lat0,lon0=geo["lat0"],geo["lon0"]
        G=np.array([to_enu(gps[n][0],gps[n][1],gps[n][2],lat0,lon0) for n in names])
        s_m,R_m,t_m=umeyama(Ccol,G)
        Tman=np.array(json.load(open(os.path.join(OUTD,"manual_align.json")))["T_splat_to_lidar"])
        self.A0=Tman[:3,:3]@(s_m*R_m); self.b0=Tman[:3,:3]@t_m+Tman[:3,3]   # colmap->lidar (Start)
        Q=(self.A0@Pc.T).T+self.b0                                          # M4T-Punkte in LiDAR-Frame
        self.centroid=Q.mean(0); self.Q0=Q-self.centroid
        self.M_rgb=((rgb[:,0]<<16)|(rgb[:,1]<<8)|rgb[:,2])
        # LiDAR grau
        self.L=read_pcd(LIDAR); self.L_rgb=np.full(len(self.L),np.uint32(0x9a9a9a),dtype=np.uint32)

        qos=QoSProfile(depth=1); qos.durability=QoSDurabilityPolicy.TRANSIENT_LOCAL; qos.reliability=QoSReliabilityPolicy.RELIABLE
        self.pub_l=self.create_publisher(PointCloud2,"/lidar_cloud",qos)
        self.pub_m=self.create_publisher(PointCloud2,"/model_cloud",qos)
        self.br=TransformBroadcaster(self)
        self.pose=Pose(); self.pose.position.x,self.pose.position.y,self.pose.position.z=[float(c) for c in self.centroid]; self.pose.orientation.w=1.0
        self.server=InteractiveMarkerServer(self,"m4t_marker")
        self.menu=MenuHandler()
        self.menu.insert("Ausrichtung speichern", callback=self.save_cb)
        self.menu.insert("Speichern + sofort colorieren", callback=self.save_color_cb)
        self.make_marker()
        self.publish_lidar(); self.create_timer(5.0,self.publish_lidar)
        self.create_timer(0.08,self.tick)
        self.get_logger().info(f"Bereit. M4T-Modell {len(self.Q0)} Pkt. Greifer ziehen, Rechtsklick->Speichern.")

    def publish_lidar(self):
        self.pub_l.publish(cloud_msg("map",self.L,self.L_rgb,self.get_clock().now().to_msg()))
    def make_marker(self):
        im=InteractiveMarker(); im.header.frame_id="map"; im.name="m4t"; im.scale=8.0; im.pose=self.pose
        sph=Marker(); sph.type=Marker.SPHERE; sph.scale.x=sph.scale.y=sph.scale.z=2.0
        sph.color.r=1.0; sph.color.g=1.0; sph.color.b=0.2; sph.color.a=0.6
        c0=InteractiveMarkerControl(); c0.always_visible=True; c0.markers.append(sph); c0.interaction_mode=InteractiveMarkerControl.MENU
        im.controls.append(c0)
        for ax,(x,y,z) in [("x",(1,0,0)),("y",(0,0,1)),("z",(0,1,0))]:
            for mode,su in [(InteractiveMarkerControl.MOVE_AXIS,"m"),(InteractiveMarkerControl.ROTATE_AXIS,"r")]:
                c=InteractiveMarkerControl(); c.name=f"{su}{ax}"; c.orientation.w=1.0
                c.orientation.x=float(x); c.orientation.y=float(y); c.orientation.z=float(z); c.interaction_mode=mode
                im.controls.append(c)
        self.server.insert(im, feedback_callback=self.fb); self.menu.apply(self.server,"m4t"); self.server.applyChanges()
    def fb(self, fb): self.pose=fb.pose
    def tick(self):
        st=self.get_clock().now().to_msg()
        t=TransformStamped(); t.header.stamp=st; t.header.frame_id="map"; t.child_frame_id="model"
        t.transform.translation.x=self.pose.position.x; t.transform.translation.y=self.pose.position.y; t.transform.translation.z=self.pose.position.z
        t.transform.rotation=self.pose.orientation; self.br.sendTransform(t)
        self.pub_m.publish(cloud_msg("model",self.Q0,self.M_rgb,st))
    def current_affine(self):
        q=self.pose.orientation; Rmk=Rot.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        tmk=np.array([self.pose.position.x,self.pose.position.y,self.pose.position.z])
        # world = Rmk*(Q0)+tmk ; Q0=A0 p - (A0*?)... Q0 = (A0 p + b0) - centroid
        # => world = Rmk*A0 p + Rmk*(b0-centroid)+tmk
        A=Rmk@self.A0; b=Rmk@(self.b0-self.centroid)+tmk
        return A,b
    def save_cb(self, fb):
        A,b=self.current_affine()
        json.dump({"A":A.tolist(),"b":b.tolist()}, open(os.path.join(OUTD,"m4t_affine.json"),"w"),indent=2)
        self.get_logger().info("GESPEICHERT -> m4t_affine.json")
    def save_color_cb(self, fb):
        self.save_cb(fb)
        self.get_logger().info("Colorierung laeuft im Hintergrund ...")
        os.system(f"cd {ROOT} && python3 project_m4t_color.py --affine=output/m4t_affine.json "
                  f"--out=../PCD-DRZ_20-05-26/merged_FinlaDRZ_m4t_manual2.ply > output/proj2.log 2>&1 &")

def main():
    rclpy.init(); rclpy.spin(Align())
if __name__=="__main__": main()
