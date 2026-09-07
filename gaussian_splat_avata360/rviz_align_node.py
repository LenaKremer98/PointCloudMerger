#!/usr/bin/env python3
"""Interaktive Ausrichtung in RViz.

- LiDAR (grau) latched auf /lidar_cloud (frame map).
- Splat (farbig) latched auf /splat_cloud (frame splat), Punkte um Schwerpunkt zentriert.
- 6-DOF Interactive-Marker steuert TF map->splat (Pfeile verschieben, Ringe drehen).
- Skalierung per Parameter 'scale' (rqt_reconfigure-Slider 0.3..3.0).
- Rechtsklick auf den Greifer -> Menue: 'Speichern' schreibt manual_align.json.

Start:  python3 rviz_align_node.py   (dann rviz2 -d align.rviz  und rqt_reconfigure)
"""
import os, json, struct, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers import InteractiveMarkerServer, MenuHandler
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as Rot

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
OUTD  = os.path.join(ROOT, "output")

def read_pcd_xyz(path):
    with open(path,"rb") as f:
        while True:
            line=f.readline()
            if line.startswith(b"POINTS"): n=int(line.split()[1])
            if line.startswith(b"DATA"): break
        return np.fromfile(f,dtype=np.float32,count=n*3).reshape(n,3).astype(np.float64)

def cloud_msg(frame, xyz, rgb_u32, stamp):
    n=len(xyz)
    arr=np.zeros(n,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4")])
    arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
    arr["rgb"]=rgb_u32.view(np.float32)
    m=PointCloud2(); m.header=Header(frame_id=frame); m.header.stamp=stamp
    m.height=1; m.width=n
    m.fields=[PointField(name=n_,offset=o,datatype=PointField.FLOAT32,count=1)
              for n_,o in (("x",0),("y",4),("z",8),("rgb",12))]
    m.is_bigendian=False; m.point_step=16; m.row_step=16*n; m.is_dense=True
    m.data=arr.tobytes(); return m

class Align(Node):
    def __init__(self):
        super().__init__("splat_align")
        # --- Daten ---
        self.L=read_pcd_xyz(LIDAR)
        import open3d as o3d
        sp=o3d.io.read_point_cloud(SPLAT)
        S=np.asarray(sp.points); C=np.asarray(sp.colors)
        lo,hi=np.percentile(S,[1,99],axis=0); m=np.all((S>=lo)&(S<=hi),1)
        S,C=S[m],C[m]
        C=np.clip(C*1.8,0,1)                        # aufhellen, damit gut sichtbar
        self.centroid=S.mean(0)
        # fuer fluessige Live-Anzeige ausgeduennt (Vollaufloesung nur fuer finales Colorieren)
        dsp=o3d.geometry.PointCloud()
        dsp.points=o3d.utility.Vector3dVector(S-self.centroid)
        dsp.colors=o3d.utility.Vector3dVector(C)
        dsp=dsp.voxel_down_sample(0.6)
        self.S0=np.asarray(dsp.points)
        Cd=np.asarray(dsp.colors)
        self.S_rgb=((Cd[:,0]*255).astype(np.uint32)<<16 |
                    (Cd[:,1]*255).astype(np.uint32)<<8 |
                    (Cd[:,2]*255).astype(np.uint32))
        g=np.uint32(0x9a9a9a)
        self.L_rgb=np.full(len(self.L), g, dtype=np.uint32)
        self.get_logger().info(f"Anzeige-Splat: {len(self.S0)} Punkte (ausgeduennt)")

        qos=QoSProfile(depth=1); qos.durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability=QoSReliabilityPolicy.RELIABLE
        self.pub_l=self.create_publisher(PointCloud2,"/lidar_cloud",qos)
        self.pub_s=self.create_publisher(PointCloud2,"/splat_cloud",qos)

        # --- Skalierung als Parameter mit Slider-Range ---
        desc=ParameterDescriptor(floating_point_range=[FloatingPointRange(from_value=0.3,to_value=3.0,step=0.01)])
        self.declare_parameter("scale",1.0,desc)
        self.scale=float(self.get_parameter("scale").value)
        self.add_on_set_parameters_callback(self.on_param)

        # --- TF + Marker ---
        self.br=TransformBroadcaster(self)
        self.pose=Pose()
        # Start NEBEN/UEBER der LiDAR-Karte (klar sichtbar), dann reinziehen:
        self.pose.position.x, self.pose.position.y, self.pose.position.z = -79.0, 110.0, 20.0
        self.pose.orientation.w=1.0
        self.server=InteractiveMarkerServer(self,"splat_marker")
        self.menu=MenuHandler()
        self.menu.insert("Ausrichtung speichern", callback=self.save_cb)
        self.menu.insert("Speichern + sofort colorieren", callback=self.save_color_cb)
        self.make_marker()

        self.publish_lidar()                 # statisch im map-Frame -> latched, selten auffrischen
        self.create_timer(5.0, self.publish_lidar)
        self.create_timer(0.08, self.tick)   # ~12 Hz: TF + Splat republizieren (folgt Greifer)
        self.get_logger().info("Bereit. RViz: align.rviz laden. Greifer ziehen, Rechtsklick->Speichern.")

    # ---- Parameter (Skalierung) ----
    def on_param(self, params):
        for p in params:
            if p.name=="scale":
                self.scale=float(p.value)   # naechster tick() republiziert das Splat skaliert
                self.get_logger().info(f"Skalierung = {self.scale:.3f}")
        return SetParametersResult(successful=True)

    # ---- Wolken publizieren ----
    def publish_lidar(self):
        self.pub_l.publish(cloud_msg("map", self.L, self.L_rgb, self.get_clock().now().to_msg()))

    def publish_splat(self, stamp):
        # aktueller Zeitstempel == TF-Stamp -> RViz transformiert mit der aktuellen Greifer-TF
        self.pub_s.publish(cloud_msg("splat", self.S0*self.scale, self.S_rgb, stamp))

    # ---- Interactive Marker mit 6-DOF ----
    def make_marker(self):
        im=InteractiveMarker(); im.header.frame_id="map"; im.name="splat"
        im.description="Splat ausrichten"; im.scale=8.0; im.pose=self.pose
        box=Marker(); box.type=Marker.SPHERE
        box.scale.x=box.scale.y=box.scale.z=2.0
        box.color.r=1.0; box.color.g=1.0; box.color.b=0.2; box.color.a=0.6
        ctrl=InteractiveMarkerControl(); ctrl.always_visible=True; ctrl.markers.append(box)
        ctrl.interaction_mode=InteractiveMarkerControl.MENU
        im.controls.append(ctrl)
        # 3x translate + 3x rotate
        for axis,(x,y,z) in [("x",(1,0,0)),("y",(0,0,1)),("z",(0,1,0))]:
            for mode,suf in [(InteractiveMarkerControl.MOVE_AXIS,"move"),
                             (InteractiveMarkerControl.ROTATE_AXIS,"rot")]:
                c=InteractiveMarkerControl(); c.name=f"{suf}_{axis}"
                c.orientation.w=1.0; c.orientation.x=float(x); c.orientation.y=float(y); c.orientation.z=float(z)
                c.interaction_mode=mode; im.controls.append(c)
        self.server.insert(im, feedback_callback=self.fb)
        self.menu.apply(self.server,"splat")
        self.server.applyChanges()

    def fb(self, fb):
        self.pose=fb.pose

    def tick(self):
        stamp=self.get_clock().now().to_msg()
        t=TransformStamped(); t.header.stamp=stamp
        t.header.frame_id="map"; t.child_frame_id="splat"
        t.transform.translation.x=self.pose.position.x
        t.transform.translation.y=self.pose.position.y
        t.transform.translation.z=self.pose.position.z
        t.transform.rotation=self.pose.orientation
        self.br.sendTransform(t)
        self.publish_splat(stamp)   # mit gleichem Stamp -> RViz zeichnet Splat an aktueller Greifer-Pose

    # ---- Speichern ----
    def current_T(self):
        q=self.pose.orientation; R=Rot.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        p=np.array([self.pose.position.x,self.pose.position.y,self.pose.position.z])
        s=self.scale
        # P_world = R*(s*(P-c)) + p = (s*R)P + (p - s*R*c)
        M=s*R; t=p - M@self.centroid
        T=np.eye(4); T[:3,:3]=M; T[:3,3]=t
        return T,s

    def save_cb(self, fb):
        T,s=self.current_T()
        json.dump({"T_splat_to_lidar":T.tolist(),"scale":s,
                   "pose":[self.pose.position.x,self.pose.position.y,self.pose.position.z,
                           self.pose.orientation.x,self.pose.orientation.y,
                           self.pose.orientation.z,self.pose.orientation.w]},
                  open(os.path.join(OUTD,"manual_align.json"),"w"),indent=2)
        self.get_logger().info(f"GESPEICHERT -> manual_align.json (scale={s:.3f}). "
                               "Jetzt: python3 colorize_from_transform.py")

    def save_color_cb(self, fb):
        self.save_cb(fb)
        self.get_logger().info("Colorierung startet im Hintergrund ...")
        os.system(f"cd {ROOT} && python3 colorize_from_transform.py > output/colorize.log 2>&1 &")

def main():
    rclpy.init(); rclpy.spin(Align())

if __name__=="__main__":
    main()
