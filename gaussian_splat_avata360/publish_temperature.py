#!/usr/bin/env python3
"""Published die Temperatur-Wolke: x,y,z + rgb (Iron) + temperature (°C, float).
Im RViz mit dem 'Select'-Werkzeug einen Punkt anklicken -> Selection-Panel zeigt 'temperature'."""
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

NPZ="output/temp_cloud.npz"

class Pub(Node):
    def __init__(self):
        super().__init__("temperature_cloud_pub")
        d=np.load(NPZ); xyz=d["xyz"]; temp=d["temp"]; rgb=d["rgb"]; N=len(xyz)
        r=(np.clip(rgb[:,0],0,1)*255).astype(np.uint32)
        g=(np.clip(rgb[:,1],0,1)*255).astype(np.uint32)
        b=(np.clip(rgb[:,2],0,1)*255).astype(np.uint32)
        rgb_u32=(r<<16)|(g<<8)|b
        arr=np.zeros(N,dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4"),("temperature","<f4")])
        arr["x"],arr["y"],arr["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
        arr["rgb"]=rgb_u32.view(np.float32); arr["temperature"]=temp
        self.buf=arr.tobytes(); self.N=N
        qos=QoSProfile(depth=1); qos.durability=QoSDurabilityPolicy.TRANSIENT_LOCAL; qos.reliability=QoSReliabilityPolicy.RELIABLE
        self.pub=self.create_publisher(PointCloud2,"/temperature_cloud",qos)
        self.create_timer(2.0,self.tick); self.tick()
        valid=np.isfinite(temp)
        self.get_logger().info(f"Published {N} Punkte (/temperature_cloud), "
                               f"Temp {np.nanmin(temp):.1f}..{np.nanmax(temp):.1f} °C, {valid.mean()*100:.0f}% mit Wert")
    def tick(self):
        h=Header(frame_id="map"); h.stamp=self.get_clock().now().to_msg()
        m=PointCloud2(); m.header=h; m.height=1; m.width=self.N
        m.fields=[PointField(name=n,offset=o,datatype=PointField.FLOAT32,count=1)
                  for n,o in (("x",0),("y",4),("z",8),("rgb",12),("temperature",16))]
        m.is_bigendian=False; m.point_step=20; m.row_step=20*self.N; m.is_dense=False; m.data=self.buf
        self.pub.publish(m)

def main(): rclpy.init(); rclpy.spin(Pub())
if __name__=="__main__": main()
