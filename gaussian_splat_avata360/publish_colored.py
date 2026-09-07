#!/usr/bin/env python3
"""Published die colorierte Wolke als latched PointCloud2 fuer RViz.
Frame: map, Topic: /colored_cloud (transient_local -> RViz sieht sie auch verspaetet)."""
import sys, struct, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

PLY = sys.argv[1] if len(sys.argv)>1 else "../PCD-DRZ_20-05-26/merged_FinlaDRZ_m4t_manual2.ply"

def read_ply(path):
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        n = 0
        for line in header.decode().splitlines():
            if line.startswith("element vertex"): n = int(line.split()[-1])
        dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
                       ("red","u1"),("green","u1"),("blue","u1")])
        data = np.frombuffer(f.read(n*dt.itemsize), dtype=dt)
    return data

class Pub(Node):
    def __init__(self):
        super().__init__("colored_cloud_pub")
        d = read_ply(PLY)
        n = len(d)
        # gepacktes RGB als float32 (0x00RRGGBB)
        rgb_u32 = (d["red"].astype(np.uint32) << 16) | (d["green"].astype(np.uint32) << 8) | d["blue"].astype(np.uint32)
        arr = np.zeros(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4")])
        arr["x"], arr["y"], arr["z"] = d["x"], d["y"], d["z"]
        arr["rgb"] = rgb_u32.view(np.float32)
        self.buf = arr.tobytes()
        self.n = n
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(PointCloud2, "/colored_cloud", qos)
        self.timer = self.create_timer(2.0, self.tick)
        self.tick()
        self.get_logger().info(f"Published {n} Punkte auf /colored_cloud (frame=map)")

    def tick(self):
        h = Header(); h.frame_id = "map"
        h.stamp = self.get_clock().now().to_msg()
        msg = PointCloud2()
        msg.header = h; msg.height = 1; msg.width = self.n
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)]
        msg.is_bigendian = False; msg.point_step = 16
        msg.row_step = 16 * self.n; msg.is_dense = True
        msg.data = self.buf
        self.pub.publish(msg)

def main():
    rclpy.init(); n = Pub(); rclpy.spin(n)

if __name__ == "__main__":
    main()
