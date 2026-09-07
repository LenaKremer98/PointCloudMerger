#!/usr/bin/env python3
"""Globale Feature-Registrierung (FPFH + RANSAC) Splat->LiDAR, danach ICP.
Anders als ICP nutzt das lokale Geometrie-Deskriptoren statt Naehe."""
import os, json, numpy as np, open3d as o3d, copy

ROOT  = os.path.dirname(os.path.abspath(__file__))
LIDAR = os.path.join(ROOT, "..", "PCD-DRZ_20-05-26", "merged_FinlaDRZ.pcd")
SPLAT = os.path.join(ROOT, "output", "avata360_splat_metric_rgb.ply")
VOXEL = 1.0

def read_pcd(path):
    with open(path,"rb") as f:
        while True:
            line=f.readline()
            if line.startswith(b"POINTS"): n=int(line.split()[1])
            if line.startswith(b"DATA"): break
        d=np.fromfile(f,dtype=np.float32,count=n*3).reshape(n,3).astype(float)
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(d); return p

def prep(pc, voxel):
    pc=pc.voxel_down_sample(voxel)
    pc,_=pc.remove_statistical_outlier(20,2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2,max_nn=30))
    f=o3d.pipelines.registration.compute_fpfh_feature(
        pc,o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=100))
    return pc,f

splat=o3d.io.read_point_cloud(SPLAT)
sxyz=np.asarray(splat.points)
lo,hi=np.percentile(sxyz,[1,99],axis=0)
splat=splat.select_by_index(np.where(np.all((sxyz>=lo)&(sxyz<=hi),1))[0])
lidar=read_pcd(LIDAR)

src,sf=prep(copy.deepcopy(splat),VOXEL)
tgt,tf=prep(copy.deepcopy(lidar),VOXEL)
print(f"FPFH auf Splat {len(src.points)} / LiDAR {len(tgt.points)} Punkten")

best=None
for trial in range(5):
    res=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,tgt,sf,tf,True,VOXEL*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL*1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(400000,1000))
    print(f"  RANSAC {trial}: fitness={res.fitness:.3f} rmse={res.inlier_rmse:.3f}")
    if best is None or res.fitness>best.fitness: best=res

print(f"Bestes RANSAC: fitness={best.fitness:.3f} rmse={best.inlier_rmse:.3f}")
icp=o3d.pipelines.registration.registration_icp(
    src,tgt,VOXEL,best.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(f"ICP-Refine: fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.3f}")

json.dump({"T_splat_to_lidar":icp.transformation.tolist(),
           "ransac_fitness":best.fitness,"icp_fitness":icp.fitness,"icp_rmse":icp.inlier_rmse},
          open(os.path.join(ROOT,"output","fpfh_result.json"),"w"),indent=2)

prev=copy.deepcopy(splat).voxel_down_sample(0.3).transform(icp.transformation)
pl=copy.deepcopy(lidar).voxel_down_sample(0.3); pl.paint_uniform_color([0.5,0.5,0.5])
o3d.io.write_point_cloud(os.path.join(ROOT,"output","preview_fpfh.ply"),prev+pl)
print("-> output/preview_fpfh.ply")
