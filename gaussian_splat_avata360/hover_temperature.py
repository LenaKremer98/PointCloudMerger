#!/usr/bin/env python3
"""Viewer mit LIVE-Temperatur an der Maus: beim Hovern ueber einen Punkt wird seine
Temperatur (°C) direkt am Cursor angezeigt. Open3D-GUI (DISPLAY noetig).

Maus: Linksklick-Ziehen = drehen, Rad = zoom, Mitte/Shift = verschieben.
"""
import os, numpy as np, open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial import cKDTree

ROOT=os.path.dirname(os.path.abspath(__file__))
d=np.load(os.path.join(ROOT,"output","temp_cloud.npz"))
xyz=d["xyz"].astype(np.float64); temp=d["temp"]; rgb=d["rgb"]
tree=cKDTree(xyz)
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(xyz)
pcd.colors=o3d.utility.Vector3dVector(np.clip(rgb,0,1))

class App:
    def __init__(self):
        gui.Application.instance.initialize()
        self.win=gui.Application.instance.create_window("Temperatur-Hover (°C an der Maus)",1400,900)
        w=self.win
        self.scene=gui.SceneWidget(); self.scene.scene=rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.12,0.12,0.12,1.0])
        mat=rendering.MaterialRecord(); mat.shader="defaultUnlit"; mat.point_size=3.0
        self.scene.scene.add_geometry("pc",pcd,mat)
        self.scene.scene.view.set_post_processing(False)
        bounds=pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(55.0,bounds,bounds.get_center())
        w.add_child(self.scene)
        # schwebendes Label am Cursor
        self.tip=gui.Label("")
        self.tip.background_color=gui.Color(0,0,0,0.7); self.tip.text_color=gui.Color(1,1,0.4,1)
        self.tip.visible=False
        w.add_child(self.tip)
        w.set_on_layout(self._layout)
        self.scene.set_on_mouse(self._mouse)
        self._tipxy=(0,0)
    def _layout(self,ctx):
        r=self.win.content_rect; self.scene.frame=r
        px,py=self._tipxy
        pref=self.tip.calc_preferred_size(ctx,gui.Widget.Constraints())
        self.tip.frame=gui.Rect(px+14,py+14,pref.width,pref.height)
    def _mouse(self,e):
        if e.type==gui.MouseEvent.Type.MOVE:
            x=int(e.x-self.scene.frame.x); y=int(e.y-self.scene.frame.y)
            W=self.scene.frame.width; H=self.scene.frame.height
            def depth_cb(dimg):
                dep=np.asarray(dimg)
                if y<0 or y>=dep.shape[0] or x<0 or x>=dep.shape[1]: return
                z=dep[y,x]
                if z>=1.0:
                    def hide(): self.tip.visible=False; self.win.set_needs_layout()
                    gui.Application.instance.post_to_main_thread(self.win,hide); return
                world=self.scene.scene.camera.unproject(x,y,z,W,H)
                wp=np.array([world[0],world[1],world[2]])
                _,idx=tree.query(wp)
                t=temp[idx]
                txt=f"  {t:.1f} °C  " if np.isfinite(t) else "  (kein Wert)  "
                def show():
                    self._tipxy=(int(e.x),int(e.y)); self.tip.text=txt
                    self.tip.visible=True; self.win.set_needs_layout()
                gui.Application.instance.post_to_main_thread(self.win,show)
            self.scene.scene.scene.render_to_depth_image(depth_cb)
        return gui.Widget.EventCallbackResult.IGNORED
    def run(self): gui.Application.instance.run()

App().run()
