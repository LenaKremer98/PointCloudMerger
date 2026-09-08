#!/usr/bin/env python3
"""
PointCloud Merge & Edit Tool
- Loads N pointclouds (PCD or ROS2 bag) into freely-addable slots
- Manual transform / ICP align between any two slots
- Edit modes: rubber-band box, polygon lasso, sphere brush, single pick, grid add
- Live highlighting of selection, undo/redo, save merged PCD
"""

from __future__ import annotations

import os
import sys
import struct
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from PyQt5 import QtCore, QtGui, QtWidgets

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# ---------------------------------------------------------------------------
# IO: PCD + ROS2 bag
# ---------------------------------------------------------------------------

def load_pcd(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points, dtype=np.float32)
    intens = None
    # Open3D doesn't keep custom fields; try reading intensity manually for ASCII/binary v0.7
    intens = _read_intensity_field(path)
    if intens is not None and intens.shape[0] != pts.shape[0]:
        intens = None
    return pts, intens


def _read_intensity_field(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, 'rb') as f:
            header = b''
            while b'DATA' not in header:
                line = f.readline()
                if not line:
                    return None
                header += line
            header_text = header.decode('ascii', errors='ignore')
            lines = header_text.splitlines()
            fields = sizes = types = counts = None
            data_kind = 'ascii'
            n_points = 0
            for ln in lines:
                if ln.startswith('FIELDS'):
                    fields = ln.split()[1:]
                elif ln.startswith('SIZE'):
                    sizes = [int(x) for x in ln.split()[1:]]
                elif ln.startswith('TYPE'):
                    types = ln.split()[1:]
                elif ln.startswith('COUNT'):
                    counts = [int(x) for x in ln.split()[1:]]
                elif ln.startswith('POINTS'):
                    n_points = int(ln.split()[1])
                elif ln.startswith('DATA'):
                    data_kind = ln.split()[1]
            if not fields or 'intensity' not in fields:
                return None
            idx = fields.index('intensity')
            if data_kind == 'binary':
                offset = sum(sizes[i] * counts[i] for i in range(idx))
                stride = sum(sizes[i] * counts[i] for i in range(len(sizes)))
                buf = f.read(stride * n_points)
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_points, stride)
                # assume float32 intensity
                size = sizes[idx]
                inten_bytes = arr[:, offset:offset + size].tobytes()
                if types[idx] == 'F' and size == 4:
                    return np.frombuffer(inten_bytes, dtype=np.float32).copy()
                if types[idx] == 'U' and size == 1:
                    return np.frombuffer(inten_bytes, dtype=np.uint8).astype(np.float32)
                return None
            elif data_kind == 'ascii':
                rest = f.read().decode('ascii', errors='ignore').strip().split()
                ncols = sum(counts)
                arr = np.array(rest, dtype=np.float32).reshape(-1, ncols)
                if arr.shape[0] != n_points:
                    return None
                col = sum(counts[:idx])
                return arr[:, col].copy()
    except Exception:
        return None
    return None


def save_pcd(path: str, points: np.ndarray, intensities: Optional[np.ndarray] = None):
    n = int(points.shape[0])
    has_i = intensities is not None and intensities.shape[0] == n
    fields = 'x y z' + (' intensity' if has_i else '')
    sizes = '4 4 4' + (' 4' if has_i else '')
    types = 'F F F' + (' F' if has_i else '')
    counts = '1 1 1' + (' 1' if has_i else '')
    with open(path, 'wb') as f:
        header = (
            f'# .PCD v0.7 - Point Cloud Data file format\n'
            f'VERSION 0.7\n'
            f'FIELDS {fields}\n'
            f'SIZE {sizes}\n'
            f'TYPE {types}\n'
            f'COUNT {counts}\n'
            f'WIDTH {n}\n'
            f'HEIGHT 1\n'
            f'VIEWPOINT 0 0 0 1 0 0 0\n'
            f'POINTS {n}\n'
            f'DATA binary\n'
        )
        f.write(header.encode('ascii'))
        if has_i:
            arr = np.empty((n, 4), dtype=np.float32)
            arr[:, :3] = points.astype(np.float32)
            arr[:, 3] = intensities.astype(np.float32)
        else:
            arr = points.astype(np.float32)
        f.write(arr.tobytes(order='C'))


# --- ROS2 bag reading -------------------------------------------------------

def list_pointcloud_topics_in_bag(bag_dir: str) -> list[str]:
    """Returns list of (topic_name) for sensor_msgs/msg/PointCloud2 topics."""
    try:
        import rosbag2_py  # type: ignore
        storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        topics = []
        for t in reader.get_all_topics_and_types():
            if t.type == 'sensor_msgs/msg/PointCloud2':
                topics.append(t.name)
        del reader
        return topics
    except Exception as e:
        print('list_pointcloud_topics_in_bag failed:', e)
        return []


def read_pointcloud_from_bag(bag_dir: str, topic: str,
                             mode: str = 'last',
                             max_messages: Optional[int] = None,
                             progress: Optional[Callable[[int, int], None]] = None
                             ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    mode:
      'last'        - return only the last message on the topic (already-accumulated map e.g. /Laser_map)
      'concat'      - concatenate ALL messages (raw scans). Voxel-downsampled to keep memory sane.
    """
    import rosbag2_py  # type: ignore
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    storage_filter = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(storage_filter)

    last_msg = None
    chunks_pts = []
    chunks_int = []
    count = 0
    while reader.has_next():
        topic_name, raw, t = reader.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(raw, PointCloud2)
        count += 1
        if mode == 'last':
            last_msg = msg
        else:
            pts, intens = pointcloud2_to_numpy(msg)
            if pts.shape[0] > 0:
                chunks_pts.append(pts)
                if intens is not None:
                    chunks_int.append(intens)
        if progress is not None and count % 50 == 0:
            progress(count, 0)
        if max_messages is not None and count >= max_messages:
            break
    del reader

    if mode == 'last':
        if last_msg is None:
            return np.zeros((0, 3), dtype=np.float32), None
        return pointcloud2_to_numpy(last_msg)
    else:
        if not chunks_pts:
            return np.zeros((0, 3), dtype=np.float32), None
        all_pts = np.concatenate(chunks_pts, axis=0)
        if len(chunks_int) == len(chunks_pts):
            all_i = np.concatenate(chunks_int, axis=0)
        else:
            all_i = None
        return all_pts, all_i


def pointcloud2_to_numpy(msg) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Decode sensor_msgs/PointCloud2 -> (Nx3 float32, intensity or None)."""
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ('x', 'y', 'z')):
        return np.zeros((0, 3), dtype=np.float32), None

    point_step = msg.point_step
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), None

    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    raw = raw[: n * point_step].reshape(n, point_step)

    def field_to_dtype(f):
        dtypes = {
            1: np.int8, 2: np.uint8,
            3: np.int16, 4: np.uint16,
            5: np.int32, 6: np.uint32,
            7: np.float32, 8: np.float64,
        }
        return dtypes.get(f.datatype, np.float32)

    def read_field(name: str):
        f = field_map[name]
        dt = np.dtype(field_to_dtype(f))
        # offset slice
        b = raw[:, f.offset:f.offset + dt.itemsize].copy()
        return np.frombuffer(b.tobytes(), dtype=dt)

    x = read_field('x').astype(np.float32, copy=False)
    y = read_field('y').astype(np.float32, copy=False)
    z = read_field('z').astype(np.float32, copy=False)
    pts = np.stack([x, y, z], axis=1)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    intens = None
    if 'intensity' in field_map:
        try:
            i = read_field('intensity').astype(np.float32, copy=False)
            intens = i[finite]
        except Exception:
            intens = None
    return pts, intens


# ---------------------------------------------------------------------------
# Editor data model + history
# ---------------------------------------------------------------------------

@dataclass
class CloudData:
    points: np.ndarray  # (N,3) float32, in WORLD frame after apply_transform
    intensities: Optional[np.ndarray] = None  # (N,) float32 or None
    name: str = ''

    def copy(self) -> 'CloudData':
        return CloudData(
            points=self.points.copy(),
            intensities=None if self.intensities is None else self.intensities.copy(),
            name=self.name,
        )


@dataclass
class History:
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)
    LIMIT: int = 30

    def push(self, snapshot: CloudData):
        self.undo.append(snapshot)
        if len(self.undo) > self.LIMIT:
            self.undo.pop(0)
        self.redo.clear()

    def can_undo(self): return bool(self.undo)
    def can_redo(self): return bool(self.redo)

    def do_undo(self, current: CloudData) -> Optional[CloudData]:
        if not self.undo:
            return None
        self.redo.append(current.copy())
        return self.undo.pop()

    def do_redo(self, current: CloudData) -> Optional[CloudData]:
        if not self.redo:
            return None
        self.undo.append(current.copy())
        return self.redo.pop()


# ---------------------------------------------------------------------------
# VTK viewer + custom interactor styles
# ---------------------------------------------------------------------------

class CloudActor:
    """Wraps a vtkActor for a numpy point cloud, with selection highlighting."""

    def __init__(self, base_color=(1.0, 1.0, 1.0)):
        self.base_color = base_color
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.intensities: Optional[np.ndarray] = None
        self.selection_mask: np.ndarray = np.zeros(0, dtype=bool)
        self.visibility_mask: np.ndarray = np.zeros(0, dtype=bool)

        self.poly = vtk.vtkPolyData()
        self.vtk_points = vtk.vtkPoints()
        self.poly.SetPoints(self.vtk_points)
        self.verts = vtk.vtkCellArray()
        self.poly.SetVerts(self.verts)

        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName('Colors')
        self.poly.GetPointData().SetScalars(self.colors)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.poly)
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.ScalarVisibilityOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(2.0)
        self.actor.GetProperty().SetRenderPointsAsSpheres(False)

    def set_points(self, points: np.ndarray, intensities: Optional[np.ndarray]):
        self.points = points.astype(np.float32, copy=False)
        self.intensities = intensities.astype(np.float32, copy=False) if intensities is not None else None
        n = len(self.points)
        self.selection_mask = np.zeros(n, dtype=bool)
        self.visibility_mask = np.ones(n, dtype=bool)
        self._rebuild_vtk()

    def add_points(self, new_pts: np.ndarray, new_intens: Optional[np.ndarray]):
        new_pts = new_pts.astype(np.float32, copy=False)
        m = len(new_pts)
        if m == 0:
            return
        if self.intensities is not None and new_intens is not None:
            self.intensities = np.concatenate([self.intensities,
                                               new_intens.astype(np.float32, copy=False)])
        elif self.intensities is None and new_intens is None:
            pass
        else:
            self.intensities = None  # mismatch — drop intensity
        self.points = np.concatenate([self.points, new_pts])
        self.visibility_mask = np.concatenate([self.visibility_mask, np.ones(m, dtype=bool)])
        self.selection_mask = np.concatenate([self.selection_mask, np.zeros(m, dtype=bool)])
        self._rebuild_vtk()

    def _visible_indices(self) -> np.ndarray:
        return np.where(self.visibility_mask)[0]

    def _rebuild_vtk(self):
        if len(self.points) == 0:
            self.vtk_points.SetNumberOfPoints(0)
            self.poly.SetVerts(vtk.vtkCellArray())
            self.colors.SetNumberOfTuples(0)
            self.poly.Modified()
            return
        vis_idx = self._visible_indices()
        n = len(vis_idx)
        if n == 0:
            self.vtk_points.SetNumberOfPoints(0)
            self.poly.SetVerts(vtk.vtkCellArray())
            self.colors.SetNumberOfTuples(0)
            self.poly.Modified()
            return
        visible = self.points[vis_idx]
        from vtkmodules.util import numpy_support as ns
        vtk_arr = ns.numpy_to_vtk(np.ascontiguousarray(visible), deep=True,
                                  array_type=vtk.VTK_FLOAT)
        self.vtk_points.SetData(vtk_arr)
        ids = np.empty((n, 2), dtype=np.int64)
        ids[:, 0] = 1
        ids[:, 1] = np.arange(n, dtype=np.int64)
        cell_arr = vtk.vtkCellArray()
        cell_arr.SetCells(n, ns.numpy_to_vtkIdTypeArray(ids.flatten(), deep=True))
        self.poly.SetVerts(cell_arr)
        self.verts = cell_arr
        self._update_colors()
        self.poly.Modified()

    def _update_colors(self):
        vis_idx = self._visible_indices()
        n = len(vis_idx)
        if n == 0:
            return
        if self.intensities is not None and len(self.intensities) == len(self.points):
            i = self.intensities[vis_idx]
            lo, hi = np.percentile(i, 2.0), np.percentile(i, 98.0)
            if hi <= lo:
                hi = lo + 1.0
            t = np.clip((i - lo) / (hi - lo), 0.0, 1.0)
            base = np.stack([0.2 + 0.8 * t,
                             0.4 + 0.6 * t,
                             0.6 + 0.4 * t], axis=1)
            base = base * np.array(self.base_color, dtype=np.float32)
        else:
            base = np.tile(np.array(self.base_color, dtype=np.float32), (n, 1))
        sel_visible = self.selection_mask[vis_idx]
        if sel_visible.any():
            base[sel_visible] = np.array([1.0, 0.15, 0.15], dtype=np.float32)
        rgb = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        from vtkmodules.util import numpy_support as ns
        vtk_arr = ns.numpy_to_vtk(np.ascontiguousarray(rgb), deep=True,
                                  array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_arr.SetName('Colors')
        self.poly.GetPointData().SetScalars(vtk_arr)
        self.colors = vtk_arr
        self.poly.Modified()

    def set_selection(self, mask: np.ndarray):
        if mask is None or mask.shape[0] != len(self.points):
            self.selection_mask = np.zeros(len(self.points), dtype=bool)
        else:
            # only visible points may be selected
            self.selection_mask = mask.astype(bool) & self.visibility_mask
        self._update_colors()

    def clear_selection(self):
        self.set_selection(np.zeros(len(self.points), dtype=bool))

    def delete_selected(self) -> int:
        if self.selection_mask is None or not self.selection_mask.any():
            return 0
        keep = ~self.selection_mask
        n_del = int(self.selection_mask.sum())
        self.points = self.points[keep]
        if self.intensities is not None:
            self.intensities = self.intensities[keep]
        self.visibility_mask = self.visibility_mask[keep]
        self.selection_mask = np.zeros(len(self.points), dtype=bool)
        self._rebuild_vtk()
        return n_del

    def set_visibility_z_range(self, zmin: float, zmax: float):
        if len(self.points) == 0:
            return
        m = (self.points[:, 2] >= zmin) & (self.points[:, 2] <= zmax)
        self.visibility_mask = m
        self.selection_mask = self.selection_mask & m
        self._rebuild_vtk()

    def show_all(self):
        if len(self.points) == 0:
            return
        self.visibility_mask = np.ones(len(self.points), dtype=bool)
        self._rebuild_vtk()

    def to_data(self, name: str = '') -> CloudData:
        return CloudData(self.points.copy(),
                         None if self.intensities is None else self.intensities.copy(),
                         name=name)

    def from_data(self, data: CloudData):
        self.set_points(data.points, data.intensities)

    def set_base_color(self, rgb: tuple):
        self.base_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        self._update_colors()

    def set_point_size(self, size: float):
        self.actor.GetProperty().SetPointSize(max(1.0, float(size)))

    def set_visible(self, b: bool):
        self.actor.SetVisibility(bool(b))


# --- Custom interactor styles ----------------------------------------------

class _BaseStyle(vtk.vtkInteractorStyleRubberBand3D):
    """A no-op style so we always have keyboard zoom+rotate available."""
    def __init__(self):
        super().__init__()


class TrackballStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()


class LiveBoxSelectStyle(vtk.vtkInteractorStyleRubberBand2D):
    """Rubber-band rectangle selection with LIVE highlight.
       Uses VTK's built-in rubber band drawing for visual feedback,
       plus extra observers for preview-during-drag and final-on-release.
    """
    def __init__(self, viewer: 'Viewer',
                 on_start: Callable[[], None],
                 on_preview: Callable[[int, int, int, int], None],
                 on_final: Callable[[int, int, int, int], None],
                 on_cancel: Callable[[], None]):
        super().__init__()
        self.viewer = viewer
        self._on_start = on_start
        self._on_preview = on_preview
        self._on_final = on_final
        self._on_cancel = on_cancel
        self._dragging = False
        self.AddObserver('LeftButtonPressEvent', self._lmb_down)
        self.AddObserver('LeftButtonReleaseEvent', self._lmb_up)
        self.AddObserver('MouseMoveEvent', self._move)
        self.AddObserver('SelectionChangedEvent', self._on_selection_done)
        self.AddObserver('KeyPressEvent', self._key)

    def _key(self, *_):
        k = self.GetInteractor().GetKeySym()
        if k == 'Escape':
            self._on_cancel()

    def _lmb_down(self, *_):
        self._dragging = True
        self._on_start()

    def _lmb_up(self, *_):
        self._dragging = False
        # actual selection mask is computed in _on_selection_done

    def _move(self, *_):
        if not self._dragging:
            return
        sp = self.GetStartPosition()
        ep = self.GetEndPosition()
        xmin, xmax = min(sp[0], ep[0]), max(sp[0], ep[0])
        ymin, ymax = min(sp[1], ep[1]), max(sp[1], ep[1])
        if xmax - xmin >= 2 and ymax - ymin >= 2:
            self._on_preview(xmin, ymin, xmax, ymax)

    def _on_selection_done(self, *_):
        sp = self.GetStartPosition()
        ep = self.GetEndPosition()
        xmin, xmax = min(sp[0], ep[0]), max(sp[0], ep[0])
        ymin, ymax = min(sp[1], ep[1]), max(sp[1], ep[1])
        if xmax - xmin >= 2 and ymax - ymin >= 2:
            self._on_final(xmin, ymin, xmax, ymax)


class PolygonLassoStyle(vtk.vtkInteractorStyleUser):
    """Left-click adds a polygon vertex; double-click or Enter closes; right-click cancels.
       on_preview(verts) is called whenever the polygon shape changes (live highlight)."""
    def __init__(self, viewer: 'Viewer',
                 on_polygon: Callable[[list[tuple[int, int]]], None],
                 on_preview: Callable[[list[tuple[int, int]]], None]):
        super().__init__()
        self.viewer = viewer
        self._on_polygon = on_polygon
        self._on_preview = on_preview
        self._verts: list[tuple[int, int]] = []
        self._line_actor: Optional[vtk.vtkActor2D] = None
        self.AddObserver('LeftButtonPressEvent', self._lmb)
        self.AddObserver('RightButtonPressEvent', self._rmb)
        self.AddObserver('MouseMoveEvent', self._move)
        self.AddObserver('KeyPressEvent', self._key)
        self.AddObserver('MouseWheelForwardEvent', self._wheel_fwd)
        self.AddObserver('MouseWheelBackwardEvent', self._wheel_bwd)

    def _wheel_fwd(self, *_):
        self.viewer.zoom(1.1)

    def _wheel_bwd(self, *_):
        self.viewer.zoom(1 / 1.1)

    def _lmb(self, *_):
        x, y = self.GetInteractor().GetEventPosition()
        # double click = finalize
        if len(self._verts) > 2:
            vx, vy = self._verts[0]
            if abs(vx - x) < 8 and abs(vy - y) < 8:
                self._finalize()
                return
        self._verts.append((x, y))
        self._draw_outline()
        if len(self._verts) >= 3:
            self._on_preview(list(self._verts))

    def _rmb(self, *_):
        self._cancel()

    def _move(self, *_):
        if not self._verts:
            return
        cur = self.GetInteractor().GetEventPosition()
        self._draw_outline(preview=cur)
        if len(self._verts) >= 2:
            self._on_preview(list(self._verts) + [cur])

    def _key(self, *_):
        k = self.GetInteractor().GetKeySym()
        if k in ('Return', 'KP_Enter'):
            self._finalize()
        elif k in ('Escape',):
            self._cancel()

    def _draw_outline(self, preview: Optional[tuple[int, int]] = None):
        self.viewer.draw_polygon_outline(self._verts, preview)

    def _finalize(self):
        if len(self._verts) < 3:
            return
        verts = list(self._verts)
        self._cleanup_outline()
        self._verts = []
        self._on_polygon(verts)

    def _cancel(self):
        self._cleanup_outline()
        self._verts = []

    def _cleanup_outline(self):
        self.viewer.clear_polygon_outline()


class SphereBrushStyle(vtk.vtkInteractorStyleUser):
    """Hover shows a 3D sphere on the picked point. Click-drag deletes points within the sphere."""
    def __init__(self, viewer: 'Viewer',
                 on_brush: Callable[[np.ndarray, float, bool], None],
                 on_radius_change: Callable[[float], None],
                 radius: float = 0.5):
        super().__init__()
        self.viewer = viewer
        self._on_brush = on_brush  # (center, radius, is_stroke_start)
        self._on_radius_change = on_radius_change
        self.radius = radius
        self._dragging = False
        self.AddObserver('MouseMoveEvent', self._move)
        self.AddObserver('LeftButtonPressEvent', self._lmb_down)
        self.AddObserver('LeftButtonReleaseEvent', self._lmb_up)
        self.AddObserver('MouseWheelForwardEvent', self._wheel_fwd)
        self.AddObserver('MouseWheelBackwardEvent', self._wheel_bwd)
        self.AddObserver('KeyPressEvent', self._key)

    def _wheel_fwd(self, *_):
        if self.GetInteractor().GetShiftKey():
            self.radius *= 1.2
            self._on_radius_change(self.radius)
            self._update_brush()
        else:
            self.viewer.zoom(1.1)

    def _wheel_bwd(self, *_):
        if self.GetInteractor().GetShiftKey():
            self.radius = max(0.01, self.radius / 1.2)
            self._on_radius_change(self.radius)
            self._update_brush()
        else:
            self.viewer.zoom(1 / 1.1)

    def _key(self, *_):
        k = self.GetInteractor().GetKeySym()
        if k == 'plus' or k == 'equal':
            self.radius *= 1.2; self._on_radius_change(self.radius); self._update_brush()
        elif k == 'minus':
            self.radius = max(0.01, self.radius / 1.2); self._on_radius_change(self.radius); self._update_brush()

    def set_radius(self, r: float):
        self.radius = max(0.001, float(r))
        self._update_brush()

    def _move(self, *_):
        self._update_brush()
        if self._dragging:
            center = self.viewer.last_brush_center
            if center is not None:
                self._on_brush(center, self.radius, False)

    def _lmb_down(self, *_):
        self._dragging = True
        self._update_brush()
        center = self.viewer.last_brush_center
        if center is not None:
            self._on_brush(center, self.radius, True)

    def _lmb_up(self, *_):
        self._dragging = False

    def _update_brush(self):
        x, y = self.GetInteractor().GetEventPosition()
        center = self.viewer.world_at_pixel(x, y)
        self.viewer.show_brush_sphere(center, self.radius)


class PointPickStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Click selects nearest visible point in the active cloud."""
    def __init__(self, on_pick: Callable[[int, int], None]):
        super().__init__()
        self._on_pick = on_pick
        self.AddObserver('LeftButtonPressEvent', self._lmb)

    def _lmb(self, obj, ev):
        x, y = self.GetInteractor().GetEventPosition()
        self._on_pick(x, y)
        # don't forward → no rotation on click
        # but allow drag rotation: only suppress if no movement (handled implicitly by caller switching mode)
        self.OnLeftButtonDown()


class PrismDrawStyle(vtk.vtkInteractorStyleUser):
    """Click adds a polygon vertex on the world plane Z=base_z (ray→plane).
       Enter / double-click on first vertex closes; Esc cancels.
       MouseMove fires preview callback so a rubber-band line can be drawn."""
    def __init__(self, viewer: 'Viewer', base_z: float,
                 on_vertex: Callable[[tuple], None],
                 on_preview: Callable[[Optional[tuple]], None],
                 on_finish: Callable[[], None],
                 on_cancel: Callable[[], None]):
        super().__init__()
        self.viewer = viewer
        self.base_z = float(base_z)
        self._on_vertex = on_vertex
        self._on_preview = on_preview
        self._on_finish = on_finish
        self._on_cancel = on_cancel
        self.AddObserver('LeftButtonPressEvent', self._lmb)
        self.AddObserver('KeyPressEvent', self._key)
        self.AddObserver('MouseMoveEvent', self._move)
        self.AddObserver('MouseWheelForwardEvent', self._wheel_fwd)
        self.AddObserver('MouseWheelBackwardEvent', self._wheel_bwd)

    def _wheel_fwd(self, *_): self.viewer.zoom(1.1)
    def _wheel_bwd(self, *_): self.viewer.zoom(1 / 1.1)

    def _world_at_screen(self, x: int, y: int) -> Optional[np.ndarray]:
        ren = self.viewer.renderer
        ren.SetDisplayPoint(x, y, 0.0)
        ren.DisplayToWorld()
        wn = ren.GetWorldPoint()
        if abs(wn[3]) > 1e-9:
            near = np.array([wn[0]/wn[3], wn[1]/wn[3], wn[2]/wn[3]])
        else:
            near = np.array(wn[:3])
        ren.SetDisplayPoint(x, y, 1.0)
        ren.DisplayToWorld()
        wf = ren.GetWorldPoint()
        if abs(wf[3]) > 1e-9:
            far = np.array([wf[0]/wf[3], wf[1]/wf[3], wf[2]/wf[3]])
        else:
            far = np.array(wf[:3])
        d = far - near
        if abs(d[2]) < 1e-9:
            return None
        t = (self.base_z - near[2]) / d[2]
        return near + t * d

    def _lmb(self, *_):
        x, y = self.GetInteractor().GetEventPosition()
        p = self._world_at_screen(x, y)
        if p is None:
            return
        self._on_vertex((float(p[0]), float(p[1])))

    def _move(self, *_):
        x, y = self.GetInteractor().GetEventPosition()
        p = self._world_at_screen(x, y)
        if p is None:
            self._on_preview(None)
        else:
            self._on_preview((float(p[0]), float(p[1])))

    def _key(self, *_):
        k = self.GetInteractor().GetKeySym()
        if k in ('Return', 'KP_Enter'):
            self._on_finish()
        elif k == 'Escape':
            self._on_cancel()


class GridPickStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Click on a point to anchor a grid (then app generates the grid)."""
    def __init__(self, on_pick: Callable[[int, int], None]):
        super().__init__()
        self._on_pick = on_pick
        self.AddObserver('LeftButtonPressEvent', self._lmb)

    def _lmb(self, obj, ev):
        x, y = self.GetInteractor().GetEventPosition()
        self._on_pick(x, y)
        self.OnLeftButtonDown()


class PairPickStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Picks the nearest point on a specific vtkActor and emits world coords.
       Used for manual point-pair alignment."""

    def __init__(self, viewer: 'Viewer',
                 on_pick: Callable[[np.ndarray], None],
                 on_cancel: Callable[[], None],
                 on_undo: Callable[[], None],
                 on_apply: Callable[[], None]):
        super().__init__()
        self.viewer = viewer
        self._on_pick = on_pick
        self._on_cancel = on_cancel
        self._on_undo = on_undo
        self._on_apply = on_apply
        self.allowed_actor: Optional[vtk.vtkActor] = None
        self.AddObserver('LeftButtonPressEvent', self._lmb)
        self.AddObserver('KeyPressEvent', self._key)

    def set_pick_actor(self, vtk_actor: vtk.vtkActor):
        self.allowed_actor = vtk_actor

    def _lmb(self, obj, ev):
        iren = self.GetInteractor()
        x, y = iren.GetEventPosition()
        ren = self.viewer.renderer
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
        if self.allowed_actor is not None:
            picker.GetPickList().RemoveAllItems()
            picker.AddPickList(self.allowed_actor)
            picker.PickFromListOn()
        ok = picker.Pick(x, y, 0, ren)
        if self.allowed_actor is not None:
            picker.PickFromListOff()
        if ok:
            wp = np.array(picker.GetPickPosition(), dtype=np.float32)
            self._on_pick(wp)
            return  # don't rotate on a successful pick
        # no point picked → fall through so user can rotate the view
        self.OnLeftButtonDown()

    def _key(self, obj, ev):
        key = self.GetInteractor().GetKeySym()
        if key in ('Escape',):
            self._on_cancel()
        elif key in ('BackSpace', 'Delete'):
            self._on_undo()
        elif key in ('Return', 'Enter', 'KP_Enter'):
            self._on_apply()


def _umeyama_rigid(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Rigid SVD alignment (no scaling). Returns 4x4 T with tgt ≈ T·src."""
    assert src.shape == tgt.shape and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    s = src - mu_s
    t = tgt - mu_t
    H = s.T @ t / float(src.shape[0])
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
    R = Vt.T @ D @ U.T
    trans = mu_t - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


# ---------------------------------------------------------------------------
# Vertical range slider (Bambu-style Z-clip)
# ---------------------------------------------------------------------------

class VerticalRangeSlider(QtWidgets.QWidget):
    """Two-handle vertical range slider.
       - Semi-transparent panel painted by the widget itself.
       - Top handle = upper Z bound, bottom handle = lower Z bound.
       - Emits rangeChanged(low_norm, high_norm) in [0,1].
       - rangeChangedFinal fires on mouse release.
    """
    rangeChanged = QtCore.pyqtSignal(float, float)
    rangeChangedFinal = QtCore.pyqtSignal(float, float)
    requestReset = QtCore.pyqtSignal()

    HANDLE_R = 9
    TRACK_W = 8
    PAD_TOP = 28      # space at top for reset button
    PAD_BOTTOM = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        # X11 + OpenGL child widgets can't reliably do real translucency.
        # We use an opaque dark panel matched to the viewer background and a
        # rounded mask to fake the floating-overlay look.
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(26, 28, 34))
        self.setPalette(pal)

        self._low = 0.0
        self._high = 1.0
        self._dragging: Optional[str] = None  # 'low' | 'high' | 'band' | None
        self._drag_offset = 0.0  # for band drag
        self._vmin = 0.0
        self._vmax = 1.0
        self._unit = 'm'

        self._reset_btn = QtWidgets.QPushButton('⤢', self)
        self._reset_btn.setToolTip('Reset Z-clip to full range')
        self._reset_btn.setFixedSize(22, 22)
        self._reset_btn.setStyleSheet(
            'QPushButton { background: rgba(40,40,46,200); color: white;'
            ' border-radius: 11px; border: 1px solid rgba(180,180,180,160); }'
            'QPushButton:hover { background: rgba(80,80,90,220); }'
        )
        self._reset_btn.clicked.connect(self._reset)

    # --- public API ---
    def low(self) -> float: return self._low
    def high(self) -> float: return self._high

    def set_value_range(self, vmin: float, vmax: float, unit: str = 'm'):
        self._vmin = float(vmin); self._vmax = float(vmax); self._unit = unit
        self.update()

    def set_normalized(self, low: float, high: float, *, emit=True):
        self._low = max(0.0, min(1.0, float(low)))
        self._high = max(self._low, min(1.0, float(high)))
        self.update()
        if emit:
            self.rangeChanged.emit(self._low, self._high)

    # --- geometry helpers ---
    def _track_rect(self) -> QtCore.QRect:
        w = self.width(); h = self.height()
        x = w // 2 - self.TRACK_W // 2
        y0 = self.PAD_TOP
        y1 = h - self.PAD_BOTTOM
        return QtCore.QRect(x, y0, self.TRACK_W, max(2, y1 - y0))

    def _y_to_norm(self, y: int) -> float:
        r = self._track_rect()
        v = 1.0 - (y - r.top()) / max(1, r.height())
        return max(0.0, min(1.0, v))

    def _norm_to_y(self, v: float) -> float:
        r = self._track_rect()
        return r.top() + (1.0 - v) * r.height()

    def _world(self, n: float) -> float:
        return self._vmin + n * (self._vmax - self._vmin)

    # --- events ---
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reset_btn.move((self.width() - self._reset_btn.width()) // 2, 4)
        # rounded mask so the (opaque) panel looks like a floating pill
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(0, 0, self.width(), self.height()), 14, 14)
        region = QtGui.QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # subtle highlighted border around the rounded panel
        panel = QtCore.QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(panel, 13, 13)

        # full track
        r = self._track_rect()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(80, 82, 90, 220))
        p.drawRoundedRect(QtCore.QRectF(r), 4, 4)

        # active band
        ylow = self._norm_to_y(self._low)
        yhigh = self._norm_to_y(self._high)
        active = QtCore.QRectF(r.left(), yhigh, r.width(), max(1.0, ylow - yhigh))
        p.setBrush(QtGui.QColor(255, 195, 80, 230))
        p.drawRoundedRect(active, 4, 4)

        # tick marks every 10%
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 70), 1))
        for i in range(11):
            ty = int(self._norm_to_y(i / 10.0))
            p.drawLine(r.left() - 4, ty, r.left() - 1, ty)
            p.drawLine(r.right() + 1, ty, r.right() + 4, ty)

        # handles
        cx = self.width() // 2
        hi_y = int(self._norm_to_y(self._high))
        lo_y = int(self._norm_to_y(self._low))
        p.setBrush(QtGui.QColor(80, 165, 255, 240))
        p.setPen(QtGui.QPen(QtGui.QColor(20, 22, 28, 250), 1))
        p.drawEllipse(QtCore.QPoint(cx, hi_y), self.HANDLE_R, self.HANDLE_R)
        p.setBrush(QtGui.QColor(245, 140, 55, 240))
        p.drawEllipse(QtCore.QPoint(cx, lo_y), self.HANDLE_R, self.HANDLE_R)

        # value labels
        p.setPen(QtGui.QColor(240, 240, 240, 250))
        f = p.font(); f.setPointSizeF(8.5); p.setFont(f)
        w_high = self._world(self._high)
        w_low = self._world(self._low)
        p.drawText(QtCore.QRect(0, hi_y - 22, self.width(), 14),
                   QtCore.Qt.AlignCenter, f'{w_high:.2f}{self._unit}')
        p.drawText(QtCore.QRect(0, lo_y + 8, self.width(), 14),
                   QtCore.Qt.AlignCenter, f'{w_low:.2f}{self._unit}')

    def _hit_handle(self, y: int) -> Optional[str]:
        ylow = self._norm_to_y(self._low)
        yhigh = self._norm_to_y(self._high)
        d_low = abs(y - ylow)
        d_high = abs(y - yhigh)
        if d_high < self.HANDLE_R + 4 and d_high <= d_low:
            return 'high'
        if d_low < self.HANDLE_R + 4:
            return 'low'
        # inside band → drag the whole band
        if min(ylow, yhigh) < y < max(ylow, yhigh):
            return 'band'
        return None

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        which = self._hit_handle(ev.y())
        if which is None:
            v = self._y_to_norm(ev.y())
            d_low = abs(v - self._low)
            d_high = abs(v - self._high)
            which = 'high' if d_high <= d_low else 'low'
            if which == 'low':
                self._low = min(v, self._high)
            else:
                self._high = max(v, self._low)
            self.update()
            self.rangeChanged.emit(self._low, self._high)
        elif which == 'band':
            v = self._y_to_norm(ev.y())
            self._drag_offset = v - 0.5 * (self._low + self._high)
        self._dragging = which

    def mouseMoveEvent(self, ev):
        if self._dragging is None:
            return
        v = self._y_to_norm(ev.y())
        if self._dragging == 'low':
            self._low = min(v, self._high)
        elif self._dragging == 'high':
            self._high = max(v, self._low)
        elif self._dragging == 'band':
            width = self._high - self._low
            center = max(width / 2, min(1.0 - width / 2, v - self._drag_offset))
            self._low = center - width / 2
            self._high = center + width / 2
        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, ev):
        if self._dragging is not None:
            self._dragging = None
            self.rangeChangedFinal.emit(self._low, self._high)

    def wheelEvent(self, ev):
        # wheel near a handle nudges it; in band area shifts the window
        delta = ev.angleDelta().y() / 1200.0
        which = self._hit_handle(ev.position().toPoint().y() if hasattr(ev, 'position') else ev.y())
        if which == 'low':
            self._low = max(0.0, min(self._high, self._low + delta))
        elif which == 'high':
            self._high = max(self._low, min(1.0, self._high + delta))
        else:
            width = self._high - self._low
            center = max(width / 2, min(1.0 - width / 2,
                                         0.5 * (self._low + self._high) + delta))
            self._low = center - width / 2
            self._high = center + width / 2
        self.update()
        self.rangeChanged.emit(self._low, self._high)
        self.rangeChangedFinal.emit(self._low, self._high)

    def mouseDoubleClickEvent(self, ev):
        self._reset()

    def _reset(self):
        self._low = 0.0; self._high = 1.0
        self.update()
        self.rangeChanged.emit(self._low, self._high)
        self.rangeChangedFinal.emit(self._low, self._high)
        self.requestReset.emit()


# ---------------------------------------------------------------------------
# Top toolbar (per-slot point color / size / visibility + grid toggle)
# ---------------------------------------------------------------------------

class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.pyqtSignal(QtGui.QColor)

    def __init__(self, initial_rgb=(1.0, 1.0, 1.0), parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 22)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._color = QtGui.QColor.fromRgbF(*initial_rgb)
        self._refresh_swatch()
        self.clicked.connect(self._open_dialog)

    def _refresh_swatch(self):
        c = self._color
        self.setStyleSheet(
            f'QPushButton {{ background-color: rgb({c.red()},{c.green()},{c.blue()});'
            f' border: 1px solid #444; border-radius: 4px; }}'
            f'QPushButton:hover {{ border: 1px solid #888; }}'
        )

    def color(self) -> QtGui.QColor:
        return self._color

    def set_color(self, c: QtGui.QColor):
        self._color = c
        self._refresh_swatch()

    def _open_dialog(self):
        c = QtWidgets.QColorDialog.getColor(self._color, self, 'Pick point color')
        if c.isValid():
            self._color = c
            self._refresh_swatch()
            self.colorChanged.emit(c)


class ViewerToolbar(QtWidgets.QFrame):
    """Slim toolbar above the 3D view: global view toggles (grid, show/hide all)."""

    showAllRequested = QtCore.pyqtSignal()
    hideAllRequested = QtCore.pyqtSignal()
    activeChanged    = QtCore.pyqtSignal(str)  # sid

    def __init__(self, viewer: 'Viewer', parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setStyleSheet(
            'ViewerToolbar { background-color: rgb(36, 38, 44); }'
            ' QLabel { color: rgb(220,220,220); }'
            ' QCheckBox { color: rgb(230,230,230); spacing: 4px; }'
            ' QPushButton { background: rgb(58,62,70); color: white; border: 1px solid #555;'
            '  padding: 2px 8px; border-radius: 3px; }'
            ' QPushButton:hover { background: rgb(72,78,90); }'
            ' QComboBox { background: rgb(48,50,58); color: white; border: 1px solid #555;'
            '  padding: 1px 6px; }'
            ' QFrame#sep { background: rgb(80,82,88); }'
        )
        self.setFixedHeight(36)
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(10, 4, 10, 4)
        h.setSpacing(8)

        h.addWidget(QtWidgets.QLabel('Active slot:'))
        self.active_combo = QtWidgets.QComboBox()
        self.active_combo.setMinimumWidth(150)
        self.active_combo.currentIndexChanged.connect(self._on_active_changed)
        h.addWidget(self.active_combo)

        sep = QtWidgets.QFrame(); sep.setObjectName('sep')
        sep.setFrameShape(QtWidgets.QFrame.VLine); sep.setFixedWidth(1)
        h.addWidget(sep)

        self.btn_show_all = QtWidgets.QPushButton('Show All')
        self.btn_show_all.clicked.connect(lambda: self.showAllRequested.emit())
        self.btn_hide_all = QtWidgets.QPushButton('Hide All')
        self.btn_hide_all.clicked.connect(lambda: self.hideAllRequested.emit())
        h.addWidget(self.btn_show_all)
        h.addWidget(self.btn_hide_all)

        h.addStretch(1)

        self.grid_chk = QtWidgets.QCheckBox('Grid')
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(self._on_toggle_grid)
        h.addWidget(self.grid_chk)

    def set_slot_choices(self, items: list[tuple[str, str]], active_sid: Optional[str]):
        """items: list of (sid, display_name). Re-populates the combo without firing signals."""
        self.active_combo.blockSignals(True)
        self.active_combo.clear()
        for sid, name in items:
            self.active_combo.addItem(name, sid)
        if active_sid is not None:
            for i in range(self.active_combo.count()):
                if self.active_combo.itemData(i) == active_sid:
                    self.active_combo.setCurrentIndex(i)
                    break
        self.active_combo.blockSignals(False)

    def _on_active_changed(self, idx: int):
        sid = self.active_combo.itemData(idx)
        if sid is not None:
            self.activeChanged.emit(sid)

    def _on_toggle_grid(self, b: bool):
        self.viewer.grid_actor.SetVisibility(bool(b))
        self.viewer.refresh()


# ---------------------------------------------------------------------------
# Viewer widget
# ---------------------------------------------------------------------------

class Viewer(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        # RViz-style gradient: lighter at top, darker at bottom
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.10, 0.11, 0.13)   # bottom
        self.renderer.SetBackground2(0.24, 0.26, 0.30)  # top
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # ground grid on XY plane (Z=0), like RViz
        self.grid_actor = _make_grid_actor(extent=200.0, step=1.0, major_step=10.0)
        self.renderer.AddActor(self.grid_actor)

        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.default_style = TrackballStyle()
        self.iren.SetInteractorStyle(self.default_style)

        # cloud actors — managed by MergeApp via add_cloud / remove_cloud (N slots)
        self.clouds: list[CloudActor] = []

        # axes
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2.0, 2.0, 2.0)
        self.orient_widget = vtk.vtkOrientationMarkerWidget()
        self.orient_widget.SetOrientationMarker(axes)
        self.orient_widget.SetInteractor(self.iren)
        self.orient_widget.SetViewport(0.0, 0.0, 0.18, 0.22)
        self.orient_widget.EnabledOn()
        self.orient_widget.InteractiveOff()

        # brush sphere preview
        self.brush_source = vtk.vtkSphereSource()
        self.brush_source.SetThetaResolution(24); self.brush_source.SetPhiResolution(24)
        self.brush_source.SetRadius(0.5)
        bm = vtk.vtkPolyDataMapper(); bm.SetInputConnection(self.brush_source.GetOutputPort())
        self.brush_actor = vtk.vtkActor(); self.brush_actor.SetMapper(bm)
        bp = self.brush_actor.GetProperty()
        bp.SetOpacity(0.20); bp.SetColor(1.0, 0.3, 0.3); bp.SetRepresentationToWireframe()
        bp.SetLineWidth(1.5)
        self.brush_actor.SetVisibility(False)
        self.renderer.AddActor(self.brush_actor)
        self.last_brush_center: Optional[np.ndarray] = None

        # 2D polygon overlay
        self._poly_actor: Optional[vtk.vtkActor2D] = None

        # cell picker (used to find world coord under cursor)
        self.world_picker = vtk.vtkWorldPointPicker()
        self.point_picker = vtk.vtkPointPicker()
        self.point_picker.SetTolerance(0.005)

        # top toolbar (per-slot color/size/visibility + grid toggle)
        self.toolbar = ViewerToolbar(self, parent=self)
        layout.insertWidget(0, self.toolbar)

        # vertical Z-clip slider overlay (right side)
        self.z_slider = VerticalRangeSlider(self)
        self.z_slider.setFixedWidth(64)
        self.z_slider.raise_()

        self.iren.Initialize()
        self.renderer.ResetCamera()
        QtCore.QTimer.singleShot(0, self._reposition_overlay)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_overlay()

    def _reposition_overlay(self):
        margin = 10
        sw = self.z_slider.width()
        toolbar_h = self.toolbar.height() if hasattr(self, 'toolbar') else 0
        avail_h = max(0, self.height() - toolbar_h - 2 * margin)
        sh = min(avail_h, max(220, int(avail_h * 0.95))) if avail_h > 220 else avail_h
        x = self.width() - sw - margin
        y = toolbar_h + (self.height() - toolbar_h - sh) // 2
        self.z_slider.setGeometry(x, y, sw, max(sh, 60))
        self.z_slider.raise_()

    # --- slot/cloud management ---
    def add_cloud(self, color: tuple, point_size: float = 2.0) -> 'CloudActor':
        c = CloudActor(base_color=color)
        c.set_point_size(point_size)
        self.renderer.AddActor(c.actor)
        self.clouds.append(c)
        return c

    def remove_cloud(self, cloud: 'CloudActor'):
        if cloud in self.clouds:
            self.renderer.RemoveActor(cloud.actor)
            self.clouds.remove(cloud)

    # --- camera helpers ---
    def reset_view(self):
        self.renderer.ResetCamera()
        self.refresh()

    def zoom(self, factor: float):
        cam = self.renderer.GetActiveCamera()
        cam.Zoom(factor)
        self.refresh()

    def refresh(self):
        self.vtk_widget.GetRenderWindow().Render()

    def set_style(self, style: vtk.vtkInteractorStyle):
        self.iren.SetInteractorStyle(style)

    def use_default_style(self):
        self.iren.SetInteractorStyle(self.default_style)

    # --- world / screen helpers ---
    def project_points_to_screen(self, world_pts: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx3 screen coords (px_x, px_y, depth_z (NDC))."""
        if world_pts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rw = self.vtk_widget.GetRenderWindow()
        w, h = rw.GetSize()
        cam = self.renderer.GetActiveCamera()
        view = np.array(cam.GetViewTransformMatrix().GetData()).reshape(4, 4)
        # GetCompositeProjectionTransformMatrix: combines model+view+proj
        comp = cam.GetCompositeProjectionTransformMatrix(
            self.renderer.GetTiledAspectRatio(), -1, 1)
        m = np.array(comp.GetData()).reshape(4, 4)
        n = world_pts.shape[0]
        homo = np.ones((n, 4), dtype=np.float64)
        homo[:, :3] = world_pts
        clip = homo @ m.T
        wcoord = clip[:, 3]
        wcoord_safe = np.where(np.abs(wcoord) < 1e-12, 1e-12, wcoord)
        ndc = clip[:, :3] / wcoord_safe[:, None]
        # NDC [-1, 1] -> screen pixels (origin bottom-left in VTK)
        sx = (ndc[:, 0] * 0.5 + 0.5) * w
        sy = (ndc[:, 1] * 0.5 + 0.5) * h
        sz = ndc[:, 2]  # depth NDC; behind camera = wcoord < 0
        # mark behind-camera points with NaN depth
        behind = wcoord < 0
        sz = np.where(behind, np.nan, sz)
        out = np.stack([sx, sy, sz], axis=1).astype(np.float32)
        return out

    def world_at_pixel(self, x: int, y: int) -> Optional[np.ndarray]:
        """Return world coordinate at pixel (x,y), prefer point pick, fallback to world picker on z buffer."""
        # Try picking nearest point first
        if self.point_picker.Pick(x, y, 0, self.renderer):
            wp = np.array(self.point_picker.GetPickPosition(), dtype=np.float32)
            self.last_brush_center = wp
            return wp
        # fallback world picker (uses depth buffer; falls back to focal plane)
        self.world_picker.Pick(x, y, 0, self.renderer)
        wp = np.array(self.world_picker.GetPickPosition(), dtype=np.float32)
        # if no actor under cursor, GetPickPosition is on focal plane — that's OK as cursor preview
        self.last_brush_center = wp
        return wp

    # --- brush sphere ---
    def show_brush_sphere(self, center: Optional[np.ndarray], radius: float):
        if center is None:
            self.brush_actor.SetVisibility(False)
        else:
            self.brush_source.SetCenter(float(center[0]), float(center[1]), float(center[2]))
            self.brush_source.SetRadius(float(radius))
            self.brush_actor.SetVisibility(True)
        self.refresh()

    def hide_brush_sphere(self):
        self.brush_actor.SetVisibility(False)
        self.refresh()

    # --- polygon outline (2D) ---
    def draw_polygon_outline(self, verts: list[tuple[int, int]],
                             preview: Optional[tuple[int, int]] = None):
        if self._poly_actor is not None:
            self.renderer.RemoveActor2D(self._poly_actor)
            self._poly_actor = None
        if not verts:
            self.refresh()
            return

        pts2d = list(verts)
        if preview is not None:
            pts2d = pts2d + [preview]

        points = vtk.vtkPoints()
        for (x, y) in pts2d:
            points.InsertNextPoint(x, y, 0.0)
        lines = vtk.vtkCellArray()
        for i in range(len(pts2d) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)

        pd = vtk.vtkPolyData()
        pd.SetPoints(points)
        pd.SetLines(lines)

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(pd)
        mapper.SetTransformCoordinate(coord)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.85, 0.2)
        actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor2D(actor)
        self._poly_actor = actor
        self.refresh()

    def clear_polygon_outline(self):
        if self._poly_actor is not None:
            self.renderer.RemoveActor2D(self._poly_actor)
            self._poly_actor = None
            self.refresh()

    def draw_rect_outline(self, xmin: int, ymin: int, xmax: int, ymax: int):
        verts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        # use polygon outline in closed mode by passing first vertex as preview
        self.draw_polygon_outline(verts, preview=(xmin, ymin))

    def clear_rect_outline(self):
        self.clear_polygon_outline()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

SLOT_PALETTE = [
    (0.95, 0.55, 0.20),  # orange
    (0.30, 0.65, 1.00),  # blue
    (0.40, 0.85, 0.45),  # green
    (1.00, 0.40, 0.60),  # pink
    (0.85, 0.75, 0.30),  # yellow
    (0.60, 0.45, 0.95),  # purple
    (0.95, 0.40, 0.40),  # red
    (0.35, 0.85, 0.85),  # cyan
    (0.95, 0.65, 0.50),  # peach
    (0.55, 0.75, 0.95),  # sky
]


@dataclass
class Slot:
    sid: str
    name: str
    actor: 'CloudActor'
    color: tuple
    history: 'History' = field(default_factory=lambda: History())
    source_label: str = '—'
    # widget refs (filled in when group is built)
    group: Optional[QtWidgets.QGroupBox] = None
    name_edit: Optional[QtWidgets.QLineEdit] = None
    label: Optional[QtWidgets.QLabel] = None
    visible_chk: Optional[QtWidgets.QCheckBox] = None
    color_btn: Optional['ColorButton'] = None
    size_spin: Optional[QtWidgets.QDoubleSpinBox] = None


class MergeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PointCloud Merge & Edit Tool')
        self.resize(1700, 1000)
        self.setMinimumSize(900, 600)

        self.status = self.statusBar()

        # central widget — QSplitter so user can drag side panel boundary
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter)

        # viewer
        self.viewer = Viewer(self)
        self.splitter.addWidget(self.viewer)

        # editor state (must be set before _build_side_panel)
        self.slots: dict[str, Slot] = {}
        self._sid_counter = 0
        self.active_sid: Optional[str] = None
        self.edit_mode = 'none'
        self._brush_radius = 0.3

        # side panel (scrollable)
        self._build_side_panel()
        self.splitter.addWidget(self._side_scroll)

        # default split: viewer ~70%, side panel ~30%
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1180, 380])

        # z-clip slider plumbing
        self._z_world_min = 0.0
        self._z_world_max = 1.0
        self._zfilter_timer = QtCore.QTimer(self)
        self._zfilter_timer.setSingleShot(True)
        self._zfilter_timer.setInterval(35)
        self._zfilter_timer.timeout.connect(self._apply_z_filter_now)
        self._pending_low = 0.0
        self._pending_high = 1.0
        self.viewer.z_slider.rangeChanged.connect(self._on_z_range_changed)
        self.viewer.z_slider.requestReset.connect(self._on_z_reset)

        # selection-preview throttle
        self._sel_preview_timer = QtCore.QTimer(self)
        self._sel_preview_timer.setSingleShot(True)
        self._sel_preview_timer.setInterval(25)
        self._sel_preview_timer.timeout.connect(self._do_sel_preview)
        self._sel_preview_kind: Optional[str] = None
        self._sel_preview_args = None
        self._proj_actor: Optional[CloudActor] = None
        self._proj_visible_idx: Optional[np.ndarray] = None
        self._proj_screen: Optional[np.ndarray] = None

        # 3D prism selection state
        self.prism = Prism3D(self.viewer)
        self.prism_state: str = 'idle'
        self._prism_draw_verts: list = []
        self._prism_draw_actor: Optional[vtk.vtkActor] = None

        # Manual pick-pair alignment state
        self._manual_active = False
        self._manual_src: Optional[Slot] = None
        self._manual_tgt: Optional[Slot] = None
        self._manual_phase: str = 'target'   # 'target' or 'source'
        self._manual_pairs: list = []        # list of [tgt_xyz, src_xyz]
        self._manual_marker_actors: list = []
        self._manual_style: Optional[PairPickStyle] = None

        # toolbar wiring
        self.viewer.toolbar.activeChanged.connect(self._on_active_slot_change)
        self.viewer.toolbar.showAllRequested.connect(self._on_show_all_slots)
        self.viewer.toolbar.hideAllRequested.connect(self._on_hide_all_slots)

        # start with two empty slots so the user can immediately load A/B
        self._add_slot()
        self._add_slot()

        self._setup_shortcuts()
        self._update_buttons()
        self._refresh_z_bounds()

    # ---------------------------------------------------------------- helpers
    def _next_sid(self) -> str:
        self._sid_counter += 1
        return f'S{self._sid_counter}'

    def _active_slot(self) -> Optional[Slot]:
        if self.active_sid is None:
            return None
        return self.slots.get(self.active_sid)

    def _active_actor_get(self) -> Optional['CloudActor']:
        s = self._active_slot()
        return s.actor if s is not None else None

    def _slot_items(self) -> list[tuple[str, str]]:
        return [(s.sid, s.name) for s in self.slots.values()]

    def _slot_display(self, sid: str) -> str:
        s = self.slots.get(sid)
        return s.name if s else sid

    def _push_history(self, sid: Optional[str] = None):
        if sid is None:
            sid = self.active_sid
        if sid is None or sid not in self.slots:
            return
        slot = self.slots[sid]
        slot.history.push(slot.actor.to_data(name=slot.name))
        self._update_buttons()

    def _update_buttons(self):
        slot = self._active_slot()
        if slot is None:
            self.btn_undo.setEnabled(False)
            self.btn_redo.setEnabled(False)
        else:
            self.btn_undo.setEnabled(slot.history.can_undo())
            self.btn_redo.setEnabled(slot.history.can_redo())
        for k, btn in self.mode_buttons.items():
            btn.setChecked(k == self.edit_mode)

    # ---------------------------------------------------------------- side panel
    def _build_side_panel(self):
        # scroll area host
        self._side_scroll = QtWidgets.QScrollArea()
        self._side_scroll.setWidgetResizable(True)
        self._side_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._side_scroll.setMinimumWidth(340)

        side = QtWidgets.QWidget()
        self._side_scroll.setWidget(side)
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # -- big prominent save button ---------------------------------------
        self.btn_save_merged_big = QtWidgets.QPushButton('💾  SAVE MERGED PCD')
        self.btn_save_merged_big.setMinimumHeight(46)
        self.btn_save_merged_big.setStyleSheet(
            'QPushButton {'
            ' background: qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            '   stop:0 #2ea04a, stop:1 #1a7a35);'
            ' color: white; font-size: 15px; font-weight: bold;'
            ' border: 2px solid #144d22; border-radius: 6px;'
            ' padding: 6px;'
            '}'
            'QPushButton:hover { background: #34b755; }'
            'QPushButton:pressed { background: #1a7a35; }'
            'QPushButton:disabled { background: #444; color: #888; border-color: #333; }'
        )
        self.btn_save_merged_big.setToolTip(
            'Combines all visible slots into a single .pcd file.')
        self.btn_save_merged_big.clicked.connect(self._on_save_merged)
        v.addWidget(self.btn_save_merged_big)

        # -- slots box -------------------------------------------------------
        slots_box = QtWidgets.QGroupBox('Point cloud slots')
        slots_box.setStyleSheet('QGroupBox { font-weight: bold; }')
        sl = QtWidgets.QVBoxLayout(slots_box)
        self._slots_container = QtWidgets.QWidget()
        self._slots_layout = QtWidgets.QVBoxLayout(self._slots_container)
        self._slots_layout.setContentsMargins(0, 0, 0, 0)
        self._slots_layout.setSpacing(6)
        sl.addWidget(self._slots_container)

        add_row = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton('＋ Add Slot')
        btn_add.setStyleSheet('font-weight: bold; padding: 4px 10px;')
        btn_add.clicked.connect(self._on_add_slot_click)
        add_row.addWidget(btn_add)
        add_row.addStretch(1)
        sl.addLayout(add_row)
        v.addWidget(slots_box)

        # -- align (transform / ICP / auto-align) ---------------------------
        tbox = QtWidgets.QGroupBox('Align — Source → Target')
        tl = QtWidgets.QGridLayout(tbox)
        tl.addWidget(QtWidgets.QLabel('Source slot:'), 0, 0)
        self.align_source_combo = QtWidgets.QComboBox()
        self.align_source_combo.currentIndexChanged.connect(lambda _i: None)
        tl.addWidget(self.align_source_combo, 0, 1)
        tl.addWidget(QtWidgets.QLabel('Target slot:'), 1, 0)
        self.align_target_combo = QtWidgets.QComboBox()
        tl.addWidget(self.align_target_combo, 1, 1)

        self._t_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        labels = [('tx', 'X (m)'), ('ty', 'Y (m)'), ('tz', 'Z (m)'),
                  ('rx', 'Roll (°)'), ('ry', 'Pitch (°)'), ('rz', 'Yaw (°)')]
        for i, (k, lbl) in enumerate(labels):
            tl.addWidget(QtWidgets.QLabel(lbl), 2 + i, 0)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e6, 1e6); sp.setDecimals(4)
            sp.setSingleStep(0.05 if k.startswith('t') else 1.0)
            tl.addWidget(sp, 2 + i, 1)
            self._t_spins[k] = sp
        row = 2 + len(labels)
        btn_apply_t = QtWidgets.QPushButton('Apply manual transform to Source')
        btn_apply_t.clicked.connect(self._on_apply_transform)
        tl.addWidget(btn_apply_t, row, 0, 1, 2); row += 1

        btn_auto = QtWidgets.QPushButton('★ Auto-Align  (Global + ICP)')
        btn_auto.setStyleSheet('font-weight: bold; padding: 6px;')
        btn_auto.clicked.connect(self._on_auto_align)
        tl.addWidget(btn_auto, row, 0, 1, 2); row += 1

        btn_icp = QtWidgets.QPushButton('ICP-only refine')
        btn_icp.clicked.connect(self._on_icp)
        tl.addWidget(btn_icp, row, 0, 1, 2); row += 1

        btn_manual = QtWidgets.QPushButton('✋ Manual Pick-Pair Align (≥3 pairs)…')
        btn_manual.setStyleSheet('font-weight: bold; padding: 6px;'
                                 ' background: rgba(60,140,200,0.20);')
        btn_manual.clicked.connect(self._start_manual_align)
        tl.addWidget(btn_manual, row, 0, 1, 2); row += 1

        # row of helper buttons that only matter while picking
        man_row = QtWidgets.QHBoxLayout()
        self.btn_manual_apply = QtWidgets.QPushButton('Apply Picked Pairs')
        self.btn_manual_apply.setStyleSheet('font-weight: bold;')
        self.btn_manual_apply.clicked.connect(self._on_manual_apply)
        self.btn_manual_apply.setEnabled(False)
        self.btn_manual_undo = QtWidgets.QPushButton('Undo Last Pick')
        self.btn_manual_undo.clicked.connect(self._on_manual_undo)
        self.btn_manual_undo.setEnabled(False)
        self.btn_manual_cancel = QtWidgets.QPushButton('Cancel')
        self.btn_manual_cancel.clicked.connect(self._on_manual_cancel)
        self.btn_manual_cancel.setEnabled(False)
        man_row.addWidget(self.btn_manual_apply)
        man_row.addWidget(self.btn_manual_undo)
        man_row.addWidget(self.btn_manual_cancel)
        man_w = QtWidgets.QWidget(); man_w.setLayout(man_row)
        tl.addWidget(man_w, row, 0, 1, 2); row += 1

        self.auto_align_check = QtWidgets.QCheckBox('Auto-Align after each load')
        self.auto_align_check.setChecked(True)
        tl.addWidget(self.auto_align_check, row, 0, 1, 2)
        v.addWidget(tbox)

        # -- merge into single slot (working cloud) -------------------------
        btn_merge = QtWidgets.QPushButton('⤵ Merge all visible → first slot (in-app)')
        btn_merge.setStyleSheet('padding: 5px;')
        btn_merge.clicked.connect(self._on_merge)
        v.addWidget(btn_merge)

        # -- edit group -----------------------------------------------------
        ebox = QtWidgets.QGroupBox('Edit (active slot)')
        el = QtWidgets.QVBoxLayout(ebox)

        self.active_label = QtWidgets.QLabel('Active: —')
        self.active_label.setStyleSheet('color: #ddd; font-weight: bold;')
        el.addWidget(self.active_label)

        mode_grid = QtWidgets.QGridLayout()
        self.mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        modes = [
            ('none',  'Pan/Rotate (Esc)'),
            ('box',   'Box Select  [B]'),
            ('poly',  'Polygon Lasso  [P]'),
            ('brush', 'Sphere Brush  [S]'),
            ('pick',  'Pick Point  [K]'),
            ('grid',  'Add Grid…  [G]'),
        ]
        for i, (key, label) in enumerate(modes):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, k=key: self.set_edit_mode(k))
            mode_grid.addWidget(btn, i // 2, i % 2)
            self.mode_buttons[key] = btn
        el.addLayout(mode_grid)

        hint = QtWidgets.QLabel(
            'Drag in Box mode → red points = selected; release → delete.\n'
            'Polygon: click vertices, double-click first to close, Esc to cancel.\n'
            'Brush: hover shows sphere; left-drag deletes inside; Shift+wheel = radius.\n'
            'Pick: click nearest point to delete it.'
        )
        hint.setWordWrap(True); hint.setStyleSheet('color:#aaa; font-size: 11px;')
        el.addWidget(hint)

        rrow = QtWidgets.QHBoxLayout()
        rrow.addWidget(QtWidgets.QLabel('Brush radius:'))
        self.brush_spin = QtWidgets.QDoubleSpinBox()
        self.brush_spin.setRange(0.001, 1e4); self.brush_spin.setDecimals(3)
        self.brush_spin.setValue(self._brush_radius)
        self.brush_spin.setSingleStep(0.05)
        self.brush_spin.valueChanged.connect(self._on_brush_radius_changed)
        rrow.addWidget(self.brush_spin)
        el.addLayout(rrow)

        srow = QtWidgets.QHBoxLayout()
        self.btn_commit = QtWidgets.QPushButton('Delete Highlighted  [Del]')
        self.btn_commit.setStyleSheet('font-weight: bold;')
        self.btn_commit.clicked.connect(self._on_commit_delete)
        self.btn_clear_sel = QtWidgets.QPushButton('Clear Highlight')
        self.btn_clear_sel.clicked.connect(self._on_clear_selection)
        srow.addWidget(self.btn_commit); srow.addWidget(self.btn_clear_sel)
        el.addLayout(srow)

        srow2 = QtWidgets.QHBoxLayout()
        btn_sel_vis = QtWidgets.QPushButton('Select All Visible (Z-Slice)')
        btn_sel_vis.clicked.connect(self._on_select_all_visible)
        btn_invert = QtWidgets.QPushButton('Invert Selection')
        btn_invert.clicked.connect(self._on_invert_selection)
        srow2.addWidget(btn_sel_vis); srow2.addWidget(btn_invert)
        el.addLayout(srow2)
        v.addWidget(ebox)

        # -- grid params ----------------------------------------------------
        gbox = QtWidgets.QGroupBox('Grid Add Parameters')
        gl = QtWidgets.QGridLayout(gbox)
        self.grid_size = QtWidgets.QDoubleSpinBox(); self.grid_size.setRange(0.05, 1e4); self.grid_size.setValue(2.0); self.grid_size.setDecimals(3)
        self.grid_step = QtWidgets.QDoubleSpinBox(); self.grid_step.setRange(0.001, 1e3); self.grid_step.setValue(0.1); self.grid_step.setDecimals(3)
        self.grid_axis = QtWidgets.QComboBox(); self.grid_axis.addItems(['XY (Z normal)', 'XZ (Y normal)', 'YZ (X normal)'])
        gl.addWidget(QtWidgets.QLabel('Edge length (m):'), 0, 0); gl.addWidget(self.grid_size, 0, 1)
        gl.addWidget(QtWidgets.QLabel('Step (m):'), 1, 0); gl.addWidget(self.grid_step, 1, 1)
        gl.addWidget(QtWidgets.QLabel('Plane:'), 2, 0); gl.addWidget(self.grid_axis, 2, 1)
        v.addWidget(gbox)

        # -- 3D Prism Selection --------------------------------------------
        pbox = QtWidgets.QGroupBox('3D Prism Selection (polygon × height)')
        pl = QtWidgets.QVBoxLayout(pbox)
        self.btn_draw_prism = QtWidgets.QPushButton('Start Polygon Draw  [N]')
        self.btn_draw_prism.setStyleSheet('font-weight: bold;')
        self.btn_draw_prism.clicked.connect(self._on_start_prism_draw)
        pl.addWidget(self.btn_draw_prism)

        plg = QtWidgets.QGridLayout()
        plg.addWidget(QtWidgets.QLabel('Base Z (m):'), 0, 0)
        self.prism_base_z = QtWidgets.QDoubleSpinBox()
        self.prism_base_z.setRange(-1e6, 1e6); self.prism_base_z.setDecimals(3)
        self.prism_base_z.setSingleStep(0.1); self.prism_base_z.setValue(0.0)
        self.prism_base_z.valueChanged.connect(self._on_prism_param_change)
        plg.addWidget(self.prism_base_z, 0, 1)
        plg.addWidget(QtWidgets.QLabel('Height (m):'), 1, 0)
        self.prism_height = QtWidgets.QDoubleSpinBox()
        self.prism_height.setRange(0.001, 1e6); self.prism_height.setDecimals(3)
        self.prism_height.setSingleStep(0.1); self.prism_height.setValue(2.0)
        self.prism_height.valueChanged.connect(self._on_prism_param_change)
        plg.addWidget(self.prism_height, 1, 1)
        pl.addLayout(plg)

        ttitle = QtWidgets.QLabel('Move (live preview):')
        ttitle.setStyleSheet('font-weight: bold; margin-top: 4px;')
        pl.addWidget(ttitle)
        plt = QtWidgets.QGridLayout()
        self.prism_tx = QtWidgets.QDoubleSpinBox(); self.prism_tx.setRange(-1e6, 1e6)
        self.prism_ty = QtWidgets.QDoubleSpinBox(); self.prism_ty.setRange(-1e6, 1e6)
        self.prism_tz = QtWidgets.QDoubleSpinBox(); self.prism_tz.setRange(-1e6, 1e6)
        for sp in (self.prism_tx, self.prism_ty, self.prism_tz):
            sp.setDecimals(3); sp.setSingleStep(0.1)
            sp.valueChanged.connect(self._on_prism_translate)
        plt.addWidget(QtWidgets.QLabel('X (m):'), 0, 0); plt.addWidget(self.prism_tx, 0, 1)
        plt.addWidget(QtWidgets.QLabel('Y (m):'), 1, 0); plt.addWidget(self.prism_ty, 1, 1)
        plt.addWidget(QtWidgets.QLabel('Z (m):'), 2, 0); plt.addWidget(self.prism_tz, 2, 1)
        pl.addLayout(plt)

        prow = QtWidgets.QHBoxLayout()
        self.btn_delete_in_prism = QtWidgets.QPushButton('Delete Points in Prism')
        self.btn_delete_in_prism.setStyleSheet('font-weight: bold; padding: 6px;')
        self.btn_delete_in_prism.clicked.connect(self._on_delete_in_prism)
        self.btn_clear_prism = QtWidgets.QPushButton('Clear Prism')
        self.btn_clear_prism.clicked.connect(self._on_clear_prism)
        prow.addWidget(self.btn_delete_in_prism)
        prow.addWidget(self.btn_clear_prism)
        pl.addLayout(prow)
        ph = QtWidgets.QLabel(
            '1) Click "Start Polygon Draw" → click vertices.\n'
            '2) Enter = finish, Esc = cancel.\n'
            '3) Adjust Height/Base Z and X/Y/Z to position the box.\n'
            '4) Only visible (Z-clip) points inside the box get deleted.'
        )
        ph.setStyleSheet('color: #aaa; font-size: 11px;'); ph.setWordWrap(True)
        pl.addWidget(ph)
        v.addWidget(pbox)

        # -- undo / redo / per-slot save ------------------------------------
        urow = QtWidgets.QHBoxLayout()
        self.btn_undo = QtWidgets.QPushButton('Undo  (Ctrl+Z)')
        self.btn_redo = QtWidgets.QPushButton('Redo  (Ctrl+Y)')
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_redo.clicked.connect(self._on_redo)
        urow.addWidget(self.btn_undo); urow.addWidget(self.btn_redo)
        v.addLayout(urow)

        save_row = QtWidgets.QHBoxLayout()
        btn_save_active = QtWidgets.QPushButton('Save Active Slot as PCD…')
        btn_save_active.clicked.connect(self._on_save_active)
        save_row.addWidget(btn_save_active)
        v.addLayout(save_row)

        v.addStretch()

    # ---------------------------------------------------------------- slot ops
    def _on_add_slot_click(self):
        sid = self._add_slot()
        self.status.showMessage(f'Added {self.slots[sid].name}', 2500)

    def _add_slot(self, name: Optional[str] = None) -> str:
        sid = self._next_sid()
        idx = len(self.slots)
        color = SLOT_PALETTE[idx % len(SLOT_PALETTE)]
        if name is None:
            name = f'Slot {idx + 1}'
        actor = self.viewer.add_cloud(color, point_size=2.0)
        slot = Slot(sid=sid, name=name, actor=actor, color=color)
        self.slots[sid] = slot
        self._build_slot_widget(slot)
        if self.active_sid is None:
            self.active_sid = sid
        self._refresh_combos()
        self._update_active_label()
        self._update_buttons()
        return sid

    def _remove_slot(self, sid: str):
        if sid not in self.slots:
            return
        if len(self.slots) <= 1:
            self.status.showMessage('Cannot remove the last slot. Use Clear instead.', 4000)
            return
        slot = self.slots[sid]
        # remove from viewer
        self.viewer.remove_cloud(slot.actor)
        # remove widget
        if slot.group is not None:
            self._slots_layout.removeWidget(slot.group)
            slot.group.setParent(None)
            slot.group.deleteLater()
        del self.slots[sid]
        # active slot fallback
        if self.active_sid == sid:
            self.active_sid = next(iter(self.slots.keys()), None)
        self._refresh_combos()
        self._update_active_label()
        self._update_buttons()
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Removed slot.', 2500)

    def _build_slot_widget(self, slot: Slot):
        gb = QtWidgets.QGroupBox()
        gb.setStyleSheet(
            f'QGroupBox {{ border: 1px solid rgba({int(slot.color[0]*255)},'
            f' {int(slot.color[1]*255)}, {int(slot.color[2]*255)}, 180);'
            f' border-radius: 4px; margin-top: 6px; }}'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }'
        )
        gl = QtWidgets.QVBoxLayout(gb)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setSpacing(4)

        # header: name edit + visible + color + size + ×
        head = QtWidgets.QHBoxLayout()
        slot.visible_chk = QtWidgets.QCheckBox()
        slot.visible_chk.setChecked(True)
        slot.visible_chk.setToolTip('Toggle slot visibility')
        slot.visible_chk.toggled.connect(
            lambda b, a=slot.actor: (a.set_visible(b), self.viewer.refresh()))
        head.addWidget(slot.visible_chk)

        slot.name_edit = QtWidgets.QLineEdit(slot.name)
        slot.name_edit.setStyleSheet('font-weight: bold;')
        slot.name_edit.editingFinished.connect(lambda s=slot: self._on_slot_renamed(s))
        head.addWidget(slot.name_edit, 1)

        slot.color_btn = ColorButton(slot.color)
        slot.color_btn.setToolTip('Slot color')
        slot.color_btn.colorChanged.connect(
            lambda c, s=slot: self._on_slot_color_changed(s, c))
        head.addWidget(slot.color_btn)

        slot.size_spin = QtWidgets.QDoubleSpinBox()
        slot.size_spin.setRange(1.0, 20.0)
        slot.size_spin.setValue(2.0)
        slot.size_spin.setDecimals(1)
        slot.size_spin.setSingleStep(0.5)
        slot.size_spin.setSuffix('px')
        slot.size_spin.setFixedWidth(64)
        slot.size_spin.valueChanged.connect(
            lambda v, a=slot.actor: (a.set_point_size(v), self.viewer.refresh()))
        head.addWidget(slot.size_spin)

        btn_x = QtWidgets.QPushButton('×')
        btn_x.setFixedSize(22, 22)
        btn_x.setToolTip('Remove this slot')
        btn_x.setStyleSheet(
            'QPushButton { background: #6a2222; color: white; font-weight: bold;'
            ' border-radius: 3px; }'
            'QPushButton:hover { background: #a83232; }'
        )
        btn_x.clicked.connect(lambda _=False, sid=slot.sid: self._remove_slot(sid))
        head.addWidget(btn_x)
        gl.addLayout(head)

        # load buttons row
        btn_row = QtWidgets.QHBoxLayout()
        btn_pcd = QtWidgets.QPushButton('Load PCD…')
        btn_bag = QtWidgets.QPushButton('Load ROS2 Bag…')
        btn_clr = QtWidgets.QPushButton('Clear')
        btn_pcd.clicked.connect(lambda _=False, sid=slot.sid: self._on_load_pcd(sid))
        btn_bag.clicked.connect(lambda _=False, sid=slot.sid: self._on_load_bag(sid))
        btn_clr.clicked.connect(lambda _=False, sid=slot.sid: self._on_clear_slot(sid))
        btn_row.addWidget(btn_pcd); btn_row.addWidget(btn_bag); btn_row.addWidget(btn_clr)
        gl.addLayout(btn_row)

        # active radio + source label
        bot = QtWidgets.QHBoxLayout()
        btn_make_active = QtWidgets.QPushButton('⊙ Make Active')
        btn_make_active.setStyleSheet(
            'QPushButton { padding: 2px 8px; font-size: 11px; }'
            'QPushButton:hover { background: rgba(255,255,255,0.10); }'
        )
        btn_make_active.clicked.connect(lambda _=False, sid=slot.sid: self._on_active_slot_change(sid))
        bot.addWidget(btn_make_active)
        slot.label = QtWidgets.QLabel('—')
        slot.label.setStyleSheet('color: #aaa; font-size: 11px;')
        bot.addWidget(slot.label, 1)
        gl.addLayout(bot)

        slot.group = gb
        self._slots_layout.addWidget(gb)

    def _on_slot_renamed(self, slot: Slot):
        new = slot.name_edit.text().strip() or slot.sid
        slot.name = new
        self._refresh_combos()
        self._update_active_label()

    def _on_slot_color_changed(self, slot: Slot, qcolor: QtGui.QColor):
        slot.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
        slot.actor.set_base_color(slot.color)
        # update group border
        slot.group.setStyleSheet(
            f'QGroupBox {{ border: 1px solid rgba({int(slot.color[0]*255)},'
            f' {int(slot.color[1]*255)}, {int(slot.color[2]*255)}, 180);'
            f' border-radius: 4px; margin-top: 6px; }}'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }'
        )
        self.viewer.refresh()

    def _refresh_combos(self):
        items = self._slot_items()
        # toolbar active combo
        self.viewer.toolbar.set_slot_choices(items, self.active_sid)
        # align combos — repopulate while restoring user's previous choice
        for combo, default_idx in ((self.align_source_combo, 1 if len(items) >= 2 else 0),
                                    (self.align_target_combo, 0)):
            combo.blockSignals(True)
            prev_sid = combo.currentData()
            combo.clear()
            for sid, name in items:
                combo.addItem(name, sid)
            if combo.count() == 0:
                combo.blockSignals(False)
                continue
            picked = -1
            if prev_sid is not None:
                for i in range(combo.count()):
                    if combo.itemData(i) == prev_sid:
                        picked = i; break
            if picked < 0:
                picked = min(default_idx, combo.count() - 1)
            combo.setCurrentIndex(picked)
            combo.blockSignals(False)

        # Ensure source != target when at least 2 slots exist.
        if (self.align_source_combo.count() >= 2
                and self.align_source_combo.currentData()
                    == self.align_target_combo.currentData()):
            src_idx = self.align_source_combo.currentIndex()
            other = (src_idx + 1) % self.align_source_combo.count()
            self.align_source_combo.blockSignals(True)
            self.align_source_combo.setCurrentIndex(other)
            self.align_source_combo.blockSignals(False)

    def _update_active_label(self):
        s = self._active_slot()
        if s is None:
            self.active_label.setText('Active: —')
        else:
            self.active_label.setText(f'Active: {s.name}')

    def _on_active_slot_change(self, sid: str):
        if sid not in self.slots:
            return
        self.active_sid = sid
        self.set_edit_mode('none')
        self._update_active_label()
        self._update_buttons()
        self.viewer.toolbar.set_slot_choices(self._slot_items(), self.active_sid)
        self.status.showMessage(f'Active edit slot: {self.slots[sid].name}', 3000)

    def _on_show_all_slots(self):
        for s in self.slots.values():
            s.visible_chk.setChecked(True)

    def _on_hide_all_slots(self):
        for s in self.slots.values():
            s.visible_chk.setChecked(False)

    # ---------------------------------------------------------------- shortcuts
    def _setup_shortcuts(self):
        def add(key, fn):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(fn)
        add('B', lambda: self.set_edit_mode('box'))
        add('P', lambda: self.set_edit_mode('poly'))
        add('S', lambda: self.set_edit_mode('brush'))
        add('K', lambda: self.set_edit_mode('pick'))
        add('G', lambda: self.set_edit_mode('grid'))
        add('N', self._on_start_prism_draw)
        add('Escape', lambda: self.set_edit_mode('none'))
        add('Ctrl+Z', self._on_undo)
        add('Ctrl+Y', self._on_redo)
        add('Ctrl+Shift+Z', self._on_redo)
        add('Delete', self._on_commit_delete)
        add('R', self.viewer.reset_view)

    # ---------------------------------------------------------------- IO
    def _on_load_pcd(self, sid: str):
        if sid not in self.slots:
            return
        slot = self.slots[sid]
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f'Load PCD ({slot.name})', '', 'PCD (*.pcd)')
        if not path:
            return
        try:
            pts, intens = load_pcd(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Load failed', str(e))
            return
        slot.actor.set_points(pts, intens)
        slot.source_label = f'{Path(path).name}  •  {len(pts):,} pts'
        slot.label.setText(slot.source_label)
        self.viewer.reset_view()
        self._refresh_z_bounds()
        self.status.showMessage(f'Loaded {len(pts):,} points into {slot.name}', 4000)
        self._maybe_auto_align()

    def _on_load_bag(self, sid: str):
        if sid not in self.slots:
            return
        slot = self.slots[sid]
        bag_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, f'Select ROS2 bag directory ({slot.name})')
        if not bag_dir:
            return
        topics = list_pointcloud_topics_in_bag(bag_dir)
        if not topics:
            QtWidgets.QMessageBox.warning(self, 'No PointCloud2 topics', 'No PointCloud2 topics found in this bag.')
            return
        topic, ok = QtWidgets.QInputDialog.getItem(self, 'Topic', 'Select PointCloud2 topic:', topics, 0, False)
        if not ok:
            return
        mode, ok2 = QtWidgets.QInputDialog.getItem(
            self, 'Mode', 'Read mode:',
            ['last  (single accumulated map message)', 'concat  (combine all messages)'], 0, False)
        if not ok2:
            return
        mode_key = 'last' if mode.startswith('last') else 'concat'
        prog = QtWidgets.QProgressDialog(f'Reading {topic}…', 'Cancel', 0, 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()
        QtWidgets.QApplication.processEvents()
        try:
            pts, intens = read_pointcloud_from_bag(bag_dir, topic, mode=mode_key)
        except Exception as e:
            prog.close()
            QtWidgets.QMessageBox.critical(self, 'Bag read failed', str(e))
            return
        prog.close()
        if pts.shape[0] == 0:
            QtWidgets.QMessageBox.information(self, 'Empty', 'No points read from this topic.')
            return
        slot.actor.set_points(pts, intens)
        slot.source_label = f'{Path(bag_dir).name} : {topic}  ({mode_key})  •  {len(pts):,} pts'
        slot.label.setText(slot.source_label)
        self.viewer.reset_view()
        self._refresh_z_bounds()
        self.status.showMessage(f'Loaded {len(pts):,} points into {slot.name}', 4000)
        self._maybe_auto_align()

    def _on_clear_slot(self, sid: str):
        if sid not in self.slots:
            return
        slot = self.slots[sid]
        slot.actor.set_points(np.zeros((0, 3), dtype=np.float32), None)
        slot.source_label = '—'
        slot.label.setText('—')
        self._refresh_z_bounds()
        self.viewer.refresh()

    # ---------------------------------------------------------------- align
    def _get_align_pair(self) -> tuple[Optional[Slot], Optional[Slot]]:
        src_sid = self.align_source_combo.currentData()
        tgt_sid = self.align_target_combo.currentData()
        return self.slots.get(src_sid), self.slots.get(tgt_sid)

    def _on_apply_transform(self):
        src, _ = self._get_align_pair()
        if src is None or len(src.actor.points) == 0:
            self.status.showMessage('Source slot is empty.', 4000)
            return
        a = src.actor
        tx = self._t_spins['tx'].value(); ty = self._t_spins['ty'].value(); tz = self._t_spins['tz'].value()
        rx = np.deg2rad(self._t_spins['rx'].value())
        ry = np.deg2rad(self._t_spins['ry'].value())
        rz = np.deg2rad(self._t_spins['rz'].value())
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = (Rz @ Ry @ Rx).astype(np.float32)
        self._push_history(sid=src.sid)
        new_pts = (a.points @ R.T) + np.array([tx, ty, tz], dtype=np.float32)
        a.points = new_pts.astype(np.float32)
        a._rebuild_vtk()
        for k in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz'):
            self._t_spins[k].setValue(0.0)
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Transform applied to {src.name}', 3000)

    def _on_icp(self):
        src, tgt = self._get_align_pair()
        if src is None or tgt is None or src.sid == tgt.sid:
            QtWidgets.QMessageBox.information(self, 'ICP', 'Pick different source and target slots.')
            return
        if len(src.actor.points) == 0 or len(tgt.actor.points) == 0:
            QtWidgets.QMessageBox.information(self, 'ICP', 'Both slots must have points.')
            return
        prog = QtWidgets.QProgressDialog('Running ICP…', None, 0, 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal); prog.show()
        QtWidgets.QApplication.processEvents()
        try:
            pa = o3d.geometry.PointCloud(); pa.points = o3d.utility.Vector3dVector(tgt.actor.points.astype(np.float64))
            pb = o3d.geometry.PointCloud(); pb.points = o3d.utility.Vector3dVector(src.actor.points.astype(np.float64))
            voxel = float(np.linalg.norm(pa.get_axis_aligned_bounding_box().get_extent()) / 200.0)
            voxel = max(0.02, voxel)
            pa_d = pa.voxel_down_sample(voxel); pb_d = pb.voxel_down_sample(voxel)
            threshold = voxel * 5.0
            res = o3d.pipelines.registration.registration_icp(
                pb_d, pa_d, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
            )
            T = res.transformation.astype(np.float32)
            self._push_history(sid=src.sid)
            b = src.actor
            ones = np.ones((b.points.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([b.points.astype(np.float32), ones], axis=1)
            new = (homo @ T.T)[:, :3]
            b.points = new.astype(np.float32)
            b.visibility_mask = np.ones(len(b.points), dtype=bool)
            b.selection_mask = np.zeros(len(b.points), dtype=bool)
            b._rebuild_vtk()
            self._refresh_z_bounds()
            self.viewer.refresh()
        finally:
            prog.close()
        self.status.showMessage(
            f'ICP done ({src.name} → {tgt.name}). fitness={res.fitness:.3f}, '
            f'rmse={res.inlier_rmse:.3f}', 6000)

    def _on_auto_align(self):
        src, tgt = self._get_align_pair()
        if src is None or tgt is None or src.sid == tgt.sid:
            QtWidgets.QMessageBox.information(self, 'Auto-Align', 'Pick different source and target slots.')
            return
        if len(src.actor.points) == 0 or len(tgt.actor.points) == 0:
            QtWidgets.QMessageBox.information(self, 'Auto-Align', 'Both slots must have points.')
            return
        prog = QtWidgets.QProgressDialog('Auto-Align (deep search, this may take a while)…',
                                         None, 0, 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.show()
        QtWidgets.QApplication.processEvents()
        try:
            pa = o3d.geometry.PointCloud()
            pa.points = o3d.utility.Vector3dVector(tgt.actor.points.astype(np.float64))
            pb = o3d.geometry.PointCloud()
            pb.points = o3d.utility.Vector3dVector(src.actor.points.astype(np.float64))
            diag = float(np.linalg.norm(pa.get_axis_aligned_bounding_box().get_extent()))
            if diag < 1e-3: diag = 1.0

            # --- three resolutions: coarse for FGR, medium/fine for ICP -------
            voxel_coarse = max(0.05, diag / 80.0)
            voxel_medium = max(0.02, diag / 200.0)
            voxel_fine   = max(0.008, diag / 600.0)

            prog.setLabelText(
                f'Downsample @ {voxel_coarse:.3f}/{voxel_medium:.3f}/'
                f'{voxel_fine:.3f} + normals…')
            QtWidgets.QApplication.processEvents()

            def _downsample_with_normals(pc, voxel):
                d = pc.voxel_down_sample(voxel)
                d.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
                return d

            pa_c = _downsample_with_normals(pa, voxel_coarse)
            pb_c = _downsample_with_normals(pb, voxel_coarse)
            pa_m = _downsample_with_normals(pa, voxel_medium)
            pb_m = _downsample_with_normals(pb, voxel_medium)
            pa_f = _downsample_with_normals(pa, voxel_fine)
            pb_f = _downsample_with_normals(pb, voxel_fine)

            # ---- candidate initial transforms ---------------------------------
            ca = np.asarray(pa_c.get_center())
            cb = np.asarray(pb_c.get_center())
            T_cent = np.eye(4); T_cent[:3, 3] = ca - cb

            def _yaw_T(deg: float) -> np.ndarray:
                a = np.deg2rad(deg)
                R = np.eye(4)
                R[0, 0] = np.cos(a); R[0, 1] = -np.sin(a)
                R[1, 0] = np.sin(a); R[1, 1] =  np.cos(a)
                T = np.eye(4)
                T[:3, 3] = ca           # rotate around target centroid
                T2 = np.eye(4); T2[:3, 3] = -cb
                return T @ R @ T2

            candidates: list[tuple[str, np.ndarray]] = [
                ('identity', np.eye(4)),
                ('centroid', T_cent),
            ]
            for yaw in (45, 90, 135, 180, 225, 270, 315):
                candidates.append((f'yaw{yaw}', _yaw_T(yaw)))

            prog.setLabelText('Computing FPFH features (coarse)…')
            QtWidgets.QApplication.processEvents()
            rf = voxel_coarse * 5.0
            fa = o3d.pipelines.registration.compute_fpfh_feature(
                pa_c, o3d.geometry.KDTreeSearchParamHybrid(radius=rf, max_nn=100))
            fb = o3d.pipelines.registration.compute_fpfh_feature(
                pb_c, o3d.geometry.KDTreeSearchParamHybrid(radius=rf, max_nn=100))

            for k in range(10):
                prog.setLabelText(f'FGR trial {k + 1}/10…')
                QtWidgets.QApplication.processEvents()
                try:
                    res_g = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                        pb_c, pa_c, fb, fa,
                        o3d.pipelines.registration.FastGlobalRegistrationOption(
                            maximum_correspondence_distance=voxel_coarse * 0.5))
                    candidates.append((f'FGR#{k+1}', np.asarray(res_g.transformation)))
                except Exception:
                    pass

            # ---- refine each candidate: coarse → medium → fine ICP ----------
            voxel_for_score = voxel_medium

            def score(res):
                if res.fitness < 0.01:
                    return -1e9
                return res.fitness - (res.inlier_rmse / max(voxel_for_score, 1e-6))

            def run_icp(pb_d, pa_d, T_in, thresh, max_it):
                return o3d.pipelines.registration.registration_icp(
                    pb_d, pa_d, thresh, T_in,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it))

            best = None
            total = len(candidates)
            for i, (name, T_init) in enumerate(candidates):
                prog.setLabelText(f'Refining {name}  ({i+1}/{total})…')
                QtWidgets.QApplication.processEvents()
                T = np.asarray(T_init, dtype=np.float64)
                res = None
                try:
                    res = run_icp(pb_c, pa_c, T, voxel_coarse * 4.0, 60); T = res.transformation
                    res = run_icp(pb_c, pa_c, T, voxel_coarse * 2.0, 60); T = res.transformation
                    res = run_icp(pb_m, pa_m, T, voxel_medium * 4.0, 80); T = res.transformation
                    res = run_icp(pb_m, pa_m, T, voxel_medium * 2.0, 80); T = res.transformation
                    res = run_icp(pb_m, pa_m, T, voxel_medium * 1.0, 80); T = res.transformation
                except Exception:
                    continue
                if res is None:
                    continue
                s = score(res)
                if (best is None) or (s > best[0]):
                    best = (s, name, T, res.fitness, res.inlier_rmse)

            if best is None:
                QtWidgets.QMessageBox.critical(
                    self, 'Auto-Align failed',
                    'All ICP trials failed. The clouds may be too different.')
                return

            _, used_name, T_best, fit, rmse = best

            # ---- fine pass on dense cloud, then Generalized ICP ----------------
            prog.setLabelText('Fine ICP on dense cloud…')
            QtWidgets.QApplication.processEvents()
            T_cur = np.asarray(T_best, dtype=np.float64)
            try:
                for thresh in (voxel_fine * 4.0, voxel_fine * 2.0, voxel_fine * 1.0):
                    res_fine = run_icp(pb_f, pa_f, T_cur, thresh, 120)
                    T_cur = np.asarray(res_fine.transformation, dtype=np.float64)
                if res_fine.fitness >= fit * 0.90:
                    T_best = T_cur
                    fit = res_fine.fitness
                    rmse = res_fine.inlier_rmse
                    used_name += ' + fine'
            except Exception:
                pass

            prog.setLabelText('Generalized ICP (final polish)…')
            QtWidgets.QApplication.processEvents()
            try:
                T_cur = np.asarray(T_best, dtype=np.float64)
                gicp = o3d.pipelines.registration.registration_generalized_icp(
                    pb_f, pa_f, voxel_fine * 2.0, T_cur,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120))
                if gicp.fitness >= fit * 0.90:
                    T_best = np.asarray(gicp.transformation, dtype=np.float64)
                    fit = gicp.fitness
                    rmse = gicp.inlier_rmse
                    used_name += ' + GICP'
            except Exception:
                pass

            T = T_best.astype(np.float32)
            # detect "essentially identity" winner (no real move needed)
            T64 = np.asarray(T_best, dtype=np.float64)
            trans_norm = float(np.linalg.norm(T64[:3, 3]))
            cos_ang = (T64[0,0] + T64[1,1] + T64[2,2] - 1) / 2.0
            rot_deg = float(np.degrees(np.arccos(max(min(cos_ang, 1.0), -1.0))))
            was_identity = (trans_norm < voxel_fine and rot_deg < 0.5)

            self._push_history(sid=src.sid)
            b = src.actor
            ones = np.ones((b.points.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([b.points.astype(np.float32), ones], axis=1)
            new = (homo @ T.T)[:, :3]
            b.points = new.astype(np.float32)
            b.visibility_mask = np.ones(len(b.points), dtype=bool)
            b.selection_mask = np.zeros(len(b.points), dtype=bool)
            b._rebuild_vtk()
            self._refresh_z_bounds()
            self.viewer.refresh()
            self.status.showMessage(
                f'Auto-Align {src.name} → {tgt.name} [winner: {used_name}]. '
                f'fitness={fit:.3f}, rmse={rmse:.3f}.  (Ctrl+Z to undo)', 12000)
            prog.close()

            # explicit result dialog so the user can't miss the outcome
            if was_identity:
                QtWidgets.QMessageBox.information(
                    self, 'Auto-Align finished',
                    f'{src.name} → {tgt.name}: clouds were already aligned.\n'
                    f'winner: {used_name}\n'
                    f'fitness = {fit:.3f}\nrmse = {rmse:.4f}\n\n'
                    f'No visible change. If you expected a movement, try '
                    f'"✋ Manual Pick-Pair Align".')
            elif fit < 0.5:
                QtWidgets.QMessageBox.warning(
                    self, 'Auto-Align quality LOW',
                    f'{src.name} → {tgt.name}\nwinner: {used_name}\n'
                    f'fitness = {fit:.3f}\nrmse = {rmse:.4f}\n\n'
                    f'The algorithm is unsure. Consider "Manual Pick-Pair Align" '
                    f'for difficult data.')
            else:
                QtWidgets.QMessageBox.information(
                    self, 'Auto-Align finished',
                    f'{src.name} → {tgt.name}\nwinner: {used_name}\n'
                    f'fitness = {fit:.3f}\nrmse = {rmse:.4f}\n'
                    f'translation = {trans_norm:.3f} m,  rotation = {rot_deg:.2f}°\n\n'
                    f'Ctrl+Z to undo if not satisfied.')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Auto-Align failed', str(e))
        finally:
            try:
                prog.close()
            except Exception:
                pass

    def _maybe_auto_align(self):
        if not getattr(self, 'auto_align_check', None) or not self.auto_align_check.isChecked():
            return
        loaded = [s for s in self.slots.values() if len(s.actor.points) > 0]
        if len(loaded) >= 2:
            QtCore.QTimer.singleShot(200, self._on_auto_align)

    # ---------------------------------------------------------------- manual align
    def _start_manual_align(self):
        src, tgt = self._get_align_pair()
        if src is None or tgt is None or src.sid == tgt.sid:
            QtWidgets.QMessageBox.information(
                self, 'Manual Align',
                'Pick different source and target slots in the Align dropdowns first.')
            return
        if len(src.actor.points) == 0 or len(tgt.actor.points) == 0:
            QtWidgets.QMessageBox.information(
                self, 'Manual Align',
                'Both slots must contain points.')
            return
        # leave any other edit mode
        self.set_edit_mode('none')

        self._manual_active = True
        self._manual_src = src
        self._manual_tgt = tgt
        self._manual_phase = 'target'
        self._manual_pairs = []
        self._manual_clear_markers()

        self._manual_style = PairPickStyle(
            self.viewer,
            on_pick=self._on_manual_pick,
            on_cancel=self._on_manual_cancel,
            on_undo=self._on_manual_undo,
            on_apply=self._on_manual_apply,
        )
        self._manual_style.set_pick_actor(tgt.actor.actor)
        self.viewer.set_style(self._manual_style)
        self._cur_style = self._manual_style

        self.btn_manual_apply.setEnabled(False)
        self.btn_manual_undo.setEnabled(False)
        self.btn_manual_cancel.setEnabled(True)

        self.status.showMessage(
            f'Pick a point on TARGET ({tgt.name}, green). Then on SOURCE '
            f'({src.name}, blue). Repeat ≥3× → Enter to apply, Esc to cancel.',
            0)

    def _on_manual_pick(self, world_pt: np.ndarray):
        if not self._manual_active:
            return
        if self._manual_phase == 'target':
            self._manual_pairs.append([world_pt, None])
            self._add_manual_marker(world_pt, kind='target',
                                    idx=len(self._manual_pairs))
            self._manual_phase = 'source'
            if self._manual_style is not None:
                self._manual_style.set_pick_actor(self._manual_src.actor.actor)
            self.status.showMessage(
                f'Pair {len(self._manual_pairs)}: now pick the SAME location '
                f'on SOURCE ({self._manual_src.name}).', 0)
        else:
            self._manual_pairs[-1][1] = world_pt
            self._add_manual_marker(world_pt, kind='source',
                                    idx=len(self._manual_pairs))
            self._add_manual_pair_line(self._manual_pairs[-1][0], world_pt)
            self._manual_phase = 'target'
            if self._manual_style is not None:
                self._manual_style.set_pick_actor(self._manual_tgt.actor.actor)
            n = len(self._manual_pairs)
            self.btn_manual_undo.setEnabled(True)
            self.btn_manual_apply.setEnabled(n >= 3)
            if n >= 3:
                self.status.showMessage(
                    f'{n} pairs picked – press Enter / "Apply" to align, '
                    f'or pick more pairs for better accuracy.', 0)
            else:
                self.status.showMessage(
                    f'{n}/3 pairs picked – pick another TARGET point '
                    f'on {self._manual_tgt.name}.', 0)
        self.viewer.refresh()

    def _on_manual_undo(self):
        if not self._manual_active or not self._manual_pairs:
            return
        last = self._manual_pairs[-1]
        if self._manual_phase == 'source' and last[1] is None:
            # remove only the target half
            self._manual_pairs.pop()
            self._manual_remove_last_markers(1)
            self._manual_phase = 'target'
            if self._manual_style is not None:
                self._manual_style.set_pick_actor(self._manual_tgt.actor.actor)
        else:
            # remove the whole pair
            self._manual_pairs.pop()
            self._manual_remove_last_markers(3)  # 2 spheres + 1 line
            self._manual_phase = 'target'
            if self._manual_style is not None:
                self._manual_style.set_pick_actor(self._manual_tgt.actor.actor)
        n = len(self._manual_pairs)
        self.btn_manual_undo.setEnabled(n > 0)
        self.btn_manual_apply.setEnabled(n >= 3)
        self.status.showMessage(
            f'Undid last pick. {n} pair(s). Next: pick TARGET '
            f'({self._manual_tgt.name}).', 4000)
        self.viewer.refresh()

    def _on_manual_cancel(self):
        if not self._manual_active:
            return
        self._manual_active = False
        self._manual_pairs = []
        self._manual_clear_markers()
        self.viewer.use_default_style()
        self._cur_style = self.viewer.default_style
        self._manual_style = None
        self.btn_manual_apply.setEnabled(False)
        self.btn_manual_undo.setEnabled(False)
        self.btn_manual_cancel.setEnabled(False)
        self.viewer.refresh()
        self.status.showMessage('Manual align cancelled.', 3000)

    def _on_manual_apply(self):
        if not self._manual_active:
            return
        complete = [p for p in self._manual_pairs if p[1] is not None]
        if len(complete) < 3:
            self.status.showMessage(
                f'Need ≥3 complete pairs (have {len(complete)}).', 4000)
            return
        tgt_pts = np.asarray([p[0] for p in complete], dtype=np.float64)
        src_pts = np.asarray([p[1] for p in complete], dtype=np.float64)
        T_um = _umeyama_rigid(src_pts, tgt_pts)

        # ICP refinement from manual init
        src = self._manual_src; tgt = self._manual_tgt
        prog = QtWidgets.QProgressDialog(
            'Manual-Align: refining with ICP…', None, 0, 0, self)
        prog.setWindowModality(QtCore.Qt.WindowModal); prog.show()
        QtWidgets.QApplication.processEvents()
        try:
            pa = o3d.geometry.PointCloud()
            pa.points = o3d.utility.Vector3dVector(tgt.actor.points.astype(np.float64))
            pb = o3d.geometry.PointCloud()
            pb.points = o3d.utility.Vector3dVector(src.actor.points.astype(np.float64))
            diag = float(np.linalg.norm(pa.get_axis_aligned_bounding_box().get_extent()))
            if diag < 1e-3: diag = 1.0
            voxel = max(0.02, diag / 250.0)
            pa_d = pa.voxel_down_sample(voxel)
            pb_d = pb.voxel_down_sample(voxel)
            pa_d.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.0, max_nn=30))
            pb_d.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.0, max_nn=30))
            T_cur = T_um
            for thresh in (voxel * 6.0, voxel * 3.0, voxel * 1.5, voxel * 0.75):
                res = o3d.pipelines.registration.registration_icp(
                    pb_d, pa_d, thresh, T_cur,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120))
                T_cur = np.asarray(res.transformation, dtype=np.float64)
            T_final = T_cur.astype(np.float32)
            fit = float(res.fitness); rmse = float(res.inlier_rmse)
        except Exception as e:
            prog.close()
            QtWidgets.QMessageBox.critical(self, 'Manual-Align failed', str(e))
            return
        finally:
            prog.close()

        self._push_history(sid=src.sid)
        b = src.actor
        ones = np.ones((b.points.shape[0], 1), dtype=np.float32)
        homo = np.concatenate([b.points.astype(np.float32), ones], axis=1)
        new = (homo @ T_final.T)[:, :3]
        b.points = new.astype(np.float32)
        b.visibility_mask = np.ones(len(b.points), dtype=bool)
        b.selection_mask = np.zeros(len(b.points), dtype=bool)
        b._rebuild_vtk()
        self._refresh_z_bounds()
        self._on_manual_cancel()  # clear markers and exit mode
        self.viewer.refresh()
        QtWidgets.QMessageBox.information(
            self, 'Manual Align done',
            f'Aligned {src.name} → {tgt.name} from {len(complete)} pairs.\n'
            f'ICP fitness = {fit:.3f}\nICP rmse = {rmse:.4f}\n\n'
            f'Undo via Ctrl+Z if not satisfied.')

    # -- manual marker / line visuals -----------------------------------------
    def _add_manual_marker(self, pos: np.ndarray, kind: str, idx: int):
        color = (0.20, 0.95, 0.30) if kind == 'target' else (0.30, 0.70, 1.00)
        # size proportional to scene
        diag = max(1.0, self._z_world_max - self._z_world_min)
        r = max(0.05, diag * 0.012)
        src = vtk.vtkSphereSource()
        src.SetThetaResolution(16); src.SetPhiResolution(16)
        src.SetRadius(r)
        src.SetCenter(float(pos[0]), float(pos[1]), float(pos[2]))
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(src.GetOutputPort())
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(*color); p.SetOpacity(0.85); p.SetLighting(False)
        actor.PickableOff()
        self.viewer.renderer.AddActor(actor)
        self._manual_marker_actors.append(actor)

    def _add_manual_pair_line(self, p_tgt: np.ndarray, p_src: np.ndarray):
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(float(p_tgt[0]), float(p_tgt[1]), float(p_tgt[2]))
        pts.InsertNextPoint(float(p_src[0]), float(p_src[1]), float(p_src[2]))
        lines = vtk.vtkCellArray()
        ln = vtk.vtkLine(); ln.GetPointIds().SetId(0, 0); ln.GetPointIds().SetId(1, 1)
        lines.InsertNextCell(ln)
        poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(1.0, 0.85, 0.20); p.SetLineWidth(2.0); p.SetLighting(False)
        actor.PickableOff()
        self.viewer.renderer.AddActor(actor)
        self._manual_marker_actors.append(actor)

    def _manual_remove_last_markers(self, n: int):
        for _ in range(n):
            if not self._manual_marker_actors:
                break
            a = self._manual_marker_actors.pop()
            self.viewer.renderer.RemoveActor(a)

    def _manual_clear_markers(self):
        for a in self._manual_marker_actors:
            self.viewer.renderer.RemoveActor(a)
        self._manual_marker_actors = []

    def _on_merge(self):
        """Combine all visible slots' points into the first slot, then clear the rest."""
        visible_slots = [s for s in self.slots.values()
                         if s.actor.actor.GetVisibility() and len(s.actor.points) > 0]
        if not visible_slots:
            self.status.showMessage('No visible slots with points.', 3000)
            return
        target = visible_slots[0]
        self._push_history(sid=target.sid)
        all_pts = [s.actor.points for s in visible_slots]
        all_int = [s.actor.intensities for s in visible_slots]
        new_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
        if all(i is not None for i in all_int):
            new_int = np.concatenate(all_int, axis=0).astype(np.float32)
        else:
            new_int = None
        target.actor.set_points(new_pts, new_int)
        target.source_label = f'merged ({len(visible_slots)} slots)  •  {len(new_pts):,} pts'
        target.label.setText(target.source_label)
        for s in visible_slots[1:]:
            s.actor.set_points(np.zeros((0, 3), dtype=np.float32), None)
            s.source_label = '— (merged into ' + target.name + ')'
            s.label.setText(s.source_label)
        self.active_sid = target.sid
        self._refresh_combos()
        self._update_active_label()
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(
            f'Merged {len(visible_slots)} slots → {target.name} ({len(new_pts):,} points).', 5000)

    # ---------------------------------------------------------------- edit modes
    def set_edit_mode(self, mode: str):
        self.edit_mode = mode
        actor = self._active_actor_get()
        if mode == 'none' or actor is None:
            self.edit_mode = 'none'
            self.viewer.use_default_style()
            self.viewer.hide_brush_sphere()
            self.viewer.clear_polygon_outline()
        elif mode == 'box':
            self._on_sel_start()
            style = LiveBoxSelectStyle(
                self.viewer,
                on_start=self._on_sel_start,
                on_preview=self._on_box_preview,
                on_final=self._on_box_final,
                on_cancel=self._on_sel_cancel)
            self.viewer.set_style(style); self._cur_style = style
            self.viewer.hide_brush_sphere(); self.viewer.clear_polygon_outline()
        elif mode == 'poly':
            self._on_sel_start()
            style = PolygonLassoStyle(
                self.viewer,
                on_polygon=self._on_poly_final,
                on_preview=self._on_poly_preview)
            self.viewer.set_style(style); self._cur_style = style
            self.viewer.hide_brush_sphere()
        elif mode == 'brush':
            style = SphereBrushStyle(
                self.viewer, self._on_brush_apply,
                on_radius_change=lambda r: self.brush_spin.setValue(r),
                radius=self._brush_radius)
            self.viewer.set_style(style); self._cur_style = style
            self.viewer.clear_polygon_outline()
        elif mode == 'pick':
            style = PointPickStyle(self._on_point_pick)
            self.viewer.set_style(style); self._cur_style = style
            self.viewer.hide_brush_sphere(); self.viewer.clear_polygon_outline()
        elif mode == 'grid':
            style = GridPickStyle(self._on_grid_pick)
            self.viewer.set_style(style); self._cur_style = style
            self.viewer.hide_brush_sphere(); self.viewer.clear_polygon_outline()
        self._update_buttons()
        self.status.showMessage(f'Mode: {self.edit_mode}', 3000)

    # ---------------------------------------------------------------- selection
    def _on_sel_start(self):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            self._proj_actor = None
            self._proj_visible_idx = None
            self._proj_screen = None
            return
        vis_idx = np.where(actor.visibility_mask)[0]
        self._proj_actor = actor
        self._proj_visible_idx = vis_idx
        if len(vis_idx) == 0:
            self._proj_screen = np.zeros((0, 3), dtype=np.float32)
        else:
            self._proj_screen = self.viewer.project_points_to_screen(
                actor.points[vis_idx])

    def _on_sel_cancel(self):
        actor = self._active_actor_get()
        if actor is not None:
            actor.clear_selection()
        self.viewer.refresh()

    def _ensure_projection(self):
        actor = self._active_actor_get()
        if actor is None:
            return
        if (getattr(self, '_proj_actor', None) is not actor
                or self._proj_visible_idx is None
                or len(self._proj_visible_idx) != int(actor.visibility_mask.sum())):
            self._on_sel_start()

    def _compute_box_mask(self, xmin, ymin, xmax, ymax) -> np.ndarray:
        actor = self._active_actor_get()
        if actor is None:
            return np.zeros(0, dtype=bool)
        self._ensure_projection()
        n = len(actor.points)
        if self._proj_visible_idx is None or len(self._proj_visible_idx) == 0:
            return np.zeros(n, dtype=bool)
        s = self._proj_screen
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        inside = ((sx >= xmin) & (sx <= xmax)
                  & (sy >= ymin) & (sy <= ymax)
                  & np.isfinite(sz) & (sz >= -1.0) & (sz <= 1.0))
        mask = np.zeros(n, dtype=bool)
        mask[self._proj_visible_idx[inside]] = True
        return mask

    def _compute_poly_mask(self, verts) -> np.ndarray:
        actor = self._active_actor_get()
        if actor is None:
            return np.zeros(0, dtype=bool)
        self._ensure_projection()
        n = len(actor.points)
        if self._proj_visible_idx is None or len(self._proj_visible_idx) == 0:
            return np.zeros(n, dtype=bool)
        s = self._proj_screen
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        valid = np.isfinite(sz) & (sz >= -1.0) & (sz <= 1.0)
        inside = _points_in_polygon(sx, sy, np.asarray(verts, dtype=np.float32))
        sel_visible = valid & inside
        mask = np.zeros(n, dtype=bool)
        mask[self._proj_visible_idx[sel_visible]] = True
        return mask

    def _box_compute_and_apply(self, xmin, ymin, xmax, ymax):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return 0
        cache_ok = (self._proj_actor is actor
                    and self._proj_screen is not None
                    and self._proj_visible_idx is not None
                    and len(self._proj_visible_idx) == int(actor.visibility_mask.sum()))
        if cache_ok:
            s = self._proj_screen
            idx = self._proj_visible_idx
            sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
            inside = ((sx >= xmin) & (sx <= xmax)
                      & (sy >= ymin) & (sy <= ymax)
                      & np.isfinite(sz) & (sz >= -1.0) & (sz <= 1.0))
            mask = np.zeros(len(actor.points), dtype=bool)
            if inside.any():
                mask[idx[inside]] = True
        else:
            screen = self.viewer.project_points_to_screen(actor.points)
            sx, sy, sz = screen[:, 0], screen[:, 1], screen[:, 2]
            mask = ((sx >= xmin) & (sx <= xmax)
                    & (sy >= ymin) & (sy <= ymax)
                    & np.isfinite(sz) & (sz >= -1.0) & (sz <= 1.0)
                    & actor.visibility_mask)
        actor.set_selection(mask)
        self.viewer.refresh()
        return int(actor.selection_mask.sum())

    def _on_box_preview(self, xmin, ymin, xmax, ymax):
        n = self._box_compute_and_apply(xmin, ymin, xmax, ymax)
        if n > 0:
            self.status.showMessage(f'Box: highlighting {n} points…', 800)

    def _on_box_final(self, xmin, ymin, xmax, ymax):
        n = self._box_compute_and_apply(xmin, ymin, xmax, ymax)
        self.status.showMessage(
            f'Highlighted {n} points. Press Delete to remove '
            f'(re-drag to refine, Esc to clear).', 6000)

    def _on_poly_preview(self, verts):
        self._sel_preview_kind = 'poly'
        self._sel_preview_args = list(verts)
        if not self._sel_preview_timer.isActive():
            self._sel_preview_timer.start()

    def _on_poly_final(self, verts):
        self._sel_preview_kind = 'poly'
        self._sel_preview_args = list(verts)
        self._do_sel_preview()
        actor = self._active_actor_get()
        n = int(actor.selection_mask.sum()) if actor is not None else 0
        self.status.showMessage(
            f'Polygon: {n} points highlighted. Press Delete to remove.', 6000)

    def _do_sel_preview(self):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        if self._sel_preview_kind == 'box':
            mask = self._compute_box_mask(*self._sel_preview_args)
        elif self._sel_preview_kind == 'poly':
            verts = self._sel_preview_args
            if len(verts) < 3:
                return
            mask = self._compute_poly_mask(verts)
        else:
            return
        actor.set_selection(mask)
        self.viewer.refresh()

    def _on_brush_apply(self, center: np.ndarray, radius: float, stroke_start: bool):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        d2 = np.sum((actor.points - center.astype(np.float32)) ** 2, axis=1)
        mask = (d2 <= radius * radius) & actor.visibility_mask
        if not mask.any():
            return
        if stroke_start:
            self._push_history()
        actor.set_selection(mask)
        n = actor.delete_selected()
        if n > 0:
            self.status.showMessage(f'Brush deleted {n} points', 1500)
        self.viewer.refresh()

    def _on_point_pick(self, x: int, y: int):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        screen = self.viewer.project_points_to_screen(actor.points)
        sx, sy, sz = screen[:, 0], screen[:, 1], screen[:, 2]
        valid = np.isfinite(sz) & (sz >= -1.0) & (sz <= 1.0) & actor.visibility_mask
        if not valid.any():
            return
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        d2[~valid] = np.inf
        idx = int(np.argmin(d2))
        if d2[idx] > 100:
            self.status.showMessage('No point near cursor.', 2000)
            return
        mask = np.zeros(len(actor.points), dtype=bool); mask[idx] = True
        actor.set_selection(mask)
        self._push_history()
        actor.delete_selected()
        self.viewer.refresh()
        self.status.showMessage(f'Deleted point #{idx}.', 2000)

    def _on_grid_pick(self, x: int, y: int):
        actor = self._active_actor_get()
        if actor is None:
            return
        center = self.viewer.world_at_pixel(x, y)
        if center is None:
            return
        size = self.grid_size.value(); step = self.grid_step.value()
        if step <= 0 or size <= 0:
            return
        n = max(2, int(round(size / step)) + 1)
        coords = np.linspace(-size / 2.0, size / 2.0, n, dtype=np.float32)
        u, v = np.meshgrid(coords, coords)
        flat = np.stack([u.flatten(), v.flatten()], axis=1)
        plane = self.grid_axis.currentIndex()
        zeros = np.zeros((flat.shape[0],), dtype=np.float32)
        if plane == 0:
            grid_pts = np.stack([flat[:, 0], flat[:, 1], zeros], axis=1)
        elif plane == 1:
            grid_pts = np.stack([flat[:, 0], zeros, flat[:, 1]], axis=1)
        else:
            grid_pts = np.stack([zeros, flat[:, 0], flat[:, 1]], axis=1)
        grid_pts = grid_pts + center.astype(np.float32)
        self._push_history()
        if actor.intensities is not None:
            new_i = np.zeros(grid_pts.shape[0], dtype=np.float32)
        else:
            new_i = None
        actor.add_points(grid_pts, new_i)
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Added grid: {grid_pts.shape[0]} points', 3000)

    def _on_commit_delete(self):
        actor = self._active_actor_get()
        if actor is None or actor.selection_mask is None or not actor.selection_mask.any():
            return
        self._push_history()
        n = actor.delete_selected()
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Deleted {n} points', 3000)

    def _on_clear_selection(self):
        actor = self._active_actor_get()
        if actor is not None:
            actor.clear_selection()
        self.viewer.refresh()

    def _on_select_all_visible(self):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        actor.set_selection(actor.visibility_mask.copy())
        self.viewer.refresh()
        n = int(actor.selection_mask.sum())
        self.status.showMessage(
            f'Selected {n} points in current Z-slice. Press Delete to remove.', 6000)

    def _on_invert_selection(self):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        new_mask = (~actor.selection_mask) & actor.visibility_mask
        actor.set_selection(new_mask)
        self.viewer.refresh()
        n = int(actor.selection_mask.sum())
        self.status.showMessage(f'Inverted: {n} points selected.', 4000)

    # ---------------------------------------------------------------- prism
    def _on_start_prism_draw(self):
        self._on_clear_prism()
        if self._z_world_min < self._z_world_max:
            self.prism_base_z.blockSignals(True)
            self.prism_base_z.setValue(float(self._z_world_min))
            self.prism_base_z.blockSignals(False)
        self._prism_draw_verts = []
        self.prism_state = 'drawing'
        style = PrismDrawStyle(
            self.viewer,
            base_z=self.prism_base_z.value(),
            on_vertex=self._on_prism_vertex,
            on_preview=self._on_prism_preview,
            on_finish=self._on_prism_finish_drawing,
            on_cancel=self._on_prism_cancel,
        )
        self.viewer.set_style(style); self._cur_style = style
        self.edit_mode = 'none'
        self._update_buttons()
        self.status.showMessage(
            'Draw polygon: click vertices in the view. Enter = finish, '
            'Esc = cancel. (Wheel = zoom)', 0)

    def _on_prism_vertex(self, xy: tuple):
        self._prism_draw_verts.append(xy)
        self._update_prism_draw_outline(preview=None)
        self.status.showMessage(
            f'Vertex {len(self._prism_draw_verts)} placed at '
            f'({xy[0]:.2f}, {xy[1]:.2f}). Enter to finish.', 4000)
        self.viewer.refresh()

    def _on_prism_preview(self, xy: Optional[tuple]):
        if not self._prism_draw_verts:
            return
        self._update_prism_draw_outline(preview=xy)
        self.viewer.refresh()

    def _update_prism_draw_outline(self, preview: Optional[tuple]):
        ren = self.viewer.renderer
        if self._prism_draw_actor is not None:
            ren.RemoveActor(self._prism_draw_actor)
            self._prism_draw_actor = None
        if not self._prism_draw_verts:
            return
        z = self.prism_base_z.value()
        verts = list(self._prism_draw_verts)
        if preview is not None:
            verts_full = verts + [preview]
        else:
            verts_full = verts
        pts = vtk.vtkPoints()
        for (x, y) in verts_full:
            pts.InsertNextPoint(float(x), float(y), float(z))
        cell_verts = vtk.vtkCellArray()
        for i in range(len(verts)):
            vx = vtk.vtkVertex(); vx.GetPointIds().SetId(0, i); cell_verts.InsertNextCell(vx)
        lines = vtk.vtkCellArray()
        for i in range(len(verts_full) - 1):
            ln = vtk.vtkLine(); ln.GetPointIds().SetId(0, i); ln.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(ln)
        if len(verts) >= 3:
            ln = vtk.vtkLine(); ln.GetPointIds().SetId(0, len(verts_full) - 1); ln.GetPointIds().SetId(1, 0)
            lines.InsertNextCell(ln)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts); poly.SetLines(lines); poly.SetVerts(cell_verts)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(1.0, 0.85, 0.2)
        prop.SetLineWidth(2.5)
        prop.SetPointSize(9.0)
        prop.SetRenderPointsAsSpheres(True)
        prop.SetLighting(False)
        actor.PickableOff()
        ren.AddActor(actor)
        self._prism_draw_actor = actor

    def _on_prism_finish_drawing(self):
        if len(self._prism_draw_verts) < 3:
            self.status.showMessage(
                f'Polygon needs ≥ 3 vertices (have {len(self._prism_draw_verts)}).', 4000)
            return
        if self._prism_draw_actor is not None:
            self.viewer.renderer.RemoveActor(self._prism_draw_actor)
            self._prism_draw_actor = None
        polygon_xy = np.array(self._prism_draw_verts, dtype=np.float32)
        base_z = self.prism_base_z.value()
        height = self.prism_height.value()
        self.prism.set_polygon(polygon_xy, base_z, height)
        self.prism_state = 'placed'
        self._prism_draw_verts = []
        for sp in (self.prism_tx, self.prism_ty, self.prism_tz):
            sp.blockSignals(True); sp.setValue(0.0); sp.blockSignals(False)
        self.viewer.use_default_style()
        self._cur_style = self.viewer.default_style
        self._update_prism_selection()
        self.status.showMessage(
            f'Prism built — {polygon_xy.shape[0]} vertices, height {height:.2f} m. '
            f'Move with X/Y/Z spinboxes, then "Delete Points in Prism".', 8000)

    def _on_prism_cancel(self):
        self._prism_draw_verts = []
        if self._prism_draw_actor is not None:
            self.viewer.renderer.RemoveActor(self._prism_draw_actor)
            self._prism_draw_actor = None
        self.prism_state = 'idle'
        self.viewer.use_default_style()
        self._cur_style = self.viewer.default_style
        self.viewer.refresh()
        self.status.showMessage('Prism drawing cancelled.', 3000)

    def _on_prism_translate(self, _val=None):
        if self.prism_state != 'placed':
            return
        self.prism.set_offset(
            self.prism_tx.value(),
            self.prism_ty.value(),
            self.prism_tz.value(),
        )
        self._update_prism_selection()

    def _on_prism_param_change(self, _val=None):
        if self.prism_state == 'drawing' and isinstance(self._cur_style, PrismDrawStyle):
            self._cur_style.base_z = self.prism_base_z.value()
            self._update_prism_draw_outline(preview=None)
            self.viewer.refresh()
        elif self.prism_state == 'placed':
            self.prism.update_base_z(self.prism_base_z.value())
            self.prism.update_height(self.prism_height.value())
            self._update_prism_selection()

    def _update_prism_selection(self):
        if not self.prism.has_prism():
            return
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        mask = self.prism.compute_mask(actor.points, actor.visibility_mask)
        actor.set_selection(mask)
        self.viewer.refresh()
        n = int(mask.sum())
        z_lo, z_hi = self.prism.world_z_range()
        name = self._slot_display(self.active_sid) if self.active_sid else '—'
        self.status.showMessage(
            f'{n} points inside prism (Z {z_lo:.2f}…{z_hi:.2f} m, {name}).', 2500)

    def _on_delete_in_prism(self):
        if not self.prism.has_prism():
            self.status.showMessage('No prism placed yet.', 3000)
            return
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        mask = self.prism.compute_mask(actor.points, actor.visibility_mask)
        n = int(mask.sum())
        if n == 0:
            self.status.showMessage(
                'No points inside prism (nothing visible there at the moment).', 4000)
            return
        self._push_history()
        actor.set_selection(mask)
        actor.delete_selected()
        self._refresh_z_bounds()
        self.viewer.refresh()
        name = self._slot_display(self.active_sid) if self.active_sid else '—'
        self.status.showMessage(f'Deleted {n} points (inside prism, {name}).', 5000)

    def _on_clear_prism(self):
        self.prism.clear()
        self.prism_state = 'idle'
        self._prism_draw_verts = []
        if self._prism_draw_actor is not None:
            self.viewer.renderer.RemoveActor(self._prism_draw_actor)
            self._prism_draw_actor = None
        actor = self._active_actor_get()
        if actor is not None:
            actor.clear_selection()
        self.viewer.refresh()
        self.status.showMessage('Prism cleared.', 2000)

    # ---------------------------------------------------------------- undo/redo
    def _on_undo(self):
        slot = self._active_slot()
        if slot is None:
            return
        cur = slot.actor.to_data(name=slot.name)
        prev = slot.history.do_undo(cur)
        if prev is None:
            return
        slot.actor.from_data(prev)
        self._update_buttons()
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Undo ({slot.name})', 2000)

    def _on_redo(self):
        slot = self._active_slot()
        if slot is None:
            return
        cur = slot.actor.to_data(name=slot.name)
        nxt = slot.history.do_redo(cur)
        if nxt is None:
            return
        slot.actor.from_data(nxt)
        self._update_buttons()
        self._refresh_z_bounds()
        self.viewer.refresh()
        self.status.showMessage(f'Redo ({slot.name})', 2000)

    def _on_brush_radius_changed(self, val: float):
        self._brush_radius = float(val)
        st = getattr(self, '_cur_style', None)
        if isinstance(st, SphereBrushStyle):
            st.set_radius(val)

    # ---------------------------------------------------------------- z-clip
    def _refresh_z_bounds(self):
        zs = []
        for s in self.slots.values():
            actor = s.actor
            if len(actor.points) > 0:
                zs.append((float(actor.points[:, 2].min()),
                           float(actor.points[:, 2].max())))
        if not zs:
            self._z_world_min, self._z_world_max = 0.0, 1.0
        else:
            self._z_world_min = min(z[0] for z in zs)
            self._z_world_max = max(z[1] for z in zs)
        eps = max(1e-3, (self._z_world_max - self._z_world_min) * 1e-4)
        self._z_world_min -= eps
        self._z_world_max += eps
        self.viewer.z_slider.set_value_range(self._z_world_min, self._z_world_max, unit='m')
        self._apply_z_filter_now()

    def _on_z_range_changed(self, low: float, high: float):
        self._pending_low = low
        self._pending_high = high
        self._zfilter_timer.start()

    def _apply_z_filter_now(self):
        low = self._pending_low; high = self._pending_high
        zmin = self._z_world_min + low * (self._z_world_max - self._z_world_min)
        zmax = self._z_world_min + high * (self._z_world_max - self._z_world_min)
        for s in self.slots.values():
            actor = s.actor
            if len(actor.points) > 0:
                actor.set_visibility_z_range(zmin, zmax)
        self._proj_actor = None
        self._proj_visible_idx = None
        self._proj_screen = None
        self.viewer.refresh()
        if low > 0.0 or high < 1.0:
            self.status.showMessage(
                f'Z-clip: {zmin:.3f} m … {zmax:.3f} m  '
                f'(slice {(high-low)*100:.0f} %)', 1500)

    def _on_z_reset(self):
        self._pending_low = 0.0
        self._pending_high = 1.0
        for s in self.slots.values():
            s.actor.show_all()
        self.viewer.refresh()
        self.status.showMessage('Z-clip reset (full range)', 2000)

    # ---------------------------------------------------------------- save
    def _on_save_active(self):
        actor = self._active_actor_get()
        if actor is None or len(actor.points) == 0:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save PCD', '', 'PCD (*.pcd)')
        if not path:
            return
        if not path.endswith('.pcd'):
            path += '.pcd'
        save_pcd(path, actor.points, actor.intensities)
        self.status.showMessage(f'Saved {len(actor.points):,} points → {path}', 4000)

    def _on_save_merged(self):
        # Combine ALL visible slots' points into one PCD.
        parts_pts = []
        parts_int = []
        names = []
        for s in self.slots.values():
            if not s.actor.actor.GetVisibility():
                continue
            if len(s.actor.points) == 0:
                continue
            parts_pts.append(s.actor.points)
            parts_int.append(s.actor.intensities)
            names.append(s.name)
        if not parts_pts:
            QtWidgets.QMessageBox.information(
                self, 'Nothing to save',
                'No visible slots contain points. Toggle slots on or load data.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save merged PCD',
            'merged.pcd', 'PCD (*.pcd)')
        if not path:
            return
        if not path.endswith('.pcd'):
            path += '.pcd'
        pts = np.concatenate(parts_pts, axis=0).astype(np.float32)
        if all(i is not None for i in parts_int) and len(parts_int) > 0:
            intens = np.concatenate(parts_int, axis=0).astype(np.float32)
        else:
            intens = None
        try:
            save_pcd(path, pts, intens)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Save failed', str(e))
            return
        self.status.showMessage(
            f'Saved {len(pts):,} points from {len(parts_pts)} slot(s) → {path}', 8000)
        QtWidgets.QMessageBox.information(
            self, 'Saved',
            f'Merged {len(parts_pts)} slot(s):\n  • '
            + '\n  • '.join(names)
            + f'\n\nTotal: {len(pts):,} points\n→ {path}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_actor(extent: float = 200.0,
                     step: float = 1.0,
                     major_step: float = 10.0) -> 'vtk.vtkActor':
    """RViz-style ground grid on the XY plane.
       - minor lines every `step` (dim grey),
       - major lines every `major_step` (lighter grey),
       - X axis line in red, Y axis line in green.
    """
    half = extent / 2.0
    n = int(round(extent / step)) + 1
    coords = np.linspace(-half, half, n, dtype=np.float32)

    minor = np.array([55, 60, 68], dtype=np.uint8)
    major = np.array([95, 105, 118], dtype=np.uint8)
    axis_x = np.array([210, 70, 70], dtype=np.uint8)
    axis_y = np.array([70, 180, 75], dtype=np.uint8)

    pts: list = []
    cols: list = []

    eps = step * 0.4

    # vertical lines: parallel to Y, at x = c
    for c in coords:
        c_f = float(c)
        if abs(c_f) < eps:
            color = axis_y  # this line goes through origin along Y → Y axis
        elif abs(c_f / major_step - round(c_f / major_step)) < 0.01:
            color = major
        else:
            color = minor
        pts.append((c_f, -half, 0.0))
        pts.append((c_f,  half, 0.0))
        cols.append(color); cols.append(color)

    # horizontal lines: parallel to X, at y = c
    for c in coords:
        c_f = float(c)
        if abs(c_f) < eps:
            color = axis_x  # this line goes through origin along X → X axis
        elif abs(c_f / major_step - round(c_f / major_step)) < 0.01:
            color = major
        else:
            color = minor
        pts.append((-half, c_f, 0.0))
        pts.append(( half, c_f, 0.0))
        cols.append(color); cols.append(color)

    pts_np = np.asarray(pts, dtype=np.float32)
    cols_np = np.asarray(cols, dtype=np.uint8)

    from vtkmodules.util import numpy_support as ns

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(pts_np),
                                    deep=True, array_type=vtk.VTK_FLOAT))

    n_lines = pts_np.shape[0] // 2
    ids = np.empty((n_lines, 3), dtype=np.int64)
    ids[:, 0] = 2
    ids[:, 1] = np.arange(n_lines) * 2
    ids[:, 2] = np.arange(n_lines) * 2 + 1
    cells = vtk.vtkCellArray()
    cells.SetCells(n_lines, ns.numpy_to_vtkIdTypeArray(ids.flatten(), deep=True))

    color_arr = ns.numpy_to_vtk(np.ascontiguousarray(cols_np),
                                deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    color_arr.SetName('Colors')

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(cells)
    poly.GetPointData().SetScalars(color_arr)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(1.0)
    actor.GetProperty().SetLighting(False)
    actor.GetProperty().SetOpacity(0.85)
    actor.PickableOff()  # grid shouldn't be pickable
    return actor


class Prism3D:
    """Extruded polygon (prism) used for 3D selection.

    - polygon_xy: base polygon in XY (in world coords)
    - base_z   : Z height of the polygon plane
    - height   : extrusion height (along +Z)
    - offset   : translation applied on top (mutable via spinboxes)

    Renders a transparent gold prism with three world-axis arrows at its
    centroid (visual indicator that it can be moved via the X/Y/Z spinboxes).
    """

    PRISM_COLOR = (1.0, 0.78, 0.20)
    EDGE_COLOR  = (1.0, 0.55, 0.0)

    def __init__(self, viewer: 'Viewer'):
        self.viewer = viewer
        self.polygon_xy: Optional[np.ndarray] = None  # (N,2) base polygon
        self.base_z: float = 0.0
        self.height: float = 1.0
        self.offset: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.actor: Optional[vtk.vtkActor] = None
        self.gizmo_actors: list = []

    def has_prism(self) -> bool:
        return self.polygon_xy is not None and self.actor is not None

    def set_polygon(self, polygon_xy: np.ndarray, base_z: float, height: float):
        self.polygon_xy = np.asarray(polygon_xy, dtype=np.float32)
        self.base_z = float(base_z)
        self.height = max(1e-3, float(height))
        self.offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._rebuild()

    def update_height(self, h: float):
        self.height = max(1e-3, float(h))
        if self.polygon_xy is not None:
            self._rebuild()

    def update_base_z(self, z: float):
        self.base_z = float(z)
        if self.polygon_xy is not None:
            self._rebuild()

    def set_offset(self, x: float, y: float, z: float):
        self.offset = np.array([float(x), float(y), float(z)], dtype=np.float32)
        if self.polygon_xy is not None:
            self._rebuild()

    def world_polygon(self) -> np.ndarray:
        return self.polygon_xy + self.offset[:2]

    def world_z_range(self) -> tuple:
        z0 = self.base_z + float(self.offset[2])
        return z0, z0 + self.height

    def compute_mask(self, points: np.ndarray, visibility_mask: np.ndarray) -> np.ndarray:
        if not self.has_prism() or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        poly_world = self.world_polygon()
        z_lo, z_hi = self.world_z_range()
        zs = points[:, 2]
        in_z = (zs >= z_lo) & (zs <= z_hi) & visibility_mask
        if not in_z.any():
            return np.zeros(len(points), dtype=bool)
        # only test x/y for points already passing z + visibility (much faster)
        idx = np.where(in_z)[0]
        in_xy = _points_in_polygon(points[idx, 0], points[idx, 1], poly_world)
        out = np.zeros(len(points), dtype=bool)
        out[idx[in_xy]] = True
        return out

    def clear(self):
        if self.actor is not None:
            self.viewer.renderer.RemoveActor(self.actor)
            self.actor = None
        for a in self.gizmo_actors:
            self.viewer.renderer.RemoveActor(a)
        self.gizmo_actors = []
        self.polygon_xy = None
        self.viewer.refresh()

    def _rebuild(self):
        # remove old visuals
        if self.actor is not None:
            self.viewer.renderer.RemoveActor(self.actor)
            self.actor = None
        for a in self.gizmo_actors:
            self.viewer.renderer.RemoveActor(a)
        self.gizmo_actors = []

        if self.polygon_xy is None or len(self.polygon_xy) < 3:
            return

        n = len(self.polygon_xy)
        poly_world = self.world_polygon()
        z_lo, z_hi = self.world_z_range()

        pts = vtk.vtkPoints()
        for i in range(n):
            pts.InsertNextPoint(float(poly_world[i, 0]),
                                float(poly_world[i, 1]),
                                float(z_lo))
        polygon_cell = vtk.vtkPolygon()
        polygon_cell.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            polygon_cell.GetPointIds().SetId(i, i)
        polys = vtk.vtkCellArray()
        polys.InsertNextCell(polygon_cell)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(pts)
        poly_data.SetPolys(polys)

        ext = vtk.vtkLinearExtrusionFilter()
        ext.SetInputData(poly_data)
        ext.SetExtrusionTypeToVectorExtrusion()
        ext.SetVector(0.0, 0.0, float(self.height))
        ext.SetCapping(True)

        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(ext.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(triangulate.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(*self.PRISM_COLOR)
        prop.SetOpacity(0.22)
        prop.SetEdgeVisibility(True)
        prop.SetEdgeColor(*self.EDGE_COLOR)
        prop.SetLineWidth(2.0)
        prop.SetLighting(False)
        self.actor.PickableOff()
        self.viewer.renderer.AddActor(self.actor)

        # axis indicator gizmo at centroid
        cx = float(np.mean(poly_world[:, 0]))
        cy = float(np.mean(poly_world[:, 1]))
        cz = (z_lo + z_hi) / 2.0
        bbox = poly_world.max(axis=0) - poly_world.min(axis=0)
        L = max(float(np.linalg.norm(bbox)) * 0.4, max(self.height, 1.0))
        for axis_vec, color in (
            (np.array([1.0, 0.0, 0.0]), (1.0, 0.30, 0.30)),
            (np.array([0.0, 1.0, 0.0]), (0.30, 1.0, 0.30)),
            (np.array([0.0, 0.0, 1.0]), (0.40, 0.65, 1.0)),
        ):
            a = self._make_arrow_actor(np.array([cx, cy, cz]), axis_vec, L, color)
            self.viewer.renderer.AddActor(a)
            self.gizmo_actors.append(a)

        self.viewer.refresh()

    def _make_arrow_actor(self, origin: np.ndarray, axis: np.ndarray,
                          length: float, color: tuple) -> vtk.vtkActor:
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(20)
        arrow.SetShaftResolution(20)
        arrow.SetTipLength(0.30)
        arrow.SetTipRadius(0.10)
        arrow.SetShaftRadius(0.04)

        axis = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / n

        x_axis = np.array([1.0, 0.0, 0.0])
        transform = vtk.vtkTransform()
        transform.Translate(float(origin[0]), float(origin[1]), float(origin[2]))
        cross = np.cross(x_axis, axis)
        cn = float(np.linalg.norm(cross))
        if cn > 1e-9:
            ang = float(np.degrees(np.arccos(np.clip(np.dot(x_axis, axis), -1.0, 1.0))))
            transform.RotateWXYZ(ang, float(cross[0]/cn), float(cross[1]/cn), float(cross[2]/cn))
        elif np.dot(x_axis, axis) < 0:
            transform.RotateY(180)
        transform.Scale(length, length, length)

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(transform)
        tf_filter.SetInputConnection(arrow.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tf_filter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetLighting(False)
        prop.SetOpacity(0.95)
        actor.PickableOff()
        return actor


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon. xs, ys: (N,). poly: (M,2)."""
    n = xs.shape[0]
    m = poly.shape[0]
    inside = np.zeros(n, dtype=bool)
    j = m - 1
    for i in range(m):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        cond = ((yi > ys) != (yj > ys))
        denom = (yj - yi)
        denom = denom if abs(denom) > 1e-12 else 1e-12
        x_at = (xj - xi) * (ys - yi) / denom + xi
        cond &= (xs < x_at)
        inside ^= cond
        j = i
    return inside


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MergeApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
