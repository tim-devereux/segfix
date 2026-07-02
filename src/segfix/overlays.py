"""Context overlays around the focus tree: unsegmented and removed points.

These mirror the plugin's cylinder overlays.  They are added as separate,
dimmed napari Points layers so they read as *context* — you can see stray
points the segmentation missed, or points removed in an earlier pass, without
them being part of the editable tree cloud.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from . import io

NON_SEG_COLOR = (1.0, 0.4, 0.1, 1.0)   # orange
REMOVED_COLOR = (0.9, 0.1, 0.5, 1.0)   # magenta


def _cylinder_mask(xy: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    return (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 <= radius**2


def load_non_seg_overlay(viewer, project, center_xy, radius: float = 10.0) -> int:
    """Add unsegmented points (``*_non_seg.ply``) within an XY cylinder."""
    files = glob.glob(str(Path(project.data_directory) / "*_non_seg.ply"))
    if not files:
        return 0
    coords = io.read_xyz(files[0])
    if coords.shape[0] == 0:
        return 0
    cx, cy = center_xy
    pts = coords[_cylinder_mask(coords[:, :2], cx, cy, radius)]
    if pts.shape[0] == 0:
        return 0
    _replace_layer(viewer, "non_seg", pts, NON_SEG_COLOR)
    return int(pts.shape[0])


def load_removed_overlay(viewer, project, center_xy, radius: float = 10.0) -> int:
    """Add previously removed points (``removed_points.xyz``) in a cylinder."""
    path = Path(project.data_directory) / "removed_points.xyz"
    if not path.exists():
        return 0
    rows = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    if rows.shape[0] == 0:
        return 0
    cx, cy = center_xy
    pts = rows[_cylinder_mask(rows[:, :2], cx, cy, radius)]
    if pts.shape[0] == 0:
        return 0
    _replace_layer(viewer, "removed", pts.astype(np.float32), REMOVED_COLOR)
    return int(pts.shape[0])


def _replace_layer(viewer, name, pts, color):
    if name in viewer.layers:
        viewer.layers.remove(name)
    layer = viewer.add_points(
        pts, name=name, size=1.0, face_color=[color], border_width=0,
        opacity=0.5, shading="none",
    )
    # Overlays are context only — never the active editing target.
    layer.editable = False
    return layer
