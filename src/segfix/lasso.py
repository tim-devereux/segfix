"""Freehand 3D lasso selection for a napari Points layer.

napari has no built-in lasso for 3D point clouds, so we implement one:

1. While the tool is *armed*, camera rotation/pan/zoom is disabled so a
   left-drag draws a polygon instead of spinning the view.
2. As the user drags, the screen-space path is painted by a transparent Qt
   overlay sitting on top of the canvas.
3. On release we project every 3D point to the same canvas pixel space using
   napari's live camera transform and test which projections fall inside the
   drawn polygon.  Those points become the new selection.

Because the test is purely in screen space, the lasso grabs every point whose
projection lands inside the outline — including points occluded behind others.
For tree clouds that is usually what you want (sweep a whole stem/canopy
through the foliage); hold the modifier to add to an existing selection.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from qtpy.QtCore import QPointF
from qtpy.QtWidgets import QWidget


# -- geometry -----------------------------------------------------------
def points_in_polygon(polygon: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test.

    Parameters
    ----------
    polygon: ``(M, 2)`` array of the closed outline vertices (x, y).
    pts: ``(N, 2)`` array of query points.

    Returns a boolean ``(N,)`` mask of points strictly inside the polygon.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if len(polygon) < 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)

    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)

    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    # For each edge, flip `inside` for points whose horizontal ray crosses it.
    for ex1, ey1, ex2, ey2 in zip(x1, y1, x2, y2):
        cond = (ey1 > y) != (ey2 > y)
        # x-coordinate of the edge at height y (guard against horizontal edges)
        denom = ey2 - ey1
        denom = denom if denom != 0 else np.finfo(np.float64).eps
        x_cross = (ex2 - ex1) * (y - ey1) / denom + ex1
        inside ^= cond & (x < x_cross)
    return inside


# -- projection ---------------------------------------------------------
def project_to_canvas(viewer, layer, coords: np.ndarray):
    """Project ``coords`` (N,3 data coords) to canvas pixels via the camera.

    Returns ``(canvas_xy, valid)`` where ``canvas_xy`` is ``(N, 2)`` and
    ``valid`` masks out points behind the camera (non-positive homogeneous
    w), whose projection is meaningless.
    """
    # The vispy node holds the displayed axes in reversed (xyz) order — match
    # that layout before mapping through the visual→canvas transform, so the
    # projection is correct whatever viewer.dims.order is set to.
    disp = list(viewer.dims.displayed)
    visual = np.asarray(coords, dtype=np.float64)[:, disp][:, ::-1]
    node = viewer.window._qt_viewer.canvas.layer_to_visual[layer].node
    transform = node.get_transform(map_from="visual", map_to="canvas")
    mapped = np.asarray(transform.map(visual))
    w = mapped[:, 3]
    valid = w > 0
    w_safe = np.where(valid, w, 1.0)
    canvas_xy = mapped[:, :2] / w_safe[:, None]
    return canvas_xy, valid


# -- Qt overlay ---------------------------------------------------------
class _LassoOverlay(QWidget):
    """Transparent widget that paints the in-progress lasso path."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._path: list[tuple[float, float]] = []
        self.hide()

    def set_path(self, path) -> None:
        self._path = list(path)
        self.update()

    def clear(self) -> None:
        self._path = []
        self.hide()
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        if len(self._path) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Scale canvas-pixel coords to this widget's logical coords (HiDPI).
        parent = self.parent()
        sw = self.width() / max(parent.width(), 1)
        sh = self.height() / max(parent.height(), 1)
        poly = QPolygonF([QPointF(x * sw, y * sh) for x, y in self._path])

        fill = QPainterPath()
        fill.addPolygon(poly)
        fill.closeSubpath()
        painter.fillPath(fill, QColor(255, 220, 0, 40))

        pen = QPen(QColor(255, 220, 0, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPolyline(poly)
        painter.drawLine(poly.last(), poly.first())  # closing segment


# -- tool ---------------------------------------------------------------
class LassoTool:
    """Manages the armed state, drawing, and selection for the lasso.

    Parameters
    ----------
    viewer: the napari Viewer.
    layer: the Points layer to select within.
    on_select: callback ``(indices: np.ndarray, additive: bool) -> None``
        invoked when a lasso completes, with the selected point indices and
        whether the shift modifier asked to add to the current selection.
    min_points: a drag shorter than this many vertices is treated as a click
        (selection cleared), not a lasso.
    """

    def __init__(self, viewer, layer, on_select, min_points: int = 3):
        self.viewer = viewer
        self.layer = layer
        self.on_select = on_select
        self.min_points = min_points
        self._armed = False
        self._saved_pan = None
        self._saved_zoom = None
        self._saved_mode = None
        self._overlay = None

    # -- arm / disarm ---------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    def _ensure_overlay(self):
        if self._overlay is None:
            native = self.viewer.window._qt_viewer.canvas.native
            self._overlay = _LassoOverlay(native.parent() or native)
        return self._overlay

    def set_armed(self, armed: bool) -> None:
        if armed == self._armed:
            return
        self._armed = armed
        cam = self.viewer.camera
        if armed:
            # Disable camera interaction so the drag draws instead of rotating.
            self._saved_pan, self._saved_zoom = cam.mouse_pan, cam.mouse_zoom
            cam.mouse_pan = False
            cam.mouse_zoom = False
            # Take the layer out of select mode: its own drag handler clears
            # the selection on press and overwrites it on release, which would
            # silently discard whatever the lasso selected.
            self._saved_mode = self.layer.mode
            self.layer.mode = "pan_zoom"
            if self._on_drag not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._on_drag)
            self.viewer.status = "Lasso armed — drag to select (Shift to add)"
        else:
            if self._saved_pan is not None:
                cam.mouse_pan = self._saved_pan
                cam.mouse_zoom = self._saved_zoom
            if self._saved_mode is not None:
                self.layer.mode = self._saved_mode
            if self._on_drag in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self._on_drag)
            if self._overlay is not None:
                self._overlay.clear()
            self.viewer.status = "Lasso off"

    def toggle(self) -> bool:
        self.set_armed(not self._armed)
        return self._armed

    def reassert(self) -> None:
        """Re-apply the camera lock while armed.

        napari re-enables camera pan/zoom whenever the active layer changes
        (e.g. a marker layer is added or removed), which would leave both the
        lasso drag and camera movement live at once.
        """
        if self._armed:
            self.viewer.camera.mouse_pan = False
            self.viewer.camera.mouse_zoom = False

    # -- drag handling --------------------------------------------------
    def _on_drag(self, viewer, event):
        if not self._armed:
            return
        overlay = self._ensure_overlay()
        native = viewer.window._qt_viewer.canvas.native
        overlay.setGeometry(0, 0, native.width(), native.height())
        overlay.show()

        path = [tuple(event.pos)]
        additive = "Shift" in getattr(event, "modifiers", ())
        yield  # wait for movement

        while event.type == "mouse_move":
            path.append(tuple(event.pos))
            overlay.set_path(path)
            yield

        overlay.clear()
        self._finish(np.asarray(path, dtype=np.float64), additive)

    def _finish(self, path: np.ndarray, additive: bool) -> None:
        if len(path) < self.min_points:
            if not additive:
                self.on_select(np.empty(0, dtype=np.int64), additive=False)
            return
        coords = self.layer.data
        canvas_xy, valid = project_to_canvas(self.viewer, self.layer, coords)
        inside = points_in_polygon(path, canvas_xy) & valid
        # Hidden points (a hide checkbox, "show unassigned", or either
        # section tool — all of which set layer.shown) are unselectable.
        shown = np.asarray(self.layer.shown, dtype=bool)
        if shown.shape[0] == inside.shape[0]:
            inside &= shown
        indices = np.flatnonzero(inside)
        self.on_select(indices, additive=additive)
