"""Freehand 3D lasso selection for the point cloud.

There is no built-in lasso for a 3D point cloud, so we implement one:

1. While the tool is *armed*, camera rotation/pan/zoom is disabled so a
   left-drag draws a polygon instead of spinning the view.
2. As the user drags, the screen-space path is painted by a transparent Qt
   overlay sitting on top of the canvas.
3. On release we project every 3D point to the same canvas pixel space using
   the live camera transform and test which projections fall inside the drawn
   polygon.  Those points become the new selection.

Because the test is purely in screen space, the lasso grabs every point whose
projection lands inside the outline — including points occluded behind others.
For tree clouds that is usually what you want (sweep a whole stem/canopy
through the foliage); hold Shift to add to an existing selection.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
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
    view: the :class:`~segfix.cloudview.CloudView`.
    on_select: callback ``(indices: np.ndarray, additive: bool) -> None``
        invoked when a lasso completes, with the selected point indices and
        whether Shift asked to add to the current selection.
    min_points: a drag shorter than this many vertices is treated as a click
        (selection cleared), not a lasso.
    """

    def __init__(self, view, on_select, min_points: int = 3):
        self.view = view
        self.on_select = on_select
        self.min_points = min_points
        self._armed = False
        self._overlay = None
        self._path: list[tuple[float, float]] | None = None
        self._additive = False
        self._canvas = view.canvas

    # -- arm / disarm ---------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    def _ensure_overlay(self):
        if self._overlay is None:
            # Parent the overlay to the canvas widget itself, so its (0, 0)
            # is the canvas' top-left — the same origin as the vispy mouse
            # ``event.pos`` and the visual→canvas projection. Parenting it to
            # an ancestor instead shifts the drawn outline by the height of
            # whatever docks sit above the canvas.
            self._overlay = _LassoOverlay(self.view.native)
        return self._overlay

    def set_armed(self, armed: bool) -> None:
        if armed == self._armed:
            return
        self._armed = armed
        if armed:
            self.view.set_camera_interactive(False)
            self._connect(True)
            self.view.status = "Lasso armed — drag to select (Shift to add)"
        else:
            self._connect(False)
            self.view.set_camera_interactive(True)
            if self._overlay is not None:
                self._overlay.clear()
            self._path = None
            self.view.status = "Lasso off"

    def toggle(self) -> bool:
        self.set_armed(not self._armed)
        return self._armed

    def reassert(self) -> None:
        """Re-apply the camera lock while armed (kept for call-site parity;
        with vispy the camera doesn't get silently re-enabled, so this is
        just a cheap idempotent nudge)."""
        if self._armed:
            self.view.set_camera_interactive(False)

    def _connect(self, on: bool) -> None:
        events = self._canvas.events
        pairs = (
            (events.mouse_press, self._on_press),
            (events.mouse_move, self._on_move),
            (events.mouse_release, self._on_release),
        )
        for signal, cb in pairs:
            try:
                signal.disconnect(cb)
            except (ValueError, TypeError):
                pass
            if on:
                signal.connect(cb)

    # -- drag handling --------------------------------------------------
    @staticmethod
    def _pos(event) -> tuple[float, float]:
        p = event.pos
        return float(p[0]), float(p[1])

    def _on_press(self, event) -> None:
        if not self._armed or event.button != 1:
            return
        self._additive = "Shift" in getattr(event, "modifiers", ())
        self._path = [self._pos(event)]
        overlay = self._ensure_overlay()
        native = self.view.native
        overlay.setGeometry(0, 0, native.width(), native.height())
        overlay.show()
        event.handled = True

    def _on_move(self, event) -> None:
        if not self._armed or self._path is None:
            return
        self._path.append(self._pos(event))
        self._ensure_overlay().set_path(self._path)
        event.handled = True

    def _on_release(self, event) -> None:
        if not self._armed or self._path is None:
            return
        path = np.asarray(self._path, dtype=np.float64)
        self._path = None
        if self._overlay is not None:
            self._overlay.clear()
        self._finish(path, self._additive)
        event.handled = True

    def _finish(self, path: np.ndarray, additive: bool) -> None:
        if len(path) < self.min_points:
            if not additive:
                self.on_select(np.empty(0, dtype=np.int64), additive=False)
            return
        coords = self.view.coords
        canvas_xy, valid = self.view.project_to_canvas(coords)
        inside = points_in_polygon(path, canvas_xy) & valid
        # Hidden points (a hide checkbox, "show unassigned", or either section
        # tool — all of which set view.shown) are unselectable.
        shown = np.asarray(self.view.shown, dtype=bool)
        if shown.shape[0] == inside.shape[0]:
            inside &= shown
        indices = np.flatnonzero(inside)
        self.on_select(indices, additive=additive)
