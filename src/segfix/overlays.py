"""Lightweight Qt widgets painted over the vispy canvas.

These sit on top of :attr:`CloudView.native` as plain child ``QWidget``s (the
same trick :mod:`segfix.lasso` uses for its drag outline) rather than as vispy
visuals — Qt text and crisp 1px rules are far easier this way, and nothing
here needs to be part of the 3-D scene.
"""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen
from qtpy.QtWidgets import QWidget

_AXIS_COLORS = (QColor("#e08080"), QColor("#86c98a"), QColor("#7fb3e0"))
_AXIS_LABELS = ("X", "Y", "Z")


class ScaleBarOverlay(QWidget):
    """A metric scale bar and an orientation axis-tripod over the canvas'
    bottom-left corner.

    Read-only: transparent to mouse events and repainted whenever the canvas
    redraws (which is whenever the camera moves), so both the bar length and
    the tripod stay in step with the live view.
    """

    _MARGIN = 10
    _PANEL_W = 188
    _PANEL_H = 92
    _TARGET_PX = 96  # rough on-screen bar length aimed for before rounding
    _ARM_PX = 20  # axis-tripod arm length

    def __init__(self, view):
        super().__init__(view.native)
        self._view = view
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._sync_geometry()
        view.canvas.events.resize.connect(lambda _e=None: self._sync_geometry())
        view.canvas.events.draw.connect(lambda _e=None: self.update())
        self.show()
        self.raise_()

    def _sync_geometry(self) -> None:
        n = self._view.native
        self.setGeometry(0, 0, n.width(), n.height())

    # -- scale maths ---------------------------------------------------
    def _projected_basis(self):
        """Screen-pixel images of the camera centre and the three world unit
        axes, or ``None`` if any land behind the camera."""
        center = np.asarray(self._view.view.camera.center, dtype=float)
        pts = np.vstack([center, center + np.eye(3)])
        xy, valid = self._view.project_to_canvas(pts)
        if not bool(np.all(valid)):
            return None
        return xy[0], xy[1:] - xy[0]

    @staticmethod
    def _metres_per_pixel(deltas: np.ndarray) -> float | None:
        """World metres spanned by one screen pixel.

        Orthographic projection of a rotated orthonormal 3-frame: the squared
        lengths of the three projected axis vectors sum to ``2 / mpp**2``
        (projecting to 2-D drops exactly one unit of squared length),
        whatever the camera orientation — no dependence on ``scale_factor``.
        """
        sumsq = float((deltas ** 2).sum())
        if sumsq <= 1e-9:
            return None
        return math.sqrt(2.0 / sumsq)

    @staticmethod
    def _nice_length(raw: float) -> float:
        """Round ``raw`` metres to the nearest 1/2/5 × 10ⁿ."""
        if raw <= 0:
            return 0.0
        exp = math.floor(math.log10(raw))
        base = raw / (10 ** exp)
        nice = 1 if base < 1.5 else 2 if base < 3.5 else 5 if base < 7.5 else 10
        return nice * (10 ** exp)

    @staticmethod
    def _format_length(metres: float) -> str:
        if metres >= 1000:
            return f"{metres / 1000:g} km"
        if metres >= 1:
            return f"{metres:g} m"
        if metres >= 0.01:
            return f"{metres * 100:g} cm"
        return f"{metres * 1000:g} mm"

    # -- painting ----------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        basis = self._projected_basis()
        if basis is None:
            return
        _, deltas = basis
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        panel = QRectF(
            self._MARGIN,
            h - self._MARGIN - self._PANEL_H,
            self._PANEL_W,
            self._PANEL_H,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 20, 20, 130))
        painter.drawRoundedRect(panel, 5, 5)

        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        painter.setFont(font)

        self._paint_axes(
            painter, panel.left() + 30, panel.top() + 32, deltas
        )
        mpp = self._metres_per_pixel(deltas)
        if mpp:
            self._paint_scale_bar(
                painter, panel.left() + 16, panel.top() + 74, mpp
            )

    def _paint_axes(self, painter, ox, oy, deltas: np.ndarray) -> None:
        for i in range(3):
            d = deltas[i]
            n = float(np.hypot(d[0], d[1]))
            if n < 1e-6:
                continue
            dx, dy = d[0] / n * self._ARM_PX, d[1] / n * self._ARM_PX
            pen = QPen(_AXIS_COLORS[i])
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(ox, oy), QPointF(ox + dx, oy + dy))
            painter.drawText(
                QPointF(ox + dx * 1.35 - 4, oy + dy * 1.35 + 4), _AXIS_LABELS[i]
            )

    def _paint_scale_bar(self, painter, x, y, mpp: float) -> None:
        length_m = self._nice_length(self._TARGET_PX * mpp)
        if length_m <= 0:
            return
        px = length_m / mpp
        pen = QPen(QColor("#e8e8e8"))
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, y), QPointF(x + px, y))
        painter.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        painter.drawLine(QPointF(x + px, y - 4), QPointF(x + px, y + 4))
        painter.drawText(QPointF(x, y - 8), self._format_length(length_m))
