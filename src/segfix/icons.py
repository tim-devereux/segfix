"""Programmatically drawn icons for the segfix buttons.

Icons are painted with QPainter at runtime rather than shipped as files:
they stay crisp at any DPI, share one visual style, and adding one is a few
lines of path code.  All drawing happens in a 32×32 coordinate space; colours
are chosen for napari's dark theme.
"""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

FG = QColor("#d8d8d8")
GREEN = QColor("#86c98a")
RED = QColor("#e08080")
BLUE = QColor("#7fb3e0")
AMBER = QColor("#e0c060")


def _pen(color=FG, width=2.4, dashed=False) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    if dashed:
        pen.setStyle(Qt.DashLine)
    return pen


def _arrowhead(p: QPainter, tip, left, right, color=FG) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygonF([QPointF(*tip), QPointF(*left), QPointF(*right)]))


def _lasso(p: QPainter) -> None:
    path = QPainterPath(QPointF(9, 19))
    path.cubicTo(QPointF(3, 10), QPointF(12, 3), QPointF(19, 5))
    path.cubicTo(QPointF(28, 7), QPointF(29, 15), QPointF(23, 18))
    path.cubicTo(QPointF(18, 21), QPointF(12, 22), QPointF(9, 19))
    p.drawPath(path)
    p.drawLine(QPointF(9, 19), QPointF(16, 28))  # rope tail


def _target(p: QPainter) -> None:
    p.drawEllipse(QRectF(8, 8, 16, 16))
    for a, b in [((16, 3), (16, 9)), ((16, 23), (16, 29)),
                 ((3, 16), (9, 16)), ((23, 16), (29, 16))]:
        p.drawLine(QPointF(*a), QPointF(*b))
    p.setPen(Qt.NoPen)
    p.setBrush(AMBER)
    p.drawEllipse(QPointF(16, 16), 2.5, 2.5)


def _reassign(p: QPainter) -> None:
    p.drawLine(QPointF(4, 16), QPointF(16, 16))
    _arrowhead(p, (20, 16), (14, 12), (14, 20))
    p.setPen(_pen(BLUE))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(19, 11, 10, 10))


def _new_tree(p: QPainter) -> None:
    p.setPen(_pen(GREEN))
    p.drawEllipse(QRectF(6, 6, 20, 20))
    p.drawLine(QPointF(16, 11), QPointF(16, 21))
    p.drawLine(QPointF(11, 16), QPointF(21, 16))


def _merge(p: QPainter) -> None:
    p.setPen(_pen(BLUE))
    p.drawEllipse(QRectF(4, 9, 14, 14))
    p.drawEllipse(QRectF(14, 9, 14, 14))


def _noise(p: QPainter) -> None:
    p.setPen(_pen(RED, 3.0))
    p.drawLine(QPointF(9, 9), QPointF(23, 23))
    p.drawLine(QPointF(23, 9), QPointF(9, 23))


def _unassign(p: QPainter) -> None:
    p.setPen(_pen(dashed=True))
    p.drawEllipse(QRectF(7, 7, 18, 18))


def _split(p: QPainter) -> None:
    p.setPen(_pen(BLUE))
    p.drawLine(QPointF(16, 12), QPointF(8, 22))
    p.drawLine(QPointF(16, 12), QPointF(24, 22))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(16, 7), 4, 4)
    p.drawEllipse(QPointF(7, 25), 4, 4)
    p.drawEllipse(QPointF(25, 25), 4, 4)


def _isolate(p: QPainter) -> None:
    path = QPainterPath(QPointF(4, 16))
    path.cubicTo(QPointF(10, 7), QPointF(22, 7), QPointF(28, 16))
    path.cubicTo(QPointF(22, 25), QPointF(10, 25), QPointF(4, 16))
    p.drawPath(path)
    p.setPen(Qt.NoPen)
    p.setBrush(FG)
    p.drawEllipse(QPointF(16, 16), 3.5, 3.5)


def _seed(p: QPainter) -> None:
    p.setPen(_pen(GREEN))
    p.drawLine(QPointF(16, 18), QPointF(16, 9))
    p.drawLine(QPointF(16, 12), QPointF(11, 7))
    p.drawLine(QPointF(16, 12), QPointF(21, 7))
    p.setPen(Qt.NoPen)
    p.setBrush(AMBER)
    p.drawEllipse(QPointF(16, 23), 5, 5)


def _grow(p: QPainter) -> None:
    p.setPen(_pen(GREEN))
    p.drawLine(QPointF(16, 29), QPointF(16, 12))
    p.drawLine(QPointF(16, 20), QPointF(7, 11))
    p.drawLine(QPointF(16, 16), QPointF(25, 7))


def _clear(p: QPainter) -> None:
    p.setPen(_pen(width=2.0))
    p.drawLine(QPointF(11, 11), QPointF(21, 21))
    p.drawLine(QPointF(21, 11), QPointF(11, 21))


def _undo(p: QPainter) -> None:
    p.drawArc(QRectF(8, 9, 16, 16), 0, 180 * 16)
    _arrowhead(p, (8, 24), (3, 16), (13, 16))


def _redo(p: QPainter) -> None:
    p.drawArc(QRectF(8, 9, 16, 16), 0, 180 * 16)
    _arrowhead(p, (24, 24), (19, 16), (29, 16))


def _save(p: QPainter) -> None:
    p.drawLine(QPointF(16, 4), QPointF(16, 15))
    _arrowhead(p, (16, 20), (11, 13), (21, 13))
    path = QPainterPath(QPointF(6, 19))
    path.lineTo(QPointF(6, 27))
    path.lineTo(QPointF(26, 27))
    path.lineTo(QPointF(26, 19))
    p.drawPath(path)


def _folder(p: QPainter) -> None:
    path = QPainterPath(QPointF(5, 25))
    path.lineTo(QPointF(5, 9))
    path.lineTo(QPointF(13, 9))
    path.lineTo(QPointF(15, 12))
    path.lineTo(QPointF(27, 12))
    path.lineTo(QPointF(27, 25))
    path.closeSubpath()
    p.drawPath(path)


def _hide(p: QPainter) -> None:
    """Crossed-out eye: hide points from view."""
    p.drawEllipse(QRectF(5, 10, 22, 12))
    p.setBrush(FG)
    p.drawEllipse(QRectF(13, 13, 6, 6))
    p.setBrush(Qt.NoBrush)
    p.setPen(_pen(RED))
    p.drawLine(QPointF(7, 27), QPointF(25, 5))


def _move(p: QPainter) -> None:
    """Four-way arrows: camera/movement mode."""
    p.drawLine(QPointF(16, 9), QPointF(16, 23))
    p.drawLine(QPointF(9, 16), QPointF(23, 16))
    _arrowhead(p, (16, 3), (11, 9), (21, 9))
    _arrowhead(p, (16, 29), (11, 23), (21, 23))
    _arrowhead(p, (3, 16), (9, 11), (9, 21))
    _arrowhead(p, (29, 16), (23, 11), (23, 21))


def _next(p: QPainter) -> None:
    p.drawLine(QPointF(4, 16), QPointF(18, 16))
    _arrowhead(p, (23, 16), (16, 10), (16, 22))
    p.drawLine(QPointF(27, 8), QPointF(27, 24))


_DRAW = {
    "hide": _hide,
    "move": _move,
    "lasso": _lasso,
    "target": _target,
    "reassign": _reassign,
    "new": _new_tree,
    "merge": _merge,
    "noise": _noise,
    "unassign": _unassign,
    "split": _split,
    "isolate": _isolate,
    "seed": _seed,
    "grow": _grow,
    "clear": _clear,
    "undo": _undo,
    "redo": _redo,
    "save": _save,
    "folder": _folder,
    "next": _next,
}

_cache: dict[str, QIcon] = {}


def icon(name: str) -> QIcon:
    """Return the named icon, drawing (and caching) it on first use."""
    if name not in _cache:
        pix = QPixmap(64, 64)
        pix.setDevicePixelRatio(2)  # crisp on HiDPI screens
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(_pen())
        p.setBrush(Qt.NoBrush)
        _DRAW[name](p)
        p.end()
        _cache[name] = QIcon(pix)
    return _cache[name]


_app_icon_cache: QIcon | None = None


def app_icon() -> QIcon:
    """The segfix window/taskbar icon: a canopy split into two colours —
    one tree wrongly merged into two, or two wrongly merged into one,
    either way the thing this tool fixes — over a short trunk. Unlike the
    button icons above this one is filled, not just stroked, so it stays
    legible at taskbar/dock sizes; several resolutions are baked in so a
    window manager picks a crisp one instead of scaling a single pixmap.
    """
    global _app_icon_cache
    if _app_icon_cache is not None:
        return _app_icon_cache

    result = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        pix = QPixmap(s, s)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)

        canopy_d = s * 0.74
        cx, cy = s * 0.5, s * 0.44
        rect = QRectF(cx - canopy_d / 2, cy - canopy_d / 2, canopy_d, canopy_d)
        p.setPen(Qt.NoPen)
        p.setBrush(GREEN)
        p.drawPie(rect, 90 * 16, 180 * 16)  # left half
        p.setBrush(AMBER)
        p.drawPie(rect, 270 * 16, 180 * 16)  # right half

        outline = QPen(QColor("#2a2a2a"))
        outline.setWidthF(max(1.0, s * 0.035))
        p.setPen(outline)
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(cx, cy - canopy_d / 2), QPointF(cx, cy + canopy_d / 2))
        p.drawEllipse(rect)

        trunk_w = s * 0.11
        trunk_h = s * 0.30
        trunk_rect = QRectF(
            cx - trunk_w / 2, cy + canopy_d / 2 - s * 0.03, trunk_w, trunk_h
        )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#8a6a4a"))
        p.drawRoundedRect(trunk_rect, trunk_w * 0.3, trunk_w * 0.3)

        p.end()
        result.addPixmap(pix)
    _app_icon_cache = result
    return result
