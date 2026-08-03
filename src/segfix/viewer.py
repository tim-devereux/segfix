"""napari viewer setup and label→colour mapping for tree point clouds.

Each tree instance gets a stable, distinct colour derived from its ID (so the
same tree keeps its colour across edits), while the special UNASSIGNED and
NOISE labels get muted greys that read as "not a tree".
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud

# Muted, recessive greys: unassigned/noise points (whole ground + understory
# on a big plot) are context and shouldn't compete with the tree colours.
UNASSIGNED_COLOR = np.array([0.4, 0.41, 0.43, 1.0], dtype=np.float32)
NOISE_COLOR = np.array([0.25, 0.25, 0.27, 1.0], dtype=np.float32)


def _hsv_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    """Vectorised HSV→RGB for hue array ``h`` in [0, 1)."""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _boost(rgb: tuple) -> tuple:
    """Brighten a source colour for display on the dark canvas.

    raycloudtools assigns arbitrary RGB per tree and many come out dark or
    washed out — at small point sizes they read as grey. Enforce a minimum
    saturation/brightness while keeping the hue, so trees stay recognisable
    by their original colour family. Display only: saving uses the untouched
    ``label_colors``.
    """
    import colorsys

    h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    return colorsys.hsv_to_rgb(h, max(s, 0.55), max(v, 0.85))


def colors_for_labels(labels: np.ndarray, label_colors=None) -> np.ndarray:
    """Return an ``(N, 4)`` RGBA array colouring points by their tree ID.

    When ``label_colors`` is given (e.g. an RGB-segmented source), each label
    keeps its original hue (brightened for the dark canvas); labels without
    an entry fall back to a hashed hue.  Without ``label_colors`` everything
    is hashed.  UNASSIGNED and NOISE always get fixed muted greys.
    """
    labels = np.asarray(labels)
    colors = np.empty((len(labels), 4), dtype=np.float32)

    is_unassigned = labels == UNASSIGNED
    is_noise = labels == NOISE
    is_tree = ~(is_unassigned | is_noise)

    colors[is_unassigned] = UNASSIGNED_COLOR
    colors[is_noise] = NOISE_COLOR

    if is_tree.any():
        ids = labels[is_tree].astype(np.int64)
        # Compute one colour per distinct label, then scatter to points via
        # searchsorted — keeps the work O(N log K), not O(N*K), on big clouds.
        uniq = np.unique(ids)
        # Multiply by the golden-ratio conjugate then take the fractional part
        # to scatter consecutive IDs across the hue circle.
        hue = np.mod(uniq * 0.61803398875, 1.0)
        per_label = _hsv_to_rgb(hue, s=0.65, v=0.95).astype(np.float32)
        if label_colors:
            for k, lab in enumerate(uniq):
                src = label_colors.get(int(lab))
                if src is not None:
                    per_label[k] = _boost(src)
        pos = np.searchsorted(uniq, ids)
        colors[is_tree, :3] = per_label[pos]
        colors[is_tree, 3] = 1.0
    return colors


def visibility_mask(labels: np.ndarray, isolated=None, hide_unassigned=False,
                    hidden=None, focus=None):
    """Per-point ``shown`` mask combining isolate mode, hide-unassigned,
    manually hidden points (``hidden`` is a per-point boolean mask), and
    focus mode (``focus`` tree IDs stay visible along with unassigned
    points; other trees and noise are hidden)."""
    shown = np.ones(len(labels), dtype=bool)
    if isolated is not None:
        ids = np.fromiter(isolated, dtype=np.int64)
        shown &= np.isin(labels, ids)
    if focus is not None:
        ids = np.fromiter(focus, dtype=np.int64)
        shown &= np.isin(labels, ids) | (labels == UNASSIGNED)
    if hide_unassigned:
        shown &= (labels != UNASSIGNED) & (labels != NOISE)
    if hidden is not None and len(hidden) == len(labels):
        shown &= ~hidden
    return shown


def add_cloud_layer(viewer, cloud: PointCloud, point_size: float = 0.01):
    """Add the point cloud to a napari viewer as a coloured Points layer.

    ``point_size`` is in data units (metres); 1 cm suits fine LiDAR. Adjust it
    live with the segfix panel's "Point size" spinner, or override here.
    """
    # napari chokes on an empty face_color array, so colour an empty cloud with
    # a plain default; real colours are applied as soon as points are loaded.
    face_color = (
        colors_for_labels(cloud.labels, cloud.label_colors)
        if cloud.n_points
        else "gray"
    )
    layer = viewer.add_points(
        cloud.coords,
        name="tree cloud",
        size=point_size,
        face_color=face_color,
        # border_width=0 is not enough: at clamped-small sizes napari still
        # rasterises the default dimgray border, which swamps the face colour
        # and turns the whole zoomed-out cloud grey.
        border_width=0,
        border_color="transparent",
        features={cloud.label_field: cloud.labels.copy()},
        shading="none",
    )
    # Zoomed out, 1 cm points fall below a pixel and antialiasing fades them
    # to grey. Clamp the on-screen size and shrink the AA band so the tree
    # colours stay visible at any zoom.
    layer.canvas_size_limits = (3, 10000)
    layer.antialiasing = 0.3
    viewer.dims.ndisplay = 3
    # Z up, like CloudCompare: put z on the vertical display axis pointing
    # upwards; x runs right and y into the screen (right-handed).
    viewer.dims.order = (1, 2, 0)
    viewer.camera.orientation = ("away", "up", "right")
    return layer


def strip_ui(viewer) -> None:
    """Hide napari chrome that has no role in the segfix workflow.

    Gone: the menu bar, the layer list/controls docks (layers are managed by
    the app; point size lives in the segfix panel), the IPython console, and
    the 2D-oriented viewer buttons (grid, roll, transpose, 2D/3D toggle —
    leaving 3D avoids scrambling the Z-up axis order).  The reset-view button
    and the status bar stay.
    """
    qt_viewer = viewer.window._qt_viewer
    for dock in (
        qt_viewer.dockLayerList,
        qt_viewer.dockLayerControls,
        qt_viewer.dockConsole,
    ):
        dock.setVisible(False)
    for name in (
        "consoleButton",
        "rollDimsButton",
        "transposeDimsButton",
        "gridViewButton",
        "ndisplayButton",
    ):
        btn = getattr(qt_viewer.viewerButtons, name, None)
        if btn is not None:
            btn.hide()
    viewer.window._qt_window.menuBar().setVisible(False)


def apply_cloudcompare_controls(viewer) -> None:
    """CloudCompare-style mouse: left-drag rotate, right-drag pan, wheel zoom.

    vispy puts zoom on right-drag and pan on Shift+left-drag.  Rather than
    reimplementing the camera maths, a right-button drag is presented to the
    camera as the drag it already knows how to pan with (Shift+left in 3D,
    left in 2D), then the event is restored so napari's own mouse handling
    still sees a right-drag.  Wheel zoom is untouched, and napari's
    mouse_pan gating (used to lock the camera while lassoing) still applies.
    """
    from vispy.util import keys

    cams = viewer.window._qt_viewer.canvas.camera
    for cam, mods in (
        (cams._3D_camera, (keys.SHIFT,)),
        (cams._2D_camera, ()),
    ):
        orig = cam.viewbox_mouse_event

        def handler(event, _orig=orig, _mods=mods):
            me = event.mouse_event
            if (
                event.type == "mouse_move"
                and 2 in me.buttons
                and 1 not in me.buttons
                and not me.modifiers
            ):
                saved = me._buttons, me._modifiers
                me._buttons, me._modifiers = [1], _mods
                try:
                    _orig(event)
                finally:
                    me._buttons, me._modifiers = saved
            else:
                _orig(event)

        cam.viewbox_mouse_event = handler


def refresh_layer(layer, cloud: PointCloud) -> None:
    """Re-apply colours/features after the labels have changed."""
    layer.features = {cloud.label_field: cloud.labels.copy()}
    layer.face_color = colors_for_labels(cloud.labels, cloud.label_colors)
    layer.refresh()
