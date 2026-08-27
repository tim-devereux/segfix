"""napari viewer setup and label→colour mapping for tree point clouds.

Each tree instance gets a stable, distinct colour derived from its ID (so the
same tree keeps its colour across edits), while the special UNASSIGNED and
NOISE labels get muted greys that read as "not a tree".
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud


def busy(viewer, message: str) -> None:
    """Show a status message and force it to paint immediately.

    A plain ``viewer.status = message`` only queues a repaint — it won't
    actually appear until control returns to the Qt event loop, which is too
    late if the very next line is a blocking load/save. Call this right
    before such a call so there's visible feedback while it runs.
    """
    viewer.status = message
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def gpu_renderer_info(viewer) -> str | None:
    """Best-effort OpenGL renderer string for the canvas's active GPU
    context (e.g. "NVIDIA RTX A1000 Laptop GPU" vs. a software/Mesa
    renderer) — lets a user confirm whether a GPU-offload env var actually
    took effect, without shelling out to ``glxinfo``. ``None`` if it can't
    be determined (context not ready yet, PyOpenGL missing, ...).

    Deliberately does *not* call ``native.makeCurrent()`` — vispy already
    has its own context current by the time a layer's been added, and
    forcing it again here (bypassing vispy's own context-tracking) corrupts
    that tracking: PyOpenGL's ``contextdata`` loses track of the "current"
    context, and the very next draw call crashes with "Attempt to retrieve
    context when no valid context" (a real crash this caused in testing).
    Just read whatever context vispy has already made current.
    """
    try:
        from OpenGL import GL

        renderer = GL.glGetString(GL.GL_RENDERER)
        return renderer.decode() if renderer else None
    except Exception:
        return None


def add_gpu_status_widget(viewer) -> None:
    """Show the active OpenGL renderer as a permanent widget in napari's own
    status bar, next to its "activity" toggle button — a persistent way to
    confirm whether a GPU-offload env var (e.g. ``__NV_PRIME_RENDER_OFFLOAD``)
    actually took effect, without shelling out to ``glxinfo``.
    """
    from qtpy.QtWidgets import QLabel

    gpu = gpu_renderer_info(viewer)
    label = QLabel(f"GPU: {gpu}" if gpu else "GPU: unknown")
    label.setStyleSheet("color: gray; padding: 0 6px;")
    label.setToolTip(
        "OpenGL renderer for this canvas. If this shows an integrated GPU "
        "(e.g. Mesa Intel) but you have a discrete one, launch with "
        "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia."
    )
    viewer.window._qt_window.statusBar().addPermanentWidget(label)

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
                    hidden=None, focus=None, cross_section=None):
    """Per-point ``shown`` mask combining isolate mode, hide-unassigned,
    manually hidden points (``hidden`` is a per-point boolean mask), focus
    mode (``focus`` tree IDs stay visible along with unassigned points;
    other trees and noise are hidden), and the cross-section tool
    (``cross_section`` is a per-point boolean mask of points inside the
    current slab — points outside it are hidden and thus unselectable, same
    as any other hidden point)."""
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
    if cross_section is not None and len(cross_section) == len(labels):
        shown &= cross_section
    return shown


_shown_patch_applied = False


def _patch_napari_shown_in_full_3d() -> None:
    """Work around a napari bug: ``Points._view_indices`` ignores ``shown``
    whenever every dimension is displayed at once (always true for our
    plain-3D clouds with ``ndisplay=3``).

    ``_PointSliceRequest.__call__`` takes a fast path when
    ``slice_input.not_displayed`` is empty — "if we want to display
    everything, use all indices" — and returns every point index without
    ever consulting ``self.shown``. That silently no-ops every
    visibility toggle in the app (focus mode, "show unassigned", and the
    per-tree hide checkbox) since they all work by setting ``layer.shown``.
    Patched to still filter by ``shown`` on that path.
    """
    global _shown_patch_applied
    if _shown_patch_applied:
        return
    try:
        from napari.layers.points._slice import (
            _PointSliceRequest,
            _PointSliceResponse,
        )
    except ImportError:
        return

    def patched_call(self):
        if len(self.data) == 0:
            return _PointSliceResponse(
                indices=np.empty(0, dtype=int),
                size=np.empty(0, dtype=float),
                slice_input=self.slice_input,
                request_id=self.id,
            )
        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            indices = np.flatnonzero(self.shown)
            return _PointSliceResponse(
                indices=indices,
                size=self.size[indices],
                slice_input=self.slice_input,
                request_id=self.id,
            )
        # napari's own _get_slice_data already filters by self.shown here.
        indices, size = self._get_slice_data(not_disp)
        return _PointSliceResponse(
            indices=indices, size=size,
            slice_input=self.slice_input, request_id=self.id,
        )

    _PointSliceRequest.__call__ = patched_call
    _shown_patch_applied = True


def add_cloud_layer(viewer, cloud: PointCloud, point_size: float = 0.01):
    """Add the point cloud to a napari viewer as a coloured Points layer.

    ``point_size`` is in data units (metres); 1 cm suits fine LiDAR. Adjust it
    live with the segfix panel's "Point size" spinner, or override here.
    """
    _patch_napari_shown_in_full_3d()
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
        # "spherical" fakes each point as a tiny shaded sphere (depth +
        # lighting cues), which helps tell overlapping/intermingled crowns
        # apart. Not true eye-dome lighting (a screen-space depth-buffer
        # post-process) — napari/vispy don't have that — but a much cheaper
        # depth cue to try first.
        shading="spherical",
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
    the app; point size lives in the segfix panel), the IPython console, the
    2D-oriented viewer buttons (grid, roll, transpose, 2D/3D toggle — leaving
    3D avoids scrambling the Z-up axis order), and most of the status bar
    (see below). The reset-view button and the status bar's plain message
    text — segfix's own feedback channel (``viewer.status = "..."``) — stay.
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

    # napari's own cursor-driven status update (coordinates/layer/source/
    # plugin fields, plus the activity toggle) fires on every mouse move —
    # disconnect the update itself rather than just hiding its widgets,
    # since ViewerStatusBar.setStatusText always calls
    # self._status.setText(text) with text='' on that path, blanking
    # segfix's own status messages (e.g. "Tree 5 marked done") within a
    # fraction of a second of the mouse being over the canvas, which it
    # almost always is. It's also of limited use here regardless: with the
    # Z-up axis remap in add_cloud_layer (dims.order), the three numbers it
    # shows are silently reordered relative to the cloud's real X/Y/Z and
    # never labelled per-axis, so "[10, 20, 30]" doesn't say which is which.
    try:
        viewer.cursor.events.position.disconnect(viewer.update_status_from_cursor)
    except (TypeError, ValueError):
        pass  # already disconnected, or a napari version wiring it differently
    # The help label (napari's "use <5> for transform" style hint) isn't
    # included here: its own widget class force-shows it on every resize
    # regardless of setVisible(False), so hiding it here would be a no-op
    # the moment the window is touched. SegFixWidget._pin_layer_mode blanks
    # its *text* instead (via viewer.help), which is the part that sticks.
    status_bar = viewer.window._qt_window.statusBar()
    for name in ("_layer_base", "_source_type", "_plugin_reader", "_coordinates"):
        widget = getattr(status_bar, name, None)
        if widget is not None:
            widget.setVisible(False)
    activity = getattr(status_bar, "_activity_item", None)
    if activity is not None:
        activity.setVisible(False)


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
