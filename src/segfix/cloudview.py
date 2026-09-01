"""The 3D point-cloud canvas, backed directly by vispy.

This is the single object the rest of the app talks to for anything on the
canvas: the point cloud, the camera, the freehand-lasso projection, the
current selection and the bounding box around the tree under review.  It
replaces what used to be a napari ``Viewer`` plus a ``Points`` layer, and
with it go all the napari workarounds that surrounded them (the
``_PointSliceRequest`` monkeypatch for ``shown`` in full 3D, the
``viewbox_mouse_event`` monkeypatch for CloudCompare-style mouse, ``strip_ui``,
and the ``_qt_viewer``/``_qt_window`` reach-throughs).

Coordinates are plain right-handed XYZ with Z up — the camera is told
``up='+z'`` and nothing is reordered, so ``coords[:, 2]`` is height
everywhere, no ``dims.order`` remap.
"""

from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.scene.cameras import TurntableCamera
from vispy.util import keys


class _CloudCompareCamera(TurntableCamera):
    """Turntable camera with CloudCompare's mouse map: left-drag orbits,
    right-drag pans, wheel zooms.

    vispy's turntable puts pan on Shift+left-drag and (a rougher) zoom on
    right-drag.  Rather than reimplement the orbit/pan maths, a plain
    right-drag is presented to the base handler as the Shift+left-drag it
    already pans with, then the event is put back.  Wheel zoom is untouched.
    """

    def viewbox_mouse_event(self, event):
        if event.type == "mouse_move" and event.press_event is not None:
            buttons = event.buttons
            mods = event.mouse_event.modifiers
            if 2 in buttons and 1 not in buttons and not mods:
                saved_buttons = list(event.mouse_event.buttons)
                saved_press = list(event.mouse_event.press_event.buttons)
                saved_mods = event.mouse_event.modifiers
                event.mouse_event.buttons[:] = [1]
                event.mouse_event.press_event.buttons[:] = [1]
                event.mouse_event.modifiers = (keys.SHIFT,)
                try:
                    super().viewbox_mouse_event(event)
                finally:
                    event.mouse_event.buttons[:] = saved_buttons
                    event.mouse_event.press_event.buttons[:] = saved_press
                    event.mouse_event.modifiers = saved_mods
                return
        super().viewbox_mouse_event(event)


class CloudView:
    """A vispy canvas showing one editable point cloud.

    The rest of the app holds one of these (built in ``app._run_scene``) and
    passes it where it used to pass a napari ``Viewer``.
    """

    def __init__(self, bgcolor: str = "#262626"):
        self.canvas = scene.SceneCanvas(
            keys=None, bgcolor=bgcolor, show=False, size=(800, 800)
        )
        grid = self.canvas.central_widget
        self.view = grid.add_view()
        # fov=0 → orthographic, like CloudCompare; Z is up.
        self.view.camera = _CloudCompareCamera(fov=0.0, up="+z")
        # Position the camera explicitly so nothing ever asks the (possibly
        # empty) scene for its bounds on the first draw.
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = 10.0

        # Fixed pixel-size flat discs. Not shaded spheres (vispy's spherical
        # shading darkens the back hemisphere until a zoomed-out plot is a
        # near-black silhouette) and not scene-unit scaling (vispy 0.16's
        # canvas_size_limits clamp doesn't hold, so metre-sized points go
        # sub-pixel and vanish). Pixels keep every point its true colour and
        # visible at any zoom — the size is the panel's "Point size" spinner.
        self.markers = scene.visuals.Markers(
            parent=self.view.scene, scaling="fixed", spherical=False
        )
        self.markers.antialias = 0.3
        # Depth-tested alpha blending: opaque points (alpha 1) composite as a
        # plain overwrite, but the panel's per-tree "fade" sets alpha to
        # FADED_ALPHA and needs real blending to show. depth_test keeps the
        # nearest point per pixel, so opaque points don't accumulate toward
        # white the way a plain translucent-without-depth pass would.
        self.markers.set_gl_state(
            "translucent", depth_test=True, cull_face=False
        )
        # A second marker set drawn on top: a translucent white halo on the
        # currently selected points (napari's Points layer did this itself).
        self.highlight = scene.visuals.Markers(
            parent=self.view.scene, scaling="fixed", spherical=False
        )
        self.highlight.set_gl_state("translucent", depth_test=False)
        # Wireframe box(es) around the tree under review.
        self.bbox = scene.visuals.Line(
            parent=self.view.scene, connect="segments", width=2, antialias=True
        )
        self.bbox.set_gl_state(depth_test=False, blend=True)

        self._coords = np.empty((0, 3), np.float32)
        self._face_color = np.empty((0, 4), np.float32)
        self._shown = np.empty(0, dtype=bool)
        self._selected: set[int] = set()
        self._size = 3.0  # marker diameter in screen pixels

        #: fn(str) -> None, set by the shell to write the status bar
        self.on_status = None
        #: fn() -> None, set by the panel to refresh its selection readout
        self.on_selection_changed = None

        self._redraw()

    # -- points ---------------------------------------------------------
    def load_cloud(self, cloud, point_size: float | None = None) -> None:
        """Show ``cloud`` (a :class:`~segfix.model.PointCloud`): fresh colours,
        everything shown, selection cleared.  Does not move the camera."""
        from .viewer import colors_for_labels

        if point_size is not None:
            self._size = float(point_size)
        self._coords = np.ascontiguousarray(cloud.coords, dtype=np.float32)
        self._face_color = (
            colors_for_labels(cloud.labels, cloud.label_colors)
            if cloud.n_points
            else np.empty((0, 4), np.float32)
        )
        self._shown = np.ones(len(self._coords), dtype=bool)
        self._selected = set()
        self._redraw()
        self._redraw_highlight()

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def face_color(self) -> np.ndarray:
        return self._face_color

    @face_color.setter
    def face_color(self, value: np.ndarray) -> None:
        self._face_color = np.asarray(value, dtype=np.float32).reshape(-1, 4)
        self._redraw()
        self._redraw_highlight()

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        self._size = float(value)
        self._redraw()
        self._redraw_highlight()

    @property
    def shown(self) -> np.ndarray:
        return self._shown

    @shown.setter
    def shown(self, mask) -> None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self._coords):
            return
        self._shown = mask
        self._redraw()
        self._redraw_highlight()

    def _redraw(self) -> None:
        vis = self._shown if len(self._shown) == len(self._coords) else None
        pos = self._coords if vis is None else self._coords[vis]
        if len(pos) == 0:
            self.markers.visible = False
            self.canvas.update()
            return
        fc = self._face_color
        if len(fc) != len(self._coords):
            fc = np.ones((len(self._coords), 4), np.float32)
        fc = fc if vis is None else fc[vis]
        self.markers.visible = True
        self.markers.set_data(
            pos=pos, face_color=fc, size=self._size, edge_width=0
        )
        self.canvas.update()

    # -- selection ----------------------------------------------------
    @property
    def selected(self) -> set[int]:
        return self._selected

    @selected.setter
    def selected(self, indices) -> None:
        self._selected = {int(i) for i in indices}
        self._redraw_highlight()
        if self.on_selection_changed is not None:
            self.on_selection_changed()

    def _redraw_highlight(self) -> None:
        idx = np.fromiter(self._selected, dtype=np.int64)
        if idx.size and len(self._shown) == len(self._coords):
            idx = idx[self._shown[idx]]
        if idx.size == 0:
            self.highlight.visible = False
            self.canvas.update()
            return
        self.highlight.visible = True
        self.highlight.set_data(
            pos=self._coords[idx],
            face_color=(1.0, 1.0, 1.0, 0.9),
            size=self._size + 1.5,
            edge_width=0,
        )
        self.canvas.update()

    # -- bounding box ----------------------------------------------------
    def set_bbox(self, segments: np.ndarray, colors: np.ndarray) -> None:
        """Draw line segments (``(2E, 3)`` endpoints, ``(2E, 4)`` colours)."""
        if segments is None or len(segments) == 0:
            self.clear_bbox()
            return
        self.bbox.set_data(
            pos=np.asarray(segments, np.float32),
            color=np.asarray(colors, np.float32),
        )
        self.bbox.visible = True
        self.canvas.update()

    def clear_bbox(self) -> None:
        self.bbox.visible = False
        self.canvas.update()

    # -- camera --------------------------------------------------------
    def reset_view(self) -> None:
        if not len(self._coords):
            return
        lo = self._coords.min(axis=0)
        hi = self._coords.max(axis=0)
        # Explicit ranges, computed from the cloud itself: vispy's own
        # ``set_range()`` would walk every sibling visual's ``.bounds`` and
        # trips over the halo/box visuals when they have no data yet.
        self.view.camera.set_range(
            x=(float(lo[0]), float(hi[0])),
            y=(float(lo[1]), float(hi[1])),
            z=(float(lo[2]), float(hi[2])),
            margin=0.05,
        )

    def fly_to(self, center_xyz, span: float) -> None:
        cam = self.view.camera
        cam.center = tuple(float(c) for c in center_xyz)
        cam.scale_factor = max(float(span), 0.5) * 1.6

    def set_camera_interactive(self, on: bool) -> None:
        self.view.camera.interactive = bool(on)

    # -- lasso support ------------------------------------------------
    @property
    def native(self):
        """The canvas' Qt widget — parent for the lasso overlay."""
        return self.canvas.native

    @property
    def canvas_size(self) -> tuple[int, int]:
        return tuple(self.canvas.size)

    def project_to_canvas(self, coords: np.ndarray):
        """Project ``coords`` (``(N, 3)`` world XYZ) to canvas pixels.

        Returns ``(xy, valid)`` — ``xy`` is ``(N, 2)`` and ``valid`` masks out
        points behind the camera (non-positive homogeneous w).
        """
        tr = self.markers.get_transform(map_from="visual", map_to="canvas")
        mapped = np.asarray(tr.map(np.asarray(coords, dtype=np.float64)))
        w = mapped[:, 3]
        valid = w > 0
        w_safe = np.where(valid, w, 1.0)
        return mapped[:, :2] / w_safe[:, None], valid

    # -- status ------------------------------------------------------
    @property
    def status(self):  # write-only in practice; getter kept for symmetry
        return None

    @status.setter
    def status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(str(message))
