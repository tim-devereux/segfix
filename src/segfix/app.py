"""Command-line entry point.

``segfix`` always opens with a startup dialog to pick a recent project or
import a new file — there's no way to pass a cloud path directly on the
command line, so every session goes through the registry. See
:mod:`registry`/:mod:`startup_ui`.

Once a path is chosen:
- A binary PLY (RGB-segmented or label-field, the common case) gets a tree
  table; double-click a row to load that tree plus its spatial neighbours,
  instead of loading the whole cloud at once. See
  :mod:`treecatalog`/:mod:`scene_ui`.
- A non-binary/ASCII PLY or LAS/LAZ loads and edits the whole cloud at once;
  memory-mapped partial loading isn't available for these, so
  ``--max-points`` is the escape hatch for very large ones.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="segfix",
        description="GUI tool to fix tree point-cloud instance segmentation.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Name of the per-point tree/instance ID field (auto-detected if omitted)",
    )
    parser.add_argument(
        "--point-size", type=float, default=0.01,
        help="Render size of points in metres (default: 0.01)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Subsample very large clouds to at most this many points for "
        "display. Only applies to LAS/LAZ or ASCII PLY input — a binary PLY "
        "loads a tree table instead and never needs subsampling. Saving "
        "writes only loaded points.",
    )
    args = parser.parse_args(argv)

    # Must import napari before any QApplication exists (including the one
    # the startup dialog below would otherwise create first): napari applies
    # a Wayland+NVIDIA OpenGL workaround at import time that only works if
    # nothing has created a Qt application yet. Doing this out of order is a
    # real, reproduced crash — the app launches but every draw call fails
    # with "OpenGL.error.GLError: invalid enumerant" / "no valid context".
    import napari

    from . import registry
    from .startup_ui import choose_project

    choice = choose_project()
    if choice is None:
        return 0
    open_path, registry_path, kind = choice
    args.cloud = open_path
    registry.add_entry(registry_path, kind=kind)

    if _is_binary_ply(args.cloud):
        return _run_scene(napari, args)

    import numpy as np

    from . import io
    from .model import PointCloud
    from .viewer import add_cloud_layer, apply_cloudcompare_controls, busy, strip_ui
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    viewer = napari.Viewer(title=f"segfix — {args.cloud}")
    viewer.window._qt_window.showMaximized()
    # Before any loading: strip_ui/apply_cloudcompare_controls only touch
    # napari's own built-in chrome (menu bar, default docks, camera), not
    # anything of ours, so they're safe this early — and doing them now
    # means the "busy" status repaint below (and the load itself) never
    # flashes the default, unstripped napari GUI first.
    apply_cloudcompare_controls(viewer)
    strip_ui(viewer)
    # An empty Points layer, added before the (potentially slow) load: with
    # zero layers, napari shows its own "drag a file here" welcome screen
    # over the canvas — adding this one dismisses that, then gets swapped
    # for the real cloud once loading finishes.
    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    layer = add_cloud_layer(viewer, empty, point_size=args.point_size)
    busy(viewer, f"Loading {args.cloud}…")

    cloud = io.load(
        args.cloud, label_field=args.label_field, max_points=args.max_points
    )
    viewer.layers.remove(layer)
    layer = add_cloud_layer(viewer, cloud, point_size=args.point_size)

    controller = SegFixController(viewer, cloud, layer)
    panel = SegFixWidget(controller)
    _dock_top(viewer, panel)
    _dock_right(viewer, panel)
    panel.size_spin.setValue(args.point_size)
    bind_shortcuts(viewer, panel)

    viewer.status = (
        f"Loaded {cloud.n_points:,} points · {len(cloud.tree_ids)} trees. "
        "Space = review first tree, L = lasso, A/N/U/X to fix, "
        "Space again when done."
    )

    napari.run()
    return 0


def _combined_panel(*widgets):
    """Stack several dock-panel widgets into one scrollable widget.

    Used so scene mode's own tree table sits above the shared segfix editing
    panel in a single right-hand dock, instead of a separate left dock —
    everything lives in one place.
    """
    from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(4, 4, 4, 4)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    return scroll


def _dock_right(viewer, widget):
    """Dock ``widget`` on the right, sized to fill the full window height.

    ``QtViewerDockWidget.__init__`` (napari) unconditionally sets the docked
    widget's vertical size policy to ``QSizePolicy.Maximum`` — capped at its
    sizeHint, regardless of available space — independently of the
    ``add_vertical_stretch`` option. That leaves a lone right-hand dock sized
    to a fraction of the window, forcing users to scroll for content that
    would otherwise fit. Override it to ``Expanding`` so the dock claims the
    rest of the window height, then nudge it with the oversized-resizeDocks
    idiom napari itself uses for its own layer-list dock at startup
    (qt_main_window.py's ``_QtMainWindow.__init__``).
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy

    dock = viewer.window.add_dock_widget(widget, name="segfix", area="right")
    policy = dock.widget().sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
    dock.widget().setSizePolicy(policy)
    viewer.window._qt_window.resizeDocks([dock], [10000], Qt.Orientation.Vertical)
    # Qt's default corner ownership gives the top-right corner to the top
    # dock area, which pushes the right dock's top edge down below it —
    # leaving a gap of bare canvas in that corner. Give the corner to the
    # right dock instead so it reaches all the way up to y=0.
    viewer.window._qt_window.setCorner(
        Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
    )
    return dock


def _dock_top(viewer, panel) -> None:
    """Dock ``panel.top_bar`` (Interaction / View / Cross section) along the
    top of the window — it's about the canvas/viewport, not the tree-review
    workflow the main side panel is organised around."""
    viewer.window.add_dock_widget(panel.top_bar, name="view", area="top")


def _is_binary_ply(path: str) -> bool:
    """Whether ``path`` looks like a binary PLY — the only format the
    default tree-table+neighbour-loading mode can memory-map partial reads
    from."""
    if not path.lower().endswith(".ply"):
        return False
    from . import io

    try:
        with open(path, "rb") as fh:
            fmt, _props, _count, _offset = io._parse_ply_header(fh)
    except OSError:
        return False
    return "binary" in fmt


def _run_scene(napari, args) -> int:
    """Default mode for a single big binary PLY: a tree table where picking
    a row loads that tree plus its neighbours, instead of the whole cloud."""
    import numpy as np

    from .model import PointCloud
    from .scene_ui import SceneController, SceneWidget
    from .treecatalog import TreeCatalog
    from .viewer import add_cloud_layer, apply_cloudcompare_controls, busy, strip_ui
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    viewer = napari.Viewer(title=f"segfix — {args.cloud}")
    viewer.window._qt_window.showMaximized()
    # Before any loading: strip_ui/apply_cloudcompare_controls only touch
    # napari's own built-in chrome (menu bar, default docks, camera), not
    # anything of ours, so they're safe this early — and doing them now
    # means the "busy" status repaint below (and the catalog scan) never
    # flashes the default, unstripped napari GUI first.
    apply_cloudcompare_controls(viewer)
    strip_ui(viewer)
    # An empty Points layer, added before any (potentially slow) loading:
    # with zero layers, napari shows its own "drag a file here" welcome
    # screen over the canvas — adding this one, even with no points yet,
    # dismisses that so the busy status below isn't fighting it for
    # attention.
    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    layer = add_cloud_layer(viewer, empty, point_size=args.point_size)
    busy(viewer, f"Scanning trees in {args.cloud}…")

    catalog = TreeCatalog(args.cloud, label_field=args.label_field)

    seg = SegFixController(viewer, empty, layer)
    panel = SegFixWidget(seg)
    scene_ctrl = SceneController(viewer, catalog, seg, point_size=args.point_size)
    scene_panel = SceneWidget(scene_ctrl)
    panel.on_done_changed = scene_panel.refresh

    _dock_top(viewer, panel)
    _dock_right(viewer, _combined_panel(scene_panel, panel))
    panel.size_spin.setValue(args.point_size)
    bind_shortcuts(viewer, panel)

    viewer.status = (
        f"{len(catalog.records)} trees in {args.cloud}. "
        "Double-click a tree to load it with neighbours."
    )
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
