"""Command-line entry point.

Two modes:
- ``segfix path/to/cloud.las`` — edit a single labelled cloud.
- ``segfix --project DIR`` — open a directory of per-tree PLY files with a
  tree table, neighbour loading, overlays and per-tree save.
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
        "cloud", nargs="?", help="Input point cloud (.las/.laz/.ply)"
    )
    parser.add_argument(
        "--project",
        metavar="DIR",
        help="Open a directory of per-tree PLY files in project mode",
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
        "display (RGB-segmented PLY only). Saving writes only loaded points.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop mode: edit a large cloud one spatial tile at a time, "
        "writing each tile back into an output copy of the file.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output file for crop mode (default: <name>_fixed.<ext>)",
    )
    args = parser.parse_args(argv)

    if not args.cloud and not args.project:
        parser.error("provide a point cloud file or --project DIR")

    import napari

    if args.project:
        return _run_project(napari, args)
    if args.crop:
        return _run_crop(napari, args)

    from . import io
    from .viewer import add_cloud_layer, apply_cloudcompare_controls, strip_ui
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    cloud = io.load(
        args.cloud, label_field=args.label_field, max_points=args.max_points
    )

    viewer = napari.Viewer(title=f"segfix — {args.cloud}")
    viewer.window._qt_window.showMaximized()
    layer = add_cloud_layer(viewer, cloud, point_size=args.point_size)

    controller = SegFixController(viewer, cloud, layer)
    panel = SegFixWidget(controller)
    viewer.window.add_dock_widget(panel, name="segfix", area="right")
    panel.size_spin.setValue(args.point_size)
    bind_shortcuts(viewer, panel)
    apply_cloudcompare_controls(viewer)
    strip_ui(viewer)

    viewer.status = (
        f"Loaded {cloud.n_points:,} points · {len(cloud.tree_ids)} trees. "
        "Space = review first tree, L = lasso, A/N/U/X to fix, "
        "Space again when done."
    )

    napari.run()
    return 0


def _run_project(napari, args) -> int:
    """Project mode: a directory of per-tree PLYs with a tree table."""
    import numpy as np

    from .model import PointCloud
    from .project import Project
    from .project_ui import ProjectController, ProjectWidget
    from .viewer import add_cloud_layer, apply_cloudcompare_controls, strip_ui
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    project = Project(args.project)

    viewer = napari.Viewer(title=f"segfix — {args.project}")
    viewer.window._qt_window.showMaximized()
    # Start with an empty editable layer; loading a tree swaps in real data.
    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    layer = add_cloud_layer(viewer, empty, point_size=args.point_size)

    seg = SegFixController(viewer, empty, layer)
    panel = SegFixWidget(seg)
    project_ctrl = ProjectController(
        viewer, project, seg, point_size=args.point_size
    )
    project_panel = ProjectWidget(project_ctrl)

    viewer.window.add_dock_widget(project_panel, name="trees", area="left")
    viewer.window.add_dock_widget(panel, name="segfix", area="right")
    panel.size_spin.setValue(args.point_size)
    bind_shortcuts(viewer, panel)
    apply_cloudcompare_controls(viewer)
    strip_ui(viewer)

    viewer.status = (
        f"{len(project.entries)} trees in {args.project}. "
        "Double-click a tree to load it with neighbours."
    )
    napari.run()
    return 0


def _run_crop(napari, args) -> int:
    """Crop mode: tile a large cloud and edit one chunk at a time."""
    import numpy as np

    from .crop import CropSession
    from .crop_ui import CropController, CropWidget
    from .model import PointCloud
    from .viewer import add_cloud_layer, apply_cloudcompare_controls, strip_ui
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    session = CropSession(args.cloud, output=args.output)

    viewer = napari.Viewer(title=f"segfix (crop) — {args.cloud}")
    viewer.window._qt_window.showMaximized()
    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    layer = add_cloud_layer(viewer, empty, point_size=args.point_size)

    seg = SegFixController(viewer, empty, layer)
    panel = SegFixWidget(seg)
    crop_ctrl = CropController(viewer, session, seg, point_size=args.point_size)
    crop_panel = CropWidget(crop_ctrl)

    viewer.window.add_dock_widget(crop_panel, name="tiles", area="left")
    viewer.window.add_dock_widget(panel, name="segfix", area="right")
    panel.size_spin.setValue(args.point_size)
    bind_shortcuts(viewer, panel)
    apply_cloudcompare_controls(viewer)
    strip_ui(viewer)

    viewer.status = (
        f"{session.count:,} points. Make a grid and double-click a tile to edit."
    )
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
