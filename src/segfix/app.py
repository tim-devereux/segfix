"""Command-line entry point.

``segfix`` always opens with a startup dialog to pick a recent project or
import a new file — there's no way to pass a cloud path directly on the
command line, so every session goes through the registry. See
:mod:`registry`/:mod:`startup_ui`.

Once a path is chosen it opens in the one editing mode: a table of every tree
in the file, where double-clicking a row loads that tree plus its spatial
neighbours rather than the whole cloud. See :mod:`treecatalog`/:mod:`scene_ui`.

That mode needs to seek to an arbitrary point without parsing everything
before it, so binary PLY — fixed-size vertex records at a known offset — is
the only format accepted; :func:`segfix.io.load` rejects the rest.
"""

from __future__ import annotations

import argparse
import sys


def _prefer_discrete_gpu() -> None:
    """Best-effort: on a hybrid-graphics machine, point OpenGL/Direct3D at
    the discrete GPU instead of whatever the platform defaults to (often
    the weaker integrated one — see the "GPU:" status-bar label added by
    :func:`viewer.add_gpu_status_widget` to confirm which one actually got
    used). Detects hardware rather than hardcoding one machine's GPU model
    name, never overrides a preference already set (env var on Linux, the
    registry key on Windows), and never raises — worst case this is a
    no-op and the platform default stands.

    Must run before ``import napari`` (like the Wayland workaround below):
    the Linux/WSL half relies on the underlying GL/EGL libraries not having
    picked an adapter yet, which happens at first context creation, not at
    process start — setting the env var any later risks losing the race
    against whichever import gets there first. The Windows half doesn't
    have that race (it's a registry key, not an env var) but only takes
    effect from the *next* launch of this Python executable, not this one:
    Windows reads a process's GPU preference at process creation, before
    any of our code has had a chance to run.
    """
    try:
        if sys.platform == "win32":
            _prefer_discrete_gpu_windows()
        elif sys.platform == "linux":
            _prefer_discrete_gpu_linux()
        # macOS: no per-process lever like the two above. Apple Silicon has
        # a single GPU anyway; Intel-Mac dual-GPU switching is a system
        # Energy Saver setting, not something a process can request.
    except Exception:
        pass  # detection failing should never block startup


def _prefer_discrete_gpu_linux() -> None:
    import os
    import shutil
    import subprocess

    is_wsl = False
    try:
        with open("/proc/version", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass
    has_nvidia = shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, timeout=2
    ).returncode == 0
    if not has_nvidia:
        return
    if is_wsl:
        # Mesa's D3D12 backend (the only GL path WSLg offers) matches this
        # against the adapter names Windows exposes, substring not exact —
        # so it still works if the discrete card is a different NVIDIA
        # model than whatever machine this was tested on.
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    else:
        # Native Linux NVIDIA Optimus/PRIME laptops: ask for the discrete
        # GPU's GLX vendor lib instead of the integrated one.
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


def _prefer_discrete_gpu_windows() -> None:
    """Set this Python executable's GPU preference to "High performance" in
    the per-app registry key Settings > System > Display > Graphics also
    writes to — the documented, vendor-agnostic way one process can ask
    Windows for the discrete GPU without touching any other app's choice.
    HKEY_CURRENT_USER, so it needs no elevation; keyed by ``sys.executable``,
    so it only affects this specific Python (e.g. one conda env), not every
    Python on the machine, and never touches a value already set — by the
    user via Settings, or by this same call on a previous run.
    """
    import winreg

    exe = sys.executable
    key = winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\DirectX\UserGpuPreferences",
        0,
        winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
    )
    try:
        try:
            winreg.QueryValueEx(key, exe)
            return  # already has a preference (ours or the user's) — leave it
        except FileNotFoundError:
            pass
        winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
    finally:
        winreg.CloseKey(key)


def main(argv=None) -> int:
    _prefer_discrete_gpu()
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

    return _run_scene(napari, args)


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
    lay.setSpacing(4)
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
    from .icons import app_icon

    viewer.window._qt_window.setWindowIcon(app_icon())
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
