"""Command-line entry point.

``segfix`` always opens with a startup dialog to pick a recent project or
import a new file — there's no way to pass a cloud path directly on the
command line, so every session goes through the registry. See
:mod:`registry`/:mod:`startup_ui`.

Once a path is chosen it opens in the one editing mode: a table of every tree
in the file, where double-clicking a row loads that tree plus its spatial
neighbours rather than the whole cloud. See :mod:`treecatalog`/:mod:`scene_ui`.

That mode needs to seek to an arbitrary point without parsing everything
before it, so the accepted formats are the ones that store points as
fixed-size records at a known offset: binary PLY and uncompressed LAS.
arbor's ``.laz`` output is decompressed to ``.las`` on import (see
:mod:`segfix.workspace`). :func:`segfix.io.load` rejects the rest.
"""

from __future__ import annotations

import argparse
import sys


def _prefer_discrete_gpu() -> None:
    """Best-effort: on a hybrid-graphics machine, point OpenGL/Direct3D at
    the discrete GPU instead of whatever the platform defaults to (often
    the weaker integrated one — see the "GPU:" status-bar label the main
    window shows to confirm which one actually got used). Detects hardware
    rather than hardcoding one machine's GPU model name, never overrides a
    preference already set (env var on Linux, the registry key on Windows),
    and never raises — worst case this is a no-op and the platform default
    stands.

    Must run before any GL context is created (and before the Qt platform
    plugin is chosen): the Linux/WSL half relies on the underlying GL/EGL
    libraries not having picked an adapter yet, which happens at first
    context creation, not at process start — setting the env var any later
    risks losing the race. The Windows half doesn't have that race (it's a
    registry key, not an env var) but only takes effect from the *next*
    launch of this Python executable, not this one: Windows reads a
    process's GPU preference at process creation, before any of our code
    has had a chance to run.
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
        "--point-size", type=float, default=3.0,
        help="Render size of points in screen pixels (default: 3)",
    )
    args = parser.parse_args(argv)

    # vispy + PyQt6 on a Wayland session with the NVIDIA driver hits GLX
    # context-creation failures; force the xcb (X11 / XWayland) platform
    # plugin before any QApplication exists. Respect an explicit override.
    import os

    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from . import registry
    from .startup_ui import choose_project

    # "Open Project…" on the menu bar re-execs segfix with this set, so the
    # freshly picked project opens straight away instead of showing the
    # startup dialog a second time. Popped so a later re-exec starts clean.
    preselected = os.environ.pop("SEGFIX_OPEN", None)
    if preselected:
        open_path = preselected
        registry_path = os.environ.pop("SEGFIX_OPEN_REGISTRY", preselected)
        kind = os.environ.pop("SEGFIX_OPEN_KIND", "file")
    else:
        choice = choose_project()
        if choice is None:
            return 0
        open_path, registry_path, kind = choice
    args.cloud = open_path
    registry.add_entry(registry_path, kind=kind)

    return _run_scene(args)


def _read_license() -> str | None:
    """The MIT licence text — from the repo checkout segfix runs from, or
    the installed distribution's metadata. ``None`` if neither is found."""
    from pathlib import Path

    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "LICENSE", here.parent / "LICENSE"):
        try:
            return cand.read_text(encoding="utf-8")
        except OSError:
            continue
    try:
        from importlib.metadata import files

        for f in files("segfix") or []:
            if Path(f.name).name in ("LICENSE", "LICENSE.txt", "COPYING"):
                return f.read_text()
    except Exception:
        pass
    return None


def _about(parent) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox

    from .icons import app_icon
    from .update import display_version

    box = QMessageBox(parent)
    box.setWindowTitle("About segfix")
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setIconPixmap(app_icon().pixmap(64, 64))
    box.setText(
        f"<b>segfix {display_version()}</b>"
        "<p>GUI tool to fix instance segmentation of tree point clouds.</p>"
        "<p>MIT Licence&nbsp;&nbsp;·&nbsp;&nbsp;© 2026 Tim Devereux<br>"
        "<a href='https://github.com/tim-devereux/segfix'>"
        "github.com/tim-devereux/segfix</a></p>"
    )
    licence = _read_license()
    if licence:
        box.setInformativeText("Full licence text under “Show Details”.")
        box.setDetailedText(licence)
    box.exec()


def _open_project(win, panel) -> None:
    """Menu "Open Project…": offer to save, pick another project in the
    startup dialog, then re-exec segfix on it. Re-exec rather than an
    in-place swap so the catalog, docks and GL context all rebuild cleanly.
    """
    import os

    from qtpy.QtWidgets import QMessageBox

    from .startup_ui import choose_project

    if panel.c.cloud.can_undo():
        answer = QMessageBox.question(
            win,
            "Open another project",
            "Save changes to the current project first?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return
        if answer == QMessageBox.StandardButton.Save:
            panel.on_save()

    choice = choose_project()
    if choice is None:
        return
    open_path, registry_path, kind = choice
    # main() records it in the registry after the re-exec.
    os.environ["SEGFIX_OPEN"] = open_path
    os.environ["SEGFIX_OPEN_REGISTRY"] = registry_path
    os.environ["SEGFIX_OPEN_KIND"] = kind
    # Replace this process; -c so it doesn't matter how segfix was launched
    # (console script, -m, IDE). User CLI flags are carried across.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-c",
            "import sys; from segfix.app import main; sys.exit(main())",
            *sys.argv[1:],
        ],
    )


def _build_menus(win, panel) -> None:
    """Window menu bar: File (open/save the project), Edit (undo/redo — the
    former "Session" panel box), Preferences (colour theme), Help (about)."""
    from qtpy.QtGui import QActionGroup
    from qtpy.QtWidgets import QApplication

    from . import theme

    bar = win.menuBar()

    file_menu = bar.addMenu("&File")
    open_act = file_menu.addAction("Open Project…")
    open_act.setShortcut("Ctrl+O")
    open_act.triggered.connect(lambda: _open_project(win, panel))
    save_act = file_menu.addAction("Save Project")
    save_act.setShortcut("Ctrl+S")
    save_act.triggered.connect(panel.on_save)

    edit_menu = bar.addMenu("&Edit")
    undo_act = edit_menu.addAction("Undo")
    undo_act.setShortcut("Ctrl+Z")
    undo_act.triggered.connect(panel.on_undo)
    redo_act = edit_menu.addAction("Redo")
    redo_act.setShortcut("Ctrl+Shift+Z")
    redo_act.triggered.connect(panel.on_redo)

    pref_menu = bar.addMenu("&Preferences")
    theme_menu = pref_menu.addMenu("Theme")
    theme_group = QActionGroup(win)
    theme_group.setExclusive(True)
    for label, mode in (("Light", "light"), ("Dark", "dark")):
        act = theme_menu.addAction(label)
        act.setCheckable(True)
        act.setChecked(theme.current() == mode)
        theme_group.addAction(act)
        act.triggered.connect(
            lambda _checked, m=mode: theme.set_mode(QApplication.instance(), m)
        )

    help_menu = bar.addMenu("&Help")
    about_act = help_menu.addAction("About segfix")
    about_act.triggered.connect(lambda: _about(win))


def _combined_panel(*widgets):
    """Stack scene mode's "All Trees" table above the shared segfix editing
    panel in one right-hand dock (rather than a separate left dock).

    A vertical splitter, so the divider is draggable; it starts at roughly
    25 % top / 75 % bottom and keeps that ratio as the dock resizes.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSplitter

    split = QSplitter(Qt.Orientation.Vertical)
    for w in widgets:
        split.addWidget(w)
    split.setChildrenCollapsible(False)
    for i in range(split.count()):
        split.setStretchFactor(i, 1 if i == 0 else 3)
    split.setSizes([1000] + [3000] * (split.count() - 1))
    # Floor so the tables get room to breathe; the dock edge stays draggable.
    split.setMinimumWidth(360)
    return split


def _bare_dock(widget, title: str):
    """A ``QDockWidget`` holding ``widget`` with no title bar and no
    float/close buttons — a plain fixed panel, like napari's stripped docks."""
    from qtpy.QtWidgets import QDockWidget, QWidget

    dock = QDockWidget(title)
    dock.setObjectName(title)
    dock.setTitleBarWidget(QWidget())  # empty widget → no visible title bar
    dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    dock.setWidget(widget)
    return dock


def _run_scene(args) -> int:
    """Default (only) mode: a tree table where picking a row loads that tree
    plus its spatial neighbours into the 3D view, instead of the whole cloud.
    """
    import numpy as np
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox

    from . import theme
    from .cloudview import CloudView
    from .icons import app_icon
    from .overlays import ScaleBarOverlay
    from .model import PointCloud
    from .scene_ui import SceneController, SceneWidget
    from .shift_ui import prompt_global_shift
    from .treecatalog import open_catalog
    from .update import display_version
    from .viewer import busy, gpu_renderer_info
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    app = QApplication.instance() or QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    theme.apply(app)  # saved light/dark palette (Preferences ▸ Theme)

    win = QMainWindow()

    def _refresh_title() -> None:
        win.setWindowTitle(f"segfix {display_version()} - {args.cloud}")

    _refresh_title()
    # Re-read the checkout version whenever the window is re-activated, so an
    # external `git pull` / reinstall (or the startup dialog's own updater)
    # shows up in the title without restarting.
    app.focusChanged.connect(
        lambda _old, now: _refresh_title()
        if now is not None and (now is win or win.isAncestorOf(now))
        else None
    )
    win.setWindowIcon(app_icon())

    view = CloudView()
    win.setCentralWidget(view.native)
    # Canvas clear colour follows the theme (points stay readable on either).
    theme.subscribe(lambda mode: view.set_background(theme.CANVAS_BG[mode]))
    # Scale bar + orientation axes over the canvas' bottom-left corner. Held
    # on the window so it outlives this function; it also parents to the
    # canvas widget, which keeps it alive regardless.
    win._scale_overlay = ScaleBarOverlay(view)

    status = win.statusBar()
    view.on_status = status.showMessage
    gpu = gpu_renderer_info()
    gpu_label = QLabel(f"GPU: {gpu}" if gpu else "GPU: unknown")
    gpu_label.setStyleSheet("color: gray; padding: 0 6px;")
    status.addPermanentWidget(gpu_label)

    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    view.load_cloud(empty, point_size=args.point_size)

    win.showMaximized()
    busy(view, f"Scanning trees in {args.cloud}…")
    try:
        catalog = open_catalog(
            args.cloud,
            label_field=args.label_field,
            shift_prompt=lambda mins, maxs, suggested:
                prompt_global_shift(win, mins, maxs, suggested),
        )
    except Exception as exc:
        # A wrong/corrupt file used to raise this far with the window already
        # shown maximized and app.exec() not yet reached — no event loop was
        # running to keep it responsive, so it looked exactly like a freeze
        # even though the process was about to exit with a traceback. Fail
        # visibly and exit cleanly instead.
        win.close()
        QMessageBox.critical(
            None, "Can't open this cloud",
            f"{args.cloud}\n\ncould not be opened:\n\n{exc}\n\n"
            "segfix reads binary PLY (RGB-segmented or with a label field) "
            "and uncompressed LAS with a treeID column.",
        )
        return 1

    seg = SegFixController(view, empty)
    panel = SegFixWidget(seg)
    scene_ctrl = SceneController(view, catalog, seg, point_size=args.point_size)
    scene_panel = SceneWidget(scene_ctrl)
    panel.on_done_changed = scene_panel.refresh

    top_dock = _bare_dock(panel.top_bar, "view")
    win.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, top_dock)
    right_dock = _bare_dock(_combined_panel(scene_panel, panel), "segfix")
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)
    # Give the top-right corner to the right dock so it reaches y=0 instead
    # of being pushed down below the top strip.
    win.setCorner(
        Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
    )
    win.resizeDocks([right_dock], [440], Qt.Orientation.Horizontal)
    panel.size_spin.setValue(args.point_size)
    _build_menus(win, panel)
    bind_shortcuts(win, panel)

    view.status = (
        f"{len(catalog.records)} trees in {args.cloud}. "
        "Double-click a tree to load it with neighbours."
    )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
