"""Startup picker: reopen a recent project, or import a point cloud file as
a new one.

``segfix`` takes no path argument, so this dialog is how every session picks
what to open — there is no way to skip it. Importing copies the chosen file
into a fresh workspace folder (:mod:`workspace`) and opens that copy, so the
source file is never the thing being edited.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import registry, workspace

_ICONS = {"workspace": "🗂", "file": "📄"}


class StartupDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("segfix")
        self.resize(560, 380)
        # What main() should open, what should be recorded in the registry,
        # and which kind it is — see create_workspace()'s docstring for why
        # the first two differ for a new workspace (open the data file
        # inside it, but register the workspace folder itself).
        self.open_path: str | None = None
        self.registry_path: str | None = None
        self.kind: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Recent projects:"))

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemDoubleClicked.connect(self._on_choose)
        layout.addWidget(self.list, stretch=1)
        self._populate()

        hint = QLabel(
            "Double-click a recent project, or start a new one by importing "
            "a point cloud file — a private copy is made in a new project "
            "folder, and edits are saved to that copy, never the original."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        row = QHBoxLayout()
        new_btn = QPushButton("New Project…")
        new_btn.clicked.connect(self._new_project)
        row.addWidget(new_btn)
        row.addStretch()
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._on_choose)
        row.addWidget(open_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

    def _populate(self) -> None:
        # registry.load_registry() is already most-recently-opened first, so
        # the newest project lands at the top (and is preselected below).
        self.list.clear()
        for entry in registry.load_registry():
            mark = _ICONS.get(entry["type"], "📄")
            age = registry.describe_age(entry.get("last_opened", ""))
            label = f"{mark}  {entry['path']}"
            if age:
                label += f"   —  {age}"
            item = QListWidgetItem(label)
            item.setToolTip(
                f"Last opened {entry.get('last_opened', 'unknown')}"
            )
            item.setData(Qt.UserRole, entry)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _new_project(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self, "Import point cloud", str(Path.home()),
            "Point clouds (*.ply *.las *.laz);;Binary PLY (*.ply);;"
            "LAS / LAZ (*.las *.laz)",
        )
        if not source:
            return
        parent, _ = str(Path.home()), None
        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to create the project folder", parent,
        )
        if not parent:
            return

        stem = Path(source).stem
        candidate = Path(parent) / stem
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            suffix += 1
            candidate = Path(parent) / f"{stem}_{suffix}"

        try:
            data_path = workspace.create_workspace(source, candidate)
        except Exception as exc:  # OSError, or laspy failing on a bad LAS/LAZ
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        self.open_path = str(data_path)
        self.registry_path = str(candidate)
        self.kind = "workspace"
        self.accept()

    def _on_choose(self, *_args) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        kind = entry["type"]
        if kind == "workspace":
            try:
                self.open_path = str(workspace.data_file(entry["path"]))
            except (OSError, KeyError, ValueError) as exc:
                QMessageBox.critical(
                    self, "Can't open project",
                    f"{entry['path']} doesn't look like a valid project "
                    f"folder anymore: {exc}",
                )
                return
        else:
            self.open_path = entry["path"]
        self.registry_path = entry["path"]
        self.kind = kind
        self.accept()


_app = None  # kept alive for the process's lifetime once created here — a
# QApplication with no surviving Python reference gets garbage-collected
# immediately (PyQt/PySide tear down the underlying app with it), which
# crashes the very next QWidget construction.


def choose_project() -> tuple[str, str, str] | None:
    """Show the startup dialog. Returns ``(open_path, registry_path, kind)``
    — the path to actually open, the path to remember in the registry
    (differs for a new workspace: open its data file, register its folder),
    and ``kind`` (see :func:`registry.add_entry`) — or ``None`` if the user
    cancelled without choosing anything.

    Creates the process-wide ``QApplication`` (reused by the main window
    afterwards). ``app.main`` has already pinned ``QT_QPA_PLATFORM`` by the
    time this runs.
    """
    global _app
    import sys

    from qtpy.QtWidgets import QApplication

    from .icons import app_icon

    _app = QApplication.instance() or QApplication(sys.argv)
    _app.setWindowIcon(app_icon())
    dialog = StartupDialog()
    if dialog.exec() == QDialog.Accepted and dialog.open_path:
        return dialog.open_path, dialog.registry_path, dialog.kind
    return None
