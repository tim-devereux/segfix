"""Project-mode UI: a tree table plus load/overlay/save controls.

Pick a tree in the table to load it together with its spatial neighbours into
the editable cloud; fix the segmentation with the segfix operation panel; then
Save Fixed to write per-tree files and record removed points.
"""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import overlays
from .icons import icon
from .project import Project
from .trees import load_scene, save_scene
from .viewer import add_cloud_layer

OVERLAY_RADIUS = 10.0


class ProjectController:
    """Owns the project, the current focus scene, and the editing controller."""

    def __init__(self, viewer, project: Project, seg_controller,
                 point_size: float = 0.01):
        self.viewer = viewer
        self.project = project
        self.seg = seg_controller
        self.point_size = point_size
        self.scene = None
        self.seg.on_save_override = self._save

    def load_tree(self, entry) -> str:
        neighbours = self.project.neighbours(entry)
        cloud, scene = load_scene(self.project, entry, neighbours)
        self.scene = scene

        if self.seg.layer is not None and self.seg.layer in self.viewer.layers:
            self.viewer.layers.remove(self.seg.layer)
        for name in ("non_seg", "removed"):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        layer = add_cloud_layer(self.viewer, cloud, point_size=self.point_size)
        self.seg.set_cloud(cloud, layer)
        self.viewer.reset_view()

        center = self.project.position(entry) or (0.0, 0.0)
        ns = overlays.load_non_seg_overlay(self.viewer, self.project, center, OVERLAY_RADIUS)
        rm = overlays.load_removed_overlay(self.viewer, self.project, center, OVERLAY_RADIUS)
        return (
            f"Loaded tree {entry.tree_id} + {len(neighbours) - 1} neighbour(s); "
            f"{ns} non-seg, {rm} removed overlay points"
        )

    def load_overlays(self) -> str:
        if self.scene is None:
            return "Load a tree first"
        center = self.project.position(self.scene.focus) or (0.0, 0.0)
        ns = overlays.load_non_seg_overlay(self.viewer, self.project, center, OVERLAY_RADIUS)
        rm = overlays.load_removed_overlay(self.viewer, self.project, center, OVERLAY_RADIUS)
        return f"Overlays: {ns} non-seg, {rm} removed points"

    def _save(self) -> str:
        if self.scene is None:
            return "Nothing loaded to save"
        return save_scene(self.seg.cloud, self.scene)


class ProjectWidget(QWidget):
    """Tree table + project actions, docked on the left."""

    COLUMNS = ["Tree ID", "File", "Done", "Notes"]

    def __init__(self, controller: ProjectController):
        super().__init__()
        self.c = controller
        layout = QVBoxLayout(self)

        self.dir_label = QLabel(str(controller.project.data_directory))
        self.dir_label.setWordWrap(True)
        layout.addWidget(self.dir_label)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(
            len(self.COLUMNS) - 1, QHeaderView.Stretch
        )
        self.table.cellDoubleClicked.connect(lambda *_: self.on_load_tree())
        layout.addWidget(self.table)

        row = QHBoxLayout()
        for text, slot, name in [
            ("Load Tree", self.on_load_tree, "folder"),
            ("Reload Overlays", self.on_overlays, "redo"),
            ("Save Fixed", self.on_save, "save"),
        ]:
            btn = QPushButton(text)
            btn.setIcon(icon(name))
            btn.setIconSize(QSize(18, 18))
            btn.clicked.connect(slot)
            row.addWidget(btn)
        layout.addLayout(row)

        self._populate()

    def _populate(self) -> None:
        entries = self.c.project.entries
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(entries))
        for r, e in enumerate(entries):
            done = "✓" if e.tree_id in self.c.project.completed else ""
            for col, value in enumerate([e.tree_id, e.mesh_file, done, e.notes]):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, r)  # stable index into entries
                self.table.setItem(r, col, item)
        self.table.setSortingEnabled(True)

    def _selected_entry(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        idx = self.table.item(row, 0).data(Qt.UserRole)
        return self.c.project.entries[idx]

    def on_load_tree(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            self.c.viewer.status = "Select a tree row first"
            return
        self.c.viewer.status = self.c.load_tree(entry)

    def on_overlays(self) -> None:
        self.c.viewer.status = self.c.load_overlays()

    def on_save(self) -> None:
        self.c.viewer.status = self.c.scene and self.c._save() or "Nothing loaded"
        self._populate()
