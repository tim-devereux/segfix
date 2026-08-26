"""Left-dock UI for the default single-file mode: a table of every tree in
the file (from :class:`~segfix.treecatalog.TreeCatalog`); double-click loads
that tree plus its spatial neighbours into the shared segfix editing panel.

Modeled on :mod:`project_ui`, minus the multi-file-specific overlay concept
(``_non_seg``/``removed_points.xyz`` are particular to the CloudCompare
plugin's per-tree-file layout, not applicable to one big labelled cloud).
"""

from __future__ import annotations

import json
import os

import numpy as np
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

from .icons import icon
from .treecatalog import TreeCatalog
from .viewer import add_cloud_layer, busy

DEFAULT_REACH = 1.0  # metres; matches SegFixWidget's own focus-mode default


class SceneController:
    """Owns the catalog and the shared segfix editing controller."""

    def __init__(self, viewer, catalog: TreeCatalog, seg_controller,
                 point_size: float = 0.01, reach: float = DEFAULT_REACH):
        self.viewer = viewer
        self.catalog = catalog
        self.seg = seg_controller
        self.point_size = point_size
        self.reach = reach
        self.current_label: int | None = None
        self._global_idx: np.ndarray | None = None
        # Set by the widget to refresh its table's point counts after a save
        # — fires regardless of which Save button triggered it (this one, or
        # the shared segfix panel's), since both route through on_save_override.
        self.on_saved = None
        self.seg.on_save_override = self._save

    def load_tree(self, label: int) -> str:
        self._flush()  # persist edits on the outgoing scene first
        neighbours = self.catalog.neighbours(label, self.reach)
        cloud, global_idx = self.catalog.load(
            [label, *sorted(neighbours)], margin=self.reach
        )
        self.current_label = label
        self._global_idx = global_idx

        if self.seg.layer is not None and self.seg.layer in self.viewer.layers:
            self.viewer.layers.remove(self.seg.layer)

        layer = add_cloud_layer(self.viewer, cloud, point_size=self.point_size)
        self.seg.set_cloud(cloud, layer)
        self.viewer.reset_view()
        return (
            f"Loaded tree {label} + {len(neighbours)} neighbour(s); "
            f"{cloud.n_points:,} points"
        )

    def _flush(self) -> None:
        if self._global_idx is not None:
            self.catalog.apply(self.seg.cloud, self._global_idx)

    def _save(self) -> str:
        self._flush()  # capture the live scene too, not just prior ones
        msg = self.catalog.save()
        if self.on_saved is not None:
            self.on_saved()
        return msg

class SceneWidget(QWidget):
    """Tree table + load/save, docked on the left."""

    COLUMNS = ["Tree ID", "Points", "Done"]

    def __init__(self, controller: SceneController):
        super().__init__()
        self.c = controller
        layout = QVBoxLayout(self)

        self.path_label = QLabel(os.path.basename(controller.catalog.path))
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMaximumHeight(200)
        self.table.cellDoubleClicked.connect(lambda *_: self.on_load_tree())
        layout.addWidget(self.table)

        row = QHBoxLayout()
        load_btn = QPushButton("Load Tree")
        load_btn.setIcon(icon("folder"))
        load_btn.setIconSize(QSize(18, 18))
        load_btn.clicked.connect(self.on_load_tree)
        row.addWidget(load_btn)
        layout.addLayout(row)

        controller.on_saved = self._populate
        self._populate()

    def _read_done(self) -> set[int]:
        """Reuse SegFixWidget's own progress sidecar (keyed by source path,
        which stays constant across tree switches in this mode) so the two
        panels' "done" state stays in sync without duplicating a file."""
        path = f"{self.c.catalog.path}.segfix.json"
        if not os.path.exists(path):
            return set()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {int(t) for t in data.get("done", [])}
        except (OSError, ValueError):
            return set()

    def _populate(self) -> None:
        done = self._read_done()
        records = sorted(self.c.catalog.records.values(), key=lambda r: r.label)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(records))
        for row, rec in enumerate(records):
            mark = "✓" if rec.label in done else ""
            for col, value in enumerate([rec.label, rec.count, mark]):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                item.setData(Qt.UserRole, rec.label)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)

    def _selected_label(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 0).data(Qt.UserRole)

    def on_load_tree(self) -> None:
        label = self._selected_label()
        if label is None:
            self.c.viewer.status = "Select a tree row first"
            return
        busy(self.c.viewer, f"Loading tree {label}…")
        self.c.viewer.status = self.c.load_tree(label)
        self._populate()
