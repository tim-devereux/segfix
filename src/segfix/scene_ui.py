"""The "All Trees" table for the default single-file mode: every tree in the
file (from a :mod:`~segfix.treecatalog` catalog); double-click loads that
tree plus its spatial neighbours into the shared segfix editing panel.

Docked at the top of the right-hand panel, directly above that editing panel
— see :func:`segfix.app._combined_panel`, which stacks the two into one dock.
"""

from __future__ import annotations

import json
import os

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .treecatalog import Catalog
from .viewer import busy

DEFAULT_REACH = 1.0  # metres; matches SegFixWidget's own "reach" spinner default


class SceneController:
    """Owns the catalog and the shared segfix editing controller."""

    def __init__(self, view, catalog: Catalog, seg_controller,
                 point_size: float = 0.01, reach: float = DEFAULT_REACH):
        self.view = view
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

        self.view.load_cloud(cloud, point_size=self.point_size)
        self.seg.set_cloud(cloud)
        self.view.reset_view()
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
    """Tree table for the whole file; stacked above the editing panel."""

    COLUMNS = ["Tree ID", "Points", "Done"]

    def __init__(self, controller: SceneController):
        super().__init__()
        self.c = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # "All Trees": every tree in the source file. Double-click a row to
        # load it plus its spatial neighbours into the "Selected Tree +
        # Neighbours" panel underneath for editing.
        self.trees_box = QGroupBox("All Trees")
        blay = QVBoxLayout(self.trees_box)
        blay.setSpacing(4)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        blay.addWidget(self.path_label)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeaderItem(2).setToolTip(
            "Whether this tree has been marked reviewed in the panel below"
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setMaximumHeight(200)
        self.table.cellDoubleClicked.connect(lambda *_: self.on_load_tree())
        blay.addWidget(self.table)

        layout.addWidget(self.trees_box)

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
        n_done = sum(1 for rec in records if rec.label in done)
        self.path_label.setText(
            f"{n_done}/{len(records)} trees done in "
            f"{os.path.basename(self.c.catalog.path)}"
        )
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

    def refresh(self) -> None:
        """Re-read done-state and point counts; call after external changes
        (e.g. the shared segfix panel marking a tree done)."""
        self._populate()

    def _selected_label(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 0).data(Qt.UserRole)

    def on_load_tree(self) -> None:
        label = self._selected_label()
        if label is None:
            self.c.view.status = "Select a tree row first"
            return
        busy(self.c.view, f"Loading tree {label}…")
        self.c.view.status = self.c.load_tree(label)
        self._populate()
