"""Crop-mode UI: a tile grid for editing a large cloud chunk by chunk.

Choose a grid over the cloud's XY extent and a context margin, then double-click
a tile to load just that region (plus margin) into the editor.  Fix it, Save
Tile to write those points back into the output file, and move to the next tile.
"""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .crop import CropSession
from .icons import icon
from .viewer import add_cloud_layer


class CropController:
    """Owns the crop session, the current tile, and the editing controller."""

    def __init__(self, viewer, session: CropSession, seg_controller,
                 point_size: float = 0.01):
        self.viewer = viewer
        self.session = session
        self.seg = seg_controller
        self.point_size = point_size
        self.info = None
        # Set by the widget to mark a tile done after any save (panel or crop).
        self.on_saved = None
        self.seg.on_save_override = self._save

    def load_tile(self, bbox, margin: float) -> str:
        cloud, info = self.session.load_crop(bbox, margin)
        self.info = info

        if self.seg.layer is not None and self.seg.layer in self.viewer.layers:
            self.viewer.layers.remove(self.seg.layer)
        layer = add_cloud_layer(self.viewer, cloud, point_size=self.point_size)
        self.seg.set_cloud(cloud, layer)
        self.viewer.reset_view()
        return (
            f"Loaded tile: {info.n_core:,} core points "
            f"(+{cloud.n_points - info.n_core:,} margin)"
        )

    def _save(self) -> str:
        if self.info is None:
            return "Load a tile first"
        msg = self.session.save_crop(self.seg.cloud, self.info)
        if self.on_saved is not None:
            self.on_saved(self.info.bbox)
        return msg


class CropWidget(QWidget):
    """Grid controls + a table of tiles, docked on the left."""

    COLUMNS = ["Tile", "X range", "Y range", "Done"]

    def __init__(self, controller: CropController):
        super().__init__()
        self.c = controller
        self._tiles = []
        self._done = set()
        layout = QVBoxLayout(self)

        x0, y0, x1, y1 = controller.session.bounds
        layout.addWidget(QLabel(
            f"Extent X [{x0:.1f}, {x1:.1f}]  Y [{y0:.1f}, {y1:.1f}]\n"
            f"{controller.session.count:,} points → {controller.session.output}"
        ))

        grid = QHBoxLayout()
        grid.addWidget(QLabel("Grid"))
        self.nx = QSpinBox(); self.nx.setRange(1, 64); self.nx.setValue(4)
        self.ny = QSpinBox(); self.ny.setRange(1, 64); self.ny.setValue(4)
        grid.addWidget(self.nx); grid.addWidget(QLabel("×")); grid.addWidget(self.ny)
        grid.addWidget(QLabel("Margin"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 1000.0); self.margin.setValue(2.0)
        self.margin.setSuffix(" m")
        grid.addWidget(self.margin)
        make_btn = QPushButton("Make grid")
        make_btn.clicked.connect(self._make_grid)
        grid.addWidget(make_btn)
        layout.addLayout(grid)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            len(self.COLUMNS) - 1, QHeaderView.Stretch
        )
        self.table.cellDoubleClicked.connect(lambda *_: self.on_load_tile())
        layout.addWidget(self.table)

        controller.on_saved = self._mark_done

        row = QHBoxLayout()
        for text, slot, name in [("Load Tile", self.on_load_tile, "folder"),
                                 ("Save Tile", self.on_save, "save"),
                                 ("Save && Next", self.on_save_next, "next")]:
            btn = QPushButton(text)
            btn.setIcon(icon(name))
            btn.setIconSize(QSize(18, 18))
            btn.clicked.connect(slot)
            row.addWidget(btn)
        layout.addLayout(row)

        self._make_grid()

    def _make_grid(self) -> None:
        self._tiles = self.c.session.tiles(self.nx.value(), self.ny.value())
        self._done.clear()
        self.table.setRowCount(len(self._tiles))
        for r, (x0, y0, x1, y1) in enumerate(self._tiles):
            cells = [str(r), f"{x0:.1f}–{x1:.1f}", f"{y0:.1f}–{y1:.1f}", ""]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                item.setData(Qt.UserRole, r)
                self.table.setItem(r, col, item)

    def _selected_tile(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 0).data(Qt.UserRole)

    def on_load_tile(self) -> None:
        idx = self._selected_tile()
        if idx is None:
            self.c.viewer.status = "Select a tile row first"
            return
        self.c.viewer.status = self.c.load_tile(self._tiles[idx], self.margin.value())

    def on_save(self) -> None:
        self.c.viewer.status = self.c._save()

    def on_save_next(self) -> None:
        """Save the current tile, then load the next tile not yet done."""
        if self.c.info is not None:
            msg = self.c._save()
            if "Saved" not in msg:
                self.c.viewer.status = msg
                return
        start = self.table.currentRow()
        order = list(range(start + 1, len(self._tiles))) + list(range(0, start + 1))
        nxt = next((r for r in order if r not in self._done), None)
        if nxt is None:
            self.c.viewer.status = "All tiles done ✓"
            return
        self.table.selectRow(nxt)
        self.c.viewer.status = self.c.load_tile(
            self._tiles[nxt], self.margin.value()
        )

    def _mark_done(self, bbox) -> None:
        for r, box in enumerate(self._tiles):
            if tuple(box) == tuple(bbox):
                self._done.add(r)
                self.table.item(r, 3).setText("✓")
                break
