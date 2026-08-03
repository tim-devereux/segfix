"""Qt dock widget wiring the napari point selection to the fix operations.

The workflow is a review queue.  The table lists every tree; Space marks the
current tree done and jumps to the next unfinished one, flying the camera to
it and focusing the view on the tree, its neighbours and the unassigned pool.
The current tree is always the implicit target: lasso points (L) and press A
to add them to it, Shift+A to absorb whole fragments, N to split a new tree
off, U/X to unassign or trash.  Seeds+grow remains for intermingled crowns.
"""

from __future__ import annotations

import json
import os

import numpy as np
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QBrush, QColor, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import operations as ops
from .icons import icon
from .lasso import LassoTool
from .model import NOISE, UNASSIGNED, PointCloud
from .viewer import colors_for_labels, refresh_layer, visibility_mask


class SegFixController:
    """Holds the editable cloud + napari layer and applies operations."""

    def __init__(self, viewer, cloud: PointCloud, layer):
        self.viewer = viewer
        self.cloud = cloud
        self.layer = layer
        self.save_path = cloud.source_path
        # Optional override: when set (project/crop mode), Save delegates here
        # instead of writing a single file. Signature: () -> str.
        self.on_save_override = None
        # Set by the panel so it can re-hook layer events after a reload.
        self.on_cloud_changed = None
        self.lasso = LassoTool(viewer, layer, self._on_lasso)

    def set_cloud(self, cloud: PointCloud, layer) -> None:
        """Re-point the controller at a freshly loaded cloud/layer."""
        was_armed = self.lasso.armed
        self.lasso.set_armed(False)
        self.cloud = cloud
        self.layer = layer
        self.save_path = cloud.source_path
        self.lasso = LassoTool(self.viewer, layer, self._on_lasso)
        if was_armed:
            self.lasso.set_armed(True)
        if self.on_cloud_changed is not None:
            self.on_cloud_changed()

    def _on_lasso(self, indices: np.ndarray, additive: bool) -> None:
        current = set(self.layer.selected_data) if additive else set()
        current.update(int(i) for i in indices)
        self.layer.selected_data = current
        self.viewer.status = f"Lasso selected {len(current)} points"

    def selected_indices(self) -> np.ndarray:
        return np.fromiter(self.layer.selected_data, dtype=np.int64)

    def _after_edit(self, message: str) -> None:
        refresh_layer(self.layer, self.cloud)
        self.layer.selected_data = set()
        self.viewer.status = message


class SegFixWidget(QWidget):
    """The dock panel: the tree queue plus the few ops the loop needs."""

    SEED_COLORS = ["red", "cyan", "yellow", "magenta", "lime", "orange"]
    DONE_BG = QColor(35, 62, 42)  # green tint marking finished rows
    BBOX_LAYER = "tree bbox"

    def __init__(self, controller: SegFixController):
        super().__init__()
        self.c = controller
        self.current: int | None = None  # the tree under review
        self.focused: set[int] | None = None  # focus mode: visible tree IDs
        self.done_ids: set[int] = set()  # trees marked done in the table
        # Fragment-merge suggestions from the scan: {fragment: (host, gap)}.
        self.suggestions: dict[int, tuple[int, float]] = {}
        self.seeds: list[np.ndarray] = []
        self._table_updating = False
        self._bbox_ids: set[int] = set()
        self._bbox_busy = False
        controller.on_cloud_changed = self._on_cloud_changed
        layout = QVBoxLayout(self)

        self.info = QLabel()
        layout.addWidget(self.info)

        # -- the queue: one row per tree, click = review that tree ------
        self.trees_box = QGroupBox("Trees")
        tlay = QVBoxLayout(self.trees_box)
        self.tree_table = QTableWidget(0, 3)
        self.tree_table.setHorizontalHeaderLabels(["✓", "Tree", "Points"])
        self.tree_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree_table.setSortingEnabled(True)
        self.tree_table.verticalHeader().setVisible(False)
        header = self.tree_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree_table.setMaximumHeight(200)
        self.tree_table.itemSelectionChanged.connect(self._on_table_selection)
        self.tree_table.itemChanged.connect(self._on_tree_item_changed)
        tlay.addWidget(self.tree_table)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav_row.addWidget(prev_btn)
        self.done_btn = QPushButton("✓ Done — next (Space)")
        self.done_btn.setIcon(icon("next"))
        self.done_btn.setIconSize(QSize(18, 18))
        self.done_btn.clicked.connect(self.on_done_next)
        nav_row.addWidget(self.done_btn, stretch=1)
        tlay.addLayout(nav_row)

        # Focus mode: only the current tree, its neighbours and unassigned
        # points stay visible, so one tree is fixed without clutter.
        focus_row = QHBoxLayout()
        self.focus_check = QCheckBox("Focus current + neighbours")
        self.focus_check.setChecked(True)
        self.focus_check.setToolTip(
            "Show only the tree under review, its neighbouring trees, and "
            "unassigned points."
        )
        self.focus_check.toggled.connect(self._on_focus_changed)
        focus_row.addWidget(self.focus_check, stretch=1)
        focus_row.addWidget(QLabel("reach"))
        self.focus_margin = QDoubleSpinBox()
        self.focus_margin.setRange(0.1, 50.0)
        self.focus_margin.setDecimals(1)
        self.focus_margin.setSingleStep(0.5)
        self.focus_margin.setValue(1.0)
        self.focus_margin.setSuffix(" m")
        self.focus_margin.setToolTip(
            "Neighbour reach: trees whose points come within this distance "
            "of the current tree's points count as neighbours."
        )
        self.focus_margin.valueChanged.connect(self._on_focus_changed)
        focus_row.addWidget(self.focus_margin)
        tlay.addLayout(focus_row)
        self._button(
            tlay, "Find fragments (F)", self.on_scan_fragments, "merge"
        )
        layout.addWidget(self.trees_box)

        # -- interaction mode: move (camera) vs lasso -----------------
        self.mode_label = QLabel()
        self.mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mode_label)
        mode_row = QHBoxLayout()
        self.move_btn = QPushButton("Move (Esc)")
        self.move_btn.setIcon(icon("move"))
        self.move_btn.setIconSize(QSize(18, 18))
        self.move_btn.setCheckable(True)
        self.move_btn.setChecked(True)
        self.move_btn.clicked.connect(self.on_move_mode)
        mode_row.addWidget(self.move_btn)
        self.lasso_btn = QPushButton("Lasso select (L)")
        self.lasso_btn.setIcon(icon("lasso"))
        self.lasso_btn.setIconSize(QSize(18, 18))
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.toggled.connect(self.on_toggle_lasso)
        mode_row.addWidget(self.lasso_btn)
        layout.addLayout(mode_row)
        self._update_mode_indicator()

        view_row = QHBoxLayout()
        self.show_unassigned = QCheckBox("Show unassigned (H)")
        self.show_unassigned.setChecked(True)
        self.show_unassigned.toggled.connect(self._on_show_unassigned)
        view_row.addWidget(self.show_unassigned, stretch=1)
        view_row.addWidget(QLabel("Point size"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.001, 1.0)
        self.size_spin.setDecimals(3)
        self.size_spin.setSingleStep(0.005)
        self.size_spin.setSuffix(" m")
        self.size_spin.setValue(0.01)
        self.size_spin.valueChanged.connect(self._on_point_size)
        view_row.addWidget(self.size_spin)
        layout.addLayout(view_row)

        # -- fixing the current tree ----------------------------------
        sel_box = QGroupBox("Current tree")
        sel = QVBoxLayout(sel_box)
        crow = QHBoxLayout()
        self.current_swatch = QLabel()
        self.current_swatch.setFixedSize(18, 18)
        crow.addWidget(self.current_swatch)
        self.current_label = QLabel()
        crow.addWidget(self.current_label, stretch=1)
        sel.addLayout(crow)
        self.suggest_label = QLabel()
        self.suggest_label.setStyleSheet("color: #e0c060; font-weight: bold;")
        self.suggest_label.setVisible(False)
        sel.addWidget(self.suggest_label)
        self.accept_btn = QPushButton()
        self.accept_btn.setIcon(icon("merge"))
        self.accept_btn.setIconSize(QSize(18, 18))
        self.accept_btn.clicked.connect(self.on_accept_suggestion)
        self.accept_btn.setVisible(False)
        sel.addWidget(self.accept_btn)
        self.sel_info = QLabel()
        sel.addWidget(self.sel_info)
        self.add_btn = QPushButton()
        self.add_btn.setIcon(icon("reassign"))
        self.add_btn.setIconSize(QSize(18, 18))
        self.add_btn.clicked.connect(self.on_add)
        sel.addWidget(self.add_btn)
        absorb_btn = QPushButton("Absorb touched trees (Shift+A)")
        absorb_btn.setIcon(icon("merge"))
        absorb_btn.setIconSize(QSize(18, 18))
        absorb_btn.setToolTip(
            "Every tree the selection touches is merged whole into the "
            "current tree — lasso a few points of each fragment."
        )
        absorb_btn.clicked.connect(self.on_absorb)
        sel.addWidget(absorb_btn)
        self._button(sel, "Split off as new tree (N)", self.on_create_new, "new")
        self._button(sel, "Unassign (U)", self.on_unassign, "unassign")
        self._button(sel, "Noise (X)", self.on_noise, "noise")
        layout.addWidget(sel_box)

        # -- seeds + grow for intermingled crowns (collapsed) ----------
        self.grow_box = QGroupBox("Untangle intermingled trees")
        self.grow_box.setCheckable(True)
        self.grow_box.setChecked(False)
        box_lay = QVBoxLayout(self.grow_box)
        grow_inner = QWidget()
        grow = QVBoxLayout(grow_inner)
        grow.setContentsMargins(0, 0, 0, 0)
        seed_row = QHBoxLayout()
        self.seed_info = QLabel("no seeds")
        seed_row.addWidget(self.seed_info, stretch=1)
        self._button(seed_row, "Add seed (D)", self.on_add_seed, "seed")
        self._button(seed_row, "Clear", self.on_clear_seeds, "clear")
        grow.addLayout(seed_row)

        params = QHBoxLayout()
        params.addWidget(QLabel("k"))
        self.grow_k = QSpinBox()
        self.grow_k.setRange(2, 32)
        self.grow_k.setValue(8)
        self.grow_k.setToolTip(
            "Neighbours per point in the growth graph. Lower follows thin "
            "strands more strictly; higher tolerates gappy scans."
        )
        params.addWidget(self.grow_k)
        params.addWidget(QLabel("max link"))
        self.grow_link = QDoubleSpinBox()
        self.grow_link.setRange(0.0, 10.0)
        self.grow_link.setDecimals(2)
        self.grow_link.setSingleStep(0.05)
        self.grow_link.setValue(0.0)
        self.grow_link.setSuffix(" m")
        self.grow_link.setSpecialValueText("off")
        self.grow_link.setToolTip(
            "Sever graph links longer than this, so growth can't leak across "
            "branches that merely touch. Cut-off points keep their label. "
            "Try 0.2–0.5 m for tangled crowns; 'off' = unlimited."
        )
        params.addWidget(self.grow_link)
        params.addStretch()
        grow.addLayout(params)

        self.grow_claim = QCheckBox("Claim unassigned points near the trees")
        self.grow_claim.setToolTip(
            "Also assign unassigned points around the involved trees (their "
            "bounding box + 2 m) to the nearest seed — pulls unsegmented "
            "canopy into the right tree."
        )
        grow.addWidget(self.grow_claim)
        self._button(grow, "Grow from seeds (G)", self.on_grow, "grow")
        box_lay.addWidget(grow_inner)
        grow_inner.setVisible(False)
        self.grow_box.toggled.connect(grow_inner.setVisible)
        layout.addWidget(self.grow_box)

        # -- history + save ------------------------------------------
        hist = QHBoxLayout()
        self._button(hist, "Undo (Ctrl+Z)", self.on_undo, "undo")
        self._button(hist, "Redo", self.on_redo, "redo")
        layout.addLayout(hist)

        save = QHBoxLayout()
        self._button(save, "Save (Ctrl+S)", self.on_save, "save")
        self._button(save, "Save As…", self.on_save_as, "save")
        layout.addLayout(save)

        layout.addStretch()
        self._on_cloud_changed()

    # -- helpers -----------------------------------------------------
    def _button(self, parent_layout, text, slot, icon_name=None) -> None:
        btn = QPushButton(text)
        if icon_name:
            btn.setIcon(icon(icon_name))
            btn.setIconSize(QSize(18, 18))
        btn.clicked.connect(slot)
        parent_layout.addWidget(btn)

    def _on_point_size(self, value: float) -> None:
        if self.c.layer is not None and len(self.c.layer.data):
            self.c.layer.size = value

    def on_toggle_lasso(self, checked: bool) -> None:
        self.c.lasso.set_armed(checked)
        self.move_btn.blockSignals(True)
        self.move_btn.setChecked(not checked)
        self.move_btn.blockSignals(False)
        self._update_mode_indicator()

    def on_move_mode(self) -> None:
        """Revert to camera/movement controls (Escape)."""
        self.lasso_btn.setChecked(False)  # disarms lasso via on_toggle_lasso
        self.move_btn.setChecked(True)  # stay checked even if already in move

    def _update_mode_indicator(self) -> None:
        if self.c.lasso.armed:
            self.mode_label.setText("LASSO — drag selects points")
            self.mode_label.setStyleSheet(
                "background: #7a6a20; color: #ffe066;"
                "border-radius: 3px; padding: 3px; font-weight: bold;"
            )
        else:
            self.mode_label.setText("MOVE — drag rotates the view")
            self.mode_label.setStyleSheet(
                "background: #2c4a33; color: #9fd8a8;"
                "border-radius: 3px; padding: 3px; font-weight: bold;"
            )

    def _on_cloud_changed(self) -> None:
        """A new cloud/layer was loaded: reset per-cloud state, re-hook."""
        self.current = None
        self.focused = None
        self.suggestions = {}
        self._load_progress()  # done-tree set lives beside the source file
        self.seeds = []
        self.seed_info.setText("no seeds")
        if "seeds" in self.c.viewer.layers:
            self.c.viewer.layers.remove("seeds")
        self._on_point_size(self.size_spin.value())  # size persists across loads
        self._update_info()
        self._update_selection()
        try:
            self.c.layer.events.highlight.connect(
                lambda *_: self._update_selection()
            )
        except AttributeError:
            pass  # selection count then only refreshes after edits
        # Layer-level bindings shadow napari's own Points shortcuts (delete
        # removes points from the layer; 'a' selects all; Space pans).
        layer_keys = {
            "Delete": self.on_noise,
            "Backspace": self.on_noise,
            "a": self.on_add,
            "Shift-a": self.on_absorb,
            "Space": self.on_done_next,
        }
        for key, fn in layer_keys.items():
            self.c.layer.bind_key(key, lambda _l, _f=fn: _f(), overwrite=True)

    def _update_info(self) -> None:
        cloud = self.c.cloud
        self.info.setText(f"{cloud.n_points:,} points · {len(cloud.tree_ids)} trees")
        self._refresh_tree_table()

    # -- tree table --------------------------------------------------
    def _refresh_tree_table(self) -> None:
        """Rebuild the table from the cloud: done, swatch+ID, point count."""
        cloud = self.c.cloud
        vals, counts = np.unique(cloud.labels, return_counts=True)
        real = (vals != UNASSIGNED) & (vals != NOISE)
        vals, counts = vals[real], counts[real]
        rgba = colors_for_labels(vals, cloud.label_colors)

        id_set = {int(t) for t in vals}
        self._table_updating = True
        self.tree_table.blockSignals(True)
        self.tree_table.setSortingEnabled(False)
        self.tree_table.clearSelection()
        self.tree_table.setRowCount(len(vals))
        for row, (tid, count) in enumerate(zip(vals, counts)):
            done = int(tid) in self.done_ids
            done_item = QTableWidgetItem()
            done_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            done_item.setCheckState(Qt.Checked if done else Qt.Unchecked)
            self.tree_table.setItem(row, 0, done_item)
            id_item = QTableWidgetItem()
            id_item.setData(Qt.DisplayRole, int(tid))  # int → numeric sort
            id_item.setData(Qt.UserRole, int(tid))
            r, g, b = (int(v * 255) for v in rgba[row][:3])
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(r, g, b))
            id_item.setData(Qt.DecorationRole, swatch)
            sug = self.suggestions.get(int(tid))
            if sug is not None and sug[0] in id_set:
                id_item.setForeground(QBrush(QColor("#e0c060")))
                id_item.setToolTip(
                    f"fragment? → tree {sug[0]} ({sug[1] * 100:.0f} cm gap) "
                    "— Y to absorb"
                )
            self.tree_table.setItem(row, 1, id_item)
            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, int(count))
            self.tree_table.setItem(row, 2, count_item)
            self._style_done_row(row, done)
        self.tree_table.setSortingEnabled(True)
        self.tree_table.blockSignals(False)
        self._table_updating = False
        self._update_done_title()
        self._sync_current({int(t) for t in vals})

    def _sync_current(self, existing: set[int]) -> None:
        """After a rebuild: keep reviewing the same tree, or advance if it
        vanished (absorbed into another tree, trashed whole, …)."""
        if self.current is not None and self.current not in existing:
            self._set_current(self._next_pending(-1))
        else:
            self._set_current(self.current, fly=False)

    def _row_of(self, tid: int) -> int | None:
        for row in range(self.tree_table.rowCount()):
            if self.tree_table.item(row, 1).data(Qt.UserRole) == tid:
                return row
        return None

    def _row_id(self, row: int) -> int:
        return int(self.tree_table.item(row, 1).data(Qt.UserRole))

    def _on_table_selection(self) -> None:
        if self._table_updating:
            return
        rows = {item.row() for item in self.tree_table.selectedItems()}
        self._set_current(self._row_id(rows.pop()) if rows else None)

    # -- the review queue ---------------------------------------------
    def _set_current(self, tid: int | None, fly: bool = True) -> None:
        """Make ``tid`` the tree under review: select its row, focus the
        view on it, box it, and (on a real change) fly the camera there."""
        changed = tid != self.current
        self.current = tid
        self._table_updating = True
        row = self._row_of(tid) if tid is not None else None
        if row is None:
            self.tree_table.clearSelection()
        else:
            self.tree_table.selectRow(row)
        self._table_updating = False
        if changed:
            self.c.layer.selected_data = set()
        self._refresh_focus()
        self._update_tree_bbox([] if tid is None else [tid])
        self._update_current_info()
        if changed and fly and tid is not None:
            self._fly_to(tid)
            self.c.viewer.status = (
                f"Tree {tid} — lasso (L) then A/N/U/X to fix, "
                "Space to mark done and continue"
            )

    def _step(self, delta: int) -> None:
        """Prev/Next: move through the table in its current order."""
        rows = self.tree_table.rowCount()
        if not rows:
            return
        cur = self._row_of(self.current) if self.current is not None else None
        row = 0 if cur is None else (cur + delta) % rows
        self._set_current(self._row_id(row))

    def _next_pending(self, after_row: int) -> int | None:
        """First not-done tree after ``after_row`` in table order, wrapping."""
        rows = self.tree_table.rowCount()
        for step in range(1, rows + 1):
            tid = self._row_id((after_row + step) % rows)
            if tid not in self.done_ids:
                return tid
        return None

    def on_done_next(self) -> None:
        """Space: mark the current tree done, save, jump to the next."""
        if not self.tree_table.rowCount():
            self.c.viewer.status = "No trees to review"
            return
        start = -1
        if self.current is not None:
            self._mark_done(self.current, True)
            start = self._row_of(self.current)
        nxt = self._next_pending(start if start is not None else -1)
        if nxt is None:
            self._set_current(None)
            self.c.viewer.status = "All trees done — save when ready"
            return
        self._set_current(nxt)

    def _fly_to(self, tid: int) -> None:
        """Centre the camera on the tree and zoom to roughly fit it."""
        cloud = self.c.cloud
        pts = cloud.coords[cloud.labels == tid]
        if not len(pts):
            return
        center = pts.mean(axis=0)
        order = self.c.viewer.dims.order  # camera axes follow display order
        self.c.viewer.camera.center = tuple(float(center[d]) for d in order)
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        canvas = getattr(self.c.viewer, "_canvas_size", (800, 800))
        self.c.viewer.camera.zoom = 0.7 * min(canvas) / max(span, 0.5)

    # -- focus mode ---------------------------------------------------
    def _on_focus_changed(self, *_) -> None:
        self._refresh_focus()

    def _refresh_focus(self) -> None:
        if self.current is None or not self.focus_check.isChecked():
            self.focused = None
        else:
            self.focused = self._neighbourhood([self.current])
        self._apply_visibility()

    def _neighbourhood(self, ids) -> set[int]:
        """The given trees plus every tree whose points come within reach."""
        from . import analysis

        focus = {int(t) for t in ids}
        for tid in list(focus):
            focus |= analysis.neighbours_by_points(
                self.c.cloud, tid, self.focus_margin.value()
            )
        return focus

    def _apply_visibility(self) -> None:
        if self.c.layer is None or not len(self.c.layer.data):
            return
        self.c.layer.shown = visibility_mask(
            self.c.cloud.labels,
            hide_unassigned=not self.show_unassigned.isChecked(),
            focus=self.focused,
        )

    def _on_show_unassigned(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.viewer.status = (
            "Unassigned/noise points shown" if checked
            else "Unassigned/noise points hidden"
        )

    # -- done tracking ------------------------------------------------
    def _on_tree_item_changed(self, item) -> None:
        if self._table_updating or item.column() != 0:
            return
        tid = self._row_id(item.row())
        self._mark_done(tid, item.checkState() == Qt.Checked)

    def _mark_done(self, tid: int, done: bool) -> None:
        if done:
            self.done_ids.add(tid)
        else:
            self.done_ids.discard(tid)
        row = self._row_of(tid)
        if row is not None:
            self._table_updating = True  # styling sets data; don't re-enter
            state = Qt.Checked if done else Qt.Unchecked
            self.tree_table.item(row, 0).setCheckState(state)
            self._style_done_row(row, done)
            self._table_updating = False
        self._update_done_title()
        path = self._save_progress()
        self.c.viewer.status = (
            f"Tree {tid} marked {'done' if done else 'not done'}"
            + (f" — saved to {os.path.basename(path)}" if path else "")
        )

    def _style_done_row(self, row: int, done: bool) -> None:
        brush = QBrush(self.DONE_BG) if done else QBrush()
        for col in range(self.tree_table.columnCount()):
            it = self.tree_table.item(row, col)
            if it is not None:
                it.setBackground(brush)

    def _update_done_title(self) -> None:
        n = self.tree_table.rowCount()
        done = sum(
            1 for row in range(n) if self._row_id(row) in self.done_ids
        )
        self.trees_box.setTitle(f"Trees — {done}/{n} done" if n else "Trees")

    def _progress_path(self) -> str | None:
        src = self.c.cloud.source_path or self.c.save_path
        return f"{src}.segfix.json" if src else None

    def _load_progress(self) -> None:
        """Restore the done-tree set from the sidecar file, if present."""
        self.done_ids = set()
        path = self._progress_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.done_ids = {int(t) for t in data.get("done", [])}
        except (OSError, ValueError) as exc:
            self.c.viewer.status = f"Could not read progress file: {exc}"

    def _save_progress(self) -> str | None:
        """Write the done-tree set next to the source file; returns the path."""
        path = self._progress_path()
        if not path:
            return None
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"done": sorted(self.done_ids)}, f)
        except OSError as exc:
            self.c.viewer.status = f"Could not save progress: {exc}"
            return None
        return path

    # -- bounding box around the current tree -------------------------
    def _update_tree_bbox(self, ids) -> None:
        """Draw a wireframe bounding box around each given tree (or clear).

        Called on every current-tree change, so it no-ops when the set of
        boxed trees is unchanged rather than rebuilding the layer.
        """
        ids = {int(t) for t in ids}
        if ids == self._bbox_ids or self._bbox_busy:
            # Never rebuild re-entrantly: adding/removing the layer fires
            # events (active-layer change → highlight) that call back in here.
            return
        self._bbox_busy = True
        try:
            self._rebuild_tree_bbox(ids)
        finally:
            self._bbox_busy = False

    def _rebuild_tree_bbox(self, ids: set[int]) -> None:
        self._bbox_ids = ids
        viewer = self.c.viewer
        if self.BBOX_LAYER in viewer.layers:
            viewer.layers.remove(self.BBOX_LAYER)
        edges, colors = [], []
        for tid in ids:
            pts = self.c.cloud.coords[self.c.cloud.labels == tid]
            if not len(pts):
                continue
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            # Corner i picks hi for each set bit of its index, lo otherwise;
            # edges join corners whose indices differ in exactly one bit.
            corners = np.array(
                [[(hi if i & (1 << b) else lo)[b] for b in range(3)]
                 for i in range(8)]
            )
            rgba = colors_for_labels(
                np.array([tid]), self.c.cloud.label_colors
            )[0]
            for i in range(8):
                for b in range(3):
                    if not i & (1 << b):
                        j = i | (1 << b)
                        edges.append([corners[i], corners[j] - corners[i]])
                        colors.append(rgba)
        if edges:
            viewer.add_vectors(
                np.asarray(edges, dtype=np.float32),
                name=self.BBOX_LAYER,
                edge_color=np.asarray(colors),
                edge_width=0.03,
                vector_style="line",
            )
            viewer.layers.selection.active = self.c.layer
        self.c.lasso.reassert()  # layer add/remove re-enabled the camera

    # -- selection info -----------------------------------------------
    def _update_selection(self) -> None:
        idx = self.c.selected_indices()
        if idx.size == 0:
            self.sel_info.setText("No selection")
            return
        trees = self._selected_trees(idx)
        self.sel_info.setText(
            f"Selected: {idx.size:,} points across {len(trees)} tree(s)"
        )

    def _selected_trees(self, idx: np.ndarray) -> np.ndarray:
        """Distinct real tree IDs (no unassigned/noise) under the selection."""
        labels = self.c.cloud.labels[idx]
        return np.unique(labels[(labels != UNASSIGNED) & (labels != NOISE)])

    def _update_current_info(self) -> None:
        self._update_suggestion_ui()
        if self.current is None:
            self.current_swatch.setStyleSheet(
                "background: none; border: 1px dashed gray;"
            )
            self.current_label.setText("none — press Space or click a row")
            self.add_btn.setText("Add selection (A)")
            return
        rgba = colors_for_labels(
            np.array([self.current]), self.c.cloud.label_colors
        )[0]
        r, g, b = (int(v * 255) for v in rgba[:3])
        self.current_swatch.setStyleSheet(f"background: rgb({r},{g},{b});")
        n = int(np.count_nonzero(self.c.cloud.labels == self.current))
        self.current_label.setText(f"tree {self.current} · {n:,} points")
        self.add_btn.setText(f"Add selection to tree {self.current} (A)")

    def _update_suggestion_ui(self) -> None:
        sug = (
            self.suggestions.get(self.current)
            if self.current is not None
            else None
        )
        show = sug is not None and bool(
            np.any(self.c.cloud.labels == sug[0])
        )
        self.suggest_label.setVisible(show)
        self.accept_btn.setVisible(show)
        if show:
            host, gap = sug
            self.suggest_label.setText(
                f"⚑ fragment of tree {host}? ({gap * 100:.0f} cm gap)"
            )
            self.accept_btn.setText(f"Absorb into tree {host} (Y)")

    # -- fragment suggestions ------------------------------------------
    def on_scan_fragments(self) -> None:
        """Scan for floating/tiny trees touching a host; flag them ⚑."""
        from . import analysis

        self.suggestions = analysis.find_fragments(self.c.cloud)
        self._refresh_tree_table()
        if not self.suggestions:
            self.c.viewer.status = (
                "No fragment suggestions — no floating or tiny tree touches "
                "another tree"
            )
            return
        self.c.viewer.status = (
            f"{len(self.suggestions)} fragment(s) flagged ⚑ in the table — "
            "Y absorbs the current one into its host"
        )
        self._set_current(self._next_suggested())

    def _next_suggested(self) -> int | None:
        """Best remaining suggestion whose fragment and host still exist."""
        labels = self.c.cloud.labels
        for frag, (host, _gap) in self.suggestions.items():
            if np.any(labels == frag) and np.any(labels == host):
                return frag
        return None

    def on_accept_suggestion(self) -> None:
        """Y: merge the current fragment into its suggested host."""
        sug = (
            self.suggestions.get(self.current)
            if self.current is not None
            else None
        )
        if sug is None:
            self.c.viewer.status = (
                "No fragment suggestion for this tree — press F to scan"
            )
            return
        host, _gap = sug
        frag = self.current
        del self.suggestions[frag]
        if not np.any(self.c.cloud.labels == host):
            self.c.viewer.status = (
                f"Host tree {host} no longer exists — press F to rescan"
            )
            self._update_current_info()
            return
        self._apply(ops.absorb_trees(self.c.cloud, [frag], host))
        nxt = self._next_suggested()
        if nxt is not None:
            self._set_current(nxt)

    def _require_selection(self) -> np.ndarray | None:
        idx = self.c.selected_indices()
        if idx.size == 0:
            self.c.viewer.status = "Select points first (L for lasso)"
            return None
        return idx

    def _require_current(self) -> int | None:
        if self.current is None:
            self.c.viewer.status = (
                "No tree under review — press Space or click a table row"
            )
        return self.current

    # -- ops on the current tree --------------------------------------
    def on_add(self) -> None:
        """Selection → the current tree (missing branches, unassigned…)."""
        tid = self._require_current()
        if tid is None:
            return
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.reassign(self.c.cloud, idx, tid), idx)

    def on_absorb(self) -> None:
        """Whole trees touched by the selection → the current tree."""
        tid = self._require_current()
        if tid is None:
            return
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(
            ops.absorb_trees(self.c.cloud, self._selected_trees(idx), tid), idx
        )

    def on_create_new(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.create_new(self.c.cloud, idx), idx)

    def on_unassign(self) -> None:
        idx = self._selection_or_current_tree("unassign")
        if idx is None:
            return
        self._apply(ops.unassign(self.c.cloud, idx), idx)

    def on_noise(self) -> None:
        idx = self._selection_or_current_tree("trash")
        if idx is None:
            return
        self._apply(ops.mark_noise(self.c.cloud, idx), idx)

    def _selection_or_current_tree(self, verb: str) -> np.ndarray | None:
        """U/X act on the selection, or the whole current tree if nothing is
        selected — that's how a non-tree blob is dismissed in one key."""
        idx = self.c.selected_indices()
        if idx.size:
            return idx
        if self.current is None:
            self.c.viewer.status = (
                f"Select points to {verb} (L), or pick a tree first"
            )
            return None
        return np.flatnonzero(self.c.cloud.labels == self.current)

    def _apply(self, msg: str, idx: np.ndarray | None = None) -> None:
        self.c._after_edit(msg)
        self._update_info()  # rebuilds the table; re-syncs focus/bbox/current
        self._update_selection()

    # -- seeds + grow -------------------------------------------------
    def on_add_seed(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self.seeds.append(idx)
        self.c.layer.selected_data = set()
        self._refresh_seed_markers()
        self.seed_info.setText(f"{len(self.seeds)} seed(s)")
        self.c.viewer.status = (
            f"Seed {len(self.seeds)}: {idx.size} points. "
            "One seed per tree, then Grow (G)."
        )

    def on_clear_seeds(self) -> None:
        self.seeds = []
        self.seed_info.setText("no seeds")
        self._refresh_seed_markers()

    def _refresh_seed_markers(self) -> None:
        viewer = self.c.viewer
        if "seeds" in viewer.layers:
            viewer.layers.remove("seeds")
        if not self.seeds:
            self.c.lasso.reassert()  # layer removal re-enabled the camera
            return
        coords = np.concatenate([self.c.cloud.coords[g] for g in self.seeds])
        colors = sum(
            (
                [self.SEED_COLORS[i % len(self.SEED_COLORS)]] * len(g)
                for i, g in enumerate(self.seeds)
            ),
            [],
        )
        size = float(np.max(self.c.layer.size)) * 5 if len(self.c.layer.data) else 0.05
        viewer.add_points(
            coords, name="seeds", size=size, face_color=colors,
            border_width=0, border_color="transparent",
        )
        viewer.layers.selection.active = self.c.layer
        self.c.lasso.reassert()  # adding a layer re-enabled the camera

    def on_grow(self) -> None:
        if len(self.seeds) < 2:
            self.c.viewer.status = (
                "Add at least two seeds first: lasso a trunk patch, press D"
            )
            return
        seed_idx = np.concatenate(self.seeds)
        msg, grown = ops.grow_from_seeds(
            self.c.cloud,
            self.seeds,
            k=self.grow_k.value(),
            max_edge=self.grow_link.value() or None,
            claim_unassigned=self.grow_claim.isChecked(),
        )
        # Seeds are kept so the result can be inspected and re-grown with
        # different k / max-link settings; Clear drops them when done.
        self._apply(msg, seed_idx)
        if grown.size:
            # Select the grown points (after _apply, which clears selection)
            # so the result is highlighted for inspection.
            self.c.layer.selected_data = set(int(i) for i in grown)
            self._update_selection()

    # -- history -----------------------------------------------------
    def on_undo(self) -> None:
        desc = self.c.cloud.undo()
        self._apply(f"Undid: {desc}" if desc else "Nothing to undo")

    def on_redo(self) -> None:
        desc = self.c.cloud.redo()
        self._apply(f"Redid: {desc}" if desc else "Nothing to redo")

    # -- save --------------------------------------------------------
    def on_save(self) -> None:
        if self.c.on_save_override is not None:
            try:
                msg = self.c.on_save_override()
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return
            self.c.viewer.status = msg
            self._save_progress()
            self._update_info()
            return
        if not self.c.save_path:
            self.on_save_as()
            return
        self._do_save(self.c.save_path)

    def on_save_as(self) -> None:
        start = self.c.save_path or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save point cloud", start,
            "Point clouds (*.las *.laz *.ply)",
        )
        if path:
            self.c.save_path = path
            self._do_save(path)

    def _do_save(self, path: str) -> None:
        from . import io

        try:
            io.save(self.c.cloud, path)
        except Exception as exc:  # surface IO errors instead of crashing
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._save_progress()
        self.c.viewer.status = f"Saved → {path}"


def bind_shortcuts(viewer, panel: SegFixWidget) -> None:
    """One-key bindings so the whole review loop stays on the canvas.

    Bound at viewer level with overwrite so they win over napari defaults
    (notably Ctrl+S, which napari uses for its own layer-save dialog).  Keys
    that clash with Points-layer defaults (a, Space, Delete) are also bound
    at layer level in ``_on_cloud_changed``.
    """
    bindings = {
        "l": lambda v: panel.lasso_btn.toggle(),
        "Escape": lambda v: panel.on_move_mode(),
        "Space": lambda v: panel.on_done_next(),
        "Left": lambda v: panel._step(-1),
        "Right": lambda v: panel._step(1),
        "a": lambda v: panel.on_add(),
        "Shift-a": lambda v: panel.on_absorb(),
        "n": lambda v: panel.on_create_new(),
        "u": lambda v: panel.on_unassign(),
        "x": lambda v: panel.on_noise(),
        "h": lambda v: panel.show_unassigned.toggle(),
        "f": lambda v: panel.on_scan_fragments(),
        "y": lambda v: panel.on_accept_suggestion(),
        "d": lambda v: panel.on_add_seed(),
        "g": lambda v: panel.on_grow(),
        "Control-z": lambda v: panel.on_undo(),
        "Control-Shift-z": lambda v: panel.on_redo(),
        "Control-s": lambda v: panel.on_save(),
    }
    for key, fn in bindings.items():
        viewer.bind_key(key, fn, overwrite=True)
