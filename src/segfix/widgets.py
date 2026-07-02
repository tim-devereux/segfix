"""Qt dock widget wiring the napari point selection to the fix operations.

The workflow is selection-first: pick points with the lasso (L) or napari's
rubber-band select, then hit a single key (or button) to act on them.  Tree IDs
are never typed — the reassign target is picked by eye: select any point of the
tree you want and press T; it shows up as a colour swatch.
"""

from __future__ import annotations

import os

import numpy as np
from qtpy.QtCore import QSize
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
    """The dock panel: lasso, eyedropper target, and one-key operations."""

    SEED_COLORS = ["red", "cyan", "yellow", "magenta", "lime", "orange"]

    def __init__(self, controller: SegFixController):
        super().__init__()
        self.c = controller
        self.target: int | None = None
        self.isolated: set[int] | None = None
        self.seeds: list[np.ndarray] = []
        controller.on_cloud_changed = self._on_cloud_changed
        layout = QVBoxLayout(self)

        self.info = QLabel()
        layout.addWidget(self.info)
        self.sel_info = QLabel()
        layout.addWidget(self.sel_info)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point size"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.001, 1.0)
        self.size_spin.setDecimals(3)
        self.size_spin.setSingleStep(0.005)
        self.size_spin.setSuffix(" m")
        self.size_spin.setValue(0.01)
        self.size_spin.valueChanged.connect(self._on_point_size)
        size_row.addWidget(self.size_spin)
        size_row.addStretch()
        layout.addLayout(size_row)

        # -- selection tool ------------------------------------------
        self.lasso_btn = QPushButton("Lasso select (L)")
        self.lasso_btn.setIcon(icon("lasso"))
        self.lasso_btn.setIconSize(QSize(18, 18))
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.toggled.connect(self.on_toggle_lasso)
        layout.addWidget(self.lasso_btn)

        # -- reassign target (eyedropper) ------------------------------
        target_box = QGroupBox("Target tree")
        trow = QHBoxLayout(target_box)
        self.target_swatch = QLabel()
        self.target_swatch.setFixedSize(18, 18)
        trow.addWidget(self.target_swatch)
        self.target_label = QLabel()
        trow.addWidget(self.target_label, stretch=1)
        self._button(trow, "Set from selection (T)", self.on_set_target, "target")
        layout.addWidget(target_box)

        # -- selection-based ops -------------------------------------
        sel_box = QGroupBox("Selected points")
        sel = QVBoxLayout(sel_box)
        self._button(sel, "Reassign to target (R)", self.on_reassign, "reassign")
        self._button(sel, "New tree (N)", self.on_create_new, "new")
        self._button(
            sel, "Merge trees in selection (M)", self.on_merge_selection, "merge"
        )
        self._button(sel, "Delete — mark noise (X)", self.on_mark_noise, "noise")
        self._button(sel, "Unassign (U)", self.on_unassign, "unassign")

        srow = QHBoxLayout()
        srow.addWidget(QLabel("Split its tree into"))
        self.split_n = QSpinBox()
        self.split_n.setRange(2, 20)
        self.split_n.setValue(2)
        srow.addWidget(self.split_n)
        split_btn = QPushButton("Split (K)")
        split_btn.setIcon(icon("split"))
        split_btn.setIconSize(QSize(18, 18))
        split_btn.clicked.connect(self.on_split_auto)
        srow.addWidget(split_btn)
        sel.addLayout(srow)
        layout.addWidget(sel_box)

        # -- untangling intermingled trees -----------------------------
        grow_box = QGroupBox("Untangle")
        grow = QVBoxLayout(grow_box)
        self.isolate_btn = QPushButton("Isolate selection's trees (I)")
        self.isolate_btn.setIcon(icon("isolate"))
        self.isolate_btn.setIconSize(QSize(18, 18))
        self.isolate_btn.setCheckable(True)
        self.isolate_btn.toggled.connect(self.on_toggle_isolate)
        grow.addWidget(self.isolate_btn)
        self.hide_btn = QPushButton("Hide unassigned/noise (H)")
        self.hide_btn.setIcon(icon("unassign"))
        self.hide_btn.setIconSize(QSize(18, 18))
        self.hide_btn.setCheckable(True)
        self.hide_btn.toggled.connect(self.on_toggle_hide)
        grow.addWidget(self.hide_btn)
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
        layout.addWidget(grow_box)

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
        self.lasso_btn.setText(
            "Lasso ON — drag to select" if checked else "Lasso select (L)"
        )

    def _on_cloud_changed(self) -> None:
        """A new cloud/layer was loaded: reset per-cloud state, re-hook."""
        self._set_target(None)
        self.isolated = None
        self.isolate_btn.blockSignals(True)
        self.isolate_btn.setChecked(False)
        self.isolate_btn.setText("Isolate selection's trees (I)")
        self.isolate_btn.blockSignals(False)
        self.seeds = []
        self.seed_info.setText("no seeds")
        if "seeds" in self.c.viewer.layers:
            self.c.viewer.layers.remove("seeds")
        self._on_point_size(self.size_spin.value())  # size persists across loads
        self._apply_visibility()  # hide-unassigned persists across loads too
        self._update_info()
        self._update_selection()
        try:
            self.c.layer.events.highlight.connect(
                lambda *_: self._update_selection()
            )
        except AttributeError:
            pass  # selection count then only refreshes after edits
        # napari's own Delete/Backspace removes points from the layer, which
        # would desync it from the cloud — route them to mark-as-noise.
        for key in ("Delete", "Backspace"):
            self.c.layer.bind_key(
                key, lambda _l: self.on_mark_noise(), overwrite=True
            )

    def _update_info(self) -> None:
        cloud = self.c.cloud
        self.info.setText(f"{cloud.n_points:,} points · {len(cloud.tree_ids)} trees")

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

    def _require_selection(self) -> np.ndarray | None:
        idx = self.c.selected_indices()
        if idx.size == 0:
            self.c.viewer.status = "Select points first (L for lasso)"
            return None
        return idx

    def _set_target(self, label: int | None) -> None:
        self.target = label
        if label is None:
            self.target_swatch.setStyleSheet("background: none; border: 1px dashed gray;")
            self.target_label.setText("none — select a point, press T")
            return
        rgba = colors_for_labels(np.array([label]), self.c.cloud.label_colors)[0]
        r, g, b = (int(v * 255) for v in rgba[:3])
        self.target_swatch.setStyleSheet(f"background: rgb({r},{g},{b});")
        self.target_label.setText(f"tree {label}")

    # -- target ------------------------------------------------------
    def on_set_target(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        labels = self.c.cloud.labels[idx]
        real = labels[(labels != UNASSIGNED) & (labels != NOISE)]
        if real.size == 0:
            self.c.viewer.status = "Selection has no tree points to target"
            return
        vals, counts = np.unique(real, return_counts=True)
        self._set_target(int(vals[np.argmax(counts)]))
        self.c.layer.selected_data = set()  # target picked; clear for next lasso
        self.c.viewer.status = f"Target set to tree {self.target}"
        self._update_selection()

    # -- selection ops ----------------------------------------------
    def on_reassign(self) -> None:
        if self.target is None:
            self.c.viewer.status = (
                "No target: select a point of the destination tree, press T"
            )
            return
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.reassign(self.c.cloud, idx, self.target), idx)

    def on_create_new(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.create_new(self.c.cloud, idx), idx)

    def on_merge_selection(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.merge_selection(self.c.cloud, idx), idx)

    def on_mark_noise(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.mark_noise(self.c.cloud, idx), idx)

    def on_unassign(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.unassign(self.c.cloud, idx), idx)

    def on_split_auto(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        trees = self._selected_trees(idx)
        if trees.size == 0:
            self.c.viewer.status = "Select points of the tree to split"
            return
        # Split the tree the selection mostly sits on.
        labels = self.c.cloud.labels[idx]
        vals, counts = np.unique(
            labels[(labels != UNASSIGNED) & (labels != NOISE)], return_counts=True
        )
        tree = int(vals[np.argmax(counts)])
        member_idx = np.flatnonzero(self.c.cloud.labels == tree)
        self._apply(
            ops.split_auto(self.c.cloud, tree, self.split_n.value()), member_idx
        )

    def _apply(self, msg: str, idx: np.ndarray | None = None) -> None:
        self.c._after_edit(msg)
        if self.isolated is not None or self.hide_btn.isChecked():
            # Keep edited points visible: trees created/targeted by the edit
            # join the isolated set (their new labels appear at the edited
            # indices), then the visibility mask is recomputed.
            if self.isolated is not None and idx is not None and len(idx):
                self.isolated |= {int(t) for t in self._selected_trees(idx)}
            self._apply_visibility()
        self._update_info()
        self._update_selection()

    # -- visibility: isolate + hide-unassigned --------------------------
    def _apply_visibility(self) -> None:
        if self.c.layer is None or not len(self.c.layer.data):
            return
        self.c.layer.shown = visibility_mask(
            self.c.cloud.labels, self.isolated, self.hide_btn.isChecked()
        )

    def on_toggle_hide(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.viewer.status = (
            "Unassigned/noise points hidden" if checked
            else "Unassigned/noise points shown"
        )

    def on_toggle_isolate(self, checked: bool) -> None:
        layer = self.c.layer
        if not checked:
            self.isolated = None
            self._apply_visibility()
            self.isolate_btn.setText("Isolate selection's trees (I)")
            self.c.viewer.status = "Showing all trees"
            return
        idx = self.c.selected_indices()
        trees = self._selected_trees(idx) if idx.size else np.empty(0)
        if trees.size == 0:
            self.c.viewer.status = "Select points of the trees to isolate first"
            self.isolate_btn.blockSignals(True)
            self.isolate_btn.setChecked(False)
            self.isolate_btn.blockSignals(False)
            return
        self.isolated = {int(t) for t in trees}
        self._apply_visibility()
        layer.selected_data = set()
        self.isolate_btn.setText("Show all (I)")
        self.c.viewer.status = (
            f"Isolated {len(self.isolated)} tree(s) — press I again to show all"
        )

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
        msg = ops.grow_from_seeds(
            self.c.cloud,
            self.seeds,
            k=self.grow_k.value(),
            max_edge=self.grow_link.value() or None,
            claim_unassigned=self.grow_claim.isChecked(),
        )
        self.on_clear_seeds()
        self._apply(msg, seed_idx)

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
        self.c.viewer.status = f"Saved → {path}"


def bind_shortcuts(viewer, panel: SegFixWidget) -> None:
    """One-key bindings so the whole fix loop stays on the canvas.

    Bound at viewer level with overwrite so they win over napari defaults
    (notably Ctrl+S, which napari uses for its own layer-save dialog).
    """
    bindings = {
        "l": lambda v: panel.lasso_btn.toggle(),
        "t": lambda v: panel.on_set_target(),
        "r": lambda v: panel.on_reassign(),
        "n": lambda v: panel.on_create_new(),
        "m": lambda v: panel.on_merge_selection(),
        "x": lambda v: panel.on_mark_noise(),
        "u": lambda v: panel.on_unassign(),
        "k": lambda v: panel.on_split_auto(),
        "i": lambda v: panel.isolate_btn.toggle(),
        "h": lambda v: panel.hide_btn.toggle(),
        "d": lambda v: panel.on_add_seed(),
        "g": lambda v: panel.on_grow(),
        "Control-z": lambda v: panel.on_undo(),
        "Control-Shift-z": lambda v: panel.on_redo(),
        "Control-s": lambda v: panel.on_save(),
    }
    for key, fn in bindings.items():
        viewer.bind_key(key, fn, overwrite=True)
