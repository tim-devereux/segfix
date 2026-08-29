"""Qt dock widget wiring the napari point selection to the fix operations.

The workflow is a review queue.  The table lists the trees currently loaded —
the one picked in scene mode's "All Trees" plus its neighbours, or the whole
file when it was loaded in one go.  Space marks the current tree done, flies
the camera to the next unfinished one and jumps to it.  The current tree is
always the implicit target: lasso points (L) and press A to add them to it,
N to split a new tree off, U/X to unassign or trash.
"""

from __future__ import annotations

import json
import os

import numpy as np
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QBrush, QColor, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import operations as ops
from .icons import icon
from .lasso import LassoTool
from .model import NOISE, UNASSIGNED, PointCloud
from .viewer import (
    busy,
    colors_for_labels,
    refresh_layer,
    visibility_mask,
)


class SegFixController:
    """Holds the editable cloud + napari layer and applies operations."""

    def __init__(self, viewer, cloud: PointCloud, layer):
        self.viewer = viewer
        self.cloud = cloud
        self.layer = layer
        self.save_path = cloud.source_path
        # Tree IDs the panel has "faded": rendered at FADED_ALPHA opacity but
        # still shown and still selectable (unlike a hidden tree). Lives here,
        # not on the widget like hidden_ids, because _after_edit re-applies the
        # face colours and must keep these ghosted through the refresh.
        self.faded_ids: set[int] = set()
        # Optional override: when set (project/scene mode), Save delegates
        # here instead of writing a single file. Signature: () -> str.
        self.on_save_override = None
        # Set by the panel so it can re-hook layer events after a reload.
        self.on_cloud_changed = None
        # Optional fn(indices)->indices narrowing a lasso's result before it
        # becomes the selection — set by the panel for the "current tree
        # only" lasso mode; persists across set_cloud (it's a mode choice,
        # not per-cloud state).
        self.lasso_filter = None
        # Optional fn(indices)->None: when set (drawing a lasso section),
        # a completed lasso is redirected here instead of becoming the
        # selection — see _on_lasso. Also a mode choice, not per-cloud state.
        self.on_lasso_section = None
        self.lasso = LassoTool(viewer, layer, self._on_lasso)

    def set_cloud(self, cloud: PointCloud, layer) -> None:
        """Re-point the controller at a freshly loaded cloud/layer."""
        was_armed = self.lasso.armed
        self.lasso.set_armed(False)
        self.cloud = cloud
        self.layer = layer
        self.save_path = cloud.source_path
        self.faded_ids = set()  # a fresh cloud starts with nothing faded
        self.lasso = LassoTool(self.viewer, layer, self._on_lasso)
        if was_armed:
            self.lasso.set_armed(True)
        if self.on_cloud_changed is not None:
            self.on_cloud_changed()

    def _on_lasso(self, indices: np.ndarray, additive: bool) -> None:
        if self.on_lasso_section is not None:
            self.on_lasso_section(indices)
            return
        if self.lasso_filter is not None:
            indices = self.lasso_filter(indices)
        current = set(self.layer.selected_data) if additive else set()
        current.update(int(i) for i in indices)
        self.layer.selected_data = current
        self.viewer.status = f"Lasso selected {len(current)} points"

    def selected_indices(self) -> np.ndarray:
        return np.fromiter(self.layer.selected_data, dtype=np.int64)

    def _after_edit(self, message: str) -> None:
        refresh_layer(self.layer, self.cloud, self.faded_ids)
        self.layer.selected_data = set()
        self.viewer.status = message


class SegFixWidget(QWidget):
    """The dock panel: the tree queue plus the few ops the loop needs."""

    DONE_BG = QColor(35, 62, 42)  # green tint marking finished rows
    BBOX_LAYER = "tree bbox"
    HIDE_COL = 3
    FADE_COL = 4

    def __init__(self, controller: SegFixController):
        super().__init__()
        self.c = controller
        self.current: int | None = None  # the tree under review
        self.done_ids: set[int] = set()  # trees marked done in the table
        self.hidden_ids: set[int] = set()  # trees manually hidden via 👁
        self._table_updating = False
        # Optional fn()->None: set by scene mode so its own tree table (which
        # mirrors done-state from the same sidecar file) refreshes the moment
        # a tree is marked done here, instead of waiting for the next save.
        self.on_done_changed = None
        self._bbox_ids: set[int] = set()
        self._bbox_busy = False
        controller.on_cloud_changed = self._on_cloud_changed
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.info = QLabel()

        # -- the queue: one row per tree, click = review that tree ------
        # "Selected Tree + Neighbours": just the trees currently loaded into
        # the 3D view (the row picked in "All Trees" above, plus whatever's
        # spatially near it) — not the whole file, which that other table
        # covers. Title is immediately overwritten with the live done count
        # by _update_done_title() below.
        self.trees_box = QGroupBox("Selected Tree + Neighbours")
        tlay = QVBoxLayout(self.trees_box)
        tlay.setSpacing(4)
        tlay.addWidget(self.info)  # "N points · M trees" for the loaded cloud
        self.tree_table = QTableWidget(0, 5)
        self.tree_table.setHorizontalHeaderLabels(
            ["Done", "Tree ID", "Points", "Hide", "Fade"]
        )
        self.tree_table.horizontalHeaderItem(0).setToolTip(
            "Whether this tree has been marked reviewed"
        )
        self.tree_table.horizontalHeaderItem(self.HIDE_COL).setIcon(icon("hide"))
        self.tree_table.horizontalHeaderItem(self.HIDE_COL).setToolTip(
            "Hide this tree from the 3D view"
        )
        self.tree_table.horizontalHeaderItem(self.FADE_COL).setIcon(icon("fade"))
        self.tree_table.horizontalHeaderItem(self.FADE_COL).setToolTip(
            "Fade this tree in the 3D view — ghosted for context, but still "
            "shown and still selectable"
        )
        self.tree_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree_table.setSortingEnabled(True)
        self.tree_table.verticalHeader().setVisible(False)
        header = self.tree_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.HIDE_COL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.FADE_COL, QHeaderView.ResizeToContents)
        self.tree_table.setMaximumHeight(200)
        self.tree_table.itemSelectionChanged.connect(self._on_table_selection)
        self.tree_table.itemChanged.connect(self._on_tree_item_changed)
        tlay.addWidget(self.tree_table)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("Prev")
        prev_btn.setIcon(icon("prev"))
        prev_btn.setIconSize(QSize(18, 18))
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav_row.addWidget(prev_btn)
        self.done_btn = QPushButton("✓ Done — next (Space)")
        self.done_btn.setIcon(icon("next"))
        self.done_btn.setIconSize(QSize(18, 18))
        self.done_btn.clicked.connect(self.on_done_next)
        nav_row.addWidget(self.done_btn, stretch=1)
        tlay.addLayout(nav_row)

        layout.addWidget(self.trees_box)

        # Interaction / View / Cross section live in a separate horizontal
        # bar (docked at the top of the window by app.py, not stacked in
        # this side panel) — they're about the canvas/viewport, not the
        # tree-review workflow the rest of this panel is organised around.
        self.top_bar = QWidget()
        top_bar_row = QHBoxLayout(self.top_bar)
        top_bar_row.setContentsMargins(0, 0, 0, 0)
        top_bar_row.setSpacing(6)

        # The top bar is a single thin strip: every group is one content row,
        # and the two section tools collapse to just their title/checkbox until
        # switched on. Keep it that way when adding controls here.

        # -- interaction mode: move (camera) vs lasso -----------------
        # The active mode shows as a coloured checked button — no separate
        # indicator. Exactly one is ever checked (see _uncheck_other_modes /
        # on_move_mode); section_draw_btn, built in the Lasso section box
        # below, is the fourth mutually-exclusive member.
        interaction_box = QGroupBox("Interaction")
        interaction = QHBoxLayout(interaction_box)
        interaction.setSpacing(3)
        self.move_btn = QPushButton("Move (Esc)")
        self.move_btn.setIcon(icon("move"))
        self.move_btn.setIconSize(QSize(18, 18))
        self.move_btn.setCheckable(True)
        self.move_btn.setChecked(True)
        self.move_btn.setToolTip("Drag rotates the view")
        self.move_btn.clicked.connect(self.on_move_mode)
        interaction.addWidget(self.move_btn)
        self.lasso_btn = QPushButton("Lasso (L)")
        self.lasso_btn.setIcon(icon("lasso"))
        self.lasso_btn.setIconSize(QSize(18, 18))
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.setToolTip("Drag selects points")
        self.lasso_btn.toggled.connect(self.on_toggle_lasso)
        interaction.addWidget(self.lasso_btn)
        self.tree_lasso_btn = QPushButton("Lasso tree (Ctrl+L)")
        self.tree_lasso_btn.setIcon(icon("lasso"))
        self.tree_lasso_btn.setIconSize(QSize(18, 18))
        self.tree_lasso_btn.setCheckable(True)
        self.tree_lasso_btn.setToolTip(
            "Freehand-select, but only points already belonging to the tree "
            "under review — grabs a clean patch out of a crowded/overlapping "
            "area without also picking up neighbouring trees."
        )
        self.tree_lasso_btn.toggled.connect(self.on_toggle_tree_lasso)
        interaction.addWidget(self.tree_lasso_btn)
        interaction.addStretch(1)
        # Each mode button lights up in its own colour while active.
        for btn, bg, fg in (
            (self.move_btn, "#2c4a33", "#9fd8a8"),
            (self.lasso_btn, "#7a6a20", "#ffe066"),
            (self.tree_lasso_btn, "#6a3a7a", "#e0b3ff"),
        ):
            btn.setStyleSheet(
                f"QPushButton:checked {{ background: {bg}; color: {fg}; "
                "font-weight: bold; }"
            )
        top_bar_row.addWidget(interaction_box)

        # -- view: what's visible/selectable, and how big it renders ---
        # One row: unassigned toggle · declutter buttons · point size. The
        # "others" buttons act on the whole loaded set, not the selection, so
        # they belong here, not with the "Current tree" fix actions.
        view_box = QGroupBox("View")
        view = QVBoxLayout(view_box)
        view.setSpacing(3)
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        self.show_unassigned = QPushButton("Show unassigned (H)")
        self.show_unassigned.setCheckable(True)
        self.show_unassigned.setChecked(True)
        self.show_unassigned.setToolTip(
            "Show or hide the unassigned + noise points"
        )
        self.show_unassigned.setStyleSheet(
            "QPushButton:checked { background: #3a4450; font-weight: bold; }"
        )
        self.show_unassigned.toggled.connect(self._on_show_unassigned)
        view_row.addWidget(self.show_unassigned)
        self._button(view_row, "Hide others", self.on_hide_neighbours, "hide")
        self._button(view_row, "Fade others", self.on_fade_neighbours, "fade")
        view_row.addStretch(1)
        view_row.addWidget(QLabel("Point size"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.001, 1.0)
        self.size_spin.setDecimals(3)
        self.size_spin.setSingleStep(0.005)
        self.size_spin.setSuffix(" m")
        self.size_spin.setValue(0.01)
        self.size_spin.valueChanged.connect(self._on_point_size)
        view_row.addWidget(self.size_spin)
        view.addLayout(view_row)

        top_bar_row.addWidget(view_box)

        # -- cross section: an interactive slab along one axis; while on,
        # only points inside it are shown and selectable (folded into the
        # same layer.shown mechanism as the per-tree hide checkboxes) -----
        self.cross_box = QGroupBox("Cross section (C)")
        self.cross_box.setCheckable(True)
        self.cross_box.setChecked(False)
        self.cross_box.setToolTip(
            "Slice the cloud to a slab along one axis. While enabled, only "
            "points inside the slab are shown or selectable."
        )
        self.cross_box.toggled.connect(self._on_cross_section_toggled)
        cross = QVBoxLayout(self.cross_box)
        cross.setContentsMargins(6, 2, 6, 4)
        # Body collapses to nothing until the tool is switched on, so an idle
        # Cross section is just its title-bar checkbox in the top strip.
        self.cross_body = QWidget()
        self.cross_body.setVisible(False)
        cross.addWidget(self.cross_body)
        cross_body_lay = QVBoxLayout(self.cross_body)
        cross_body_lay.setContentsMargins(0, 0, 0, 0)
        cross_body_lay.setSpacing(3)
        cross_top = QHBoxLayout()
        cross_top.addWidget(QLabel("Axis"))
        self.cross_axis_combo = QComboBox()
        self.cross_axis_combo.addItems(["X", "Y", "Z"])
        self.cross_axis_combo.setCurrentIndex(2)  # Z: height, usually most useful
        self.cross_axis_combo.currentIndexChanged.connect(self._on_cross_axis_changed)
        cross_top.addWidget(self.cross_axis_combo)
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Widen the slab back out to the full extent")
        reset_btn.clicked.connect(self._on_cross_reset)
        cross_top.addWidget(reset_btn)
        self.cross_range_label = QLabel()
        cross_top.addWidget(self.cross_range_label, stretch=1)
        cross_body_lay.addLayout(cross_top)
        slab_row = QHBoxLayout()
        slab_row.addWidget(QLabel("Min"))
        self.cross_min_slider = QSlider(Qt.Horizontal)
        self.cross_min_slider.valueChanged.connect(self._on_cross_range_changed)
        slab_row.addWidget(self.cross_min_slider)
        slab_row.addWidget(QLabel("Max"))
        self.cross_max_slider = QSlider(Qt.Horizontal)
        self.cross_max_slider.valueChanged.connect(self._on_cross_range_changed)
        slab_row.addWidget(self.cross_max_slider)
        cross_body_lay.addLayout(slab_row)
        top_bar_row.addWidget(self.cross_box)
        self._cross_lo, self._cross_hi = 0.0, 1.0
        self._reset_cross_section_range()

        # -- lasso section: same idea as cross section, but the kept
        # region is a hand-drawn outline instead of an axis-aligned slab.
        # Draw arms the shared lasso tool in "section" mode (see
        # SegFixController.on_lasso_section); the outline it produces is
        # a one-shot boolean mask, not a live screen region, so it stays
        # put as the camera moves — same persistence model as the slab.
        self.lasso_section_box = QGroupBox("Lasso section (Shift+C)")
        self.lasso_section_box.setCheckable(True)
        self.lasso_section_box.setChecked(False)
        self.lasso_section_box.setToolTip(
            "Keep only points inside a hand-drawn outline. Like cross "
            "section, but the kept region is whatever shape you draw "
            "instead of a slab along one axis."
        )
        self.lasso_section_box.toggled.connect(self._on_lasso_section_toggled)
        lsec = QVBoxLayout(self.lasso_section_box)
        lsec.setContentsMargins(6, 2, 6, 4)
        self.lasso_body = QWidget()  # collapses until the tool is on
        self.lasso_body.setVisible(False)
        lsec.addWidget(self.lasso_body)
        lsec_body_lay = QVBoxLayout(self.lasso_body)
        lsec_body_lay.setContentsMargins(0, 0, 0, 0)
        lsec_body_lay.setSpacing(3)
        lsec_row = QHBoxLayout()
        self.section_draw_btn = QPushButton("Draw (Shift+L)")
        self.section_draw_btn.setIcon(icon("lasso"))
        self.section_draw_btn.setIconSize(QSize(18, 18))
        self.section_draw_btn.setCheckable(True)
        self.section_draw_btn.setToolTip(
            "Drag on the canvas to outline the region to keep"
        )
        self.section_draw_btn.setStyleSheet(
            "QPushButton:checked { background: #3a5a7a; color: #a8d4ff; "
            "font-weight: bold; }"
        )
        self.section_draw_btn.toggled.connect(self.on_toggle_lasso_section)
        lsec_row.addWidget(self.section_draw_btn)
        section_reset_btn = QPushButton("Reset")
        section_reset_btn.setToolTip("Clear the outline — show every point again")
        section_reset_btn.clicked.connect(self._on_lasso_section_reset)
        lsec_row.addWidget(section_reset_btn)
        lsec_body_lay.addLayout(lsec_row)
        self.lasso_section_label = QLabel()
        lsec_body_lay.addWidget(self.lasso_section_label)
        top_bar_row.addWidget(self.lasso_section_box)
        self._lasso_section_mask: np.ndarray | None = None
        self._update_lasso_section_label()

        top_bar_row.addStretch()

        # -- fixing the current tree ----------------------------------
        sel_box = QGroupBox("Current tree")
        sel = QVBoxLayout(sel_box)
        sel.setSpacing(3)
        crow = QHBoxLayout()
        self.current_swatch = QLabel()
        self.current_swatch.setFixedSize(18, 18)
        crow.addWidget(self.current_swatch)
        self.current_label = QLabel()
        crow.addWidget(self.current_label, stretch=1)
        sel.addLayout(crow)
        self.sel_info = QLabel()
        sel.addWidget(self.sel_info)

        sel.addWidget(self._subheading("Move selection into a tree"))
        self.add_btn = QPushButton()
        self.add_btn.setIcon(icon("reassign"))
        self.add_btn.setIconSize(QSize(18, 18))
        self.add_btn.clicked.connect(self.on_add)
        sel.addWidget(self.add_btn)
        neighbour_header = QHBoxLayout()
        self.neighbour_label = QLabel("Send selection to neighbour:")
        self.neighbour_label.setVisible(False)
        neighbour_header.addWidget(self.neighbour_label, stretch=1)
        neighbour_header.addWidget(QLabel("reach"))
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
        self.focus_margin.valueChanged.connect(self._update_neighbour_picker)
        neighbour_header.addWidget(self.focus_margin)
        sel.addLayout(neighbour_header)
        self.neighbour_row = QHBoxLayout()
        self.neighbour_row.setContentsMargins(0, 0, 0, 0)
        sel.addLayout(self.neighbour_row)

        sel.addWidget(self._subheading("Remove selection from its tree"))
        self._button(sel, "Split off as new tree (N)", self.on_create_new, "new")
        self._button(sel, "Unassign (U)", self.on_unassign, "unassign")
        self._button(sel, "Noise (X)", self.on_noise, "noise")
        layout.addWidget(sel_box)

        # -- session: history + save -----------------------------------
        session_box = QGroupBox("Session")
        session = QVBoxLayout(session_box)
        session.setSpacing(3)
        hist = QHBoxLayout()
        self._button(hist, "Undo (Ctrl+Z)", self.on_undo, "undo")
        self._button(hist, "Redo (Ctrl+Shift+Z)", self.on_redo, "redo")
        session.addLayout(hist)

        save = QHBoxLayout()
        self._button(save, "Save Project (Ctrl+S)", self.on_save, "save")
        session.addLayout(save)
        layout.addWidget(session_box)

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

    def _subheading(self, text: str) -> QLabel:
        """A small bold label dividing a group box into sub-sections,
        lighter-weight than nesting another QGroupBox."""
        label = QLabel(text)
        label.setStyleSheet(
            "font-weight: bold; color: gray; margin-top: 2px;"
        )
        return label

    def _on_point_size(self, value: float) -> None:
        if self.c.layer is not None and len(self.c.layer.data):
            self.c.layer.size = value

    def on_toggle_lasso(self, checked: bool) -> None:
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = None
        self.c.on_lasso_section = None
        if checked:
            self._uncheck_other_modes(self.lasso_btn)
        else:
            self.move_btn.setChecked(True)

    def on_toggle_tree_lasso(self, checked: bool) -> None:
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = self._filter_to_current_tree if checked else None
        self.c.on_lasso_section = None
        if checked:
            self._uncheck_other_modes(self.tree_lasso_btn)
        else:
            self.move_btn.setChecked(True)

    def on_toggle_lasso_section(self, checked: bool) -> None:
        """Arm/disarm the shared lasso tool in "section" mode: a completed
        drag becomes the kept-region outline (via _on_lasso_section_drawn)
        instead of a selection. See SegFixController._on_lasso."""
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = None
        self.c.on_lasso_section = self._on_lasso_section_drawn if checked else None
        if checked:
            self._uncheck_other_modes(self.section_draw_btn)
            self.c.viewer.status = "Lasso section: drag to outline the kept region"
        else:
            self.move_btn.setChecked(True)

    def _filter_to_current_tree(self, indices: np.ndarray) -> np.ndarray:
        """Keep only the points among ``indices`` already in the current
        tree — no restriction if nothing's under review yet."""
        if self.current is None:
            return indices
        return indices[self.c.cloud.labels[indices] == self.current]

    def _uncheck_other_modes(self, active_btn) -> None:
        for btn in (
            self.move_btn, self.lasso_btn, self.tree_lasso_btn, self.section_draw_btn,
        ):
            if btn is active_btn:
                continue
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def on_move_mode(self) -> None:
        """Revert to camera/movement controls (Escape)."""
        self.lasso_btn.setChecked(False)  # disarms lasso via on_toggle_lasso
        self.tree_lasso_btn.setChecked(False)  # ditto, tree-only lasso
        self.section_draw_btn.setChecked(False)  # ditto, lasso-section drawing
        self.move_btn.setChecked(True)  # stay checked even if already in move

    def _on_cloud_changed(self) -> None:
        """A new cloud/layer was loaded: reset per-cloud state, re-hook."""
        self.current = None
        self.hidden_ids = set()
        # A slab computed for the previous cloud's coordinate space doesn't
        # carry over; turn the tool off and recompute its range for this one.
        self.cross_box.setChecked(False)
        self._reset_cross_section_range()
        # Same for a lasso section: its mask is indices into the *previous*
        # cloud's points array, meaningless for a new one.
        self._lasso_section_mask = None
        self.lasso_section_box.setChecked(False)
        self._update_lasso_section_label()
        self._load_progress()  # done-tree set lives beside the source file
        self._on_point_size(self.size_spin.value())  # size persists across loads
        self._update_info()
        self._update_selection()
        try:
            self.c.layer.events.highlight.connect(
                lambda *_: self._update_selection()
            )
        except AttributeError:
            pass  # selection count then only refreshes after edits
        # Layer-level bindings shadow napari's own Points shortcuts ('1',
        # Delete and Backspace all hard-delete points from the layer; 'a'
        # selects all; Space pans).
        layer_keys = {
            "1": self.on_noise,
            "Delete": self.on_noise,
            "Backspace": self.on_noise,
            "a": self.on_add,
            "Space": self.on_done_next,
        }
        for key, fn in layer_keys.items():
            self.c.layer.bind_key(key, lambda _l, _f=fn: _f(), overwrite=True)
        # napari's Points layer also offers Add/Select/Transform modes (keys
        # 2/3/5, or P/S) that bypass segfix's own tools entirely: Add drops
        # arbitrary new points, Transform rotates/scales/translates the
        # whole layer instead of the tree data, silently desyncing the view
        # from cloud.coords. Pin the layer to pan/zoom and refuse any other
        # mode, and replace napari's default status-bar hint (always "use
        # <2> for add points, ... <5> for transform" — it lists every mode
        # *other* than the current one, so this is shown constantly, not
        # just while one of those modes is active) with segfix's real ones.
        self._pin_layer_mode()
        self.c.layer.events.mode.connect(self._pin_layer_mode)

    def _pin_layer_mode(self, *_args) -> None:
        """Force the Points layer back to pan/zoom if anything switches it
        into one of napari's own edit modes, and blank the status-bar help
        text napari would otherwise show for it (its default, which lists
        those other modes). See _on_cloud_changed.

        Blanking needs both ``layer.help`` *and* ``viewer.help`` — they're
        separate properties, and only the latter actually reaches the
        status bar (``layer.help`` alone was this method's bug for a
        while: it looked fixed in a quick check, but the real text sticks
        to whatever ``viewer.help`` last was, which napari sets itself, on
        its own schedule, from the active layer's ``help`` — so this needs
        to run every time, not just once at setup, to keep winning that
        race. Its widget's own visibility isn't ours to control either:
        napari's StatusBarWidget.do_layout() force-shows it back on every
        resize regardless of any setVisible(False) — blank text is the
        only thing that reliably stops it from being shown."""
        layer = self.c.layer
        if layer.mode != "pan_zoom":
            layer.mode = "pan_zoom"
        layer.help = ""
        self.c.viewer.help = ""

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
        self.hidden_ids &= id_set  # drop ids for trees that no longer exist
        self.c.faded_ids &= id_set
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
            self.tree_table.setItem(row, 1, id_item)
            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, int(count))
            self.tree_table.setItem(row, 2, count_item)
            hidden = int(tid) in self.hidden_ids
            hide_item = QTableWidgetItem()
            hide_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            hide_item.setCheckState(Qt.Checked if hidden else Qt.Unchecked)
            hide_item.setToolTip("Hide this tree's points in the 3D view")
            self.tree_table.setItem(row, self.HIDE_COL, hide_item)
            faded = int(tid) in self.c.faded_ids
            fade_item = QTableWidgetItem()
            fade_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            fade_item.setCheckState(Qt.Checked if faded else Qt.Unchecked)
            fade_item.setToolTip(
                "Fade this tree — ghosted for context, still selectable"
            )
            self.tree_table.setItem(row, self.FADE_COL, fade_item)
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

    def on_hide_neighbours(self) -> None:
        """Toggle hiding every other tree currently loaded, leaving only the
        tree under review visible — shows them again if they're all already
        hidden.

        This used to recompute "neighbouring" via a fresh distance test
        against ``self.current`` every click. That under-hid in scene mode:
        the loaded cloud is fixed at load time to one tree plus its one-hop
        neighbours, but "current" moves to a different member of that group
        as the review queue advances (Space/table clicks), and two loaded
        trees needn't be within reach of *each other* even though both were
        within reach of the tree originally picked. A tree loaded alongside
        current is a neighbour regardless of which one is under review now,
        so hide by loaded-set membership instead of recomputing distances.
        """
        if self.current is None:
            self.c.viewer.status = (
                "No tree under review — press Space or click a table row"
            )
            return
        neighbours = {int(t) for t in self.c.cloud.tree_ids} - {self.current}
        if neighbours and neighbours <= self.hidden_ids:
            self.hidden_ids -= neighbours
            verb = "Shown"
        else:
            self.hidden_ids |= neighbours
            verb = "Hid"
        self._refresh_tree_table()  # updates the 👁 checkboxes to match
        self._apply_visibility()
        self.c.viewer.status = f"{verb} {len(neighbours)} other tree(s)"

    def on_fade_neighbours(self) -> None:
        """Toggle fading every other tree currently loaded, so only the tree
        under review is at full opacity — un-fades them if they're all faded
        already. Same loaded-set logic as :meth:`on_hide_neighbours`, but the
        others stay visible and selectable."""
        if self.current is None:
            self.c.viewer.status = (
                "No tree under review — press Space or click a table row"
            )
            return
        neighbours = {int(t) for t in self.c.cloud.tree_ids} - {self.current}
        if neighbours and neighbours <= self.c.faded_ids:
            self.c.faded_ids -= neighbours
            verb = "Restored"
        else:
            self.c.faded_ids |= neighbours
            verb = "Faded"
        self._refresh_tree_table()  # updates the Fade checkboxes to match
        self._apply_transparency()
        self.c.viewer.status = f"{verb} {len(neighbours)} other tree(s)"

    def _apply_visibility(self) -> None:
        if self.c.layer is None or not len(self.c.layer.data):
            return
        hidden = (
            np.isin(self.c.cloud.labels, list(self.hidden_ids))
            if self.hidden_ids else None
        )
        self.c.layer.shown = visibility_mask(
            self.c.cloud.labels,
            hide_unassigned=not self.show_unassigned.isChecked(),
            hidden=hidden,
            cross_section=self._combined_section_mask(),
        )

    def _combined_section_mask(self) -> np.ndarray | None:
        """AND of the cross section and lasso section, whichever are
        currently enabled — both fold into the same "shown" mechanism as
        one combined region, so this is the only thing _apply_visibility
        needs to pass along."""
        cross = self._cross_section_mask()
        lasso = self._lasso_section_mask_active()
        if cross is None:
            return lasso
        if lasso is None:
            return cross
        return cross & lasso

    def _on_show_unassigned(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.viewer.status = (
            "Unassigned/noise points shown" if checked
            else "Unassigned/noise points hidden"
        )

    # -- cross section --------------------------------------------------
    CROSS_SECTION_STEPS = 1000

    def _cross_axis_bounds(self) -> tuple[float, float]:
        """(lo, hi) of the current cloud's extent along the selected axis."""
        coords = self.c.cloud.coords
        if not len(coords):
            return 0.0, 1.0
        axis = self.cross_axis_combo.currentIndex()
        lo, hi = float(coords[:, axis].min()), float(coords[:, axis].max())
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def _reset_cross_section_range(self) -> None:
        """(Re)initialise the slab to the current cloud's full extent on the
        selected axis — called on axis change, Reset, and on a new cloud."""
        self._cross_lo, self._cross_hi = self._cross_axis_bounds()
        for slider, value in (
            (self.cross_min_slider, 0),
            (self.cross_max_slider, self.CROSS_SECTION_STEPS),
        ):
            slider.blockSignals(True)
            slider.setRange(0, self.CROSS_SECTION_STEPS)
            slider.setValue(value)
            slider.blockSignals(False)
        self._update_cross_range_label()

    def _slider_to_value(self, slider_value: int) -> float:
        lo, hi = self._cross_lo, self._cross_hi
        return lo + (hi - lo) * (slider_value / self.CROSS_SECTION_STEPS)

    def _on_cross_section_toggled(self, checked: bool) -> None:
        self.cross_body.setVisible(checked)
        self._apply_visibility()
        self.c.viewer.status = (
            "Cross section on — only the slab is shown/selectable"
            if checked else "Cross section off"
        )

    def _on_cross_axis_changed(self, _index: int) -> None:
        self._reset_cross_section_range()
        self._apply_visibility()

    def _on_cross_reset(self) -> None:
        self._reset_cross_section_range()
        self._apply_visibility()

    def _on_cross_range_changed(self, _value: int) -> None:
        # Keep min <= max by nudging the other slider if a drag crosses it;
        # setValue re-enters this handler, which then just updates in place.
        if self.cross_min_slider.value() > self.cross_max_slider.value():
            if self.sender() is self.cross_min_slider:
                self.cross_max_slider.setValue(self.cross_min_slider.value())
            else:
                self.cross_min_slider.setValue(self.cross_max_slider.value())
            return
        self._update_cross_range_label()
        self._apply_visibility()

    def _update_cross_range_label(self) -> None:
        axis_name = "XYZ"[self.cross_axis_combo.currentIndex()]
        lo = self._slider_to_value(self.cross_min_slider.value())
        hi = self._slider_to_value(self.cross_max_slider.value())
        self.cross_range_label.setText(f"{axis_name}: {lo:.2f} – {hi:.2f} m")

    def _cross_section_mask(self) -> np.ndarray | None:
        """Per-point mask, True inside the current slab; None when off."""
        if not self.cross_box.isChecked() or not len(self.c.cloud.coords):
            return None
        axis = self.cross_axis_combo.currentIndex()
        lo = self._slider_to_value(self.cross_min_slider.value())
        hi = self._slider_to_value(self.cross_max_slider.value())
        coords_axis = self.c.cloud.coords[:, axis]
        return (coords_axis >= lo) & (coords_axis <= hi)

    # -- lasso section ----------------------------------------------------
    def _on_lasso_section_drawn(self, indices: np.ndarray) -> None:
        """Completed drag while section_draw_btn is armed: freeze it as a
        per-point mask (not a live screen region — the camera can move
        freely afterwards) and switch the section on to show it."""
        n = len(self.c.cloud.coords)
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        self._lasso_section_mask = mask
        self.lasso_section_box.blockSignals(True)
        self.lasso_section_box.setChecked(True)
        self.lasso_section_box.blockSignals(False)
        self._update_lasso_section_label()
        self._apply_visibility()
        self.c.viewer.status = (
            f"Lasso section drawn — kept {mask.sum():,} of {n:,} points"
        )

    def _on_lasso_section_toggled(self, checked: bool) -> None:
        self.lasso_body.setVisible(checked)
        self._apply_visibility()
        self.c.viewer.status = (
            "Lasso section on — only the outline is shown/selectable"
            if checked else "Lasso section off"
        )

    def _on_lasso_section_reset(self) -> None:
        self._lasso_section_mask = None
        self.lasso_section_box.setChecked(False)
        self._update_lasso_section_label()
        self._apply_visibility()
        self.c.viewer.status = "Lasso section cleared"

    def _update_lasso_section_label(self) -> None:
        mask = self._lasso_section_mask
        if mask is None:
            self.lasso_section_label.setText("No outline drawn")
        else:
            self.lasso_section_label.setText(
                f"{int(mask.sum()):,} / {len(mask):,} points kept"
            )

    def _lasso_section_mask_active(self) -> np.ndarray | None:
        """The drawn mask, if the section is on and it still matches the
        currently loaded cloud (a new cloud invalidates it — see
        _on_cloud_changed); None otherwise, meaning no restriction."""
        mask = self._lasso_section_mask
        if not self.lasso_section_box.isChecked() or mask is None:
            return None
        if len(mask) != len(self.c.cloud.coords):
            return None
        return mask

    # -- done tracking ------------------------------------------------
    def _on_tree_item_changed(self, item) -> None:
        if self._table_updating:
            return
        tid = self._row_id(item.row())
        if item.column() == 0:
            self._mark_done(tid, item.checkState() == Qt.Checked)
        elif item.column() == self.HIDE_COL:
            self._set_hidden(tid, item.checkState() == Qt.Checked)
        elif item.column() == self.FADE_COL:
            self._set_faded(tid, item.checkState() == Qt.Checked)

    def _set_hidden(self, tid: int, hidden: bool) -> None:
        if hidden:
            self.hidden_ids.add(tid)
        else:
            self.hidden_ids.discard(tid)
        self._apply_visibility()
        self.c.viewer.status = f"Tree {tid} {'hidden' if hidden else 'shown'}"

    def _set_faded(self, tid: int, faded: bool) -> None:
        if faded:
            self.c.faded_ids.add(tid)
        else:
            self.c.faded_ids.discard(tid)
        self._apply_transparency()
        self.c.viewer.status = (
            f"Tree {tid} {'faded' if faded else 'back to full opacity'}"
        )

    def _apply_transparency(self) -> None:
        """Re-colour the layer so trees in ``faded_ids`` render at
        FADED_ALPHA. Parallel to _apply_visibility, but for opacity — faded
        points stay shown and selectable."""
        if self.c.layer is None or not len(self.c.layer.data):
            return
        refresh_layer(self.c.layer, self.c.cloud, self.c.faded_ids)

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
        if self.on_done_changed is not None:
            self.on_done_changed()
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
        title = "Selected Tree + Neighbours"
        self.trees_box.setTitle(f"{title} — {done}/{n} done" if n else title)

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
        self._update_neighbour_picker()
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

    def _update_neighbour_picker(self) -> None:
        """Rebuild the "send selection to neighbour" button row for the
        current tree — a quicker path than switching current tree and
        pressing Add when moving a patch to an adjacent tree."""
        while self.neighbour_row.count():
            item = self.neighbour_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if self.current is None:
            self.neighbour_label.setVisible(False)
            return
        from . import analysis

        neighbours = analysis.neighbours_by_points(
            self.c.cloud, self.current, self.focus_margin.value()
        )
        self.neighbour_label.setVisible(bool(neighbours))
        for nid in sorted(neighbours):
            rgba = colors_for_labels(
                np.array([nid]), self.c.cloud.label_colors
            )[0]
            r, g, b = (int(v * 255) for v in rgba[:3])
            btn = QPushButton(f"→ {nid}")
            btn.setStyleSheet(f"border-left: 4px solid rgb({r},{g},{b});")
            btn.setToolTip(f"Move the current selection to tree {nid}")
            btn.clicked.connect(
                lambda _checked=False, n=nid: self.on_send_to_neighbour(n)
            )
            self.neighbour_row.addWidget(btn)
        self.neighbour_row.addStretch()

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
        self._apply(ops.reassign(self.c.cloud, idx, tid))

    def on_send_to_neighbour(self, target_id: int) -> None:
        """Selection → a neighbouring tree, without switching current."""
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.reassign(self.c.cloud, idx, target_id))

    def on_create_new(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.create_new(self.c.cloud, idx))

    def on_unassign(self) -> None:
        idx = self._selection_or_current_tree("unassign")
        if idx is None:
            return
        self._apply(ops.unassign(self.c.cloud, idx))

    def on_noise(self) -> None:
        idx = self._selection_or_current_tree("trash")
        if idx is None:
            return
        self._apply(ops.mark_noise(self.c.cloud, idx))

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

    def _apply(self, msg: str) -> None:
        self.c._after_edit(msg)
        self._update_info()  # rebuilds the table; re-syncs bbox/current tree
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
            busy(self.c.viewer, "Saving…")
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
            self.c.viewer.status = "Nothing loaded to save"
            return
        self._do_save(self.c.save_path)

    def _do_save(self, path: str) -> None:
        from . import io

        busy(self.c.viewer, f"Saving to {path}…")
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
        "Control-l": lambda v: panel.tree_lasso_btn.toggle(),
        "Escape": lambda v: panel.on_move_mode(),
        "Space": lambda v: panel.on_done_next(),
        "Left": lambda v: panel._step(-1),
        "Right": lambda v: panel._step(1),
        "a": lambda v: panel.on_add(),
        "n": lambda v: panel.on_create_new(),
        "u": lambda v: panel.on_unassign(),
        "x": lambda v: panel.on_noise(),
        "h": lambda v: panel.show_unassigned.toggle(),
        "c": lambda v: panel.cross_box.setChecked(not panel.cross_box.isChecked()),
        "Shift-l": lambda v: panel.section_draw_btn.toggle(),
        "Shift-c": lambda v: panel.lasso_section_box.setChecked(
            not panel.lasso_section_box.isChecked()
        ),
        "Control-z": lambda v: panel.on_undo(),
        "Control-Shift-z": lambda v: panel.on_redo(),
        "Control-s": lambda v: panel.on_save(),
    }
    for key, fn in bindings.items():
        viewer.bind_key(key, fn, overwrite=True)
