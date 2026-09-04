"""Core data model for a segmented tree point cloud.

A :class:`PointCloud` holds the point coordinates, the per-point instance
label (``tree_id``), and any extra scalar attributes carried through from the
source file.  Every mutation goes through :meth:`PointCloud.set_labels`, which
records an undo entry, so the GUI can offer undo/redo cheaply even on large
clouds (we store only the indices and previous values that changed, never a
full copy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Sentinel label for points that belong to no tree.  Kept as a module
# constant so IO, operations and widgets all agree on its meaning.
UNASSIGNED = 0
NOISE = -1


@dataclass
class _Edit:
    """A reversible change to the label array: ``labels[indices] = new``."""

    indices: np.ndarray
    old_values: np.ndarray
    new_value: int
    description: str


@dataclass
class PointCloud:
    """A tree point cloud with editable per-point instance labels.

    Attributes
    ----------
    coords:
        ``(N, 3)`` float32 array of XYZ coordinates.
    labels:
        ``(N,)`` int32 array of tree instance IDs.  ``UNASSIGNED`` (0) means
        "no tree", ``NOISE`` (-1) marks points flagged as noise/non-tree.
    attributes:
        Extra per-point scalar columns from the source file (e.g. intensity),
        keyed by name.  Carried through on save when the format supports it.
    source_path / label_field:
        Provenance, used to round-trip the file on save.
    global_shift:
        ``(dx, dy, dz)`` already added to ``coords`` relative to the source
        file, or ``None`` if the cloud wasn't shifted. See
        :func:`segfix.treecatalog.suggest_global_shift`.
    """

    coords: np.ndarray
    labels: np.ndarray
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    source_path: str | None = None
    label_field: str = "treeID"
    # How the cloud was loaded, so save() can round-trip the original format.
    # "auto" uses the path extension; "raycloud_rgb" means the segmentation is
    # encoded as per-point RGB (see io.load_rgb_segmented); "las" means it came
    # from a LAS/LAZ and save() clones that file to keep every other column.
    source_format: str = "auto"
    # For RGB-segmented clouds: the original colour of each tree label, used
    # both for display and to write colours back on save.
    label_colors: dict | None = None
    # (dx, dy, dz) added to the source file's raw coordinates to bring a
    # large-magnitude cloud (e.g. UTM-referenced LiDAR) near the origin
    # before it's stored as float32 — see treecatalog.suggest_global_shift.
    # Display/analysis metadata only: coordinates are never written back to
    # disk, so this is never subtracted back out on save.
    global_shift: tuple[float, float, float] | None = None

    # Indices touched by the most recent set_labels/undo/redo (empty on a
    # no-op, None before the first edit). Lets the viewer recolour just the
    # points that changed instead of the whole cloud on every keystroke.
    last_changed: np.ndarray | None = field(default=None, repr=False)

    _undo: list[_Edit] = field(default_factory=list, repr=False)
    _redo: list[_Edit] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float32).reshape(-1, 3)
        self.labels = np.asarray(self.labels, dtype=np.int32).reshape(-1)
        if len(self.coords) != len(self.labels):
            raise ValueError(
                f"coords ({len(self.coords)}) and labels ({len(self.labels)}) "
                "length mismatch"
            )

    # -- queries ---------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.coords)

    @property
    def tree_ids(self) -> np.ndarray:
        """Sorted unique tree IDs, excluding UNASSIGNED and NOISE."""
        ids = np.unique(self.labels)
        return ids[(ids != UNASSIGNED) & (ids != NOISE)]

    def next_free_id(self) -> int:
        """Smallest positive integer not currently used as a tree ID."""
        ids = self.tree_ids
        return int(ids.max()) + 1 if len(ids) else 1

    # -- mutation --------------------------------------------------------
    def set_labels(self, indices, new_value: int, description: str) -> int:
        """Set ``labels[indices] = new_value``, recording an undo entry.

        Returns the number of points actually changed (0 means no-op, and no
        undo entry is recorded).
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        empty = np.empty(0, dtype=np.int64)
        if indices.size == 0:
            self.last_changed = empty
            return 0
        old = self.labels[indices].copy()
        changed = old != new_value
        if not changed.any():
            self.last_changed = empty
            return 0
        indices = indices[changed]
        old = old[changed]
        self.labels[indices] = new_value
        self._undo.append(_Edit(indices, old, int(new_value), description))
        self._redo.clear()
        self.last_changed = indices
        return int(indices.size)

    # -- undo / redo -----------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str | None:
        if not self._undo:
            self.last_changed = np.empty(0, dtype=np.int64)
            return None
        edit = self._undo.pop()
        # Restore old values, capturing what they were so redo can re-apply.
        current = self.labels[edit.indices].copy()
        self.labels[edit.indices] = edit.old_values
        self._redo.append(
            _Edit(edit.indices, current, edit.new_value, edit.description)
        )
        self.last_changed = edit.indices
        return edit.description

    def redo(self) -> str | None:
        if not self._redo:
            self.last_changed = np.empty(0, dtype=np.int64)
            return None
        edit = self._redo.pop()
        current = self.labels[edit.indices].copy()
        self.labels[edit.indices] = edit.new_value
        self._undo.append(
            _Edit(edit.indices, current, edit.new_value, edit.description)
        )
        self.last_changed = edit.indices
        return edit.description
