"""Label-grouped index over one big binary PLY, for on-demand tree+neighbour
loading instead of loading (and rendering) the whole cloud at once.

Memory-maps the file once, keeps a few cheap resident arrays for fast
repeated queries, and loads/saves only the points a given operation actually
needs — grouped by tree label, since the review workflow already works one
tree (+ neighbours) at a time.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np

from . import io
from .model import NOISE, UNASSIGNED, PointCloud


@dataclass
class TreeRecord:
    """Cheap, always-available stats for one tree — no points loaded."""

    label: int
    count: int
    centroid: tuple[float, float]
    bbox: tuple[np.ndarray, np.ndarray]


class TreeCatalog:
    """A memory-mapped binary PLY, grouped by tree label.

    ``labels`` and ``coords`` are decoded/copied once and kept resident
    (tens of MB even for multi-million-point clouds) so neighbour search and
    the tree table work instantly; the bulk per-point data (attributes, and
    coords/labels for anything not currently loaded) stays memory-mapped and
    is only read for the points actually requested via :meth:`load`.
    """

    def __init__(self, path: str, label_field: str | None = None):
        with open(path, "rb") as fh:
            fmt, props, count, offset = io._parse_ply_header(fh)
        if "binary" not in fmt:
            raise ValueError("Tree catalog requires a binary PLY")

        self.path = path
        self.offset = offset
        self.count = count
        byteorder = ">" if fmt == "binary_big_endian" else "<"
        self.dtype = np.dtype([
            (name, byteorder + io.PLY_TYPE_MAP.get(ptype.lower(), "f4"))
            for name, ptype in props
        ])
        self._names = {n.lower(): n for n in self.dtype.names}
        self.is_rgb = io._is_rgb_segmented(props, label_field)
        self.label_field = (
            None if self.is_rgb
            else io._pick_label_field(list(self.dtype.names), label_field)
        )

        self._mm = np.memmap(path, dtype=self.dtype, mode="r",
                              offset=offset, shape=(count,))

        self.coords = np.column_stack([
            self._mm[self._names["x"]],
            self._mm[self._names["y"]],
            self._mm[self._names["z"]],
        ]).astype(np.float32)

        if self.is_rgb:
            self.labels, self.label_colors = self._labels_from_rgb(self._mm)
        else:
            self.labels = np.asarray(self._mm[self.label_field]).astype(np.int32)
            self.label_colors = None

        # Snapshot to diff against on save — see save().
        self._original_labels = self.labels.copy()
        self._build_index()
        self._next_id = int(self.labels.max()) + 1 if self.labels.size else 1

    def _labels_from_rgb(self, sub):
        r = sub[self._names["red"]].astype(np.uint32)
        g = sub[self._names["green"]].astype(np.uint32)
        b = sub[self._names["blue"]].astype(np.uint32)
        packed = (r << 16) | (g << 8) | b
        uniq, inverse = np.unique(packed, return_inverse=True)
        labels = (inverse + 1).astype(np.int32)
        labels[packed == 0] = UNASSIGNED
        label_colors = {
            int(k + 1): (int((u >> 16) & 255), int((u >> 8) & 255), int(u & 255))
            for k, u in enumerate(uniq)
            if u != 0
        }
        return labels, label_colors

    # -- grouping index --------------------------------------------------
    def _build_index(self) -> None:
        """Sort-by-label once so any tree's point indices are an O(1) slice.

        Rebuilt (O(N log N)) after every :meth:`apply` — that's only on a
        tree switch or Save, never per-render, so it's not on the hot path
        that made loading the whole cloud slow.
        """
        self.order = np.argsort(self.labels, kind="stable")
        sorted_labels = self.labels[self.order]
        uniq, counts = np.unique(sorted_labels, return_counts=True)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._starts = dict(zip(uniq.tolist(), starts.tolist()))
        self._counts = dict(zip(uniq.tolist(), counts.tolist()))
        self.records: dict[int, TreeRecord] = {}
        for lab, start, cnt in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
            if lab in (UNASSIGNED, NOISE):
                continue
            idx = self.order[start:start + cnt]
            self.records[lab] = self._make_record(lab, idx)

    def _make_record(self, label: int, idx: np.ndarray) -> TreeRecord:
        pts = self.coords[idx]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centroid = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
        return TreeRecord(label=label, count=int(idx.size), centroid=centroid,
                           bbox=(lo, hi))

    def indices_for(self, label: int) -> np.ndarray:
        start = self._starts.get(label)
        if start is None:
            return np.empty(0, dtype=np.int64)
        return self.order[start:start + self._counts[label]]

    # -- neighbour search --------------------------------------------------
    def neighbours(self, label: int, reach: float) -> set[int]:
        """Trees whose points come within ``reach`` of tree ``label``.

        Delegates to :func:`analysis.neighbours_by_points` against the
        resident (not memory-mapped) coords/labels — same algorithm the
        review panel's own focus-mode already uses, just pointed at the
        whole file's data instead of whatever's currently loaded.
        """
        from . import analysis

        tmp = PointCloud(coords=self.coords, labels=self.labels)
        return analysis.neighbours_by_points(tmp, label, reach)

    # -- loading -----------------------------------------------------------
    def load(self, labels: list[int], margin: float = 1.0) -> tuple[PointCloud, np.ndarray]:
        """Load ``labels`` plus nearby unassigned points into a real cloud.

        Unassigned points near the loaded trees are included (not just the
        labelled ones) because the review workflow relies on lassoing them
        and pressing A to add them to the current tree.
        """
        labels = [int(t) for t in labels]
        parts = [self.indices_for(t) for t in labels]
        tree_idx = (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        )
        if tree_idx.size == 0:
            raise ValueError("No points for the requested trees")

        pts = self.coords[tree_idx]
        lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
        in_box = np.all((self.coords >= lo) & (self.coords <= hi), axis=1)
        unassigned_idx = np.flatnonzero(in_box & (self.labels == UNASSIGNED))

        global_idx = np.union1d(tree_idx, unassigned_idx).astype(np.int64)

        sub = self._mm[global_idx]
        coords = np.column_stack([
            sub[self._names["x"]], sub[self._names["y"]], sub[self._names["z"]],
        ]).astype(np.float32)
        cloud_labels = self.labels[global_idx].copy()

        skip = {self._names.get(n) for n in ("x", "y", "z")}
        if self.is_rgb:
            skip |= {self._names.get(n) for n in ("red", "green", "blue")}
        else:
            skip.add(self.label_field)
        attributes = {
            name: np.ascontiguousarray(sub[name])
            for name in self.dtype.names
            if name not in skip
        }

        cloud = PointCloud(
            coords=coords,
            labels=cloud_labels,
            attributes=attributes,
            source_path=self.path,
            label_field=self.label_field or "treeID",
            source_format="raycloud_rgb" if self.is_rgb else "auto",
            label_colors=self.label_colors,
        )
        # New tree IDs (splits, grow) must be unique across the *whole* file,
        # not just this subset — see next_free_id(). This mirrors the
        # instance-level override slot SegFixController.on_save_override
        # already uses elsewhere in this codebase.
        cloud.next_free_id = self.next_free_id
        return cloud, global_idx

    def next_free_id(self) -> int:
        """A session-wide-unique tree ID, unlike PointCloud's own (subset-
        scoped) implementation. Stateful/incrementing by design: repeated
        calls before an intervening apply() must still not collide."""
        nid = self._next_id
        self._next_id += 1
        return nid

    # -- persisting edits ----------------------------------------------
    def apply(self, cloud: PointCloud, global_idx: np.ndarray) -> None:
        """Scatter a loaded (and possibly edited) subset's labels back into
        the master array, so the catalog and any later `load()` reflect it."""
        if global_idx is None or global_idx.size == 0:
            return
        self.labels[global_idx] = cloud.labels
        self._build_index()
        if self.labels.size:
            self._next_id = max(self._next_id, int(self.labels.max()) + 1)

    # -- saving --------------------------------------------------------
    def save(self, output: str | None = None) -> str:
        """Write only the points whose label changed since the last save.

        Diffs against the snapshot taken at open time / after the previous
        save, so this reflects edits made across *any* tree visited this
        session, not just whatever is currently loaded.
        """
        changed = np.flatnonzero(self.labels != self._original_labels)
        target = output or self.path
        is_new_target = os.path.abspath(target) != os.path.abspath(self.path)

        if not is_new_target and changed.size == 0:
            return "Nothing changed since last save"

        if is_new_target and not os.path.exists(target):
            # A full copy of the whole file, not just whatever's loaded —
            # needed even with zero edits so "Save As" still exports a
            # complete copy at the new path, not nothing.
            shutil.copyfile(self.path, target)

        if changed.size == 0:
            return f"Saved (no changes) → {target}"

        out = np.memmap(target, dtype=self.dtype, mode="r+",
                         offset=self.offset, shape=(self.count,))
        if self.is_rgb:
            colour = self._colours_for(self.labels[changed])
            out[self._names["red"]][changed] = colour[:, 0]
            out[self._names["green"]][changed] = colour[:, 1]
            out[self._names["blue"]][changed] = colour[:, 2]
        else:
            out[self.label_field][changed] = self.labels[changed]
        out.flush()
        del out

        if not is_new_target:
            # Only a write to self.path advances the diff baseline — a
            # Save As to a different file doesn't touch self.path, so an
            # in-place Save afterwards must still see these as pending.
            self._original_labels = self.labels.copy()
        return f"Saved {changed.size:,} changed point(s) → {target}"

    def _colours_for(self, labels: np.ndarray) -> np.ndarray:
        colour = np.zeros((labels.size, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab in (UNASSIGNED, NOISE):
                continue
            colour[labels == lab] = io.color_for_label(int(lab), self.label_colors)
        return colour
