"""Label-grouped index over one big point cloud, for on-demand tree+neighbour
loading instead of loading (and rendering) the whole cloud at once.

Memory-maps the file once, keeps a few cheap resident arrays for fast
repeated queries, and loads/saves only the points a given operation actually
needs — grouped by tree label, since the review workflow already works one
tree (+ neighbours) at a time.

Two on-disk formats get this treatment, because both store their points as
fixed-size records at a known offset that ``numpy.memmap`` can index and patch
in place:

* **binary PLY** (:class:`TreeCatalog`) — the raycloudtools / label-field clouds
  segfix has always handled.
* **uncompressed LAS** (:class:`LasCatalog`) — what arbor's pipeline produces
  (a ``treeID`` Extra-Bytes column). arbor writes ``.laz``; :mod:`workspace`
  decompresses it to a ``.las`` working copy on import, and :meth:`LasCatalog.save`
  re-compresses back to ``.laz`` so arbor can re-read the corrected cloud.

:func:`open_catalog` picks the backend from the path extension.
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


class _BaseCatalog:
    """Format-agnostic label index over a memory-mapped point cloud.

    Subclasses implement the file-format specifics (:meth:`_open`,
    :meth:`_decode_coords`, :meth:`_decode_labels`, :meth:`_write_labels`,
    :meth:`_finalize_save`); everything else here — the grouping index,
    neighbour search, subset load/apply and the diff-based save — is shared.

    ``labels`` and ``coords`` are decoded/copied once and kept resident
    (tens of MB even for multi-million-point clouds) so neighbour search and
    the tree table work instantly; the bulk per-point data (attributes, and
    coords/labels for anything not currently loaded) stays memory-mapped and
    is only read for the points actually requested via :meth:`load`.
    """

    def __init__(self, path: str, label_field: str | None = None):
        self.path = path
        self._label_field_req = label_field
        # Subclass fills in: offset, count, dtype, _names, is_rgb, label_field,
        # and _mm (the memmap of fixed-size point records).
        self._open()

        self.coords = self._decode_coords(self._mm)
        self.labels, self.label_colors = self._decode_labels(self._mm)
        self.labels = np.asarray(self.labels).astype(np.int32)

        # Snapshot to diff against on save — see save().
        self._original_labels = self.labels.copy()
        self._build_index()
        self._next_id = int(self.labels.max()) + 1 if self.labels.size else 1

    # -- format hooks (subclass) -----------------------------------------
    def _open(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decode_coords(self, sub) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decode_labels(self, sub):  # pragma: no cover - abstract
        raise NotImplementedError

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        raise NotImplementedError  # pragma: no cover - abstract

    def _finalize_save(self, target: str, is_new_target: bool) -> None:
        """Hook after the in-place patch is flushed (e.g. re-export LAZ)."""

    # -- grouping index --------------------------------------------------
    def _build_index(self) -> None:
        """Sort-by-label once so any tree's point indices are an O(1) slice.

        Rebuilt (O(N log N)) after every :meth:`apply` — that's only on a
        tree switch or Save, never per-render, so it's not on the hot path
        that made loading the whole cloud slow.
        """
        self.order = np.argsort(self.labels, kind="stable")
        sorted_labels = self.labels[self.order]
        # On a sorted array np.unique's first-occurrence index is the group
        # start, so this replaces a separate cumsum.
        uniq, starts, counts = np.unique(
            sorted_labels, return_index=True, return_counts=True
        )
        self._starts = dict(zip(uniq.tolist(), starts.tolist()))
        self._counts = dict(zip(uniq.tolist(), counts.tolist()))

        # Per-tree bbox/centroid in two vectorised reductions over the
        # label-sorted coords, instead of a fancy-index + min/max per tree —
        # this runs on every apply() (tree switch, Save), so the Python loop
        # it replaces was O(trees) numpy calls on multi-thousand-tree plots.
        self.records: dict[int, TreeRecord] = {}
        if not uniq.size:
            return
        sorted_coords = self.coords[self.order]
        mins = np.minimum.reduceat(sorted_coords, starts, axis=0)
        maxs = np.maximum.reduceat(sorted_coords, starts, axis=0)
        for k, lab in enumerate(uniq.tolist()):
            if lab in (UNASSIGNED, NOISE):
                continue
            lo, hi = mins[k], maxs[k]
            self.records[lab] = TreeRecord(
                label=lab,
                count=int(counts[k]),
                centroid=(float((lo[0] + hi[0]) / 2),
                          float((lo[1] + hi[1]) / 2)),
                bbox=(lo, hi),
            )

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
        review panel's own neighbour picker uses, just pointed at the whole
        file's data instead of whatever's currently loaded.
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
        coords = self._decode_coords(sub)
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
            self._finalize_save(target, is_new_target)
            return f"Saved (no changes) → {target}"

        out = np.memmap(target, dtype=self.dtype, mode="r+",
                         offset=self.offset, shape=(self.count,))
        self._write_labels(out, changed, self.labels[changed])
        out.flush()
        del out
        self._finalize_save(target, is_new_target)

        if not is_new_target:
            # Only a write to self.path advances the diff baseline — a
            # Save As to a different file doesn't touch self.path, so an
            # in-place Save afterwards must still see these as pending.
            self._original_labels = self.labels.copy()
        return f"Saved {changed.size:,} changed point(s) → {target}"


class TreeCatalog(_BaseCatalog):
    """A memory-mapped binary PLY, grouped by tree label.

    The instance label lives either in a per-point field (``treeID``,
    ``PredInstance``, ... — auto-detected) or, for raycloudtools clouds, in the
    point's RGB colour.
    """

    def _open(self) -> None:
        with open(self.path, "rb") as fh:
            fmt, props, count, offset = io._parse_ply_header(fh)
        if "binary" not in fmt:
            raise ValueError("Tree catalog requires a binary PLY")

        self.offset = offset
        self.count = count
        byteorder = ">" if fmt == "binary_big_endian" else "<"
        self.dtype = np.dtype([
            (name, byteorder + io.PLY_TYPE_MAP.get(ptype.lower(), "f4"))
            for name, ptype in props
        ])
        self._names = {n.lower(): n for n in self.dtype.names}
        self.is_rgb = io._is_rgb_segmented(props, self._label_field_req)
        self.label_field = (
            None if self.is_rgb
            else io._pick_label_field(list(self.dtype.names), self._label_field_req)
        )
        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r",
                              offset=self.offset, shape=(self.count,))

    def _decode_coords(self, sub) -> np.ndarray:
        return np.column_stack([
            sub[self._names["x"]],
            sub[self._names["y"]],
            sub[self._names["z"]],
        ]).astype(np.float32)

    def _decode_labels(self, sub):
        if self.is_rgb:
            return self._labels_from_rgb(sub)
        return np.asarray(sub[self.label_field]).astype(np.int32), None

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

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        if self.is_rgb:
            colour = self._colours_for(values)
            out[self._names["red"]][changed] = colour[:, 0]
            out[self._names["green"]][changed] = colour[:, 1]
            out[self._names["blue"]][changed] = colour[:, 2]
        else:
            out[self.label_field][changed] = values

    def _colours_for(self, labels: np.ndarray) -> np.ndarray:
        colour = np.zeros((labels.size, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab in (UNASSIGNED, NOISE):
                continue
            colour[labels == lab] = io.color_for_label(int(lab), self.label_colors)
        return colour


class LasCatalog(_BaseCatalog):
    """A memory-mapped uncompressed LAS, grouped by its ``treeID`` column.

    Reads the header and Extra-Bytes layout with laspy but never lets laspy
    decode the points: the fixed-size records are indexed and patched with
    ``numpy.memmap``, exactly like the PLY path. On save the in-place ``.las``
    patch is mirrored to a ``.laz`` beside it when the project came from one
    (see :meth:`_finalize_save`), so arbor can re-read the corrected cloud.
    """

    is_rgb = False  # LAS always carries an explicit label column

    def _open(self) -> None:
        import laspy

        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".laz":
            raise ValueError(
                "LasCatalog needs an uncompressed .las; compressed .laz cannot "
                "be memory-mapped (workspace.create_workspace decompresses on "
                "import)"
            )

        with laspy.open(self.path) as fh:
            header = fh.header
        self.dtype = header.point_format.dtype()
        self.offset = int(header.offset_to_point_data)
        self.count = int(header.point_count)
        self._scales = np.asarray(header.scales, dtype=np.float64)
        self._offsets = np.asarray(header.offsets, dtype=np.float64)
        self._names = {n.lower(): n for n in self.dtype.names}

        expected = self.offset + self.count * self.dtype.itemsize
        actual = os.path.getsize(self.path)
        if self.dtype.itemsize != header.point_format.size or actual < expected:
            raise ValueError(
                f"{self.path}: LAS point records don't line up "
                f"(record {self.dtype.itemsize}B x {self.count} + {self.offset}B "
                f"header = {expected}B, file is {actual}B) — not a plain "
                "uncompressed LAS?"
            )

        self.label_field = io._pick_label_field(
            list(self.dtype.names), self._label_field_req
        )
        if self.label_field is None:
            raise ValueError(
                f"{self.path}: no treeID / instance-ID column "
                f"(looked for {', '.join(io.LABEL_CANDIDATES)}). "
                "Is this an arbor '_segmented' cloud? Pass --label-field to "
                "name it explicitly."
            )
        self._label_is_unsigned = self.dtype[self.label_field].kind == "u"

        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r",
                              offset=self.offset, shape=(self.count,))

    def _decode_coords(self, sub) -> np.ndarray:
        xyz = np.column_stack([
            sub[self._names["x"]].astype(np.float64) * self._scales[0] + self._offsets[0],
            sub[self._names["y"]].astype(np.float64) * self._scales[1] + self._offsets[1],
            sub[self._names["z"]].astype(np.float64) * self._scales[2] + self._offsets[2],
        ])
        return xyz.astype(np.float32)

    def _decode_labels(self, sub):
        return np.asarray(sub[self.label_field]).astype(np.int32), None

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        if self._label_is_unsigned:
            # An unsigned Extra-Bytes column can't hold NOISE (-1); write it
            # back as UNASSIGNED (0). segfix's own <cloud>.segfix.json sidecar
            # still remembers which points were dismissed as noise, but a
            # reader of the LAS alone (arbor, lidR) sees them as unassigned.
            values = np.where(values == NOISE, UNASSIGNED, values)
        out[self.label_field][changed] = values.astype(self.dtype[self.label_field])

    def _finalize_save(self, target: str, is_new_target: bool) -> None:
        laz = _laz_export_path(self.path, target)
        if laz is None:
            return
        import laspy

        laspy.read(target).write(laz)


def _laz_export_path(source_las: str, target_las: str) -> str | None:
    """Where to re-export a corrected LAS as LAZ, or ``None`` if not needed.

    Returns a sibling ``.laz`` only when the project's working copy was
    decompressed from one on import — recorded as a ``.laz`` ``source`` in the
    :mod:`workspace` manifest next to the ``.las``.
    """
    from . import workspace

    manifest = os.path.join(os.path.dirname(source_las), workspace.MANIFEST_NAME)
    if not os.path.exists(manifest):
        return None
    try:
        import json

        with open(manifest, encoding="utf-8") as fh:
            src = json.load(fh).get("source", "")
    except (OSError, ValueError):
        return None
    if os.path.splitext(src)[1].lower() != ".laz":
        return None
    return os.path.splitext(target_las)[0] + ".laz"


def open_catalog(path: str, label_field: str | None = None) -> _BaseCatalog:
    """Open ``path`` with the backend its extension calls for.

    ``.ply`` → :class:`TreeCatalog`; ``.las`` → :class:`LasCatalog`. ``.laz`` is
    rejected here — :mod:`workspace` decompresses it to ``.las`` on import.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return TreeCatalog(path, label_field=label_field)
    if ext in (".las", ".laz"):
        return LasCatalog(path, label_field=label_field)
    raise ValueError(f"segfix opens .ply and .las clouds, not {ext or path!r}")


# Backwards-compatible alias for type hints that just want "some catalog".
Catalog = _BaseCatalog
