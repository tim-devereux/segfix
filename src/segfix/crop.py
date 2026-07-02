"""Edit a very large cloud one spatial tile at a time.

Loading 14M points at once is fine for a quick look but heavy to edit
interactively.  A :class:`CropSession` memory-maps the file once, then loads
only the points inside a chosen XY box (optionally with a context *margin* of
surrounding points), so you fix the forest tile by tile.

Saving writes just that tile's points back into an output copy of the file via
a vectorised memory-map assignment — the rest of the file is untouched and no
full in-memory rewrite is needed.  Successive tile saves accumulate into the
same output file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np

from . import io
from .model import NOISE, UNASSIGNED, PointCloud


@dataclass
class CropInfo:
    """Where a loaded crop came from, so edits can be written back.

    ``global_idx`` maps each loaded point to its row in the full file;
    ``core_mask`` marks the points inside the real tile box (as opposed to the
    surrounding margin, which is read-only context and not saved).
    """

    bbox: tuple[float, float, float, float]
    margin: float
    global_idx: np.ndarray
    core_mask: np.ndarray

    @property
    def n_core(self) -> int:
        return int(self.core_mask.sum())


def default_output(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}_fixed{ext}"


class CropSession:
    """A memory-mapped large cloud that loads/saves rectangular XY tiles."""

    def __init__(self, path: str, output: str | None = None, label_field=None):
        with open(path, "rb") as fh:
            fmt, props, count, offset = io._parse_ply_header(fh)
        if "binary" not in fmt:
            raise ValueError("Crop mode requires a binary PLY")

        self.path = path
        self.output = output or default_output(path)
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

        # working_path is the current source of truth for labels/colours; it
        # advances to the output file after the first save so later tile loads
        # reflect edits already made.
        self.working_path = path
        self._mm = np.memmap(path, dtype=self.dtype, mode="r",
                             offset=offset, shape=(count,))

        # Cache XY once (float32) for fast repeated cropping.
        self._x = np.asarray(self._mm[self._names["x"]], dtype=np.float32)
        self._y = np.asarray(self._mm[self._names["y"]], dtype=np.float32)
        self.bounds = (
            float(self._x.min()), float(self._y.min()),
            float(self._x.max()), float(self._y.max()),
        )

    # -- tiling --------------------------------------------------------
    def tiles(self, nx: int, ny: int) -> list[tuple[float, float, float, float]]:
        """Split the XY extent into an ``nx`` × ``ny`` grid of boxes."""
        x0, y0, x1, y1 = self.bounds
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        boxes = []
        for j in range(ny):
            for i in range(nx):
                boxes.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))
        return boxes

    # -- loading -------------------------------------------------------
    def load_crop(self, bbox, margin: float = 0.0) -> tuple[PointCloud, CropInfo]:
        x0, y0, x1, y1 = bbox
        core = (self._x >= x0) & (self._x < x1) & (self._y >= y0) & (self._y < y1)
        load = (
            (self._x >= x0 - margin) & (self._x < x1 + margin)
            & (self._y >= y0 - margin) & (self._y < y1 + margin)
        )
        global_idx = np.flatnonzero(load)
        if global_idx.size == 0:
            raise ValueError("No points in this crop")

        sub = self._mm[global_idx]
        coords = np.column_stack([
            sub[self._names["x"]], sub[self._names["y"]], sub[self._names["z"]]
        ]).astype(np.float32)

        if self.is_rgb:
            labels, label_colors = self._labels_from_rgb(sub)
            source_format = "raycloud_rgb"
        else:
            labels = np.asarray(sub[self.label_field]).astype(np.int32)
            label_colors = None
            source_format = "auto"

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
            labels=labels,
            attributes=attributes,
            source_path=self.path,
            label_field=self.label_field or "treeID",
            source_format=source_format,
            label_colors=label_colors,
        )
        info = CropInfo(
            bbox=tuple(bbox), margin=margin,
            global_idx=global_idx, core_mask=core[global_idx],
        )
        return cloud, info

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

    # -- saving --------------------------------------------------------
    def save_crop(self, cloud: PointCloud, info: CropInfo) -> str:
        """Write the crop's *core* points back into the output file.

        Only points inside the tile box are written (margin points are context
        and belong to neighbouring tiles).  The first save copies the source to
        the output path; later saves edit that output in place.
        """
        if not os.path.exists(self.output) or self.working_path == self.path:
            if os.path.abspath(self.output) != os.path.abspath(self.path):
                shutil.copyfile(self.working_path, self.output)

        core_local = np.flatnonzero(info.core_mask)
        core_global = info.global_idx[core_local]
        core_labels = cloud.labels[core_local]

        out = np.memmap(self.output, dtype=self.dtype, mode="r+",
                        offset=self.offset, shape=(self.count,))
        if self.is_rgb:
            colour = self._colours_for(core_labels, cloud.label_colors)
            out[self._names["red"]][core_global] = colour[:, 0]
            out[self._names["green"]][core_global] = colour[:, 1]
            out[self._names["blue"]][core_global] = colour[:, 2]
        else:
            out[self.label_field][core_global] = core_labels
        out.flush()
        del out

        # Subsequent loads/saves work from the output file.
        self.working_path = self.output
        self._mm = np.memmap(self.output, dtype=self.dtype, mode="r",
                             offset=self.offset, shape=(self.count,))
        return f"Saved {core_global.size} points → {self.output}"

    @staticmethod
    def _colours_for(labels, label_colors):
        colour = np.zeros((labels.size, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab in (UNASSIGNED, NOISE):
                continue  # leave black
            colour[labels == lab] = io.color_for_label(int(lab), label_colors)
        return colour
