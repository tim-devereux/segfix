"""Load several per-tree PLY files into one labelled cloud, and save back.

This bridges the plugin's "one file per tree" world to segfix's single
:class:`PointCloud` editing model: each loaded file becomes a distinct integer
label tagged with its tree id, so the existing merge/split/reassign/lasso
operations Just Work.  On save we group points by label, write each tree to
``fixed/{tree_id}.ply``, and append any deleted points to
``removed_points.xyz`` in the plugin's column format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io
from .model import NOISE, UNASSIGNED, PointCloud
from .project import Project, TreeEntry

# Columns of removed_points.xyz: x y z r g b nx ny nz alpha time group_id
_REMOVED_ATTR_DEFAULTS = {
    "red": 0.0, "green": 0.0, "blue": 0.0,
    "nx": 0.0, "ny": 0.0, "nz": 0.0,
    "alpha": 0.0, "time": 0.0,
}


@dataclass
class TreeScene:
    """Bookkeeping for one focus-tree editing session."""

    project: Project
    focus: TreeEntry
    entries: list[TreeEntry]
    label_tree_id: dict[int, str] = field(default_factory=dict)
    original_labels: np.ndarray | None = None

    @property
    def focus_tree_id(self) -> str:
        return self.focus.tree_id


def load_scene(project: Project, focus: TreeEntry, entries) -> tuple[PointCloud, TreeScene]:
    """Load ``entries`` (focus first) into one labelled :class:`PointCloud`.

    Every file gets its own label; files sharing a ``tree_id`` (a tree's
    primary + stem files) keep their distinct labels but map to the same id.
    Per-point colour/normal attributes are carried through (unioned across
    files) so deleted points can be written to ``removed_points.xyz`` intact.
    """
    coords_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    label_tree_id: dict[int, str] = {}

    # First pass: read each file, collect coords/labels and the attribute keys.
    loaded = []
    attr_keys: set[str] = set()
    next_label = 1
    for entry in entries:
        for mesh_file in entry.all_files:
            path = project.data_directory / mesh_file
            if not path.exists():
                continue
            sub = io.load(str(path))
            if sub.n_points == 0:
                continue
            label = next_label
            next_label += 1
            label_tree_id[label] = entry.tree_id
            coords_parts.append(sub.coords)
            label_parts.append(np.full(sub.n_points, label, dtype=np.int32))
            attr_keys.update(sub.attributes.keys())
            loaded.append(sub)

    if not coords_parts:
        raise ValueError(f"No loadable points for tree {focus.tree_id}")

    coords = np.vstack(coords_parts)
    labels = np.concatenate(label_parts)

    # Second pass: stack attributes, filling files missing a key with zeros.
    attributes: dict[str, np.ndarray] = {}
    for key in attr_keys:
        cols = [
            sub.attributes[key]
            if key in sub.attributes
            else np.zeros(sub.n_points, dtype=np.float32)
            for sub in loaded
        ]
        attributes[key] = np.concatenate(cols)

    cloud = PointCloud(
        coords=coords,
        labels=labels,
        attributes=attributes,
        source_path=str(project.data_directory / focus.mesh_file),
        label_field="treeID",
    )
    scene = TreeScene(
        project=project,
        focus=focus,
        entries=list(entries),
        label_tree_id=label_tree_id,
        original_labels=labels.copy(),
    )
    return cloud, scene


# -- save ----------------------------------------------------------------
def _next_group_id(removed_path: Path) -> int:
    """Largest existing group_id in removed_points.xyz, or 0 if none."""
    if not removed_path.exists():
        return 0
    max_gid = 0
    with open(removed_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 12:
                try:
                    max_gid = max(max_gid, int(float(parts[11])))
                except ValueError:
                    pass
    return max_gid


def _append_removed(cloud: PointCloud, indices: np.ndarray, removed_path: Path) -> int:
    """Append deleted points to removed_points.xyz, returning the count."""
    if indices.size == 0:
        return 0
    gid = _next_group_id(removed_path) + 1
    xyz = cloud.coords[indices]

    def col(name):
        arr = cloud.attributes.get(name)
        return arr[indices] if arr is not None else None

    rgb = [col("red"), col("green"), col("blue")]
    nrm = [col("nx"), col("ny"), col("nz")]
    alpha, t = col("alpha"), col("time")

    with open(removed_path, "a", encoding="utf-8") as fh:
        for i in range(indices.size):
            def g(arr, default=0.0):
                return float(arr[i]) if arr is not None else default

            r = int(max(0, min(255, round(g(rgb[0])))))
            gr = int(max(0, min(255, round(g(rgb[1])))))
            b = int(max(0, min(255, round(g(rgb[2])))))
            a = int(max(0, min(255, round(g(alpha)))))
            fh.write(
                f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                f"{r} {gr} {b} "
                f"{g(nrm[0]):.6f} {g(nrm[1]):.6f} {g(nrm[2]):.6f} "
                f"{a} {g(t):.6f} {gid}\n"
            )
    return int(indices.size)


def save_scene(cloud: PointCloud, scene: TreeScene) -> str:
    """Write the edited scene to ``fixed/`` and track removed points.

    Each label is written to ``fixed/{tree_id}.ply`` (or ``{tree_id}_{n}.ply``
    when one tree id ends up with several labels, e.g. after a split).  Points
    marked NOISE/UNASSIGNED that were originally assigned are appended to
    ``removed_points.xyz``.  The focus tree is marked complete.
    """
    data_dir = scene.project.data_directory
    fixed = data_dir / "fixed"
    fixed.mkdir(exist_ok=True)

    # Deleted points: originally assigned, now noise/unassigned.
    orig = scene.original_labels
    gone = (cloud.labels == NOISE) | (cloud.labels == UNASSIGNED)
    if orig is not None:
        gone &= orig > 0
    removed_idx = np.flatnonzero(gone)
    removed_n = _append_removed(cloud, removed_idx, data_dir / "removed_points.xyz")

    # Group surviving labels by tree id (new split labels inherit the focus id).
    labels = cloud.labels
    tree_to_labels: dict[str, list[int]] = {}
    for label in np.unique(labels):
        if label in (NOISE, UNASSIGNED):
            continue
        tree_id = scene.label_tree_id.get(int(label), scene.focus_tree_id)
        tree_to_labels.setdefault(tree_id, []).append(int(label))

    saved_files = []
    for tree_id, lbls in tree_to_labels.items():
        for n, label in enumerate(lbls, start=1):
            idx = np.flatnonzero(labels == label)
            suffix = "" if len(lbls) == 1 else f"_{n}"
            out = fixed / f"{tree_id}{suffix}.ply"
            sub = PointCloud(
                coords=cloud.coords[idx],
                labels=np.zeros(idx.size, dtype=np.int32),
                attributes={k: v[idx] for k, v in cloud.attributes.items()},
                label_field="treeID",
            )
            io.save(sub, str(out))
            saved_files.append(out.name)

    scene.project.mark_completed(scene.focus_tree_id)
    return (
        f"Saved {len(saved_files)} cloud(s) to fixed/ "
        f"({', '.join(saved_files)}); {removed_n} points removed."
    )
