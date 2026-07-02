"""Project/session layer: discover per-tree PLY files in a directory and find
spatial neighbours, mirroring the CloudCompare plugin's no-CSV workflow.

In this dataset each PLY file *is* one tree; the tree identity comes from the
filename, not a per-point label.  A :class:`Project` scans a directory into
:class:`TreeEntry` rows (the table), derives each tree's XY position, and —
when you focus a tree — gathers it plus nearby trees so they can be loaded
together for editing.
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io

# Neighbour-search tuning (ported from the plugin's config.py).
DISTANCE_THRESHOLD = 10.0
NEIGHBOUR_QUERY_RADIUS = 1.0
BBOX_CANDIDATE_MARGIN = 0.5
SETTINGS_FILE_NAME = "segfix_settings.json"


# -- filename parsing ----------------------------------------------------
def extract_tree_id(filename: str):
    """Parse ``(tree_id, stem_id, file_type)`` from a PLY filename.

    Mirrors the plugin: ``{tree}_matched.ply``, ``{tree}_{stem}_uncertain.ply``
    etc. yield a tree id; anything without ``_matched``/``_uncertain`` is
    unmatched and returns ``(None, None, None)``.
    """
    if not filename.endswith(".ply"):
        return None, None, None
    stem = filename[:-4]
    if "_matched" not in stem and "_uncertain" not in stem:
        return None, None, None

    if "_matched" in stem:
        base, _ = stem.rsplit("_matched", 1)
        file_type = "matched"
    else:
        base, _ = stem.rsplit("_uncertain", 1)
        file_type = "uncertain"

    if "_" in base:
        tree_id, stem_id = base.rsplit("_", 1)
        if tree_id and stem_id and tree_id[0].isalnum() and stem_id[0].isalnum():
            return tree_id, stem_id, file_type
    elif base and base[0].isalnum():
        return base, None, file_type
    return None, None, None


@dataclass
class TreeEntry:
    """One row in the tree table: a primary file plus any sibling stems."""

    mesh_file: str
    tree_id: str
    secondary_files: list[str] = field(default_factory=list)
    file_format: str = "matched_uncertain"
    notes: str = ""
    # XY position derived from the cloud, filled lazily; used for neighbours.
    position: tuple[float, float] | None = None

    @property
    def all_files(self) -> list[str]:
        return [self.mesh_file, *self.secondary_files]


def discover(data_directory: str | Path) -> list[TreeEntry]:
    """Scan ``data_directory`` for tree PLY files and group them by tree id."""
    data_directory = Path(data_directory)
    entries: list[TreeEntry] = []
    groups: dict[str, dict[str, list[str]]] = {}

    for path in sorted(glob.glob(str(data_directory / "*.ply"))):
        filename = os.path.basename(path)
        if "_non_seg" in filename:
            continue  # overlay data, not a tree
        tree_id, _stem, file_type = extract_tree_id(filename)
        if tree_id is None:
            entries.append(
                TreeEntry(
                    mesh_file=filename,
                    tree_id="No Match",
                    file_format="unmatched",
                    notes="No _matched/_uncertain suffix",
                )
            )
            continue
        groups.setdefault(tree_id, {"matched": [], "uncertain": []})
        groups[tree_id].setdefault(file_type, []).append(filename)

    for tree_id, by_type in groups.items():
        matched = sorted(by_type.get("matched", []))
        uncertain = sorted(by_type.get("uncertain", []))
        if matched:
            primary, secondary = matched[0], matched[1:] + uncertain
            notes = "Auto-detected"
        elif uncertain:
            primary, secondary = uncertain[0], uncertain[1:]
            notes = "MISSING MATCHED - auto-detected"
        else:
            continue
        n = len(secondary) + 1
        if n > 1:
            notes += f" ({n} stems)"
        entries.append(
            TreeEntry(
                mesh_file=primary,
                tree_id=tree_id,
                secondary_files=secondary,
                notes=notes,
            )
        )
    return entries


# -- project -------------------------------------------------------------
class Project:
    """A data directory of tree PLY files plus session state."""

    def __init__(self, data_directory: str | Path):
        self.data_directory = Path(data_directory)
        self.entries: list[TreeEntry] = discover(self.data_directory)
        self.completed: set[str] = set()
        self._bbox_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        self.load_settings()

    # -- geometry helpers ----------------------------------------------
    def _bbox(self, mesh_file: str):
        if mesh_file in self._bbox_cache:
            return self._bbox_cache[mesh_file]
        path = self.data_directory / mesh_file
        try:
            pts = io.read_xyz(str(path))
        except Exception:
            pts = np.empty((0, 3), np.float32)
        bbox = None if pts.shape[0] == 0 else (pts.min(0), pts.max(0))
        self._bbox_cache[mesh_file] = bbox
        return bbox

    def position(self, entry: TreeEntry) -> tuple[float, float] | None:
        """XY centre of the tree, derived from its primary cloud (cached)."""
        if entry.position is not None:
            return entry.position
        bbox = self._bbox(entry.mesh_file)
        if bbox is None:
            return None
        lo, hi = bbox
        entry.position = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
        return entry.position

    # -- neighbour finding ---------------------------------------------
    def neighbours(
        self,
        focus: TreeEntry,
        distance_threshold: float = DISTANCE_THRESHOLD,
        bbox_margin: float = BBOX_CANDIDATE_MARGIN,
        query_radius: float = NEIGHBOUR_QUERY_RADIUS,
    ) -> list[TreeEntry]:
        """Return ``focus`` plus nearby trees to load together.

        Three stages, matching the plugin: (1) all trees within
        ``distance_threshold`` of the focus centre, (2) expand to any tree whose
        bbox intersects the combined bbox grown by ``bbox_margin``, (3) keep
        only trees with a point within ``query_radius`` of the running mask.
        """
        target = self.position(focus)
        if target is None:
            return [focus]

        # Stage 1: distance threshold.
        within = []
        for e in self.entries:
            if e is focus or e.tree_id == "No Match":
                continue
            pos = self.position(e)
            if pos is None:
                continue
            d2 = (target[0] - pos[0]) ** 2 + (target[1] - pos[1]) ** 2
            if d2 <= distance_threshold**2:
                within.append((d2, e))
        within.sort(key=lambda t: t[0])
        candidates = [e for _, e in within]

        # Stage 2: bbox-margin expansion off the combined loaded bbox.
        candidates = self._expand_by_bbox(focus, candidates, bbox_margin)

        # Stage 3: nearest-neighbour proximity filter.
        return self._filter_by_proximity(focus, candidates, query_radius)

    def _expand_by_bbox(self, focus, candidates, margin):
        if margin <= 0:
            return candidates
        boxes = [self._bbox(focus.mesh_file)]
        boxes += [self._bbox(e.mesh_file) for e in candidates]
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return candidates
        lo = np.min([b[0] for b in boxes], axis=0) - margin
        hi = np.max([b[1] for b in boxes], axis=0) + margin
        chosen = {focus.mesh_file, *(e.mesh_file for e in candidates)}
        expanded = list(candidates)
        for e in self.entries:
            if e.mesh_file in chosen or e.tree_id == "No Match":
                continue
            b = self._bbox(e.mesh_file)
            if b is None:
                continue
            if np.all(hi >= b[0]) and np.all(b[1] >= lo):
                expanded.append(e)
                chosen.add(e.mesh_file)
        return expanded

    def _filter_by_proximity(self, focus, candidates, radius):
        from sklearn.neighbors import KDTree

        base = io.read_xyz(str(self.data_directory / focus.mesh_file))
        if base.shape[0] == 0:
            return [focus]
        mask_pts = base
        tree = KDTree(mask_pts)
        kept = [focus]
        for e in candidates:
            pts = io.read_xyz(str(self.data_directory / e.mesh_file))
            if pts.shape[0] == 0:
                continue
            counts = tree.query_radius(pts, r=radius, count_only=True)
            if np.any(counts > 0):
                kept.append(e)
                mask_pts = np.vstack([mask_pts, pts])
                tree = KDTree(mask_pts)
        return kept

    # -- settings ------------------------------------------------------
    @property
    def settings_path(self) -> Path:
        return self.data_directory / SETTINGS_FILE_NAME

    def load_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text())
        except Exception:
            return
        self.completed = set(data.get("completed_trees", []))

    def save_settings(self) -> None:
        payload = {
            "data_directory": str(self.data_directory),
            "completed_trees": sorted(self.completed),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.settings_path.write_text(json.dumps(payload, indent=2))

    def mark_completed(self, tree_id: str) -> None:
        self.completed.add(tree_id)
        self.save_settings()
