"""Segmentation-quality analysis: spatial neighbour queries.

UI-agnostic, like :mod:`~segfix.operations`.  Point-set distances are
approximated by sampling each tree down to a few thousand points and querying
a KD-tree — exact enough for "does this touch?" questions at forest scale,
and fast enough to run per keypress.
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud

_SAMPLE_CAP = 3000


def _sample(rng, idx: np.ndarray, cap: int = _SAMPLE_CAP) -> np.ndarray:
    if idx.size <= cap:
        return idx
    return rng.choice(idx, cap, replace=False)


def build_cluster_index(coords: np.ndarray, rng=None):
    """Build the KD-tree + a gap threshold for :func:`cluster_from_seed`.

    The gap is derived from the cloud's own point spacing (median
    nearest-neighbour distance on a sample), scaled up so a region grow
    bridges the normal gaps between adjacent points but not the empty space
    between separate structures.  Returned as ``(kdtree, eps)`` so a caller
    can cache it for the lifetime of a loaded cloud.
    """
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    tree = cKDTree(coords)
    rng = rng or np.random.default_rng(0)
    n = len(coords)
    sample = coords if n <= 20000 else coords[rng.choice(n, 20000, replace=False)]
    d, _ = tree.query(sample, k=2, workers=-1)
    spacing = float(np.median(d[:, 1])) or 0.01
    return tree, spacing * 4.0


def cluster_from_seed(
    coords: np.ndarray, seed: int, eps: float, *, mask=None,
    kdtree=None, max_points: int = 400_000,
) -> np.ndarray:
    """Indices of the connected patch of points reachable from ``seed`` by
    hops of at most ``eps`` metres (a breadth-first region grow).

    ``mask`` (per-point bool) restricts the grow to a subset — e.g. only the
    points currently shown.  Stops early at ``max_points`` so a click that
    lands on a giant connected component can't lock the UI.
    """
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    tree = kdtree if kdtree is not None else cKDTree(coords)
    allowed = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    seed = int(seed)
    if not (0 <= seed < n) or not allowed[seed]:
        return np.array([seed] if 0 <= seed < n else [], dtype=np.int64)

    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    frontier = np.array([seed], dtype=np.int64)
    while frontier.size:
        balls = tree.query_ball_point(coords[frontier], eps, workers=-1)
        cand = np.unique(np.concatenate(balls)) if len(balls) else np.empty(0, int)
        cand = cand[allowed[cand] & ~visited[cand]]
        if cand.size == 0:
            break
        room = max_points - int(visited.sum())
        if cand.size > room:
            cand = cand[:room]
            visited[cand] = True
            break
        visited[cand] = True
        frontier = cand
    return np.flatnonzero(visited)


def neighbours_by_points(
    cloud: PointCloud, tid: int, reach: float, rng=None
) -> set[int]:
    """IDs of trees whose points come within ``reach`` metres of tree
    ``tid``'s points.

    Bounding-box tests massively over-count neighbours in a closed canopy
    (a tall tree's box spans its whole crown), so boxes are only used as a
    prefilter; candidates are confirmed by sampled point-to-point distance.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    labels, coords = cloud.labels, cloud.coords
    mine = np.flatnonzero(labels == tid)
    if mine.size == 0:
        return set()
    lo = coords[mine].min(axis=0) - reach
    hi = coords[mine].max(axis=0) + reach
    # Per-axis chained test avoids two full (N, 3) boolean temporaries.
    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
    in_box = (
        (cx >= lo[0]) & (cx <= hi[0])
        & (cy >= lo[1]) & (cy <= hi[1])
        & (cz >= lo[2]) & (cz <= hi[2])
    )
    in_box[mine] = False
    # Everything below works off the in-box points only, so the per-candidate
    # loop no longer rescans all N labels once per candidate.
    box_idx = np.flatnonzero(in_box)
    box_labels = labels[box_idx]
    cand = np.unique(box_labels)
    cand = cand[(cand != UNASSIGNED) & (cand != NOISE) & (cand != tid)]
    if not cand.size:
        return set()
    kd = cKDTree(coords[_sample(rng, mine)])
    out: set[int] = set()
    for t in cand:
        theirs = box_idx[box_labels == t]
        d, _ = kd.query(
            coords[_sample(rng, theirs)], k=1, distance_upper_bound=reach
        )
        if np.isfinite(d).any():
            out.add(int(t))
    return out
