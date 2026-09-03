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


def point_spacing(coords: np.ndarray, rng=None) -> float:
    """Median nearest-neighbour distance over a sample — the cloud's typical
    point pitch.  Used to pick the gap a region grow is allowed to bridge.
    """
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return 0.01
    rng = rng or np.random.default_rng(0)
    sample = coords if n <= 20000 else coords[rng.choice(n, 20000, replace=False)]
    d, _ = cKDTree(coords).query(sample, k=2, workers=-1)
    return float(np.median(d[:, 1])) or 0.01


def connected_components_within(coords: np.ndarray, eps: float) -> np.ndarray:
    """Label every point 0..k-1 by which ``eps``-connected blob it belongs to.

    Runs on the given points only (call it per tree, not per plot), so it
    stays fast even on a big cloud.  Points within ``eps`` of each other are
    in the same blob.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(coords).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    data = np.ones(len(pairs), dtype=np.int8)
    graph = coo_matrix(
        (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    _, comp = connected_components(graph, directed=False)
    return comp


def connected_blob(
    coords: np.ndarray, seed: int, eps: float, tree=None, cap: int = 400_000
) -> np.ndarray:
    """Indices of the points reachable from ``coords[seed]`` by hops of at
    most ``eps`` — one physically continuous blob around the seed, with no
    regard for labels (so it can span several trees / unassigned points
    where they actually touch).

    Breadth-first over a KD-tree, so the cost tracks the blob size, not the
    whole cloud; pass a ``tree`` already built over the same ``coords`` to
    skip rebuilding it each call. ``cap`` stops the runaway case where a
    large ``eps`` percolates the whole cloud into one mass.
    """
    from scipy.spatial import cKDTree

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    tree = tree if tree is not None else cKDTree(coords)
    seed = int(seed)
    seen = np.zeros(n, dtype=bool)
    seen[seed] = True
    frontier = np.array([seed], dtype=np.int64)
    while frontier.size:
        groups = tree.query_ball_point(coords[frontier], eps, workers=-1)
        cand = np.unique(np.concatenate(
            [np.asarray(g, dtype=np.int64) for g in groups]
        )) if len(groups) else np.empty(0, dtype=np.int64)
        frontier = cand[~seen[cand]]
        seen[frontier] = True
        if seen.sum() >= cap:
            break
    return np.flatnonzero(seen)


def connected_patch(
    coords: np.ndarray, labels: np.ndarray, seed: int, eps: float
) -> np.ndarray:
    """Indices of the ``eps``-connected blob of the seed's own tree that the
    seed sits in — i.e. one physically continuous lump of a single tree ID.
    """
    seed = int(seed)
    same = np.flatnonzero(labels == labels[seed])
    if same.size <= 1:
        return same
    comp = connected_components_within(coords[same], eps)
    local = int(np.searchsorted(same, seed))
    return same[comp == comp[local]]


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
