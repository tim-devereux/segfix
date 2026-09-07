"""Point-density estimation and voxel decimation for very dense clouds.

A terrestrial or drone LiDAR plot can carry a point every few millimetres.
Nothing in segfix *needs* that detail to fix a segmentation — a tree's
extent is decided at the decimetre scale — but every millimetre of it costs
memory, GPU upload and lasso time, so a sub-2cm cloud makes the review
workflow crawl for no reviewing benefit.

This module supplies the three pieces of the answer:

* :func:`estimate_spacing` — the cloud's typical point pitch, measured
  without ever building a KD-tree over the whole thing (see its docstring;
  that's the part that has to work on a 200-million-point file).
* :func:`voxel_indices` — a "keep one real point per voxel" decimation,
  returning *indices* rather than new points, so labels, attributes and the
  file rows they came from all stay addressable.
* :func:`class_trees` / :func:`expand_labels` — the way back: which
  full-resolution points an edit on the decimated cloud stands for.

Editing then happens on the decimated cloud, and
:meth:`segfix.treecatalog._BaseCatalog.save` interpolates the edited labels
back onto every full-resolution point before writing, so the file keeps all
of its points. The prompt itself is :mod:`segfix.density_ui`.
"""

from __future__ import annotations

import numpy as np

#: Below this typical point spacing (metres) a cloud is dense enough that
#: reviewing it at full resolution is mostly wasted work — 2cm still
#: resolves a small branch, which is well past what re-labelling a stem
#: needs.
DENSE_SPACING = 0.02

# How many points a spacing measurement aims to look at, and how many query
# points it takes nearest-neighbour distances from. Both are "enough for a
# median", not tuning knobs.
_BLOCK_TARGET = 40_000
_QUERY_CAP = 20_000
_BLOCK_SEEDS = 3


def _median_nn(points: np.ndarray, rng, cap: int = _QUERY_CAP) -> float:
    """Median distance from a sample of ``points`` to their nearest other
    point in the *whole* set.

    Sampling the query points (not the tree's points) is what keeps this
    unbiased: thinning the set itself would push every nearest neighbour
    further away and overstate the spacing.
    """
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    queries = pts if len(pts) <= cap else pts[rng.choice(len(pts), cap, replace=False)]
    d, _ = cKDTree(pts).query(queries, k=2, workers=-1)
    return float(np.median(d[:, 1]))


def in_box(coords: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside an axis-aligned box, tested axis by axis
    so a huge cloud never materialises two full ``(N, 3)`` temporaries."""
    return (
        (coords[:, 0] >= lo[0]) & (coords[:, 0] <= hi[0])
        & (coords[:, 1] >= lo[1]) & (coords[:, 1] <= hi[1])
        & (coords[:, 2] >= lo[2]) & (coords[:, 2] <= hi[2])
    )


def estimate_spacing(coords: np.ndarray, rng=None) -> float:
    """The cloud's typical nearest-neighbour distance, in metres.

    :func:`segfix.analysis.point_spacing` answers the same question but
    builds a KD-tree over every point, which is exactly what can't be done
    on the clouds this check exists for (a 200M-point tree is tens of
    gigabytes). Instead this measures inside a few small boxes: local
    neighbourhoods are complete within a box, so the nearest-neighbour
    distances measured there are the real ones, and the median over several
    randomly placed boxes doesn't care that most of the cloud was never
    looked at.

    Returns ``0.0`` for a cloud too small or too sparse to measure.
    """
    coords = np.asarray(coords)
    n = len(coords)
    if n < 2:
        return 0.0
    rng = rng or np.random.default_rng(0)
    if n <= _BLOCK_TARGET:
        return _median_nn(coords, rng)

    lo_all, hi_all = coords.min(axis=0), coords.max(axis=0)
    extent = float(np.max(hi_all - lo_all))
    if extent <= 0:
        return 0.0
    # First guess at a box side holding ~_BLOCK_TARGET points, assuming the
    # points were spread evenly through the bounding box. They never are, so
    # the loop below corrects it.
    half = 0.5 * extent * max((_BLOCK_TARGET / n) ** (1 / 3), 1e-4)

    measured: list[float] = []
    for _ in range(_BLOCK_SEEDS):
        centre = coords[int(rng.integers(n))].astype(np.float64)
        block = np.empty((0, 3))
        for _ in range(8):
            mask = in_box(coords, centre - half, centre + half)
            count = int(mask.sum())
            if count > 8 * _BLOCK_TARGET:
                half *= 0.5
                continue
            if count < _BLOCK_TARGET // 8:
                half *= 2.0
                continue
            block = coords[mask]
            break
        else:
            block = coords[mask]
        if len(block) >= 50:
            measured.append(_median_nn(block, rng))
    if not measured:
        return 0.0
    return float(np.median(measured))


def suggest_voxel(spacing: float) -> float:
    """Voxel size to offer for a cloud measured at ``spacing``.

    :data:`DENSE_SPACING` is both the threshold and the target: a cloud is
    only offered decimation because it is finer than 2cm, and thinning it to
    2cm is what makes it comparable to the clouds segfix already handles
    comfortably.
    """
    return DENSE_SPACING


def voxel_indices(coords: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per occupied ``voxel``-sized cube.

    Returns *indices into* ``coords`` — real points, not voxel centroids —
    so the caller keeps every per-point label and attribute, and can still
    map each kept point back to the file row it came from.

    The point kept from each voxel is the one nearest its centre, not
    whichever came first in the file. That matters for what happens on save:
    a full-resolution point is given the label of the nearest kept point
    (:func:`nearest_label_expansion`), and centred representatives make
    "nearest kept point" line up with "own voxel". Keeping an arbitrary point
    instead leaves representatives bunched against voxel corners, where a
    neighbouring voxel's representative can be the closer one — which shows
    up as a stripe of points along an edited tree's edge quietly keeping
    their old label.
    """
    coords = np.asarray(coords)
    if voxel <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel}")
    if len(coords) == 0:
        return np.empty(0, dtype=np.int64)

    origin = coords.min(axis=0)
    local = (coords.astype(np.float64) - origin) / voxel
    keys = np.floor(local).astype(np.int64)
    # Distance to the voxel centre, in voxel units — the tie-break that
    # decides which point of each voxel is kept.
    offset = ((local - keys - 0.5) ** 2).sum(axis=1).astype(np.float32)

    dims = keys.max(axis=0) + 1
    # Flattening three axes into one int64 is much faster than a structured
    # unique, but only while the product fits; fall back when it doesn't
    # (an enormous extent against a tiny voxel).
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2.0**62:
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    else:  # pragma: no cover - needs a >10^6 m extent at millimetre voxels
        flat = np.unique(keys, axis=0, return_inverse=True)[1]

    # Sorted by voxel, and within a voxel by distance to its centre, so the
    # first row of each voxel's run is the representative.
    order = np.lexsort((offset, flat))
    _, first = np.unique(flat[order], return_index=True)
    return np.sort(order[first].astype(np.int64))


def class_trees(coords: np.ndarray, codes: np.ndarray, candidates: np.ndarray):
    """One KD-tree per original label, over the working points in
    ``candidates``.

    Grouping by label is what makes the expansion faithful. A plain nearest
    working point is not enough: a trunk point at ground level has ground
    points within a centimetre, so re-labelling the trunk would leave a ring
    of stragglers whose nearest working point was a piece of ground nobody
    touched. Asking instead for the nearest working point *that was the same
    tree* answers the question actually being asked — which edited point does
    this point belong with — and keeps the seam between two trees exactly
    where the decimated cloud put it.

    Returns ``{code: (tree, positions)}`` where ``positions`` indexes
    ``coords``.
    """
    from scipy.spatial import cKDTree

    out = {}
    if not len(candidates):
        return out
    sub_codes = codes[candidates]
    for code in np.unique(sub_codes):
        positions = candidates[sub_codes == code]
        pts = np.ascontiguousarray(coords[positions], dtype=np.float64)
        out[int(code)] = (cKDTree(pts), positions)
    return out


def expand_labels(
    trees: dict,
    query_coords: np.ndarray,
    query_codes: np.ndarray,
    changed: np.ndarray,
    n_working: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Which of ``query_coords`` inherit an edited label, and from where.

    Each query point is matched to the nearest working point that carried the
    same original label (``trees``, from :func:`class_trees`), within
    ``max_distance``. Points whose match is not one of the ``changed``
    working positions are left out entirely, so a save only ever touches the
    neighbourhoods of points the user actually re-labelled.

    Returns ``(local_indices, working_positions)`` — where the pairs are in
    ``query_coords``, and which working point each takes its label from.
    """
    empty = np.empty(0, dtype=np.int64)
    if not len(query_coords) or not trees:
        return empty, empty

    is_changed = np.zeros(n_working, dtype=bool)
    is_changed[changed] = True

    local_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for code, (tree, positions) in trees.items():
        rows = np.flatnonzero(query_codes == code)
        if not rows.size:
            continue
        dist, hit = tree.query(
            np.ascontiguousarray(query_coords[rows], dtype=np.float64),
            k=1, workers=-1, distance_upper_bound=max_distance,
        )
        # Out-of-range queries come back as len(positions) with inf distance.
        found = np.isfinite(dist)
        if not found.any():
            continue
        rows, hit = rows[found], hit[found]
        source = positions[hit]
        keep = is_changed[source]
        if not keep.any():
            continue
        local_parts.append(rows[keep])
        source_parts.append(source[keep])

    if not local_parts:
        return empty, empty
    return np.concatenate(local_parts), np.concatenate(source_parts)
