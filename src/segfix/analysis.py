"""Segmentation-quality analysis: fragment suggestions and neighbour queries.

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
    in_box = np.all((coords >= lo) & (coords <= hi), axis=1)
    in_box[mine] = False
    cand = np.unique(labels[in_box])
    cand = cand[(cand != UNASSIGNED) & (cand != NOISE) & (cand != tid)]
    if not cand.size:
        return set()
    kd = cKDTree(coords[_sample(rng, mine)])
    out: set[int] = set()
    for t in cand:
        theirs = np.flatnonzero(in_box & (labels == t))
        d, _ = kd.query(
            coords[_sample(rng, theirs)], k=1, distance_upper_bound=reach
        )
        if np.isfinite(d).any():
            out.add(int(t))
    return out


def find_fragments(
    cloud: PointCloud,
    max_gap: float = 0.3,
    float_base: float = 5.0,
    tiny: int = 500,
    rng=None,
) -> dict[int, tuple[int, float]]:
    """Suggest fragment → host merges for likely over-segmentation.

    A tree is a fragment *candidate* when it floats (its lowest point is more
    than ``float_base`` metres above the ground, taken as the 1st percentile
    of all z) or is ``tiny``.  A candidate becomes a *suggestion* when its
    points come within ``max_gap`` metres of another tree — the closest such
    tree is the proposed host.

    Returns ``{fragment_id: (host_id, gap_m)}`` ordered best (smallest gap)
    first.  Hosts that are themselves suggested fragments are followed to
    their final host, so suggestions can be accepted in any order.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    labels, coords = cloud.labels, cloud.coords
    ids = cloud.tree_ids
    if ids.size < 2:
        return {}
    ground = float(np.percentile(coords[:, 2], 1))

    fragments: list[int] = []
    for tid in ids:
        mine = np.flatnonzero(labels == tid)
        zmin = float(coords[mine, 2].min())
        if mine.size < tiny or zmin > ground + float_base:
            fragments.append(int(tid))
    frag_set = set(fragments)

    raw: dict[int, tuple[int, float]] = {}
    for tid in fragments:
        mine = np.flatnonzero(labels == tid)
        lo = coords[mine].min(axis=0) - max_gap
        hi = coords[mine].max(axis=0) + max_gap
        in_box = np.all((coords >= lo) & (coords <= hi), axis=1)
        in_box[mine] = False
        cand = np.unique(labels[in_box])
        cand = cand[(cand != UNASSIGNED) & (cand != NOISE) & (cand != tid)]
        kd = cKDTree(coords[_sample(rng, mine)])
        best: tuple[int, float] | None = None
        best_real = False  # prefer hosts that are not fragments themselves
        for t in cand:
            theirs = np.flatnonzero(in_box & (labels == t))
            d, _ = kd.query(coords[_sample(rng, theirs)], k=1)
            gap = float(d.min())
            if gap > max_gap:
                continue
            real = int(t) not in frag_set
            if (
                best is None
                or (real and not best_real)
                or (real == best_real and gap < best[1])
            ):
                best = (int(t), gap)
                best_real = real
        if best is not None:
            raw[tid] = best

    resolved: dict[int, tuple[int, float]] = {}
    for frag, (host, gap) in raw.items():
        seen = {frag}
        while host in raw and host not in seen:
            seen.add(host)
            host = raw[host][0]
        if host == frag:  # two fragments touching each other: keep direct
            host = raw[frag][0]
        resolved[frag] = (host, gap)
    return dict(sorted(resolved.items(), key=lambda kv: kv[1][1]))
