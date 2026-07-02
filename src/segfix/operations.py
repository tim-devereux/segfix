"""Segmentation-fixing operations on a :class:`~segfix.model.PointCloud`.

Each function mutates the cloud's label array through ``set_labels`` (so every
op is undoable) and returns a short human-readable summary for the status bar.
The functions are deliberately UI-agnostic: the GUI passes in point indices
(from the napari selection) or tree IDs, and these decide what the new labels
should be.
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud


def reassign(cloud: PointCloud, indices, target_id: int) -> str:
    """Move the selected points to an existing tree ``target_id``."""
    n = cloud.set_labels(indices, int(target_id), f"reassign → {target_id}")
    return f"Reassigned {n} points to tree {target_id}"


def create_new(cloud: PointCloud, indices) -> str:
    """Split the selected points off into a brand-new tree instance.

    This covers both "create a new tree from unassigned points" and "split an
    under-segmented blob": select the subset that should become its own tree
    and the rest keeps its original ID.
    """
    new_id = cloud.next_free_id()
    n = cloud.set_labels(indices, new_id, f"new tree {new_id}")
    return f"Created tree {new_id} from {n} points"


def mark_noise(cloud: PointCloud, indices) -> str:
    """Flag the selected points as noise / non-tree (kept, not deleted)."""
    n = cloud.set_labels(indices, NOISE, "mark noise")
    return f"Marked {n} points as noise"


def unassign(cloud: PointCloud, indices) -> str:
    """Return the selected points to the unassigned pool."""
    n = cloud.set_labels(indices, UNASSIGNED, "unassign")
    return f"Unassigned {n} points"


def merge_ids(cloud: PointCloud, tree_ids) -> str:
    """Merge several whole trees into one (fixes over-segmentation).

    All points belonging to any of ``tree_ids`` are relabelled to the smallest
    ID in the set, which becomes the surviving tree.
    """
    ids = sorted({int(t) for t in tree_ids})
    if len(ids) < 2:
        return "Merge needs at least two tree IDs"
    keep = ids[0]
    mask = np.isin(cloud.labels, ids)
    indices = np.flatnonzero(mask)
    n = cloud.set_labels(indices, keep, f"merge {ids} → {keep}")
    return f"Merged trees {ids} into {keep} ({n} points)"


def merge_selection(cloud: PointCloud, indices) -> str:
    """Merge every tree touched by the selection into one.

    Convenience for "draw across the fragments that should be one tree": we
    look at which tree IDs the selected points belong to and merge them.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return "Nothing selected"
    touched = np.unique(cloud.labels[indices])
    touched = touched[(touched != UNASSIGNED) & (touched != NOISE)]
    if touched.size < 2:
        return "Selection spans fewer than two trees"
    return merge_ids(cloud, touched)


def grow_from_seeds(
    cloud: PointCloud,
    seed_groups,
    k: int = 8,
    max_edge: float | None = None,
    claim_unassigned: bool = False,
) -> str:
    """Re-partition intermingled trees by growing outward from seed patches.

    ``seed_groups`` is a list of point-index arrays, one patch per intended
    tree (typically a small lasso on each trunk).  Every point belonging to a
    tree the seeds touch is reassigned to the seed it is closest to *through
    the cloud* — multi-source shortest path over a k-nearest-neighbour graph —
    so a branch follows its physical attachment rather than straight-line
    distance to a crown centre.

    Parameters
    ----------
    k:
        Neighbours per point in the graph. Lower follows thin strands more
        strictly; higher tolerates gappy scans but bridges more easily.
    max_edge:
        Maximum link length in metres. Edges longer than this are severed, so
        growth cannot leak across branches that merely come close; points cut
        off from every seed keep their current label. ``None`` = unlimited.
    claim_unassigned:
        Also grow over unassigned points near the involved trees (within
        their bounding box + 2 m), assigning them to the nearest seed —
        useful for pulling unsegmented canopy into the right tree.

    Each seed keeps the dominant existing label of its own patch; when two
    seeds sit inside the same (fused) tree, the later one gets a fresh ID.
    """
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import kneighbors_graph

    groups = [np.unique(np.asarray(g, dtype=np.int64)) for g in seed_groups]
    groups = [g for g in groups if g.size]
    if len(groups) < 2:
        return "Grow needs at least two seed patches"

    all_seeds = np.concatenate(groups)
    touched = np.unique(cloud.labels[all_seeds])
    touched = touched[(touched != UNASSIGNED) & (touched != NOISE)]
    work_mask = np.isin(cloud.labels, touched)
    work_mask[all_seeds] = True
    if claim_unassigned and work_mask.any():
        lo = cloud.coords[work_mask].min(axis=0) - 2.0
        hi = cloud.coords[work_mask].max(axis=0) + 2.0
        in_box = np.all((cloud.coords >= lo) & (cloud.coords <= hi), axis=1)
        work_mask |= in_box & (cloud.labels == UNASSIGNED)
    work = np.flatnonzero(work_mask)
    if work.size < 2:
        return "Seeds touch no tree points to grow over"

    graph = kneighbors_graph(
        cloud.coords[work], min(k, work.size - 1), mode="distance"
    )
    if max_edge:
        # Sever long links so growth can't jump across touching branches.
        graph.data[graph.data > max_edge] = 0
        graph.eliminate_zeros()
    graph = graph.maximum(graph.T)  # make edges usable in both directions

    seed_local = [np.searchsorted(work, g) for g in groups]
    sources = np.concatenate(seed_local)
    _, _, src_node = dijkstra(
        graph,
        directed=False,
        indices=sources,
        return_predecessors=True,
        min_only=True,
    )
    node_group = np.full(work.size, -1, dtype=np.int64)
    for gi, s in enumerate(seed_local):
        node_group[s] = gi
    reached = src_node >= 0
    assign = np.full(work.size, -1, dtype=np.int64)
    assign[reached] = node_group[src_node[reached]]

    next_id = cloud.next_free_id()
    target_ids: list[int] = []
    for g in groups:
        labs = cloud.labels[g]
        labs = labs[(labs != UNASSIGNED) & (labs != NOISE)]
        dom = None
        if labs.size:
            vals, counts = np.unique(labs, return_counts=True)
            dom = int(vals[counts.argmax()])
        if dom is None or dom in target_ids:
            dom = next_id
            next_id += 1
        target_ids.append(dom)

    moved = 0
    for gi, tid in enumerate(target_ids):
        moved += cloud.set_labels(
            work[assign == gi], tid, f"grow seed {gi + 1} → {tid}"
        )
    msg = (
        f"Grew {len(groups)} seeds over {work.size:,} points; "
        f"{moved:,} reassigned → trees {target_ids}"
    )
    stranded = int((assign < 0).sum())
    if stranded:
        msg += f" ({stranded:,} unreachable, unchanged)"
    return msg


def split_auto(cloud: PointCloud, tree_id: int, n_clusters: int = 2) -> str:
    """Auto-split one tree into ``n_clusters`` trees via KMeans on XYZ.

    A quick way to break an under-segmented blob without hand-selecting every
    point.  The largest resulting cluster keeps the original ID; the others
    get new IDs.  For precise control, use the lasso + "create new" instead.
    """
    from sklearn.cluster import KMeans

    mask = cloud.labels == int(tree_id)
    member_idx = np.flatnonzero(mask)
    if member_idx.size < n_clusters:
        return f"Tree {tree_id} has too few points to split"

    xyz = cloud.coords[member_idx]
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    sub = km.fit_predict(xyz)

    # Keep the biggest sub-cluster on the original ID; assign fresh IDs to the
    # rest.  We allocate new IDs up front so they don't collide with each other.
    counts = np.bincount(sub, minlength=n_clusters)
    keep_cluster = int(counts.argmax())
    summary_ids = [int(tree_id)]
    for c in range(n_clusters):
        if c == keep_cluster:
            continue
        new_id = cloud.next_free_id()
        cloud.set_labels(member_idx[sub == c], new_id, f"split {tree_id}")
        summary_ids.append(new_id)
    return f"Split tree {tree_id} into {summary_ids}"
