import numpy as np

from segfix import analysis
from segfix.model import PointCloud


def _line(start, end, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(start) + t * (np.subtract(end, start))


def _cloud():
    """A host tree, a touching crown fragment, a far real tree, an isolated
    sapling, and a two-point decoy whose bbox overlaps the host's."""
    host = _line((0, 0, 0), (0, 0, 10), n=600)             # label 1
    frag = _line((0.05, 0, 8), (0.6, 0, 9), n=50)          # label 2: floating+tiny, touches host
    far = _line((20, 0, 0), (20, 0, 10), n=600)            # label 3: real tree
    sapling = _line((40, 40, 0), (40, 40, 1), n=30)        # label 4: tiny but isolated
    decoy = np.array([[-0.5, 5, 0], [0.5, -5, 10]])        # label 5: bbox spans host, points far
    coords = np.vstack([host, frag, far, sapling, decoy]).astype(np.float32)
    labels = np.concatenate([
        np.full(600, 1), np.full(50, 2), np.full(600, 3),
        np.full(30, 4), np.full(2, 5),
    ]).astype(np.int32)
    return PointCloud(coords=coords, labels=labels)


def test_neighbours_by_points_ignores_bbox_only_overlap():
    cloud = _cloud()
    # The decoy's bbox intersects the host's expanded box, but its points
    # are metres away — a bbox rule would include it, point distance must not.
    near = analysis.neighbours_by_points(cloud, 1, reach=1.0)
    assert 5 not in near
    assert 2 in near  # the touching fragment is a true neighbour
    assert 3 not in near  # 20 m away


def test_neighbours_by_points_respects_reach():
    a = _line((0, 0, 0), (10, 0, 0))
    b = _line((0, 1.5, 0), (10, 1.5, 0))
    cloud = PointCloud(
        coords=np.vstack([a, b]).astype(np.float32),
        labels=np.concatenate([np.full(200, 1), np.full(200, 2)]).astype(np.int32),
    )
    assert analysis.neighbours_by_points(cloud, 1, reach=1.0) == set()
    assert analysis.neighbours_by_points(cloud, 1, reach=2.0) == {2}


def test_cluster_from_seed_grows_one_connected_blob():
    """Two dense blobs with a wide empty gap: a region grow from a seed in
    one must stay inside it."""
    rng = np.random.default_rng(0)
    a = rng.normal((0, 0, 0), 0.05, (4000, 3))
    b = rng.normal((5, 0, 0), 0.05, (4000, 3))
    coords = np.vstack([a, b]).astype(np.float32)
    tree, eps = analysis.build_cluster_index(coords)

    got_a = analysis.cluster_from_seed(coords, 0, eps, kdtree=tree)
    assert (got_a < 4000).all()
    assert len(got_a) > 3500  # nearly the whole blob is connected

    got_b = analysis.cluster_from_seed(coords, 5000, eps, kdtree=tree)
    assert (got_b >= 4000).all()


def test_cluster_from_seed_respects_mask_and_cap():
    rng = np.random.default_rng(1)
    coords = rng.normal((0, 0, 0), 0.1, (6000, 3)).astype(np.float32)
    tree, eps = analysis.build_cluster_index(coords)

    mask = np.zeros(6000, dtype=bool)
    mask[:2000] = True
    masked = analysis.cluster_from_seed(coords, 0, eps, mask=mask, kdtree=tree)
    assert masked.max() < 2000

    capped = analysis.cluster_from_seed(
        coords, 0, eps * 50, kdtree=tree, max_points=500
    )
    assert len(capped) <= 500

    # a seed the mask forbids yields just that seed (or nothing)
    forbidden = analysis.cluster_from_seed(
        coords, 5000, eps, mask=mask, kdtree=tree
    )
    assert set(forbidden.tolist()) <= {5000}
