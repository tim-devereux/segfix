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


def test_connected_patch_isolates_the_blob_around_the_seed():
    """One tree ID split into two physically separate blobs plus a second
    tree: connected_patch from a seed in one blob returns just that blob."""
    rng = np.random.default_rng(0)
    blob_a = rng.normal((0, 0, 0), 0.05, (2000, 3))
    blob_b = rng.normal((5, 0, 0), 0.05, (2000, 3))  # same label, far away
    other = rng.normal((0.1, 0, 0), 0.05, (2000, 3))  # different label, touching a
    coords = np.vstack([blob_a, blob_b, other]).astype(np.float32)
    labels = np.concatenate([
        np.full(2000, 7), np.full(2000, 7), np.full(2000, 9)
    ]).astype(np.int32)
    gap = analysis.point_spacing(coords) * 4.0

    got = analysis.connected_patch(coords, labels, seed=10, eps=gap)
    assert (got < 2000).all()          # only blob A
    assert len(got) > 1800             # nearly all of it
    assert 9 not in labels[got]        # never crosses into the other tree

    got_b = analysis.connected_patch(coords, labels, seed=2500, eps=gap)
    assert (got_b >= 2000).all() and (got_b < 4000).all()  # only blob B


def test_connected_components_within_labels_disjoint_blobs():
    rng = np.random.default_rng(1)
    a = rng.normal((0, 0, 0), 0.03, (500, 3))
    b = rng.normal((3, 0, 0), 0.03, (500, 3))
    coords = np.vstack([a, b]).astype(np.float32)
    comp = analysis.connected_components_within(coords, eps=0.2)
    assert len(np.unique(comp)) == 2
    assert len(np.unique(comp[:500])) == 1
    assert comp[0] != comp[500]


def test_point_spacing_matches_a_regular_grid():
    g = np.mgrid[0:10, 0:10, 0:3].reshape(3, -1).T.astype(np.float32) * 0.5
    assert abs(analysis.point_spacing(g) - 0.5) < 1e-6
