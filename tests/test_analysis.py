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


def test_find_fragments_suggests_touching_host():
    cloud = _cloud()
    sugg = analysis.find_fragments(cloud)
    assert 2 in sugg
    host, gap = sugg[2]
    assert host == 1
    assert gap < 0.3
    # grounded, substantial trees are never fragments
    assert 1 not in sugg and 3 not in sugg
    # the isolated sapling is a candidate but touches nothing
    assert 4 not in sugg


def test_find_fragments_follows_chains():
    # frag B touches frag A which touches host H: B must resolve to H.
    host = _line((0, 0, 0), (0, 0, 10), n=600)
    frag_a = _line((0.05, 0, 8), (1.0, 0, 8.5), n=40)
    frag_b = _line((1.05, 0, 8.5), (2.0, 0, 9), n=40)
    cloud = PointCloud(
        coords=np.vstack([host, frag_a, frag_b]).astype(np.float32),
        labels=np.concatenate(
            [np.full(600, 1), np.full(40, 2), np.full(40, 3)]
        ).astype(np.int32),
    )
    sugg = analysis.find_fragments(cloud)
    assert sugg[2][0] == 1
    assert sugg[3][0] == 1  # chained through fragment 2


def test_find_fragments_empty_cases():
    one = PointCloud(
        coords=_line((0, 0, 0), (0, 0, 10)).astype(np.float32),
        labels=np.full(200, 1, np.int32),
    )
    assert analysis.find_fragments(one) == {}
