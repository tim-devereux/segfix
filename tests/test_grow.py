import numpy as np

from segfix import operations as ops
from segfix.model import PointCloud


def _line(start, end, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(start) + t * (np.subtract(end, start))


def test_grow_splits_fused_tree_by_connectivity():
    # Two parallel "stems" fused under one label. Point spacing along each
    # stem (~0.17) is much smaller than the stem separation (1.0), so the
    # kNN graph keeps the stems as separate strands.
    a = _line((0, 0, 0), (0, 0, 10))
    b = _line((1, 0, 0), (1, 0, 10))
    cloud = PointCloud(
        coords=np.vstack([a, b]).astype(np.float32),
        labels=np.full(120, 7, np.int32),
    )
    msg = ops.grow_from_seeds(cloud, [np.arange(0, 5), np.arange(60, 65)])

    assert (cloud.labels[:60] == 7).all()  # first seed keeps the label
    new_id = cloud.labels[60]
    assert new_id != 7
    assert (cloud.labels[60:] == new_id).all()
    assert "2 seeds" in msg


def test_grow_follows_attachment_not_crown_distance():
    # A drooping branch: attached to stem A at the top, but its tip hangs
    # much closer (straight-line) to stem B. Geodesic growth must give the
    # whole branch to A; nearest-centroid logic would hand the tip to B.
    # (The 1.2 m tip gap stays outside the kNN graph's reach — points in
    # actual contact are genuinely ambiguous and may go either way.)
    stem_a = _line((0, 0, 0), (0, 0, 10))
    stem_b = _line((6, 0, 0), (6, 0, 10))
    branch = _line((0, 0, 10), (4.8, 0, 2), n=80)  # tip 1.2 from stem B
    coords = np.vstack([stem_a, stem_b, branch]).astype(np.float32)
    labels = np.concatenate(
        [np.full(60, 1), np.full(60, 2), np.full(80, 2)]  # branch mislabelled
    ).astype(np.int32)
    cloud = PointCloud(coords=coords, labels=labels)

    ops.grow_from_seeds(
        cloud, [np.arange(0, 5), np.arange(60, 65)]  # base of each stem
    )
    assert (cloud.labels[:60] == 1).all()
    assert (cloud.labels[60:120] == 2).all()
    branch_labels = cloud.labels[120:]
    # the whole branch follows its attachment to stem A
    assert (branch_labels == 1).all(), np.unique(branch_labels)


def test_grow_is_undoable():
    a = _line((0, 0, 0), (0, 0, 10))
    b = _line((2, 0, 0), (2, 0, 10))
    cloud = PointCloud(
        coords=np.vstack([a, b]).astype(np.float32),
        labels=np.full(120, 3, np.int32),
    )
    ops.grow_from_seeds(cloud, [np.arange(0, 5), np.arange(60, 65)])
    assert len(np.unique(cloud.labels)) == 2
    while cloud.can_undo:
        cloud.undo()
    assert (cloud.labels == 3).all()


def test_grow_max_edge_severs_contact_bridge():
    # A branch from stem A whose tip comes within 0.4 m of stem B — inside
    # kNN reach, so unlimited growth leaks across the contact. A max_edge
    # below the contact gap severs the bridge and the branch stays with A.
    stem_a = _line((0, 0, 0), (0, 0, 10))
    stem_b = _line((6, 0, 0), (6, 0, 10))
    branch = _line((0, 0, 10), (5.6, 0, 2), n=80)  # tip 0.4 from stem B

    def build():
        coords = np.vstack([stem_a, stem_b, branch]).astype(np.float32)
        labels = np.concatenate(
            [np.full(60, 1), np.full(60, 2), np.full(80, 1)]
        ).astype(np.int32)
        return PointCloud(coords=coords, labels=labels)

    seeds = [np.arange(0, 5), np.arange(60, 65)]

    leaky = build()
    ops.grow_from_seeds(leaky, seeds)
    assert (leaky.labels[120:] == 2).any()  # unlimited growth leaks

    strict = build()
    ops.grow_from_seeds(strict, seeds, max_edge=0.3)
    assert (strict.labels[120:] == 1).all()  # bridge severed


def test_grow_claims_nearby_unassigned():
    from segfix.model import UNASSIGNED

    stem_a = _line((0, 0, 0), (0, 0, 10))
    stem_b = _line((4, 0, 0), (4, 0, 10))
    fringe = _line((0.1, 0, 5), (0.5, 0, 8), n=20)   # unsegmented bits near A
    far = _line((50, 50, 0), (50, 50, 5), n=10)       # unrelated, far away
    coords = np.vstack([stem_a, stem_b, fringe, far]).astype(np.float32)
    labels = np.concatenate(
        [np.full(60, 1), np.full(60, 2), np.zeros(30)]
    ).astype(np.int32)
    cloud = PointCloud(coords=coords, labels=labels)
    seeds = [np.arange(0, 5), np.arange(60, 65)]

    ops.grow_from_seeds(cloud, seeds, claim_unassigned=True)
    assert (cloud.labels[120:140] == 1).all()          # fringe joins stem A
    assert (cloud.labels[140:] == UNASSIGNED).all()    # far points untouched


def test_grow_needs_two_seeds():
    cloud = PointCloud(
        coords=np.random.rand(10, 3).astype(np.float32),
        labels=np.ones(10, np.int32),
    )
    msg = ops.grow_from_seeds(cloud, [np.arange(3)])
    assert "two seed" in msg
    assert (cloud.labels == 1).all()
