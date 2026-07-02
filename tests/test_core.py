import numpy as np
import pytest

from segfix import NOISE, UNASSIGNED, PointCloud
from segfix import io, operations as ops
from segfix.lasso import points_in_polygon


def make_cloud():
    # Three little trees of 4 points each, IDs 1, 2, 3.
    coords = np.random.RandomState(0).rand(12, 3).astype(np.float32)
    labels = np.repeat([1, 2, 3], 4).astype(np.int32)
    return PointCloud(coords=coords, labels=labels)


def test_reassign_and_undo():
    c = make_cloud()
    ops.reassign(c, [0, 1], 2)
    assert list(c.labels[:4]) == [2, 2, 1, 1]
    c.undo()
    assert list(c.labels[:4]) == [1, 1, 1, 1]
    c.redo()
    assert list(c.labels[:4]) == [2, 2, 1, 1]


def test_create_new_allocates_fresh_id():
    c = make_cloud()
    ops.create_new(c, [0, 1])  # split first two points off tree 1
    assert c.labels[0] == 4 and c.labels[1] == 4
    assert set(c.tree_ids.tolist()) == {1, 2, 3, 4}


def test_merge_ids_keeps_smallest():
    c = make_cloud()
    ops.merge_ids(c, [2, 3])
    assert 3 not in c.tree_ids
    assert (c.labels[4:] == 2).all()


def test_merge_selection_spans_trees():
    c = make_cloud()
    # indices 3 (tree1) and 4 (tree2) span two trees
    ops.merge_selection(c, [3, 4])
    assert c.labels[3] == c.labels[4] == 1


def test_mark_noise_and_unassign():
    c = make_cloud()
    ops.mark_noise(c, [0])
    assert c.labels[0] == NOISE
    ops.unassign(c, [1])
    assert c.labels[1] == UNASSIGNED
    # noise/unassigned excluded from tree_ids
    assert NOISE not in c.tree_ids and UNASSIGNED not in c.tree_ids


def test_split_auto():
    rs = np.random.RandomState(1)
    a = rs.rand(50, 3) + [0, 0, 0]
    b = rs.rand(50, 3) + [10, 0, 0]  # well separated cluster
    coords = np.vstack([a, b]).astype(np.float32)
    labels = np.ones(100, dtype=np.int32)
    c = PointCloud(coords=coords, labels=labels)
    ops.split_auto(c, 1, 2)
    assert len(c.tree_ids) == 2
    # the two clusters should be cleanly separated
    ids = c.labels
    assert len(set(ids[:50])) == 1 and len(set(ids[50:])) == 1
    assert ids[0] != ids[50]


def test_no_op_records_no_undo():
    c = make_cloud()
    ops.reassign(c, [0], 1)  # already tree 1
    assert not c.can_undo


def test_points_in_polygon_square():
    # unit square polygon
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    pts = np.array(
        [[5, 5], [1, 1], [9, 9], [-1, 5], [5, 11], [10.5, 5]], dtype=float
    )
    mask = points_in_polygon(poly, pts)
    assert list(mask) == [True, True, True, False, False, False]


def test_points_in_polygon_concave():
    # an L / concave shape; the notch must read as outside
    poly = np.array(
        [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], dtype=float
    )
    inside = points_in_polygon(poly, np.array([[2, 8]], dtype=float))[0]
    outside = points_in_polygon(poly, np.array([[8, 8]], dtype=float))[0]
    assert inside and not outside


def test_points_in_polygon_degenerate():
    poly = np.array([[0, 0], [1, 1]], dtype=float)  # < 3 verts
    assert not points_in_polygon(poly, np.array([[0.5, 0.5]])).any()


@pytest.mark.parametrize("ext", [".ply", ".las"])
def test_io_roundtrip(tmp_path, ext):
    c = make_cloud()
    c.attributes["intensity"] = np.arange(12, dtype=np.uint16)
    path = str(tmp_path / f"cloud{ext}")
    io.save(c, path)
    loaded = io.load(path)
    assert loaded.n_points == 12
    np.testing.assert_array_equal(loaded.labels, c.labels)
    np.testing.assert_allclose(loaded.coords, c.coords, atol=1e-3)
    assert "intensity" in loaded.attributes
