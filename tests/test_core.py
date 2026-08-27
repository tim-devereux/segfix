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


def test_mark_noise_and_unassign():
    c = make_cloud()
    ops.mark_noise(c, [0])
    assert c.labels[0] == NOISE
    ops.unassign(c, [1])
    assert c.labels[1] == UNASSIGNED
    # noise/unassigned excluded from tree_ids
    assert NOISE not in c.tree_ids and UNASSIGNED not in c.tree_ids


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


def test_ply_roundtrip_keeps_labels_coords_and_attributes(tmp_path):
    c = make_cloud()
    c.attributes["intensity"] = np.arange(12, dtype=np.uint16)
    path = str(tmp_path / "cloud.ply")
    io.save(c, path)
    loaded = io.load(path)
    assert loaded.n_points == 12
    np.testing.assert_array_equal(loaded.labels, c.labels)
    np.testing.assert_allclose(loaded.coords, c.coords, atol=1e-3)
    np.testing.assert_array_equal(
        loaded.attributes["intensity"], c.attributes["intensity"]
    )


@pytest.mark.parametrize("name,make", [
    ("cloud.las", lambda p: p.write_bytes(b"LASF")),
    ("cloud.laz", lambda p: p.write_bytes(b"LASF")),
    ("cloud.txt", lambda p: p.write_text("1 2 3")),
])
def test_load_rejects_non_ply(tmp_path, name, make):
    path = tmp_path / name
    make(path)
    with pytest.raises(ValueError, match="binary PLY only"):
        io.load(str(path))


def test_load_rejects_ascii_ply(tmp_path):
    """ASCII PLY has variable-length records, so treecatalog cannot seek to a
    tree without parsing the whole file — it is refused rather than silently
    loading down a path that no longer exists."""
    from plyfile import PlyData, PlyElement

    v = np.empty(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("treeID", "i4")])
    v["x"] = v["y"] = v["z"] = 0.0
    v["treeID"] = [1, 2, 3]
    path = tmp_path / "ascii.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=True).write(str(path))

    with pytest.raises(ValueError, match="ASCII PLY"):
        io.load(str(path))


def test_save_rejects_non_ply(tmp_path):
    with pytest.raises(ValueError, match="binary PLY only"):
        io.save(make_cloud(), str(tmp_path / "out.las"))
