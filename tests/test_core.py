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


def test_load_rejects_unknown_extension(tmp_path):
    path = tmp_path / "cloud.txt"
    path.write_text("1 2 3")
    with pytest.raises(ValueError, match="binary PLY and LAS/LAZ"):
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


def test_save_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="binary PLY and LAS/LAZ"):
        io.save(make_cloud(), str(tmp_path / "out.xyz"))


# -- LAS / LAZ (arbor output) -----------------------------------------------
def _write_arbor_las(path, coords, tree_id, *, fmt=7, extra_unsigned=False):
    """A minimal arbor-shaped LAS: XYZ + an int ``treeID`` Extra-Bytes column."""
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=fmt)
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID",
        type=np.uint32 if extra_unsigned else np.int32,
        description="Unique ID per tree",
    ))
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    las.treeID = np.asarray(tree_id, dtype=np.int64)
    if fmt in (2, 3, 7, 8):
        las.red = np.arange(len(coords), dtype=np.uint16)
    las.write(str(path))


def test_las_catalog_patches_treeid_in_place(tmp_path):
    from segfix.treecatalog import LasCatalog, open_catalog

    coords = np.random.RandomState(1).rand(60, 3).astype(np.float64) * 20
    tree_id = np.repeat([0, 1, 2, 3, 4, 5], 10)
    path = tmp_path / "plot_segmented.las"
    _write_arbor_las(path, coords, tree_id)
    before = path.read_bytes()

    cat = open_catalog(str(path))
    assert isinstance(cat, LasCatalog)
    assert sorted(cat.records) == [1, 2, 3, 4, 5]

    cloud, gidx = cat.load([2], margin=0.0)
    cloud.set_labels(np.flatnonzero(cloud.labels == 2), 3, "merge 2->3")
    cat.apply(cloud, gidx)
    msg = cat.save()
    assert "Saved" in msg

    import laspy

    reread = laspy.read(str(path))
    np.testing.assert_array_equal(
        np.sort(np.unique(reread.treeID)), [0, 1, 3, 4, 5]
    )
    # Only the treeID bytes moved: strip that column and the rest is identical.
    after = path.read_bytes()
    assert len(after) == len(before)
    a0 = np.frombuffer(before, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    a1 = np.frombuffer(after, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    for name in cat.dtype.names:
        if name == "treeID":
            continue
        np.testing.assert_array_equal(a0[name], a1[name])


def test_las_catalog_requires_a_label_field(tmp_path):
    import laspy
    from segfix.treecatalog import open_catalog

    header = laspy.LasHeader(version="1.4", point_format=6)
    las = laspy.LasData(header)
    las.x = las.y = las.z = np.zeros(5)
    path = tmp_path / "bare.las"
    las.write(str(path))
    with pytest.raises(ValueError, match="treeID"):
        open_catalog(str(path))


def test_open_catalog_dispatches_by_extension(tmp_path):
    from segfix.treecatalog import LasCatalog, TreeCatalog, open_catalog

    c = make_cloud()
    ply = tmp_path / "c.ply"
    io.save(c, str(ply))
    assert isinstance(open_catalog(str(ply)), TreeCatalog)

    las = tmp_path / "c.las"
    _write_arbor_las(las, c.coords.astype(np.float64), c.labels)
    assert isinstance(open_catalog(str(las)), LasCatalog)

    with pytest.raises(ValueError):
        open_catalog(str(tmp_path / "c.e57"))


def test_las_io_roundtrip_via_source_clone(tmp_path):
    coords = np.random.RandomState(2).rand(20, 3).astype(np.float64) * 5
    src = tmp_path / "src.las"
    _write_arbor_las(src, coords, np.repeat([1, 2], 10))

    cloud = io.load(str(src))
    assert cloud.source_format == "las"
    cloud.set_labels(np.arange(10), 9, "relabel first tree")

    out = tmp_path / "out.laz"
    io.save(cloud, str(out))

    import laspy

    back = laspy.read(str(out))
    np.testing.assert_array_equal(
        np.sort(np.unique(back.treeID)), [2, 9]
    )


def test_workspace_decompresses_laz_on_import(tmp_path):
    import json
    import laspy
    from segfix import workspace

    coords = np.random.RandomState(3).rand(15, 3).astype(np.float64) * 4
    src = tmp_path / "plot.laz"
    _write_arbor_las(src, coords, np.repeat([1, 2, 3], 5))

    dest = workspace.create_workspace(str(src), tmp_path / "proj")
    assert dest.suffix == ".las" and dest.exists()
    assert src.exists()  # original .laz untouched

    manifest = json.loads((tmp_path / "proj" / workspace.MANIFEST_NAME).read_text())
    assert manifest["data_file"] == dest.name
    assert manifest["source"].endswith(".laz")

    # Saving the working copy re-exports a sibling .laz for arbor to re-read.
    from segfix.treecatalog import open_catalog

    cat = open_catalog(str(dest))
    cloud, gidx = cat.load([1], margin=0.0)
    cloud.set_labels(np.flatnonzero(cloud.labels == 1), 2, "merge")
    cat.apply(cloud, gidx)
    cat.save()
    exported = tmp_path / "proj" / "plot.laz"
    assert exported.exists()
    assert 1 not in np.unique(laspy.read(str(exported)).treeID)
