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


def test_last_changed_tracks_touched_indices():
    c = make_cloud()
    assert c.last_changed is None  # nothing edited yet

    ops.reassign(c, [0, 1, 2], 2)
    assert sorted(c.last_changed.tolist()) == [0, 1, 2]

    ops.reassign(c, [3], 3)
    assert c.last_changed.tolist() == [3]

    c.undo()  # undoes the [3] edit
    assert c.last_changed.tolist() == [3]
    c.redo()
    assert c.last_changed.tolist() == [3]

    # A no-op (already tree 2) and an exhausted undo both report "nothing".
    ops.reassign(c, [0], 2)
    assert c.last_changed.size == 0
    for _ in range(10):
        c.undo()
    assert c.last_changed.size == 0


def test_refresh_view_incremental_matches_full_recompute():
    """The per-edit fast path (recolour only `changed` rows) must land on the
    exact same face_color array as a whole-cloud recompute."""
    from segfix.viewer import colors_for_labels, refresh_view

    class _StubLayer:
        def __init__(self, n):
            self.face_color = np.zeros((n, 4), np.float32)
            self.features = None
            self._refreshed = 0

        def refresh(self):
            self._refreshed += 1

    rng = np.random.RandomState(0)
    coords = rng.rand(400, 3).astype(np.float32)
    labels = rng.randint(1, 9, size=400).astype(np.int32)
    cloud = PointCloud(coords=coords, labels=labels)

    layer = _StubLayer(cloud.n_points)
    refresh_view(layer, cloud)  # initial full paint
    np.testing.assert_array_equal(
        layer.face_color, colors_for_labels(cloud.labels)
    )

    # Each edit refreshes incrementally (as _after_edit does per op); the
    # running array must stay identical to a from-scratch recompute.
    for _ in range(5):
        ops.reassign(
            cloud, rng.choice(400, 40, replace=False), int(rng.randint(1, 9))
        )
        refresh_view(layer, cloud, changed=cloud.last_changed)
        np.testing.assert_array_equal(
            layer.face_color, colors_for_labels(cloud.labels)
        )
    ops.create_new(cloud, rng.choice(400, 15, replace=False))  # brand-new id
    refresh_view(layer, cloud, changed=cloud.last_changed)
    np.testing.assert_array_equal(
        layer.face_color, colors_for_labels(cloud.labels)
    )
    cloud.undo()
    refresh_view(layer, cloud, changed=cloud.last_changed)
    np.testing.assert_array_equal(
        layer.face_color, colors_for_labels(cloud.labels)
    )

    # An empty `changed` (no-op edit) leaves the layer untouched.
    before = layer.face_color.copy()
    hits = layer._refreshed
    refresh_view(layer, cloud, changed=np.empty(0, np.int64))
    np.testing.assert_array_equal(layer.face_color, before)
    assert layer._refreshed == hits


def test_colors_for_labels_fades_only_requested_trees():
    from segfix.viewer import FADED_ALPHA, colors_for_labels

    labels = np.array([UNASSIGNED, NOISE, 1, 1, 2, 3])
    rgba = colors_for_labels(labels, faded={1, 3})

    assert list(rgba[:, 3]) == pytest.approx(
        [1.0, 1.0, FADED_ALPHA, FADED_ALPHA, 1.0, FADED_ALPHA]
    )
    # colour (RGB) is unchanged by fading — only alpha moves.
    assert (colors_for_labels(labels)[:, :3] == rgba[:, :3]).all()
    # no faded set (or an empty one) => everything opaque.
    assert (colors_for_labels(labels, faded=set())[:, 3] == 1.0).all()
    assert (colors_for_labels(labels)[:, 3] == 1.0).all()


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
def _write_arbor_las(path, coords, tree_id, *, fmt=7, extra_unsigned=False,
                      offsets=(0.0, 0.0, 0.0)):
    """A minimal arbor-shaped LAS: XYZ + an int ``treeID`` Extra-Bytes column.

    ``offsets`` matters for large (e.g. UTM-referenced) ``coords``: LAS
    stores XYZ as ``(value - offset) / scale`` in a 32-bit int, so without a
    matching offset a large coordinate simply doesn't fit.
    """
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=fmt)
    header.offsets = list(offsets)
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


def test_las_catalog_signed_treeid_keeps_noise(tmp_path):
    """A signed treeID column can hold NOISE (-1); it must round-trip so a
    reopened project still shows those points as dismissed."""
    import laspy
    from segfix.treecatalog import open_catalog

    coords = np.random.RandomState(7).rand(40, 3).astype(np.float64) * 10
    path = tmp_path / "signed.las"
    _write_arbor_las(path, coords, np.repeat([1, 2, 3, 4], 10))

    cat = open_catalog(str(path))
    assert cat._label_is_unsigned is False
    cloud, gidx = cat.load([2], margin=0.0)
    cloud.set_labels(np.flatnonzero(cloud.labels == 2), NOISE, "dismiss 2")
    cat.apply(cloud, gidx)
    cat.save()

    back = np.asarray(laspy.read(str(path)).treeID)
    assert (back == NOISE).sum() == 10
    assert sorted(np.unique(back).tolist()) == [NOISE, 1, 3, 4]


def test_las_catalog_unsigned_treeid_writes_noise_as_unassigned(tmp_path):
    """An unsigned treeID column can't store -1 — NOISE points are written
    back as UNASSIGNED (0) rather than wrapping to a huge value."""
    import laspy
    from segfix.treecatalog import open_catalog

    coords = np.random.RandomState(8).rand(40, 3).astype(np.float64) * 10
    path = tmp_path / "unsigned.las"
    _write_arbor_las(path, coords, np.repeat([1, 2, 3, 4], 10),
                     extra_unsigned=True)

    cat = open_catalog(str(path))
    assert cat._label_is_unsigned is True
    cloud, gidx = cat.load([2], margin=0.0)
    cloud.set_labels(np.flatnonzero(cloud.labels == 2), NOISE, "dismiss 2")
    cat.apply(cloud, gidx)
    cat.save()

    back = np.asarray(laspy.read(str(path)).treeID)
    assert back.min() >= 0  # no unsigned wraparound
    assert (back == UNASSIGNED).sum() == 10
    assert sorted(np.unique(back).tolist()) == [0, 1, 3, 4]


def test_las_catalog_rejects_laz_path(tmp_path):
    """workspace decompresses to .las on import; a .laz handed straight to the
    catalog is refused rather than silently failing to memory-map."""
    from segfix.treecatalog import LasCatalog, open_catalog

    coords = np.random.RandomState(9).rand(12, 3).astype(np.float64) * 3
    las_path = tmp_path / "p.las"
    _write_arbor_las(las_path, coords, np.repeat([1, 2], 6))
    laz_path = tmp_path / "p.laz"
    import laspy

    laspy.read(str(las_path)).write(str(laz_path))

    with pytest.raises(ValueError, match="uncompressed"):
        LasCatalog(str(laz_path))
    with pytest.raises(ValueError, match="uncompressed"):
        open_catalog(str(laz_path))


def test_las_catalog_rejects_truncated_las(tmp_path):
    """If the point records don't fill the file the header claims, bail with
    an explanation instead of a later out-of-bounds memmap read."""
    from segfix.treecatalog import open_catalog

    coords = np.random.RandomState(10).rand(30, 3).astype(np.float64) * 5
    path = tmp_path / "trunc.las"
    _write_arbor_las(path, coords, np.repeat([1, 2, 3], 10))
    path.write_bytes(path.read_bytes()[:-40])  # drop a whole point record

    with pytest.raises(ValueError, match="line up"):
        open_catalog(str(path))


def test_las_save_as_reexports_laz_in_a_laz_project(tmp_path):
    """Save As inside a project imported from .laz drops a matching .laz next
    to the new .las target too, not just on in-place saves."""
    import json
    import laspy
    from segfix import workspace
    from segfix.treecatalog import open_catalog

    coords = np.random.RandomState(11).rand(30, 3).astype(np.float64) * 6
    src = tmp_path / "stand.laz"
    _write_arbor_las(src, coords, np.repeat([1, 2, 3], 10))
    dest = workspace.create_workspace(str(src), tmp_path / "proj")

    cat = open_catalog(str(dest))
    cloud, gidx = cat.load([1], margin=0.0)
    cloud.set_labels(np.flatnonzero(cloud.labels == 1), 2, "merge 1->2")
    cat.apply(cloud, gidx)

    target = tmp_path / "proj" / "corrected.las"
    msg = cat.save(output=str(target))
    assert msg.startswith("Saved")
    assert target.exists()
    sibling_laz = tmp_path / "proj" / "corrected.laz"
    assert sibling_laz.exists()
    assert 1 not in np.unique(laspy.read(str(sibling_laz)).treeID)

    # The in-place .las / .laz and the original .laz are all left alone.
    assert 1 in np.unique(laspy.read(str(dest)).treeID)
    assert 1 in np.unique(laspy.read(str(src)).treeID)
    manifest = json.loads(
        (tmp_path / "proj" / workspace.MANIFEST_NAME).read_text()
    )
    assert manifest["source"] == str(src.resolve())
