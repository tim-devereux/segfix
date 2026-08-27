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
    # Compare values, not just presence: a LAS point format defines
    # "intensity" whether or not anything was ever written into it, so
    # `"intensity" in loaded.attributes` passes even on a save that dropped it.
    np.testing.assert_array_equal(
        loaded.attributes["intensity"], c.attributes["intensity"]
    )


# Minimal projected CRS, carried in a WKT VLR (point formats 6+).
_WKT = (
    'PROJCS["GDA94 / MGA zone 56",GEOGCS["GDA94",DATUM["GDA94",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
    'UNIT["metre",1],AUTHORITY["EPSG","28356"]]'
)


def test_las_save_preserves_crs_scale_offset_and_attributes(tmp_path):
    """Fixing labels must not cost the file its georeferencing or its other
    per-point columns.

    Regression: the save header used to be rebuilt from only the source's
    point format and version, which dropped the CRS VLR (and the global
    encoding bit that says how to read it) and left every dimension except
    XYZ and the label field written out as zeros.
    """
    import laspy

    n = 20
    intensity = np.arange(n, dtype=np.uint16) + 7
    gps_time = np.linspace(1000.0, 1060.0, n)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([100.0, 200.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.vlrs.append(laspy.vlrs.known.WktCoordinateSystemVlr(_WKT))
    header.global_encoding.wkt = True
    las = laspy.LasData(header)
    las.x = np.linspace(100, 110, n)
    las.y = np.linspace(200, 210, n)
    las.z = np.linspace(0, 5, n)
    las.intensity = intensity
    las.classification = np.full(n, 5, dtype=np.uint8)
    las.gps_time = gps_time
    las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las.treeID = np.repeat([1, 2], n // 2)
    src = str(tmp_path / "src.las")
    las.write(src)

    cloud = io.load(src)
    ops.reassign(cloud, [0, 1, 2], 2)
    out = str(tmp_path / "out.las")
    io.save(cloud, out)

    rt = laspy.read(out)
    wkt_vlrs = [
        v for v in rt.header.vlrs
        if isinstance(v, laspy.vlrs.known.WktCoordinateSystemVlr)
    ]
    assert [v.string for v in wkt_vlrs] == [_WKT]
    assert rt.header.global_encoding.wkt
    np.testing.assert_array_equal(rt.header.scales, header.scales)
    np.testing.assert_array_equal(rt.header.offsets, header.offsets)
    assert rt.header.point_count == n

    np.testing.assert_array_equal(rt.intensity, intensity)
    np.testing.assert_array_equal(rt.classification, np.full(n, 5))
    np.testing.assert_allclose(rt.gps_time, gps_time)
    # The edit landed, and only where it was asked for.
    np.testing.assert_array_equal(rt.treeID, cloud.labels)
    assert list(rt.treeID[:4]) == [2, 2, 2, 1]


def test_las_save_carries_non_las_attributes_it_can_represent(tmp_path):
    """A PLY-sourced cloud written to LAS puts its extra columns in extra
    dimensions, skipping only those LAS extra bytes cannot express."""
    import laspy

    c = make_cloud()
    c.attributes["time"] = np.linspace(0, 1, 12)  # f8 -> extra dim
    c.attributes["flag"] = np.ones(12, dtype=bool)  # no LAS equivalent
    c.attributes["x" * 40] = np.arange(12, dtype=np.int32)  # name too long

    path = str(tmp_path / "fromply.las")
    io.save(c, path)

    rt = laspy.read(path)
    names = set(rt.point_format.dimension_names)
    assert "time" in names
    assert "flag" not in names and "x" * 40 not in names
    np.testing.assert_allclose(rt.time, c.attributes["time"])
    np.testing.assert_array_equal(rt.treeID, c.labels)
