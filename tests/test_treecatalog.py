"""Catalog scenarios run against both backends.

The tree-at-a-time index, neighbour search, subset load/apply and the
diff-based save live in ``treecatalog._BaseCatalog``; ``TreeCatalog`` (binary
PLY, tree = colour) and ``LasCatalog`` (uncompressed LAS, tree = ``treeID``)
only fill in the file-format specifics. The ``three_trees`` fixture is
parametrised over the two so every scenario below exercises both.
"""

import os

import numpy as np
import pytest

from segfix import analysis, io, operations as ops
from segfix.model import UNASSIGNED
from segfix.treecatalog import open_catalog
from tests.test_core import _write_arbor_las
from tests.test_rgb import _write_raycloud_ply

# Three trees along x: A and B are close (gap 0.6m, within reach); C is far.
# A few unassigned points sit between A and B.
_A = np.column_stack([np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_B = np.column_stack([1.0 + np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_C = np.column_stack([10.0 + np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_UNASSIGNED_PTS = np.column_stack([[0.6, 0.7, 0.8], [0, 0, 0], [0, 0, 0]])

_COLOR_A, _COLOR_B, _COLOR_C = (200, 0, 0), (0, 200, 0), (0, 0, 200)


@pytest.fixture(params=["ply", "las"])
def three_trees(tmp_path, request):
    """(path, ids) for a 3-tree plot in the requested format, where ``ids``
    maps ``"A"``/``"B"``/``"C"`` to that backend's tree id."""
    xyz = np.vstack([_A, _B, _C, _UNASSIGNED_PTS])
    if request.param == "ply":
        rgb = np.vstack([
            np.tile(_COLOR_A, (5, 1)),
            np.tile(_COLOR_B, (5, 1)),
            np.tile(_COLOR_C, (5, 1)),
            np.zeros((3, 3)),  # black = unassigned
        ]).astype(np.uint8)
        path = tmp_path / "three_trees.ply"
        _write_raycloud_ply(path, xyz, rgb)
        # RGB-segmented labels are numbered by the loader, not by us — resolve
        # each tree's id from the colour it was written with.
        colours = open_catalog(str(path)).label_colors
        ids = {
            name: next(lab for lab, c in colours.items() if c == col)
            for name, col in (("A", _COLOR_A), ("B", _COLOR_B), ("C", _COLOR_C))
        }
        return str(path), ids
    tid = np.repeat([1, 2, 3, 0], [5, 5, 5, 3])
    path = tmp_path / "three_trees.las"
    _write_arbor_las(path, xyz, tid)
    return str(path), {"A": 1, "B": 2, "C": 3}


def test_build_index_records_match_hand_computed_stats(three_trees):
    """Locks the vectorised (reduceat) per-tree bbox/centroid/count path."""
    path, ids = three_trees
    catalog = open_catalog(path)

    for name, geom in (("A", _A), ("B", _B), ("C", _C)):
        rec = catalog.records[ids[name]]
        lo, hi = geom.min(axis=0), geom.max(axis=0)
        assert rec.count == len(geom)
        np.testing.assert_allclose(rec.bbox[0], lo, atol=1e-5)
        np.testing.assert_allclose(rec.bbox[1], hi, atol=1e-5)
        np.testing.assert_allclose(
            rec.centroid, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), atol=1e-5
        )
    # Unassigned / noise never get a record.
    assert 0 not in catalog.records and -1 not in catalog.records


def test_neighbours_matches_full_cloud_analysis(three_trees):
    path, ids = three_trees
    catalog = open_catalog(path)

    got = catalog.neighbours(ids["A"], reach=1.0)
    full_cloud = io.load(path)
    expected = analysis.neighbours_by_points(full_cloud, ids["A"], 1.0)

    assert got == expected
    assert ids["B"] in got
    assert ids["C"] not in got


def test_load_includes_neighbours_and_nearby_unassigned_not_far_tree(three_trees):
    path, ids = three_trees
    catalog = open_catalog(path)

    neighbours = catalog.neighbours(ids["A"], reach=1.0)
    cloud, _ = catalog.load([ids["A"], *neighbours], margin=1.0)

    # A (5) + B (5) + the 3 unassigned points between them; not C.
    assert cloud.n_points == 13
    assert (cloud.labels == UNASSIGNED).sum() == 3
    assert set(cloud.labels[cloud.labels != UNASSIGNED].tolist()) == {
        ids["A"], ids["B"],
    }
    assert ids["C"] not in cloud.labels


def test_apply_round_trips_edits_into_master(three_trees):
    path, ids = three_trees
    catalog = open_catalog(path)

    cloud, global_idx = catalog.load([ids["A"], ids["B"]], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == ids["B"]), ids["A"])
    catalog.apply(cloud, global_idx)

    # Master state reflects the merge: B is gone, A grew.
    assert ids["B"] not in catalog.records
    assert catalog.records[ids["A"]].count == 10  # 5 (A) + 5 (former B)
    assert (catalog.labels[global_idx] == cloud.labels).all()


def test_save_only_touches_changed_points(three_trees):
    path, ids = three_trees
    catalog = open_catalog(path)

    cloud, global_idx = catalog.load([ids["A"], ids["B"]], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == ids["B"]), ids["A"])
    catalog.apply(cloud, global_idx)
    msg = catalog.save()

    assert msg.startswith("Saved 5")  # exactly the 5 ex-B points

    reopened = open_catalog(path)
    # B merged into A: two trees left (A's survivor + untouched C), and the
    # three unassigned points are unchanged.
    assert len(reopened.records) == 2
    assert (reopened.labels == UNASSIGNED).sum() == 3

    # A second save with nothing new to write is a no-op.
    assert catalog.save() == "Nothing changed since last save"


def test_next_free_id_avoids_unloaded_tree_collision(three_trees):
    path, ids = three_trees
    catalog = open_catalog(path)

    # Only load A (no neighbours) and split part of it into a new tree.
    cloud, global_idx = catalog.load([ids["A"]], margin=0.0)
    new_id = cloud.next_free_id()
    assert new_id != ids["C"]
    assert new_id not in catalog.records

    ops.create_new(cloud, np.flatnonzero(cloud.labels == ids["A"])[:2])
    catalog.apply(cloud, global_idx)

    # A further split must still not collide with C, which was never loaded.
    another_id = cloud.next_free_id()
    assert another_id != ids["C"]
    assert another_id not in catalog.records


def test_save_as_new_path_copies_whole_file_and_keeps_pending_edits(
    three_trees, tmp_path
):
    path, ids = three_trees
    ext = os.path.splitext(path)[1]
    catalog = open_catalog(path)

    # Save As with zero edits still produces a full copy at the new path —
    # not "nothing changed", which would otherwise skip the copy entirely.
    copy_path = tmp_path / f"copy{ext}"
    msg = catalog.save(output=str(copy_path))
    assert copy_path.exists()
    assert open_catalog(str(copy_path)).count == catalog.count
    assert "no changes" in msg

    # Make an edit, Save As to a second new path...
    cloud, global_idx = catalog.load([ids["A"], ids["B"]], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == ids["B"]), ids["A"])
    catalog.apply(cloud, global_idx)
    copy2_path = tmp_path / f"copy2{ext}"
    msg2 = catalog.save(output=str(copy2_path))
    assert msg2.startswith("Saved 5")
    assert len(open_catalog(str(copy2_path)).records) == 2

    # ...the original file must be untouched by that Save As...
    assert len(open_catalog(path).records) == 3
    # ...and an in-place Save afterwards must still see the edit as pending,
    # not silently skip it because Save As already "used up" the diff.
    msg3 = catalog.save()
    assert msg3.startswith("Saved 5")
    assert len(open_catalog(path).records) == 2


def test_out_of_range_and_negative_labels_fold_into_unassigned(tmp_path):
    """A pipeline "no tree" sentinel (INT32_MIN, from the CHERLET LAS) and a
    plain negative id must not surface as trees — they load as UNASSIGNED,
    while NOISE (-1) is left alone."""
    coords = np.random.RandomState(0).rand(60, 3).astype(np.float64) * 10
    tid = np.repeat([1, 2, -(2**31), -7, -1, 0], 10)  # 6 groups of 10
    path = tmp_path / "sentinels.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path))
    assert sorted(cat.records) == [1, 2]          # only the real trees
    folded = cat.labels[np.isin(tid, [-(2**31), -7])]
    assert (folded == UNASSIGNED).all()
    assert (cat.labels[tid == -1] == -1).all()    # NOISE preserved
    assert (cat.labels[tid == 0] == UNASSIGNED).all()


def test_unsigned_label_column_out_of_range_folds_into_unassigned(tmp_path):
    """A uint32 treeID column whose "unassigned" sentinel is 2**32-1 (or any
    value past int32 max) must not wrap to a bogus negative tree."""
    coords = np.random.RandomState(1).rand(40, 3).astype(np.float64) * 10
    tid = np.repeat([5, 6, 2**32 - 1, 2**31 + 3], 10)
    path = tmp_path / "unsigned.las"
    _write_arbor_las(path, coords, tid, extra_unsigned=True)

    cat = open_catalog(str(path))
    assert sorted(cat.records) == [5, 6]
    assert (cat.labels[tid >= 2**31] == UNASSIGNED).all()


# -- global shift ("large coordinates") --------------------------------------
def test_needs_global_shift_thresholds():
    from segfix.treecatalog import GLOBAL_SHIFT_THRESHOLD, needs_global_shift

    small = np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 3.0])
    assert not needs_global_shift(*small)

    large = np.array([204277.0, 7223248.0, -0.4]), np.array([204303.0, 7223264.0, 14.0])
    assert needs_global_shift(*large)

    # exactly at the threshold is fine; just past it needs a shift
    at = np.array([-GLOBAL_SHIFT_THRESHOLD]), np.array([GLOBAL_SHIFT_THRESHOLD])
    assert not needs_global_shift(*at)
    past = np.array([-GLOBAL_SHIFT_THRESHOLD - 1]), np.array([GLOBAL_SHIFT_THRESHOLD])
    assert needs_global_shift(*past)


def test_suggest_global_shift_moves_min_corner_to_the_origin():
    from segfix.treecatalog import suggest_global_shift

    mins = np.array([204277.15, 7223248.32, -0.37])
    maxs = np.array([204302.76, 7223264.30, 14.04])
    shift = suggest_global_shift(mins, maxs)
    shifted_min = mins + shift
    assert np.all(shifted_min >= 0) and np.all(shifted_min < 1)  # floor of mins


def test_catalog_defaults_to_unshifted_without_a_prompt(tmp_path):
    """Backwards compatible: open_catalog with no shift_prompt never shifts,
    however large the coordinates — every existing caller keeps working."""
    big = np.array([204300.0, 7223250.0, 5.0]) + np.random.RandomState(2).rand(30, 3) * 5
    tid = np.repeat([1, 2, 0], 10)
    path = tmp_path / "big.las"
    _write_arbor_las(path, big, tid, offsets=(204000.0, 7223000.0, 0.0))

    cat = open_catalog(str(path))
    assert cat.global_shift is None
    # Unshifted at this magnitude is genuinely imprecise -- that's the whole
    # reason the feature exists -- so this only checks it's in the right
    # ballpark, not tight to the metre.
    np.testing.assert_allclose(cat.coords[:, :2].mean(axis=0), big[:, :2].mean(axis=0),
                                atol=5.0)


def test_catalog_applies_a_shift_when_prompt_accepts_it(tmp_path):
    big = np.array([204300.0, 7223250.0, 5.0]) + np.random.RandomState(3).rand(30, 3) * 5
    tid = np.repeat([1, 2, 0], 10)
    path = tmp_path / "big.las"
    _write_arbor_las(path, big, tid, offsets=(204000.0, 7223000.0, 0.0))

    seen = {}

    def prompt(mins, maxs, suggested):
        seen["mins"], seen["maxs"], seen["suggested"] = mins, maxs, suggested
        return tuple(suggested)  # accept the suggestion

    cat = open_catalog(str(path), shift_prompt=prompt)
    assert seen["mins"] is not None  # prompt was actually consulted
    assert cat.global_shift is not None
    # shifted coordinates land near the origin, at full precision
    assert np.abs(cat.coords).max() < 100
    # and match the original points once the shift is undone
    unshifted = cat.coords.astype(np.float64) - np.asarray(cat.global_shift)
    np.testing.assert_allclose(np.sort(unshifted[:, 0]), np.sort(big[:, 0]), atol=0.01)

    # a per-tree load() carries the same shift and stays consistent with the
    # resident (whole-catalog) coordinates
    cloud, gidx = cat.load([1], margin=1.0)
    assert cloud.global_shift == tuple(cat.global_shift.tolist())
    np.testing.assert_allclose(cloud.coords, cat.coords[gidx], atol=1e-3)


def test_catalog_prompt_declining_leaves_coordinates_unshifted(tmp_path):
    big = np.array([204300.0, 7223250.0, 5.0]) + np.random.RandomState(4).rand(20, 3) * 5
    tid = np.repeat([1, 0], 10)
    path = tmp_path / "big.las"
    _write_arbor_las(path, big, tid, offsets=(204000.0, 7223000.0, 0.0))

    cat = open_catalog(str(path), shift_prompt=lambda mins, maxs, suggested: None)
    assert cat.global_shift is None


def test_shift_never_touches_saved_coordinate_bytes(tmp_path):
    """The global shift is display/analysis-only: save() must still only
    patch the label column, never coordinates -- on disk or off."""
    big = np.array([204300.0, 7223250.0, 5.0]) + np.random.RandomState(5).rand(30, 3) * 5
    tid = np.repeat([1, 2, 0], 10)
    path = tmp_path / "big.las"
    _write_arbor_las(path, big, tid, offsets=(204000.0, 7223000.0, 0.0))
    before = path.read_bytes()

    cat = open_catalog(
        str(path), shift_prompt=lambda mins, maxs, suggested: tuple(suggested)
    )
    cloud, gidx = cat.load([2], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == 2), 1)
    cat.apply(cloud, gidx)
    cat.save()

    after = path.read_bytes()
    assert len(after) == len(before)
    a0 = np.frombuffer(before, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    a1 = np.frombuffer(after, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    for name in cat.dtype.names:
        if name == "treeID":
            continue
        np.testing.assert_array_equal(a0[name], a1[name])


# -- wrong-format guard ("app freezes") --------------------------------------
def test_sniff_format_identifies_real_files_and_rejects_garbage(tmp_path):
    ply = tmp_path / "a.ply"
    _write_raycloud_ply(
        ply, np.zeros((3, 3)), np.zeros((3, 3), dtype=np.uint8)
    )
    assert io.sniff_format(str(ply)) == "ply"

    las = tmp_path / "a.las"
    _write_arbor_las(las, np.zeros((10, 3)), np.zeros(10, dtype=int))
    assert io.sniff_format(str(las)) == "las"

    garbage = tmp_path / "bogus.ply"
    garbage.write_bytes(os.urandom(4096))
    assert io.sniff_format(str(garbage)) is None


def test_wrong_extension_content_rejected_fast_not_scanned(tmp_path):
    """A .ply-named file that is not actually PLY must fail immediately from
    the magic-byte check, not fall into _parse_ply_header's line-by-line scan
    of the (here, large) binary content -- the freeze this guards against."""
    import time

    fake = tmp_path / "fake.ply"
    fake.write_bytes(os.urandom(20_000_000))  # no 'ply' magic, no newlines guaranteed

    start = time.monotonic()
    with pytest.raises(ValueError, match="doesn't look like"):
        open_catalog(str(fake))
    assert time.monotonic() - start < 1.0

    with pytest.raises(ValueError, match="doesn't look like"):
        io.load(str(fake))
