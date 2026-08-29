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
