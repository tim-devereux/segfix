import numpy as np

from segfix import analysis, io, operations as ops
from segfix.model import PointCloud, UNASSIGNED
from segfix.treecatalog import TreeCatalog
from tests.test_rgb import _write_raycloud_ply

# Three trees along x: A and B are close (gap 0.6m, within reach); C is far.
# A few unassigned (black) points sit between A and B.
_A = np.column_stack([np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_B = np.column_stack([1.0 + np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_C = np.column_stack([10.0 + np.arange(5) * 0.1, np.zeros(5), np.zeros(5)])
_UNASSIGNED_PTS = np.column_stack([[0.6, 0.7, 0.8], [0, 0, 0], [0, 0, 0]])

_COLOR_A, _COLOR_B, _COLOR_C = (200, 0, 0), (0, 200, 0), (0, 0, 200)


def _three_trees_cloud(tmp_path):
    xyz = np.vstack([_A, _B, _C, _UNASSIGNED_PTS])
    rgb = np.vstack([
        np.tile(_COLOR_A, (5, 1)),
        np.tile(_COLOR_B, (5, 1)),
        np.tile(_COLOR_C, (5, 1)),
        np.zeros((3, 3)),  # black = unassigned
    ]).astype(np.uint8)
    path = tmp_path / "three_trees.ply"
    _write_raycloud_ply(path, xyz, rgb)
    return path


def _label_for_color(catalog, color):
    return next(lab for lab, c in catalog.label_colors.items() if c == color)


def test_neighbours_matches_full_cloud_analysis(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_b = _label_for_color(catalog, _COLOR_B)
    label_c = _label_for_color(catalog, _COLOR_C)

    got = catalog.neighbours(label_a, reach=1.0)
    full_cloud = io.load(str(path))
    expected = analysis.neighbours_by_points(full_cloud, label_a, 1.0)

    assert got == expected
    assert label_b in got
    assert label_c not in got


def test_load_includes_neighbours_and_nearby_unassigned_not_far_tree(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_b = _label_for_color(catalog, _COLOR_B)
    label_c = _label_for_color(catalog, _COLOR_C)

    neighbours = catalog.neighbours(label_a, reach=1.0)
    cloud, global_idx = catalog.load([label_a, *neighbours], margin=1.0)

    # A (5) + B (5) + the 3 unassigned points between them; not C.
    assert cloud.n_points == 13
    assert (cloud.labels == UNASSIGNED).sum() == 3
    assert set(cloud.labels[cloud.labels != UNASSIGNED].tolist()) == {label_a, label_b}
    assert label_c not in cloud.labels


def test_apply_round_trips_edits_into_master(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_b = _label_for_color(catalog, _COLOR_B)

    cloud, global_idx = catalog.load([label_a, label_b], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == label_b), label_a)
    catalog.apply(cloud, global_idx)

    # Master state reflects the merge: B is gone, A grew.
    assert label_b not in catalog.records
    assert catalog.records[label_a].count == 10  # 5 (A) + 5 (former B)
    assert (catalog.labels[global_idx] == cloud.labels).all()


def test_save_only_touches_changed_points(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_b = _label_for_color(catalog, _COLOR_B)

    cloud, global_idx = catalog.load([label_a, label_b], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == label_b), label_a)
    catalog.apply(cloud, global_idx)
    msg = catalog.save()

    assert msg.startswith("Saved 5")  # exactly the 5 ex-B points changed colour
    reloaded = io.load(str(path))
    # Label ids are reassigned fresh on reload, so identify trees by colour:
    # B merged into A, so only two colours remain — A's (survivor) and C's
    # (untouched, since only points inside A+B's margin were written).
    assert set(reloaded.label_colors.values()) == {_COLOR_A, _COLOR_C}
    assert len(reloaded.tree_ids) == 2
    assert (reloaded.labels == UNASSIGNED).sum() == 3

    # A second save with nothing new to write is a no-op.
    assert catalog.save() == "Nothing changed since last save"


def test_next_free_id_avoids_unloaded_tree_collision(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_c = _label_for_color(catalog, _COLOR_C)  # not loaded below

    # Only load A (no neighbours) and split part of it into a new tree.
    cloud, global_idx = catalog.load([label_a], margin=0.0)
    new_id = cloud.next_free_id()
    assert new_id != label_c
    assert new_id not in catalog.records

    ops.create_new(cloud, np.flatnonzero(cloud.labels == label_a)[:2])
    catalog.apply(cloud, global_idx)

    # A further split must still not collide with C, which was never loaded.
    another_id = cloud.next_free_id()
    assert another_id != label_c
    assert another_id not in catalog.records


def test_save_as_new_path_copies_whole_file_and_keeps_pending_edits(tmp_path):
    path = _three_trees_cloud(tmp_path)
    catalog = TreeCatalog(str(path))
    label_a = _label_for_color(catalog, _COLOR_A)
    label_b = _label_for_color(catalog, _COLOR_B)

    # Save As with zero edits still produces a full copy at the new path —
    # not "nothing changed", which would otherwise skip the copy entirely.
    copy_path = tmp_path / "copy.ply"
    msg = catalog.save(output=str(copy_path))
    assert copy_path.exists()
    assert io.load(str(copy_path)).n_points == catalog.count
    assert "no changes" in msg

    # Make an edit, Save As to a second new path...
    cloud, global_idx = catalog.load([label_a, label_b], margin=1.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == label_b), label_a)
    catalog.apply(cloud, global_idx)
    copy2_path = tmp_path / "copy2.ply"
    msg2 = catalog.save(output=str(copy2_path))
    assert msg2.startswith("Saved 5")
    assert set(io.load(str(copy2_path)).label_colors.values()) == {
        _COLOR_A, _COLOR_C,
    }

    # ...the original file must be untouched by that Save As...
    assert set(io.load(str(path)).label_colors.values()) == {
        _COLOR_A, _COLOR_B, _COLOR_C,
    }
    # ...and an in-place Save afterwards must still see the edit as pending,
    # not silently skip it because Save As already "used up" the diff.
    msg3 = catalog.save()
    assert msg3.startswith("Saved 5")
    assert set(io.load(str(path)).label_colors.values()) == {
        _COLOR_A, _COLOR_C,
    }
