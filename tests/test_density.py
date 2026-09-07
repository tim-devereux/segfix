"""Dense-cloud check, voxel decimation, and the save-time interpolation back
onto every full-resolution point.

The measurement and decimation live in :mod:`segfix.density`; the catalog
wires them into a session (``density_prompt``) exactly like the global-shift
prompt, and ``save()`` is where a decimated session has to put its edits back
onto the points it never loaded.
"""

import numpy as np
import pytest

from segfix import density, operations as ops
from segfix.model import UNASSIGNED
from segfix.treecatalog import open_catalog
from tests.test_core import _write_arbor_las
from tests.test_rgb import _write_raycloud_ply


def _grid(spacing, nx, ny, nz=1, origin=(0.0, 0.0, 0.0)):
    """A regular lattice with a known nearest-neighbour distance."""
    g = np.stack(np.meshgrid(
        np.arange(nx) * spacing + origin[0],
        np.arange(ny) * spacing + origin[1],
        np.arange(nz) * spacing + origin[2],
        indexing="ij",
    ), axis=-1)
    return g.reshape(-1, 3)


# -- measuring ---------------------------------------------------------------
def test_estimate_spacing_matches_a_known_lattice():
    coords = _grid(0.005, 60, 60, 5)  # 18k points, 5 mm apart
    assert density.estimate_spacing(coords) == pytest.approx(0.005, rel=0.05)


def test_estimate_spacing_survives_a_cloud_too_big_to_tree_whole():
    """Past the block target the estimate comes from a few small boxes, not
    a KD-tree over everything — it must still land on the real spacing."""
    coords = _grid(0.01, 160, 160, 4)  # 102k points, past _BLOCK_TARGET
    assert len(coords) > density._BLOCK_TARGET
    assert density.estimate_spacing(coords) == pytest.approx(0.01, rel=0.1)


def test_estimate_spacing_on_a_sparse_cloud_reads_above_the_threshold():
    coords = _grid(0.15, 20, 20, 2)
    assert density.estimate_spacing(coords) > density.DENSE_SPACING


def test_estimate_spacing_handles_degenerate_clouds():
    assert density.estimate_spacing(np.empty((0, 3))) == 0.0
    assert density.estimate_spacing(np.zeros((1, 3))) == 0.0


# -- decimating --------------------------------------------------------------
def test_voxel_indices_keeps_one_real_point_per_voxel():
    coords = _grid(0.005, 40, 40, 4)  # 5 mm points
    keep = density.voxel_indices(coords, 0.02)

    # One per occupied 2cm voxel: the 0.195 x 0.195 x 0.015 m lattice spans
    # 10 x 10 x 1 of them.
    assert keep.size == 10 * 10 * 1
    # Indices into the original array (real points, not centroids), unique
    # and in file order.
    assert keep.max() < len(coords)
    assert np.unique(keep).size == keep.size
    assert (np.diff(keep) > 0).all()
    # No two kept points share a voxel.
    cells = np.floor((coords[keep] - coords.min(axis=0)) / 0.02).astype(int)
    assert np.unique(cells, axis=0).shape[0] == keep.size


def test_voxel_indices_keeps_the_point_nearest_each_voxel_centre():
    """Not "whichever came first": corner-bunched representatives are what
    let a neighbouring voxel steal points on save (see the docstring)."""
    # One voxel's worth of points, the closest to its centre listed last.
    coords = np.array([
        [0.000, 0.000, 0.000],   # corner, first in file
        [0.019, 0.019, 0.019],   # far corner
        [0.011, 0.009, 0.010],   # nearest the centre (0.01, 0.01, 0.01)
    ])
    keep = density.voxel_indices(coords, 0.02)
    assert keep.tolist() == [2]


def test_voxel_indices_rejects_a_nonpositive_voxel():
    with pytest.raises(ValueError, match="must be positive"):
        density.voxel_indices(_grid(0.01, 4, 4), 0.0)


# -- catalog integration -----------------------------------------------------
def _dense_plot(tmp_path, name="dense.las"):
    """Two 5 mm-spaced trees, far enough apart not to be neighbours."""
    a = _grid(0.005, 30, 30, 6)
    b = _grid(0.005, 30, 30, 6, origin=(5.0, 0.0, 0.0))
    coords = np.vstack([a, b])
    tid = np.repeat([1, 2], [len(a), len(b)])
    path = tmp_path / name
    _write_arbor_las(path, coords, tid)
    return str(path), len(a), len(b)


def test_catalog_stays_full_resolution_without_a_prompt(tmp_path):
    """Backwards compatible: no density_prompt means no check and no
    decimation, however dense the cloud."""
    path, n_a, n_b = _dense_plot(tmp_path)

    cat = open_catalog(path)
    assert not cat.is_decimated
    assert cat.voxel_size is None and cat.spacing is None
    assert cat.working_count == cat.count == n_a + n_b


def test_catalog_offers_decimation_only_for_a_dense_cloud(tmp_path):
    sparse = _grid(0.1, 20, 20, 2)
    path = tmp_path / "sparse.las"
    _write_arbor_las(path, sparse, np.ones(len(sparse), dtype=int))

    asked = []
    cat = open_catalog(
        str(path),
        density_prompt=lambda *a: asked.append(a) or 0.02,
    )
    assert asked == []                      # never even offered
    assert not cat.is_decimated


def test_catalog_decimates_when_the_prompt_accepts(tmp_path):
    path, n_a, n_b = _dense_plot(tmp_path)
    seen = {}

    def prompt(spacing, n_points, suggested):
        seen.update(spacing=spacing, n_points=n_points, suggested=suggested)
        return suggested

    cat = open_catalog(path, density_prompt=prompt)

    assert seen["spacing"] == pytest.approx(0.005, rel=0.2)
    assert seen["n_points"] == n_a + n_b
    assert seen["suggested"] == density.DENSE_SPACING
    assert cat.is_decimated and cat.voxel_size == density.DENSE_SPACING
    # Far fewer points to draw and lasso, but both trees still there.
    assert cat.working_count < (n_a + n_b) / 4
    assert cat.count == n_a + n_b          # the file itself is untouched
    assert sorted(cat.records) == [1, 2]
    assert cat.coords.shape[0] == cat.labels.shape[0] == cat.working_count


def test_catalog_prompt_declining_keeps_every_point(tmp_path):
    path, n_a, n_b = _dense_plot(tmp_path)
    cat = open_catalog(path, density_prompt=lambda *a: None)
    assert not cat.is_decimated
    assert cat.working_count == n_a + n_b


def test_decimated_load_returns_coordinates_matching_the_working_set(tmp_path):
    path, _, _ = _dense_plot(tmp_path)
    cat = open_catalog(path, density_prompt=lambda *a: a[2])

    cloud, gidx = cat.load([1], margin=0.5)
    # load() has to map working-set positions through to file rows; if it
    # didn't, these coordinates would be some unrelated points'.
    np.testing.assert_allclose(cloud.coords, cat.coords[gidx], atol=1e-4)
    assert (cloud.labels == cat.labels[gidx]).all()


def test_decimated_save_interpolates_edits_onto_every_original_point(tmp_path):
    """The point of the whole feature: edit a thinned cloud, save, and the
    full-resolution file must come back re-labelled point for point."""
    path, n_a, n_b = _dense_plot(tmp_path)
    cat = open_catalog(path, density_prompt=lambda *a: a[2])
    assert cat.is_decimated

    # Merge tree 2 into tree 1 on the decimated cloud.
    cloud, gidx = cat.load([2], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == 2), 1)
    cat.apply(cloud, gidx)
    msg = cat.save()

    # Every one of tree 2's original points was written, not just the
    # decimated stand-ins.
    assert f"{n_b:,}" in msg
    reopened = open_catalog(path)          # full resolution
    assert reopened.count == n_a + n_b
    assert sorted(reopened.records) == [1]
    assert reopened.records[1].count == n_a + n_b


def test_decimated_save_leaves_untouched_trees_byte_identical(tmp_path):
    """Interpolation must not bleed into a tree the user never edited, and
    must not rewrite any other column."""
    path, n_a, n_b = _dense_plot(tmp_path)
    before = np.fromfile(path, dtype=np.uint8).copy()

    cat = open_catalog(path, density_prompt=lambda *a: a[2])
    cloud, gidx = cat.load([2], margin=0.0)
    ops.unassign(cloud, np.flatnonzero(cloud.labels == 2))
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(path)
    assert reopened.records[1].count == n_a       # tree 1 untouched
    assert 2 not in reopened.records
    # Coordinates (and every other column) are unchanged: same file length,
    # and the same bytes outside the treeID column.
    after = np.fromfile(path, dtype=np.uint8)
    assert after.size == before.size
    a0 = np.frombuffer(before, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    a1 = np.frombuffer(after, dtype=cat.dtype, offset=cat.offset, count=cat.count)
    for name in cat.dtype.names:
        if name == "treeID":
            continue
        np.testing.assert_array_equal(a0[name], a1[name])


def test_decimated_save_of_a_partial_edit_only_relabels_that_part(tmp_path):
    """A lasso over part of a tree must relabel the full-resolution points
    under that part — and only those."""
    coords = _grid(0.005, 40, 40, 4)
    tid = np.ones(len(coords), dtype=int)
    path = tmp_path / "one_tree.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a: a[2])
    cloud, gidx = cat.load([1], margin=0.0)
    # Split off the half of the tree with x below the midpoint.
    mid = float(np.median(cloud.coords[:, 0]))
    left = np.flatnonzero(cloud.coords[:, 0] < mid)
    ops.reassign(cloud, left, 7)
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(str(path))
    assert sorted(reopened.records) == [1, 7]
    assert reopened.records[1].count + reopened.records[7].count == len(coords)
    # The split lands on the geometry, not on some arbitrary subset: every
    # relabelled point sits on the left of the cut (within a voxel of it).
    moved = reopened.coords[reopened.labels == 7]
    assert moved[:, 0].max() < mid + cat.voxel_size
    assert (reopened.labels[reopened.coords[:, 0] > mid + cat.voxel_size] == 1).all()


def test_decimated_new_ids_clear_labels_the_working_set_dropped(tmp_path):
    """A tree thinned out of the working set entirely still owns its ID."""
    coords = np.vstack([_grid(0.005, 20, 20, 4), [[9.0, 9.0, 9.0]]])
    tid = np.append(np.ones(len(coords) - 1, dtype=int), 40)
    path = tmp_path / "with_speck.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a: a[2])
    assert cat.next_free_id() > 40


def test_unassigned_points_are_decimated_alongside_the_trees(tmp_path):
    """The unassigned points the review workflow lassoes are part of the
    working set too, thinned the same way."""
    tree = _grid(0.005, 20, 20, 4)
    ground = _grid(0.005, 20, 20, 1, origin=(0.0, 0.0, -1.0))
    coords = np.vstack([tree, ground])
    tid = np.repeat([1, UNASSIGNED], [len(tree), len(ground)])
    path = tmp_path / "with_ground.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a: a[2])
    assert (cat.labels == UNASSIGNED).sum() > 0
    assert (cat.labels == UNASSIGNED).sum() < len(ground)


def test_decimated_save_round_trips_an_rgb_segmented_ply(tmp_path):
    """The other backend: a raycloudtools PLY carries its labels as colour,
    so the interpolated edits have to come back as recoloured points."""
    a = _grid(0.005, 24, 24, 6)
    b = _grid(0.005, 24, 24, 6, origin=(5.0, 0.0, 0.0))
    xyz = np.vstack([a, b])
    rgb = np.vstack([
        np.tile((200, 0, 0), (len(a), 1)),
        np.tile((0, 200, 0), (len(b), 1)),
    ]).astype(np.uint8)
    path = tmp_path / "dense.ply"
    _write_raycloud_ply(path, xyz, rgb)

    cat = open_catalog(str(path), density_prompt=lambda *a_: a_[2])
    assert cat.is_decimated
    red, green = (
        next(lab for lab, c in cat.label_colors.items() if c == col)
        for col in ((200, 0, 0), (0, 200, 0))
    )

    cloud, gidx = cat.load([green], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == green), red)
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(str(path))
    assert len(reopened.records) == 1
    # Every original point is still there, now all one colour.
    only = next(iter(reopened.records.values()))
    assert only.count == len(xyz)


def test_decimated_save_does_not_bleed_into_a_touching_tree(tmp_path):
    """Two trees whose points meet: relabelling one must not drag any of the
    other's full-resolution points with it, even though the working set only
    kept one point per voxel along the seam."""
    a = _grid(0.005, 40, 40, 4)
    b = _grid(0.005, 40, 40, 4, origin=(0.2, 0.0, 0.0))  # abuts a
    coords = np.vstack([a, b])
    tid = np.repeat([1, 2], [len(a), len(b)])
    path = tmp_path / "touching.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a_: a_[2])
    cloud, gidx = cat.load([1], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == 1), 9)
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(str(path))
    assert sorted(reopened.records) == [2, 9]
    assert reopened.records[9].count == len(a)
    assert reopened.records[2].count == len(b)


def test_decimated_save_moves_a_whole_tree_that_sits_in_dense_ground(tmp_path):
    """A trunk standing in ground points: every one of the tree's points has
    an unassigned point within millimetres, so matching each full-resolution
    point to the nearest *working* point would strand a ring of them on the
    old label. Matching within the same original tree must not."""
    trunk = _grid(0.004, 6, 6, 60)                      # a dense column
    ground = _grid(0.004, 60, 60, 3)                    # dense ground through it
    coords = np.vstack([trunk, ground])
    tid = np.repeat([1, UNASSIGNED], [len(trunk), len(ground)])
    path = tmp_path / "trunk_in_ground.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a: a[2])
    cloud, gidx = cat.load([1], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == 1), 3)
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(str(path))
    assert sorted(reopened.records) == [3]
    assert reopened.records[3].count == len(trunk)      # all of it, not most
    assert (reopened.labels == UNASSIGNED).sum() == len(ground)


def test_decimated_save_promotes_only_the_lassoed_unassigned_points(tmp_path):
    """The other half of the same problem: adding ground points to a tree
    (the workflow's A key) must carry their full-resolution neighbours along
    without also sweeping up ground the user left outside the lasso."""
    tree = _grid(0.004, 10, 10, 20)
    ground = _grid(0.004, 60, 10, 2, origin=(0.0, 0.0, -0.05))
    coords = np.vstack([tree, ground])
    tid = np.repeat([1, UNASSIGNED], [len(tree), len(ground)])
    path = tmp_path / "add_ground.las"
    _write_arbor_las(path, coords, tid)

    cat = open_catalog(str(path), density_prompt=lambda *a: a[2])
    cloud, gidx = cat.load([1], margin=1.0)
    # Lasso the near end of the ground strip only.
    cut = 0.1
    near = np.flatnonzero((cloud.labels == UNASSIGNED) & (cloud.coords[:, 0] < cut))
    assert near.size
    ops.reassign(cloud, near, 1)
    cat.apply(cloud, gidx)
    cat.save()

    reopened = open_catalog(str(path))
    promoted = reopened.records[1].count - len(tree)
    # Full-resolution ground came along with the working points that were
    # lassoed, but nowhere near all of it.
    assert 0 < promoted < len(ground)
    # The far end of the strip, well outside the lasso, is untouched.
    far = reopened.coords[:, 0] > cut + 4 * cat.voxel_size
    assert far.any() and (reopened.labels[far] == UNASSIGNED).all()


# -- interaction with the global-shift prompt --------------------------------
def _dense_utm_plot(tmp_path):
    """A dense plot standing at a UTM easting/northing, so opening it raises
    both load-time questions at once."""
    a = _grid(0.005, 30, 30, 6, origin=(204300.0, 7223250.0, 12.0))
    b = _grid(0.005, 30, 30, 6, origin=(204305.0, 7223250.0, 12.0))
    coords = np.vstack([a, b])
    tid = np.repeat([1, 2], [len(a), len(b)])
    path = tmp_path / "dense_utm.las"
    _write_arbor_las(path, coords, tid,
                     offsets=(204000.0, 7223000.0, 0.0))
    return str(path), len(a) + len(b)


def test_dense_utm_cloud_asks_both_questions_and_decimates(tmp_path):
    path, n = _dense_utm_plot(tmp_path)
    asked = []

    cat = open_catalog(
        path,
        shift_prompt=lambda mins, maxs, sug: (asked.append("shift"), tuple(sug))[1],
        density_prompt=lambda sp, n_, sug: (asked.append("density"), sug)[1],
    )

    assert asked == ["shift", "density"]
    assert cat.global_shift is not None
    # Measured on the shifted coordinates, so it's the real spacing.
    assert cat.spacing == pytest.approx(0.005, rel=0.3)
    assert cat.is_decimated and cat.working_count < n / 4


def test_declining_the_shift_also_declines_the_density_check(tmp_path):
    """Unshifted UTM coordinates have already lost sub-metre detail to the
    float32 cast, so a spacing measured on them is quantisation, not the
    cloud -- it reads far finer than the truth and voxelises far harder than
    asked. The offer must not be made on those numbers."""
    path, n = _dense_utm_plot(tmp_path)
    asked = []

    cat = open_catalog(
        path,
        shift_prompt=lambda mins, maxs, sug: (asked.append("shift"), None)[1],
        density_prompt=lambda sp, n_, sug: (asked.append("density"), sug)[1],
    )

    assert asked == ["shift"]          # never got as far as the density check
    assert cat.spacing is None
    assert not cat.is_decimated and cat.working_count == n


def test_large_coordinates_without_a_shift_prompt_are_left_alone(tmp_path):
    """Same guard when there is no shift prompt wired up at all: the
    coordinates are just as quantised, nobody was asked, so nothing is
    decimated."""
    path, n = _dense_utm_plot(tmp_path)

    cat = open_catalog(path, density_prompt=lambda *a: a[2])

    assert not cat.is_decimated and cat.working_count == n
