import numpy as np

from segfix import io, operations as ops
from segfix.crop import CropSession
from tests.test_rgb import _write_raycloud_ply


def _grid_cloud(tmp_path):
    # a 6x6 grid of points spanning [0,6) x [0,6); colour = a per-quadrant tree
    xs, ys = np.meshgrid(np.arange(6) + 0.5, np.arange(6) + 0.5)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(36)])
    # two colours: left half red, right half green
    rgb = np.where(
        (xyz[:, 0:1] < 3), np.array([200, 0, 0]), np.array([0, 200, 0])
    ).astype(np.uint8)
    path = tmp_path / "grid.ply"
    _write_raycloud_ply(path, xyz, rgb)
    return path


def test_bounds_and_tiles(tmp_path):
    sess = CropSession(str(_grid_cloud(tmp_path)))
    x0, y0, x1, y1 = sess.bounds
    assert (x0, y0) == (0.5, 0.5) and (x1, y1) == (5.5, 5.5)
    tiles = sess.tiles(2, 2)
    assert len(tiles) == 4


def test_load_crop_core_vs_margin(tmp_path):
    sess = CropSession(str(_grid_cloud(tmp_path)))
    # bottom-left box [0,3)x[0,3) with a 1 m margin
    cloud, info = sess.load_crop((0, 0, 3, 3), margin=1.0)
    # core points are those strictly inside [0,3): x,y in {0.5,1.5,2.5} -> 3x3
    assert info.n_core == 9
    # margin pulls in neighbours, so more points load than the core
    assert cloud.n_points > info.n_core


def test_save_crop_writes_back_only_core(tmp_path):
    path = _grid_cloud(tmp_path)
    sess = CropSession(str(path))

    # load the right half (green) with margin, recolour it to the red tree
    cloud, info = sess.load_crop((3, 0, 6, 6), margin=1.0)
    red_label = next(
        lab for lab, col in cloud.label_colors.items() if col == (200, 0, 0)
    )
    ops.reassign(cloud, np.arange(cloud.n_points), red_label)
    msg = sess.save_crop(cloud, info)

    out = io.load(sess.output)
    # right-half points (x>=3) are now red; left half untouched
    xs = out.coords[:, 0]
    reds = {tuple(c) for c in out.label_colors.values()}
    # everything is red now (we recoloured the right half to match the left)
    assert reds == {(200, 0, 0)}
    assert "Saved" in msg
    # only core points (x in [3,6)) were written, but since left half was
    # already red, the whole cloud is red — verify core count saved
    assert info.n_core == 18  # right half of the 6x6 grid


def test_save_crop_accumulates_across_tiles(tmp_path):
    path = _grid_cloud(tmp_path)
    sess = CropSession(str(path))

    # tile 1: delete (mark noise) the bottom-left core points
    c1, i1 = sess.load_crop((0, 0, 3, 3), margin=0.0)
    ops.mark_noise(c1, np.flatnonzero(i1.core_mask))
    sess.save_crop(c1, i1)

    # tile 2 loads from the updated working file and should see tile 1's edits
    c2, i2 = sess.load_crop((0, 0, 3, 3), margin=0.0)
    # those points are now black -> unassigned
    from segfix.model import UNASSIGNED
    assert (c2.labels == UNASSIGNED).all()
