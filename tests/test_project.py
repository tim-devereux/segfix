import numpy as np
import pytest

from segfix import PointCloud, NOISE, io
from segfix import operations as ops
from segfix.project import Project, discover, extract_tree_id
from segfix.trees import load_scene, save_scene


# -- filename parsing ----------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("12_matched.ply", ("12", None, "matched")),
        ("12_3_matched.ply", ("12", "3", "matched")),
        ("7_uncertain.ply", ("7", None, "uncertain")),
        ("7_2_uncertain.ply", ("7", "2", "uncertain")),
        ("site_non_seg.ply", (None, None, None)),
        ("random.ply", (None, None, None)),
    ],
)
def test_extract_tree_id(name, expected):
    assert extract_tree_id(name) == expected


def _write(path, xyz, color=(10, 20, 30)):
    n = len(xyz)
    io.save(
        PointCloud(
            coords=np.asarray(xyz, np.float32),
            labels=np.zeros(n, np.int32),
            attributes={
                "red": np.full(n, color[0], np.uint8),
                "green": np.full(n, color[1], np.uint8),
                "blue": np.full(n, color[2], np.uint8),
            },
        ),
        str(path),
    )


def _make_project(tmp_path):
    # trees 1 & 2 overlap (touching, within the 1 m proximity radius); tree 3
    # is far away. a non_seg file must be ignored by discovery.
    _write(tmp_path / "1_matched.ply", np.random.RandomState(0).rand(200, 3) * 2 + [0, 0, 0])
    _write(tmp_path / "2_matched.ply", np.random.RandomState(1).rand(200, 3) * 2 + [1.5, 0, 0])
    _write(tmp_path / "3_matched.ply", np.random.RandomState(2).rand(200, 3) * 2 + [40, 0, 0])
    _write(tmp_path / "site_non_seg.ply", np.random.RandomState(3).rand(50, 3) * 50)
    return Project(tmp_path)


def test_discover_ignores_non_seg(tmp_path):
    proj = _make_project(tmp_path)
    ids = sorted(e.tree_id for e in proj.entries)
    assert ids == ["1", "2", "3"]
    assert all("non_seg" not in e.mesh_file for e in proj.entries)


def test_neighbours_within_threshold(tmp_path):
    proj = _make_project(tmp_path)
    focus = next(e for e in proj.entries if e.tree_id == "1")
    names = {e.tree_id for e in proj.neighbours(focus)}
    # tree 2 is 4 m away (loaded); tree 3 is 40 m away (excluded)
    assert "1" in names and "2" in names and "3" not in names


def test_load_scene_labels_per_file(tmp_path):
    proj = _make_project(tmp_path)
    focus = next(e for e in proj.entries if e.tree_id == "1")
    neighbours = proj.neighbours(focus)
    cloud, scene = load_scene(proj, focus, neighbours)
    # one distinct label per loaded file
    assert len(np.unique(cloud.labels)) == len(neighbours)
    assert scene.focus_tree_id == "1"
    assert "red" in cloud.attributes


def test_save_scene_writes_fixed_and_removed(tmp_path):
    proj = _make_project(tmp_path)
    focus = next(e for e in proj.entries if e.tree_id == "1")
    neighbours = proj.neighbours(focus)
    cloud, scene = load_scene(proj, focus, neighbours)

    # delete a few points from the focus tree → should be tracked as removed
    focus_label = min(np.unique(cloud.labels))
    focus_idx = np.flatnonzero(cloud.labels == focus_label)
    ops.mark_noise(cloud, focus_idx[:5])

    msg = save_scene(cloud, scene)

    fixed = tmp_path / "fixed"
    assert (fixed / "1.ply").exists()
    assert (tmp_path / "removed_points.xyz").exists()
    removed = np.loadtxt(tmp_path / "removed_points.xyz", ndmin=2)
    assert removed.shape[0] == 5 and removed.shape[1] == 12
    assert "1" in proj.completed
    assert "removed" in msg

    # saved focus file has the surviving points only
    saved = io.read_xyz(str(fixed / "1.ply"))
    assert saved.shape[0] == focus_idx.size - 5


def test_save_scene_split_creates_indexed_files(tmp_path):
    proj = _make_project(tmp_path)
    focus = next(e for e in proj.entries if e.tree_id == "1")
    cloud, scene = load_scene(proj, focus, [focus])  # focus only

    # split the focus tree into a second label → 1.ply + 1_*.ply
    focus_idx = np.flatnonzero(cloud.labels == min(np.unique(cloud.labels)))
    ops.create_new(cloud, focus_idx[: focus_idx.size // 2])
    save_scene(cloud, scene)

    fixed = tmp_path / "fixed"
    produced = sorted(p.name for p in fixed.glob("1*.ply"))
    assert produced == ["1_1.ply", "1_2.ply"]
