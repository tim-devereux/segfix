"""Generate a sample *project directory* of per-tree PLY files.

Mimics the plugin's no-CSV input: one ``{tree_id}_matched.ply`` per tree, a
``site_non_seg.ply`` of unsegmented understory, and deliberate errors to fix:
trees 4 & 5 are two fragments of one stem (over-segmented → merge), tree 6
contains two stems fused together (under-segmented → split).

Usage:  python scripts/make_project_sample.py sample_project/
"""

import sys
from pathlib import Path

import numpy as np

from segfix import PointCloud
from segfix import io


def tree(center, height=8.0, radius=1.6, n=1500, rng=None, color=(40, 120, 40)):
    rng = rng or np.random.default_rng()
    z = rng.random(n) ** 0.5 * height
    r = (1 - z / height) * radius * rng.random(n) ** 0.5
    theta = rng.random(n) * 2 * np.pi
    xyz = np.column_stack(
        [center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), center[2] + z]
    ).astype(np.float32)
    return xyz


def write_tree(path, xyz, color):
    n = len(xyz)
    cloud = PointCloud(
        coords=xyz,
        labels=np.zeros(n, dtype=np.int32),
        attributes={
            "red": np.full(n, color[0], dtype=np.uint8),
            "green": np.full(n, color[1], dtype=np.uint8),
            "blue": np.full(n, color[2], dtype=np.uint8),
        },
    )
    io.save(cloud, str(path))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    write_tree(out / "1_matched.ply", tree((0, 0, 0), rng=rng), (200, 60, 60))
    write_tree(out / "2_matched.ply", tree((5, 0, 0), rng=rng), (60, 200, 60))
    write_tree(out / "3_matched.ply", tree((0, 5, 0), rng=rng), (60, 60, 200))

    # Over-segmented: one stem split across two files.
    full = tree((5, 5, 0), rng=rng)
    write_tree(out / "4_matched.ply", full[full[:, 2] < 4], (200, 200, 60))
    write_tree(out / "5_matched.ply", full[full[:, 2] >= 4], (200, 120, 60))

    # Under-segmented: two stems fused in one file.
    fused = np.vstack([tree((10, 2, 0), radius=1.2, rng=rng),
                       tree((11.5, 2, 0), radius=1.2, rng=rng)])
    write_tree(out / "6_matched.ply", fused, (160, 60, 200))

    # Unsegmented understory / ground.
    ground = rng.random((4000, 3)).astype(np.float32) * [14, 8, 0.4]
    write_tree(out / "site_non_seg.ply", ground, (140, 140, 140))

    print(f"Wrote project sample to {out}/ (6 trees + non_seg)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample_project")
