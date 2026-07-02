"""Generate a synthetic forest plot with deliberate segmentation errors.

Creates a handful of simple cone-ish "trees" plus two classic mistakes to
practise on: an over-segmented tree (one tree split into two IDs) and an
under-segmented pair (two trees sharing one ID).

Usage:  python scripts/make_sample.py sample.las
"""

import sys

import numpy as np

from segfix import PointCloud
from segfix import io


def tree(center, height=8.0, radius=2.0, n=2000, rng=None):
    rng = rng or np.random.default_rng()
    z = rng.random(n) ** 0.5 * height          # denser near the base
    r = (1 - z / height) * radius * rng.random(n) ** 0.5
    theta = rng.random(n) * 2 * np.pi
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y, center[2] + z]).astype(np.float32)


def main(out):
    rng = np.random.default_rng(42)
    parts, labels = [], []

    def add(pts, lab):
        parts.append(pts)
        labels.append(np.full(len(pts), lab, dtype=np.int32))

    # Three clean trees: IDs 1, 2, 3
    add(tree((0, 0, 0), rng=rng), 1)
    add(tree((10, 0, 0), rng=rng), 2)
    add(tree((0, 10, 0), rng=rng), 3)

    # Over-segmented: one physical tree at (10,10) split into IDs 4 and 5
    t = tree((10, 10, 0), rng=rng)
    add(t[t[:, 2] < 4], 4)
    add(t[t[:, 2] >= 4], 5)

    # Under-segmented: two trees both labelled ID 6
    add(tree((20, 5, 0), radius=1.5, rng=rng), 6)
    add(tree((22.5, 5, 0), radius=1.5, rng=rng), 6)

    coords = np.vstack(parts)
    labels = np.concatenate(labels)
    cloud = PointCloud(coords=coords, labels=labels)
    io.save(cloud, out)
    print(f"Wrote {cloud.n_points:,} points, trees {sorted(cloud.tree_ids.tolist())} → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.las")
