"""Generate a synthetic forest plot with deliberate segmentation errors.

Creates a handful of simple cone-ish "trees" plus two classic mistakes to
practise on: an over-segmented tree (one tree split into two IDs) and an
under-segmented pair (two trees sharing one ID) — plus the two non-tree
labels real scans always carry: a scatter of unassigned ground/understory
points, and a handful of noise points floating with nothing near them.

Usage:  python scripts/make_sample.py sample.ply
"""

import sys

import numpy as np

from segfix import PointCloud
from segfix import io
from segfix.model import NOISE, UNASSIGNED


def tree(center, height=8.0, radius=2.0, n=2000, rng=None):
    rng = rng or np.random.default_rng()
    z = rng.random(n) ** 0.5 * height          # denser near the base
    r = (1 - z / height) * radius * rng.random(n) ** 0.5
    theta = rng.random(n) * 2 * np.pi
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y, center[2] + z]).astype(np.float32)


def ground(bounds, n=3000, z=0.0, jitter=0.15, rng=None):
    """A scatter of unassigned points across the plot's footprint, at
    roughly ground level — the terrain/understory real scans are full of,
    which the review workflow lassoes and assigns into a tree with A."""
    rng = rng or np.random.default_rng()
    (xlo, xhi), (ylo, yhi) = bounds
    x = rng.uniform(xlo, xhi, n)
    y = rng.uniform(ylo, yhi, n)
    z = z + rng.normal(0, jitter, n)
    return np.column_stack([x, y, z]).astype(np.float32)


def floaters(bounds, n=15, z_range=(3.0, 12.0), rng=None):
    """A few stray points scattered mid-air with no tree nearby — sensor
    noise, to practise marking Noise (X) instead of assigning to a tree."""
    rng = rng or np.random.default_rng()
    (xlo, xhi), (ylo, yhi) = bounds
    x = rng.uniform(xlo, xhi, n)
    y = rng.uniform(ylo, yhi, n)
    z = rng.uniform(*z_range, n)
    return np.column_stack([x, y, z]).astype(np.float32)


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

    # Ground/understory (unassigned) and a few stray noise points, spread
    # across the footprint of every tree above.
    bounds = ((-3, 25), (-3, 13))
    add(ground(bounds, rng=rng), UNASSIGNED)
    add(floaters(bounds, rng=rng), NOISE)

    coords = np.vstack(parts)
    labels = np.concatenate(labels)
    cloud = PointCloud(coords=coords, labels=labels)
    io.save(cloud, out)
    n_unassigned = int(np.sum(labels == UNASSIGNED))
    n_noise = int(np.sum(labels == NOISE))
    print(
        f"Wrote {cloud.n_points:,} points, trees {sorted(cloud.tree_ids.tolist())} "
        f"+ {n_unassigned:,} unassigned + {n_noise:,} noise → {out}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.ply")
