"""Generate a synthetic forest plot with deliberate segmentation errors.

Creates a handful of simple cone-ish "trees" plus two classic mistakes to
practise on: an over-segmented tree (one tree split into two IDs) and an
under-segmented pair (two trees sharing one ID) — plus the two non-tree
labels real scans always carry: a scatter of unassigned ground/understory
points, and a handful of noise points floating with nothing near them.

Usage:  python scripts/make_sample.py sample.ply
        python scripts/make_sample.py --format las sample.las

The ``.las`` form mimics arbor's output: XYZ plus a ``treeID`` Extra-Bytes
column, so the LAS editing path (:class:`segfix.treecatalog.LasCatalog`) has
something to open without running the arbor pipeline.
"""

import argparse
import os

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


def _write_las(coords: np.ndarray, labels: np.ndarray, out: str) -> None:
    """Write an arbor-shaped LAS/LAZ: XYZ + an int ``treeID`` Extra-Bytes
    column (``0`` = unassigned, as arbor writes it)."""
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(coords.min(axis=0))
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="treeID", type=np.int32,
                               description="Unique ID per tree")
    )
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    tid = np.where(labels == NOISE, UNASSIGNED, labels)  # LAS has no noise id
    las.treeID = tid.astype(np.int32)
    las.write(out)


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

    if os.path.splitext(out)[1].lower() in (".las", ".laz"):
        _write_las(coords, labels, out)
        n_points = len(coords)
        tree_ids = sorted(set(labels.tolist()) - {UNASSIGNED, NOISE})
    else:
        cloud = PointCloud(coords=coords, labels=labels)
        io.save(cloud, out)
        n_points = cloud.n_points
        tree_ids = sorted(cloud.tree_ids.tolist())

    n_unassigned = int(np.sum(labels == UNASSIGNED))
    n_noise = int(np.sum(labels == NOISE))
    print(
        f"Wrote {n_points:,} points, trees {tree_ids} "
        f"+ {n_unassigned:,} unassigned + {n_noise:,} noise → {out}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", help="output path (extension picks format)")
    ap.add_argument("--format", choices=["ply", "las"], default="ply",
                    help="format when OUT is omitted (default: ply)")
    args = ap.parse_args()
    main(args.out or f"sample.{args.format}")
