"""Generate a synthetic forest plot with deliberate segmentation errors.

Creates a handful of simple cone-ish "trees" plus two classic mistakes to
practise on: an over-segmented tree (one tree split into two IDs) and an
under-segmented pair (two trees sharing one ID) — plus the two non-tree
labels real scans always carry: a scatter of unassigned ground/understory
points, and a handful of noise points floating with nothing near them.

Usage:  python scripts/make_sample.py sample.ply
        python scripts/make_sample.py --format las sample.las
        python scripts/make_sample.py --spacing 0.01 dense.las
        python scripts/make_sample.py --spacing 0.01 --origin 204300 7223250 12 \
            dense_utm.las

The ``.las`` form mimics arbor's output: XYZ plus a ``treeID`` Extra-Bytes
column, so the LAS editing path (:class:`segfix.treecatalog.LasCatalog`) has
something to open without running the arbor pipeline.

``--spacing`` switches to a scan-like cloud: instead of a few thousand points
filling each tree's volume, points are laid on the surfaces a scanner
actually sees (trunk, crown shell, ground) at the requested pitch. That is
what makes a cloud *dense* — a metre of trunk is one ring of points, but at
5mm there are two hundred of them — and it is how to get a file that trips
segfix's "points closer than 2cm" check and its downsample offer (see
:mod:`segfix.density`). The same deliberate segmentation errors are in it,
so the whole fix-and-save round trip can be exercised at full density.

``--origin`` puts the plot somewhere georeferenced (a UTM easting/northing,
say) instead of at the origin, which is what a real survey looks like and
what trips segfix's *other* load-time question, the global shift (see
:mod:`segfix.shift_ui`). Combined with ``--spacing`` it produces a cloud that
asks both questions in one open, which is the only way to exercise them
together.
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


def _jittered_grid(u_size, v_size, spacing, rng):
    """``(u, v)`` samples covering a ``u_size`` x ``v_size`` patch at roughly
    ``spacing`` pitch.

    A jittered grid rather than uniform-random points: random points on a
    surface clump, and their median nearest-neighbour distance comes out
    around half the pitch you asked for, which would make "generate a 1cm
    cloud" quietly produce a 5mm one. Jittering a grid by a fraction of a
    cell keeps the measured spacing close to the requested one while still
    looking scanned rather than plotted.
    """
    nu = max(1, int(round(u_size / spacing)))
    nv = max(1, int(round(v_size / spacing)))
    u, v = np.meshgrid(
        (np.arange(nu) + 0.5) * (u_size / nu),
        (np.arange(nv) + 0.5) * (v_size / nv),
        indexing="ij",
    )
    u = u.ravel() + rng.uniform(-0.2, 0.2, u.size) * spacing
    v = v.ravel() + rng.uniform(-0.2, 0.2, v.size) * spacing
    return u, v


def dense_tree(center, height=8.0, radius=2.0, spacing=0.01, rng=None):
    """A tree as a scanner sees it: a trunk cylinder and a conical crown
    shell, both sampled at ``spacing``.

    Surfaces, not a filled volume — that is the difference between a cloud
    that is merely large and one that is genuinely dense. Filling the same
    cone's 34 cubic metres at 5mm would take 270 million points; its
    surfaces take two million.
    """
    rng = rng or np.random.default_rng()
    trunk_r = max(0.04, radius * 0.07)
    crown_base = 0.3 * height

    # Trunk: unroll the cylinder to a (circumference x height) patch.
    u, z = _jittered_grid(2 * np.pi * trunk_r, height, spacing, rng)
    theta = u / trunk_r
    r = trunk_r + rng.normal(0, spacing * 0.3, theta.size)  # bark roughness
    trunk = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        center[2] + z,
    ])

    # Crown: the cone's lateral surface, unrolled the same way. Its radius
    # tapers to nothing at the top, so the grid is laid out in slant length
    # against the *widest* circumference and then thinned per ring - which
    # is also what keeps the crown from turning into a solid cap up top.
    crown_h = height - crown_base
    slant = float(np.hypot(radius, crown_h))
    u, s = _jittered_grid(2 * np.pi * radius, slant, spacing, rng)
    frac = np.clip(s / slant, 0, 1)          # 0 at the base, 1 at the tip
    ring_r = radius * (1 - frac)
    # Keep each ring's points at the requested pitch: a ring of radius
    # ring_r only has room for a fraction of the widest ring's points.
    keep = rng.random(u.size) < np.maximum(ring_r / radius, 1e-6)
    theta = u[keep] / radius
    ring_r, frac = ring_r[keep], frac[keep]
    jitter = rng.normal(0, spacing * 0.5, theta.size)
    crown = np.column_stack([
        center[0] + (ring_r + jitter) * np.cos(theta),
        center[1] + (ring_r + jitter) * np.sin(theta),
        center[2] + crown_base + frac * crown_h,
    ])
    return np.vstack([trunk, crown]).astype(np.float32)


def dense_ground(centers, spacing=0.01, reach=3.0, rng=None):
    """Ground under the trees, at ``spacing``: one grid over the whole plot,
    keeping only what falls within ``reach`` of a trunk.

    Near the trees rather than across the whole rectangle, because that is
    all the review workflow ever lassoes — and at these pitches the empty
    gaps between trees would otherwise be most of the file. One shared grid
    rather than a disc each, so overlapping discs can't quietly double the
    density where two trees stand close together.
    """
    rng = rng or np.random.default_rng()
    centers = np.asarray(centers, dtype=np.float64)
    lo = centers[:, :2].min(axis=0) - reach
    hi = centers[:, :2].max(axis=0) + reach
    u, v = _jittered_grid(hi[0] - lo[0], hi[1] - lo[1], spacing, rng)
    x, y = lo[0] + u, lo[1] + v

    near = np.zeros(x.size, dtype=bool)
    for cx, cy, _cz in centers:
        near |= (x - cx) ** 2 + (y - cy) ** 2 <= reach * reach
    x, y = x[near], y[near]
    z = centers[:, 2].mean() + rng.normal(0, spacing, x.size)  # slight relief
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


# Where the trees stand, and how wide, shared by both modes: three clean
# trees, one that is over-segmented, and a close pair sharing an ID.
_CENTERS = [
    ((0, 0, 0), 2.0),
    ((10, 0, 0), 2.0),
    ((0, 10, 0), 2.0),
    ((10, 10, 0), 2.0),
    ((20, 5, 0), 1.5),
    ((22.5, 5, 0), 1.5),
]


def main(out, spacing=None, origin=None):
    rng = np.random.default_rng(42)
    parts, labels = [], []

    def add(pts, lab):
        parts.append(pts)
        labels.append(np.full(len(pts), lab, dtype=np.int32))

    def make(center, radius):
        if spacing is None:
            return tree(center, radius=radius, rng=rng)
        return dense_tree(center, radius=radius, spacing=spacing, rng=rng)

    # Three clean trees: IDs 1, 2, 3
    for lab, (center, radius) in zip((1, 2, 3), _CENTERS[:3]):
        add(make(center, radius), lab)

    # Over-segmented: one physical tree at (10,10) split into IDs 4 and 5
    t = make(*_CENTERS[3])
    add(t[t[:, 2] < 4], 4)
    add(t[t[:, 2] >= 4], 5)

    # Under-segmented: two trees both labelled ID 6
    for center, radius in _CENTERS[4:]:
        add(make(center, radius), 6)

    # Ground/understory (unassigned) and a few stray noise points, spread
    # across the footprint of every tree above.
    bounds = ((-3, 25), (-3, 13))
    if spacing is None:
        add(ground(bounds, rng=rng), UNASSIGNED)
    else:
        add(dense_ground([c for c, _ in _CENTERS], spacing=spacing, rng=rng),
            UNASSIGNED)
    add(floaters(bounds, rng=rng), NOISE)

    # float64 from here on: adding a UTM-sized origin to float32 coordinates
    # would round away the millimetres this just went to the trouble of
    # generating, before laspy ever sees them. (LAS itself is safe -- it
    # stores xyz as scaled offsets from a header origin, not as floats.)
    coords = np.vstack(parts).astype(np.float64)
    if origin is not None:
        coords += np.asarray(origin, dtype=np.float64)
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
        f"+ {n_unassigned:,} unassigned + {n_noise:,} noise → {out} "
        f"({os.path.getsize(out) / 1e6:,.0f} MB)"
    )
    if origin is not None:
        from segfix.treecatalog import needs_global_shift

        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        verdict = (
            "opening this also offers a global shift"
            if needs_global_shift(mins, maxs) else
            "not large enough for segfix to offer a global shift"
        )
        print(
            f"Plot placed at ({mins[0]:,.0f}, {mins[1]:,.0f}, {mins[2]:,.0f}) "
            f"- {verdict}"
        )
    if spacing is not None:
        # Measured with the same estimator segfix runs on load, so the
        # number printed here is the one the app will report back.
        from segfix.density import DENSE_SPACING, estimate_spacing

        measured = estimate_spacing(coords)
        verdict = (
            "below segfix's 2cm threshold: opening this offers to downsample"
            if measured < DENSE_SPACING else
            "above segfix's 2cm threshold: no downsample will be offered"
        )
        print(f"Measured point spacing {measured * 100:.2f} cm - {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", help="output path (extension picks format)")
    ap.add_argument("--format", choices=["ply", "las"], default="ply",
                    help="format when OUT is omitted (default: ply)")
    ap.add_argument(
        "--spacing", type=float, default=None, metavar="METRES",
        help="generate a scan-like cloud with points this far apart on the "
             "trunk, crown and ground surfaces, instead of a few thousand "
             "points per tree. 0.01 gives ~4M points (~140 MB LAS); 0.005 "
             "gives ~16M (~540 MB). Anything under 0.02 trips segfix's "
             "dense-cloud check.",
    )
    ap.add_argument(
        "--origin", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="place the plot here instead of at (0, 0, 0), e.g. --origin "
             "204300 7223250 12 for a UTM-referenced cloud that also trips "
             "segfix's large-coordinate (global shift) prompt. LAS only past "
             "10 km from the origin: a PLY sample stores xyz as float32, "
             "which at those magnitudes quantises the cloud to metres.",
    )
    args = ap.parse_args()
    if args.spacing is not None and args.spacing <= 0:
        ap.error("--spacing must be positive")
    default_name = "dense" if args.spacing is not None else "sample"
    out = args.out or f"{default_name}.{args.format}"
    if args.origin is not None and os.path.splitext(out)[1].lower() == ".ply":
        from segfix.treecatalog import GLOBAL_SHIFT_THRESHOLD

        if max(abs(v) for v in args.origin) > GLOBAL_SHIFT_THRESHOLD:
            ap.error(
                f"--origin {args.origin} needs a .las output: this script's "
                "PLY writer stores xyz as float32, which past "
                f"{GLOBAL_SHIFT_THRESHOLD:,.0f} m keeps under a metre of "
                "precision - the cloud would arrive already ruined, which is "
                "not a useful test of the prompt that exists to prevent that."
            )
    main(out, spacing=args.spacing, origin=args.origin)
