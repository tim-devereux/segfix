# segfix

A GUI tool to **fix the instance segmentation of tree point clouds**. Load a
segmented LiDAR cloud, see each tree in its own colour, and correct mistakes by
selecting points and merging / splitting / reassigning them — then save back to
LAS/LAZ or PLY.

Built on [napari](https://napari.org) for fast 3D rendering and point selection.

## Install

Requires Python 3.10–3.12 (napari/Qt do not yet ship wheels for 3.13+).

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # fish: source .venv/bin/activate.fish
pip install -e .
```

## Run

```bash
segfix path/to/cloud.las
# or generate a practice cloud with built-in segmentation errors:
python scripts/make_sample.py sample.las
segfix sample.las
```

The per-point tree ID field is auto-detected (`treeID`, `PredInstance`,
`label`, …); override with `--label-field NAME` if needed.

### RGB-segmented clouds (raycloudtools)

Binary PLYs where the segmentation is encoded as **per-point colour** (each
tree a distinct RGB, e.g. raycloudtools output with `double x/y/z`, normals,
`time`, `rgba`) are detected automatically — no label field needed. Each
distinct colour becomes a tree; pure black `(0,0,0)` is treated as
unsegmented. Trees render in their original colours, and **Save** writes the
file back in the same raycloud format, recolouring edited points (merged trees
take the surviving tree's colour; new/split trees get a fresh distinct colour).

```bash
segfix WYTHAM_CHERLET_raycloud_segmented.ply
# 14M-point clouds load in a couple of seconds; for snappier rotation:
segfix WYTHAM_CHERLET_raycloud_segmented.ply --max-points 2000000
```

`--max-points` subsamples for display only; note that saving then writes just
the loaded (subsampled) points, so omit it when producing a corrected file.

### Crop mode — edit a huge cloud in chunks

For large clouds (millions of points) it's faster to fix one region at a time.
Crop mode memory-maps the file, loads only the points in a chosen tile, and
writes each fixed tile back into an output copy — no full in-memory rewrite.

```bash
segfix WYTHAM_CHERLET_raycloud_segmented.ply --crop
segfix WYTHAM_CHERLET_raycloud_segmented.ply --crop --output corrected.ply
```

- The **tiles** dock (left) shows the cloud's XY extent. Set a grid (e.g. 4×4)
  and a **context margin** (surrounding points shown for reference but not
  saved), click *Make grid*, then double-click a tile to load it.
- Fix it with the segfix panel (lasso + merge/split/reassign/delete), then
  *Save Tile* (or `Ctrl+S`) to write just that tile's points back — or
  **Save & Next** to save and jump straight to the next unfinished tile.
  Saved tiles get a ✓; later tiles load with earlier edits already applied.
- Tile load/save on the 14M-point Wytham cloud take well under a second each.

Note: a tree straddling a tile boundary is fixed per tile (the margin lets you
see the boundary); only the points inside the tile box are written on save.

## Project mode (forestry QA workflow)

Reimplements the CloudCompare `cc-tree-segmentation-fix` plugin's workflow as a
standalone app. Point it at a directory of **per-tree PLY files** (one tree per
file, named `{tree_id}_matched.ply` / `_uncertain.ply`; `*_non_seg.ply` is
treated as overlay data):

```bash
python scripts/make_project_sample.py sample_project   # demo data with errors
segfix --project sample_project
```

- **Tree table** (left dock): every discovered tree, with a ✓ when its fix is
  saved. Double-click a row (or *Load Tree*) to load that tree **plus its
  spatial neighbours** into one editable cloud — each source file a distinct
  colour/label. Neighbours are found by distance threshold → bbox-margin
  expansion → nearest-neighbour radius, matching the plugin.
- **Overlays**: unsegmented points (`*_non_seg.ply`, orange) and previously
  removed points (`removed_points.xyz`, magenta) are shown as dimmed context
  within a cylinder around the focus tree.
- **Fix** the segmentation with the segfix panel (lasso + merge/split/reassign/
  delete) — e.g. merge two over-segmented fragments, or lasso-split a fused
  pair into separate trees.
- **Save Fixed** writes each resulting tree to `fixed/{tree_id}.ply` (indexed
  `_N.ply` when a tree splits), appends deleted points to `removed_points.xyz`
  in the plugin's `x y z r g b nx ny nz alpha time group_id` format, and marks
  the tree complete (tracked in `segfix_settings.json`).

### Deviations from the CloudCompare plugin
- Editing is one labelled cloud rather than separate DB-tree entities; the
  segment/merge/delete tools are reimplemented natively (see operations below).
- Saved files are named by **each tree's own id** (the plugin named every
  loaded entity after the focus tree). The CSV "assessment" mode and the
  taper/girth polyline are not yet ported (the PLY auto-discovery path is).

## Editing workflow

Navigation matches CloudCompare: clouds open **Z-up**, **left-drag rotates,
right-drag pans, wheel zooms**. The window opens maximised with napari's
stock chrome (menu bar, layer list/controls, console) hidden — everything
you need is in the segfix panel, including a **Point size** spinner that
applies across tile/tree loads.

Everything is selection-first and driven by single keys, so the mouse stays on
the canvas:

1. The cloud opens in 3D, each tree a distinct colour. Grey = unassigned,
   dark grey = noise. The mouse navigates by default; the panel shows a live
   count of what's selected.
2. **Select** points with the lasso: press **L**, drag a freehand loop
   (Shift adds to the selection), press **L** again to go back to navigating.
3. **Pick a target tree by eye** (for reassignment): select any point of the
   destination tree and press **T** — it appears as a colour swatch in the
   panel. No tree IDs to type, ever.
4. **Fix** with one key:

   | Key | Operation |
   |-----|-----------|
   | `R` | Reassign selection to the target tree |
   | `N` | New tree from selection (split off) |
   | `M` | Merge all trees the selection touches |
   | `X` | Delete — mark selection as noise |
   | `U` | Unassign selection |
   | `K` | KMeans-split the tree under the selection into N parts |
   | `I` | Isolate the selection's trees (press again to show all) |
   | `H` | Hide/show unassigned + noise points |
   | `D` | Add the selection as a grow seed |
   | `G` | Grow from seeds |
   | `Delete` | Same as `X` (mark noise) |
   | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
   | `Ctrl+S` | Save |

### Untangling intermingled crowns

For two (or more) trees whose branches interweave:

1. Select a few points on each tree involved and press **I** — everything
   else is hidden so stray lassos can't touch bystander trees.
2. Lasso a small patch on each trunk, pressing **D** after each — the seeds
   show as coloured markers, one colour per future tree.
3. Press **G**. Every point of the trees the seeds touch is reassigned to the
   seed it is closest to *through the cloud* (shortest path over a
   k-nearest-neighbour graph), so branches follow their physical attachment
   instead of straight-line distance to a crown centre. This both splits a
   fused tree (one label, two seeds) and re-partitions a mislabelled tangle.
4. Review, touch up borders with the lasso, press **I** to un-hide the rest.

Growth parameters (in the Untangle box):

- **k** — neighbours per point in the graph (default 8). Lower follows thin
  strands more strictly; higher tolerates gappy scans but bridges more.
- **max link** — severs graph links longer than this, so growth can't leak
  across branches that merely touch (the main failure mode). Points cut off
  from every seed keep their current label. Try 0.2–0.5 m for tangled
  crowns; *off* = unlimited.
- **Claim unassigned** — also grow over unassigned points around the
  involved trees (bounding box + 2 m), pulling unsegmented canopy into the
  nearest tree.

5. **Save** writes back preserving the original header (CRS, scale, offset)
   and extra attributes; in crop/project mode it saves the tile/tree instead.

## Layout

| File | Responsibility |
|------|----------------|
| `model.py` | `PointCloud` data + undo/redo (diff-based) |
| `io.py` | load/save PLY + LAS/LAZ, label-field detection |
| `operations.py` | pure, UI-agnostic label edits (merge/split/reassign/…) |
| `lasso.py` | 3D screen-space lasso: camera projection + polygon test |
| `viewer.py` | napari layer + label→colour mapping |
| `widgets.py` | Qt dock panel wiring selection → operations |
| `project.py` | project mode: PLY discovery, neighbour finding, settings |
| `trees.py` | load many per-tree PLYs into one cloud; save + removed tracking |
| `overlays.py` | non-seg / removed-point cylinder overlays |
| `project_ui.py` | tree table + project controller |
| `crop.py` | crop mode: memory-mapped tile load + write-back |
| `crop_ui.py` | tile grid table + crop controller |
| `app.py` | `segfix` CLI entry point (single-file / `--project` / `--crop`) |

## Tests

```bash
pytest        # core model, operations, and IO round-trip (no GUI needed)
```

## Notes & next steps

- "Delete" is modelled as **mark-as-noise** (points kept, label `-1`) so files
  round-trip 1:1; an optional "drop noise on export" can be added later.
