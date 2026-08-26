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

Running `segfix` with **no path** opens a startup dialog instead: double-click
a recent project to reopen it, or click **New Project…** to import a point
cloud file. Importing copies the file into a new project folder (created
next to wherever you choose) and opens that copy — edits are always saved to
the copy, never the original source file. ("Open Project Folder…" is the
separate `--project DIR` forestry-QA workflow below, a directory of per-tree
PLY files rather than one imported cloud.)

Every project opened this way — or given directly on the command line, which
skips the dialog entirely and edits the given file in place as before — is
recorded in `~/.config/segfix/registry.json` (a plain JSON file, not a
database) so it shows up in the "Recent projects" list next time.

### Binary PLYs: a tree table by default

Opening a **binary PLY** (RGB-segmented or with a label field) shows a table
of every tree in the file instead of loading the whole cloud at once —
double-click a row to load that tree plus its spatial neighbours for editing.
This is what makes multi-million-point clouds open instantly: the file is
memory-mapped and only the current tree's neighbourhood is ever loaded into
the 3D view.

```bash
segfix WYTHAM_CHERLET_raycloud_segmented.ply
```

**Save** (or the left-dock Save button) writes back only the points whose
tree changed since the last save — not a full-file rewrite — so it stays
fast regardless of file size, and captures edits made across *every* tree
visited in the session, not just the one currently on screen.

For RGB-segmented PLYs specifically (e.g. raycloudtools output with
`double x/y/z`, normals, `time`, `rgba`), the segmentation is encoded as
**per-point colour** and detected automatically — no label field needed. Each
distinct colour becomes a tree; pure black `(0,0,0)` is treated as
unsegmented. Trees render in their original colours, and saved edits
recolour points (merged trees take the surviving tree's colour; new/split
trees get a fresh distinct colour).

`--max-points N` subsamples the display and only applies to LAS/LAZ or ASCII
PLY input, which can't be memory-mapped this way and still load eagerly in
full — a binary PLY never needs it.

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
- **Fix** the segmentation with the segfix panel (lasso + add/absorb/split/
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

The panel is a **review queue**: the Trees table lists every tree with a
done checkbox, and progress is saved to a `<cloud>.segfix.json` sidecar so a
half-finished plot resumes where you left off.

1. Press **Space** (or click a table row) to start reviewing. The camera
   flies to the tree and a wireframe box marks it. To declutter a crowded
   view, use the 👁 column in the table (or **Hide all neighbours** in the
   Current tree panel) to hide specific or all neighbouring trees.
2. Inspect it. If it's correct, press **Space** — the tree is marked done,
   progress is saved, and the next unfinished tree loads. That's the loop.
3. If it needs fixing, **select** points with the lasso: press **L**, drag a
   freehand loop (Shift adds), **Esc** to go back to navigating. The tree
   under review is always the implicit target — no tree IDs, no eyedropper:

   | Key | Operation |
   |-----|-----------|
   | `Space` | Mark current tree done, jump to next unfinished |
   | `←` / `→` | Previous / next tree (without marking done) |
   | `A` | Add selection to the current tree (missing branches, unassigned canopy) |
   | `N` | Split selection off as a new tree (it joins the queue unreviewed) |
   | `U` | Unassign selection — or the whole current tree if nothing is selected |
   | `X` | Mark selection as noise — or the whole current tree if nothing is selected (dismiss a bush/wall in one key) |
   | `H` | Show/hide the unassigned + noise points |
   | `Delete` | Same as `X` (mark noise) |
   | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
   | `Ctrl+S` | Save Project |

   To move stray points *to a neighbour* instead, click the neighbour's row
   (one click — it becomes current), lasso the strays, press `A`.
4. After the last tree, any leftover unassigned points may hide missed
   trees: with no tree selected everything is shown — lasso and press `N`
   to create them (they join the queue).

5. **Save** writes back preserving the original header (CRS, scale, offset)
   and extra attributes; in project mode it saves per-tree files instead.

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
| `treecatalog.py` | default mode: memory-mapped tree-label grouping, neighbour load + write-back |
| `scene_ui.py` | tree table + scene controller for the default mode |
| `registry.py` | on-disk list of recently opened files/projects |
| `workspace.py` | project folders: copy an imported file, never touch the source |
| `startup_ui.py` | startup dialog: pick a recent entry, new project, or `--project` folder |
| `app.py` | `segfix` CLI entry point (default table mode / `--project`) |

## Tests

```bash
pytest        # core model, operations, and IO round-trip (no GUI needed)
```

## Notes & next steps

- "Delete" is modelled as **mark-as-noise** (points kept, label `-1`) so files
  round-trip 1:1; an optional "drop noise on export" can be added later.
