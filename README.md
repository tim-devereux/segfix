# segfix

A GUI tool to **fix the instance segmentation of tree point clouds**. Load a
segmented LiDAR cloud, see each tree in its own colour, and correct mistakes by
selecting points and merging / splitting / reassigning them — then save back to
LAS/LAZ or PLY.

Built on [napari](https://napari.org) for fast 3D rendering and point selection.

<img width="1920" height="1134" alt="image" src="https://github.com/user-attachments/assets/a3386292-249b-4866-93c9-8ca07f1dccd1" />

## Install

Requires Python 3.10–3.12 (napari/Qt do not yet ship wheels for 3.13+).

```bash
conda create -n segfix python=3.11
conda activate segfix
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
the copy, never the original source file.

Every project opened this way — or given directly on the command line, which
skips the dialog entirely and edits the given file in place as before — is
recorded in `~/.config/segfix/registry.json` (a plain JSON file, not a
database) so it shows up in the "Recent projects" list next time.

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
   and extra attributes.

## Layout

| File | Responsibility |
|------|----------------|
| `model.py` | `PointCloud` data + undo/redo (diff-based) |
| `io.py` | load/save PLY + LAS/LAZ, label-field detection |
| `operations.py` | pure, UI-agnostic label edits (merge/split/reassign/…) |
| `lasso.py` | 3D screen-space lasso: camera projection + polygon test |
| `viewer.py` | napari layer + label→colour mapping |
| `widgets.py` | Qt dock panel wiring selection → operations |
| `treecatalog.py` | default mode: memory-mapped tree-label grouping, neighbour load + write-back |
| `scene_ui.py` | tree table + scene controller for the default mode |
| `registry.py` | on-disk list of recently opened files/projects |
| `workspace.py` | project folders: copy an imported file, never touch the source |
| `startup_ui.py` | startup dialog: pick a recent entry or start a new project |
| `app.py` | `segfix` CLI entry point |

## Tests

```bash
pytest        # core model, operations, and IO round-trip (no GUI needed)
```

## Notes & next steps

- "Delete" is modelled as **mark-as-noise** (points kept, label `-1`) so files
  round-trip 1:1; an optional "drop noise on export" can be added later.
