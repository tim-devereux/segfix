# segfix

A GUI tool to **fix the instance segmentation of tree point clouds**. Load a
segmented LiDAR cloud, see each tree in its own colour, and correct mistakes by
lassoing points and reassigning, splitting off, or dismissing them — then save
back to the PLY you started from.

Built on [napari](https://napari.org) for fast 3D rendering and point selection.

<img width="1920" height="1013" alt="image" src="https://github.com/user-attachments/assets/dc454e6e-93ad-45f9-aa20-a61d225c0a94" />

## Contributing

Feedback, issues, and PRs all welcome. For issues please use [GitHub issues](https://github.com/tim-devereux/segfix/issues) (not a personal message) so the community can benefit.
## Install

Requires Python 3.10–3.12 (napari/Qt do not yet ship wheels for 3.13+).

```bash
git clone https://github.com/tim-devereux/segfix.git
cd segfix
conda create -n segfix python=3.11
conda activate segfix
pip install -e .
```

## Run

```bash
segfix
# or generate a practice cloud with built-in segmentation errors first:
python scripts/make_sample.py sample.ply
python scripts/make_sample.py --format las sample.las   # arbor-shaped LAS
```

`segfix` will open a startup dialog. Double-click
a recent project to reopen it, or click **New Project…** to import a point cloud
file. Importing copies the file into a new project folder (created inside the
directory you pick, named after the source file) and opens that copy — edits are
always saved to the copy, never the original source file.

Every project opened this way is recorded in `~/.config/segfix/registry.json`
(a plain JSON file) so it shows up in the "Recent projects"
list next time — most-recently-opened at the top and preselected, each row
showing how long ago it was last opened.

The per-point tree ID field is auto-detected (`treeID`, `PredInstance`,
`label`, …); override with `--label-field NAME` if needed.

### Accepted formats

**Binary PLY** — RGB-segmented (raycloudtools) or with a label field — and
**LAS**. Both store their points as fixed-size records at a known offset, which
is what lets `treecatalog.py` memory-map a whole plot, read back just the points
of one tree, and on save patch only the label bytes that changed.

### raycloudtools output

[raycloudtools](https://github.com/csiro-robotics/raycloudtools)' `rayextract
trees` writes `<plot>_segmented.ply` — a binary PLY with no label column, each
point instead **coloured by tree** (`x y z time nx ny nz red green blue alpha`,
double xyz). segfix detects the RGB encoding, maps each distinct colour to a
tree, and treats pure black `(0, 0, 0)` as unsegmented. Import the `.ply`
directly; **Save** patches the colour bytes of points whose tree changed, in
place, so the file stays in exactly the format `rayextract` produced.

One caveat: noise (`X`) and unassigned points are **both** written back as black,
so once such a file is reloaded the two are indistinguishable (segfix's
`.segfix.json` sidecar still remembers which were noise for the current
project). New trees created during editing (`N`, splits) get a deterministic
colour derived from their id.

### arbor output

[arbor](https://github.com/r-lidar/arbor)'s pipeline (`arbor segment …`) writes
`<plot>_output/<plot>_segmented.laz` — a point cloud with a per-point `treeID`
Extra-Bytes column (`0` = unassigned). Import that `.laz` directly: segfix
decompresses it to a `.las` working copy in the project folder (the original
`.laz` is never touched), you fix the `treeID`s with the workflow below, and
**Save** patches the `.las` in place *and* re-compresses a fresh `.laz` beside
it for arbor to re-read.

One caveat: if a cloud's `treeID` column is an *unsigned* type, points you
dismiss as noise (`X`) are written back as `0` (unassigned) — segfix's own
`.segfix.json` sidecar still remembers they were noise, but a reader of the LAS
alone cannot tell noise from unassigned. (arbor writes a signed `treeID`, so
this does not apply to its output.)

## Editing workflow

Navigation matches CloudCompare: clouds open **Z-up**, **left-drag rotates,
right-drag pans, wheel zooms**.

Two tables stack in the right-hand panel. **All Trees** (top) lists every tree
in the file, with a `✓` once it has been reviewed — **double-click a row** to
load that tree plus its spatial neighbours into the 3D view. **Selected Tree +
Neighbours** (below) is the review queue for what's currently loaded: a Done
checkbox per tree, a 👁 column to hide one from the view, a **Fade** column to
ghost one (still shown, still selectable), and the Prev / Done buttons. Both
read the same `<cloud>.segfix.json` sidecar, written next to the working copy,
so a half-finished plot resumes where you left off.

1. Double-click a tree in **All Trees**. The camera flies to it and a
   wireframe box marks it. To declutter a crowded view, use the 👁 (hide) or
   **Fade** column in the lower table on specific trees — or **Hide others** /
   **Fade others** in the top-bar **View** group to do it to every loaded tree
   except the one under review. Fading keeps a tree visible as faint context
   and still lets the lasso grab its points; hiding removes it from both.
2. Inspect it. If it's correct, press **Space** — the tree is marked done,
   progress is saved, and the next unfinished tree in the loaded set becomes
   current. That's the loop.
3. If it needs fixing, **select** points with the lasso: press **L**, drag a
   freehand loop (Shift adds), **Esc** to go back to navigating. The tree
   under review is always the implicit target.

   | Key | Operation |
   |-----|-----------|
   | `Space` | Mark current tree done, jump to next unfinished |
   | `←` / `→` | Previous / next tree (without marking done) |
   | `L` | Lasso select |
   | `Ctrl+L` | Lasso, but only points already in the current tree — grabs a clean patch out of an overlapping crown |
   | `Esc` | Back to camera / navigation |
   | `A` | Add selection to the current tree (missing branches, unassigned canopy) |
   | `N` | Split selection off as a new tree (it joins the queue unreviewed) |
   | `U` | Unassign selection — or the whole current tree if nothing is selected |
   | `X` | Mark selection as noise — or the whole current tree if nothing is selected (dismiss a bush/wall in one key) |
   | `Delete` / `Backspace` | Same as `X` (mark noise) |
   | `H` | Show/hide the unassigned + noise points |
   | `C` | Cross section on/off |
   | `Shift+L` | Draw a lasso-section outline |
   | `Shift+C` | Lasso section on/off |
   | `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / Redo |
   | `Ctrl+S` | Save Project |

   To move stray points *to a neighbour* instead, lasso them and click one of
   the **→ id** buttons in the Current tree panel — one per tree within
   "reach" metres of this one. Clicking that neighbour's table row to make it
   current and pressing `A` does the same thing.

   To **merge** an over-segmented fragment back in, lasso the whole fragment
   and press `A`; there is no separate merge key.
4. For a crowded canopy, two tools in the top bar cut the view down. Both fold
   into the same visibility as the 👁 column, so hidden points are also
   unselectable and the lasso can't grab through them:
   - **Cross section (`C`)** — a slab along X, Y or Z, set with two sliders.
   - **Lasso section (`Shift+C`)** — same idea, but the kept region is an
     outline you draw (`Shift+L`, then drag). It's frozen into a point mask
     as you release, so the camera moves freely afterwards.

   Both reset when a new tree is loaded.
5. Leftover unassigned points may hide missed trees: they're loaded alongside
   every tree you open, so lasso one and press `N` to promote it to a tree of
   its own (it joins the queue unreviewed).
6. **Save** (`Ctrl+S`) writes to the project copy, never the original import.
   It patches only the points whose label changed, in place, so the header and
   every other column are untouched byte for byte.

## Layout

| File | Responsibility |
|------|----------------|
| `model.py` | `PointCloud` data + undo/redo (diff-based) |
| `io.py` | whole-cloud load/save (binary PLY, LAS/LAZ), label-field and RGB-segmentation detection |
| `operations.py` | pure, UI-agnostic label edits (reassign/split/unassign/noise) |
| `analysis.py` | which trees touch which, by sampled point distance (KD-tree) |
| `lasso.py` | 3D screen-space lasso: camera projection + polygon test |
| `viewer.py` | napari layer + label→colour mapping |
| `widgets.py` | Qt dock panel wiring selection → operations |
| `icons.py` | inline SVG icons for the panel buttons and window |
| `treecatalog.py` | default mode: memory-mapped tree-label grouping, neighbour load + write-back (`TreeCatalog` = PLY, `LasCatalog` = LAS, `open_catalog` picks) |
| `scene_ui.py` | tree table + scene controller for the default mode |
| `registry.py` | on-disk list of recently opened files/projects |
| `workspace.py` | project folders: copy (or decompress `.laz`→`.las`) an imported file, never touch the source |
| `startup_ui.py` | startup dialog: pick a recent entry or start a new project |
| `app.py` | `segfix` CLI entry point |

## Tests

```bash
pytest        # core model, operations, and IO round-trip (no GUI needed)
```
