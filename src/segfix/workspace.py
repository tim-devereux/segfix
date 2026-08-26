"""A workspace is a folder holding a private working copy of an imported
point cloud, plus a small manifest recording where it came from. All edits
happen on the copy — the original source file is never opened for writing.

Distinct from ``--project DIR`` (a directory of many per-tree PLY files, the
CloudCompare-plugin-style workflow — see :mod:`project`): a workspace wraps
exactly one imported file for the default single-cloud/scene-mode workflow.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

MANIFEST_NAME = "segfix_project.json"


def is_workspace(path: str | Path) -> bool:
    """Whether ``path`` is a folder created by :func:`create_workspace`."""
    return (Path(path) / MANIFEST_NAME).exists()


def data_file(workspace_dir: str | Path) -> Path:
    """The working copy's path inside a workspace folder."""
    manifest = json.loads((Path(workspace_dir) / MANIFEST_NAME).read_text())
    return Path(workspace_dir) / manifest["data_file"]


def source_file(workspace_dir: str | Path) -> str:
    """The original file this workspace's copy was imported from."""
    manifest = json.loads((Path(workspace_dir) / MANIFEST_NAME).read_text())
    return manifest["source"]


def create_workspace(source: str | Path, workspace_dir: str | Path) -> Path:
    """Copy ``source`` into a new ``workspace_dir``, write a manifest, and
    return the path to the working copy.

    Raises ``FileExistsError`` if ``workspace_dir`` already exists and is
    non-empty, so callers needing an available name (e.g. the startup
    dialog, auto-naming from the source's filename) can retry with a
    different one rather than silently mixing two projects together.
    """
    source = Path(source)
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and any(workspace_dir.iterdir()):
        raise FileExistsError(f"{workspace_dir} already exists and is not empty")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    dest = workspace_dir / source.name
    shutil.copyfile(source, dest)
    manifest = {
        "source": str(source.resolve()),
        "data_file": source.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (workspace_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return dest
