"""A small on-disk registry of recently opened files/projects, so segfix can
show a startup picker instead of requiring a CLI path every time.

A JSON file under the user's config directory — deliberately not a database;
this only ever holds a short list of paths, matching the JSON-sidecar
convention already used elsewhere in this app (``<cloud>.segfix.json`` for
review progress, ``segfix_project.json`` for a workspace manifest). Point
cloud data itself always stays in its native PLY files — the registry
only remembers where they are.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_ENTRIES = 20
KINDS = ("file", "workspace")


def registry_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(config_home).expanduser() / "segfix" / "registry.json"


def load_registry(registry_file: Path | None = None) -> list[dict]:
    """Known entries, most-recently-opened first.

    Entries whose path no longer exists on disk are silently dropped (and
    the pruned list re-saved), so the startup picker never offers a dead
    file/directory. ``registry_file`` overrides the on-disk location — used
    by tests so they never touch the real user config.
    """
    path = registry_file or registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    entries = data.get("entries", [])
    live = [e for e in entries if Path(e.get("path", "")).exists()]
    if len(live) != len(entries):
        _write(live, path)
    return sorted(live, key=lambda e: e.get("last_opened", ""), reverse=True)


def add_entry(path: str, kind: str, registry_file: Path | None = None) -> None:
    """Record ``path`` as just-opened, bumping it to the front if already
    known. Trims to :data:`MAX_ENTRIES`, dropping the oldest first.

    ``kind`` is one of :data:`KINDS`: ``"file"`` (a bare point cloud opened
    directly, no workspace) or ``"workspace"`` (a folder created by
    :func:`segfix.workspace.create_workspace` — ``path`` is the *folder*,
    not the data file inside it).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    target = registry_file or registry_path()
    entries = load_registry(target)
    resolved = str(Path(path).resolve())
    entries = [e for e in entries if e.get("path") != resolved]
    entries.insert(0, {
        "path": resolved,
        "type": kind,
        "last_opened": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _write(entries[:MAX_ENTRIES], target)


def describe_age(last_opened: str, *, now: float | None = None) -> str:
    """Human phrasing of when an entry was last opened, for the startup list.

    ``"just now"`` / ``"5 min ago"`` / ``"3 hr ago"`` / ``"yesterday"`` /
    ``"4 days ago"`` for the past week, then the plain ``YYYY-MM-DD`` date.
    Returns ``""`` if ``last_opened`` is missing or unparseable. The stored
    timestamps are local-time and naive (see :func:`add_entry`), so they are
    read back as local time here too.
    """
    if not last_opened:
        return ""
    try:
        struct = time.strptime(last_opened, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return ""
    delta = (time.time() if now is None else now) - time.mktime(struct)
    minutes = delta / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        n = int(minutes)
        return f"{n} min ago"
    hours = minutes / 60
    if hours < 24:
        n = int(hours)
        return f"{n} hr ago"
    days = int(hours / 24)
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return time.strftime("%Y-%m-%d", struct)


def _write(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=2))
