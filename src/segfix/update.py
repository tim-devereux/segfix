"""Update check for the git checkout segfix runs from.

segfix has no installer or PyPI package — the README's install path is a git
clone plus an editable `pip install -e .` into it. So "update" here just
means "pull the repo and reinstall it", and the check below shells out to
``git`` inside whichever repo the running package lives in, rather than
hitting a package index or the GitHub API. If the running copy isn't a git
checkout (e.g. vendored some other way) or there's no network, everything
here just returns ``None`` / raises for the caller to swallow — this is a
best-effort startup nicety, never something that should block opening a
project.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UpdateStatus:
    commits_behind: int
    repo_root: Path


def _repo_root() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return Path(out.stdout.strip())


def checkout_version() -> str | None:
    """``git describe`` for the checkout segfix runs from — e.g. ``0.3.2`` on
    a release tag, ``0.3.2-5-gabc1234`` a few commits past it, with a
    ``-dirty`` suffix when the working tree has uncommitted changes.
    ``None`` when segfix isn't running from a git checkout.

    ``segfix.__version__`` comes from the installed distribution's metadata,
    frozen at ``pip install`` time — for the README's editable git install
    that goes stale the moment ``pyproject.toml`` is bumped or the branch
    moves, until the next reinstall. This tracks the working tree instead,
    which is what actually runs.
    """
    root = _repo_root()
    if root is None:
        return None
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    tag = out.stdout.strip()
    if tag.startswith("v") and tag[1:2].isdigit():
        tag = tag[1:]  # normalise the "v0.3.2" tag style to match __version__
    return tag or None


def display_version() -> str:
    """Version string for UI chrome: the live checkout's ``git describe`` if
    segfix runs from a clone, otherwise the installed distribution version."""
    from . import __version__

    return checkout_version() or __version__


def check_for_update(timeout: float = 8.0) -> UpdateStatus | None:
    """Fetch the upstream remote and report how many commits the local
    checkout is behind it. ``None`` means "nothing to offer" — not a git
    checkout, no upstream configured, or the fetch failed (most commonly:
    offline). Never raises; meant to be called from a background thread on
    every startup.
    """
    root = _repo_root()
    if root is None:
        return None
    try:
        subprocess.run(
            ["git", "fetch", "--quiet"],
            cwd=root, capture_output=True, text=True, timeout=timeout, check=True,
        )
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        count = subprocess.run(
            ["git", "rev-list", "--count", f"HEAD..{upstream}"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    behind = int(count) if count.isdigit() else 0
    if behind <= 0:
        return None
    return UpdateStatus(commits_behind=behind, repo_root=root)


def apply_update(root: Path) -> str:
    """Pull the upstream branch and reinstall in place. Returns combined
    stdout from both steps for display; raises ``subprocess.CalledProcessError``
    if either step fails so the caller can show the user what went wrong.

    ``--ff-only`` refuses rather than merges/rebases if the local checkout
    has diverged (e.g. someone hacking on it directly) — silently rewriting
    history out from under a working copy would be worse than just failing.
    """
    pull = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=root, capture_output=True, text=True, timeout=60, check=True,
    )
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=root, capture_output=True, text=True, timeout=600, check=True,
    )
    return pull.stdout + install.stdout
