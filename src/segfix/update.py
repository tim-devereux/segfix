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
