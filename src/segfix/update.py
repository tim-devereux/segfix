"""Update check: new PyPI releases for an installed copy, new commits for a
git checkout.

segfix reaches people two ways, and "update" means something different for
each:

* **Installed from PyPI** (``pip install segfix``, what the README says): the
  check asks PyPI's JSON API for the newest release, and updating is
  ``pip install --upgrade segfix==<that version>`` into the running
  interpreter's environment. Pinned to the version the banner showed, so what
  gets installed is what was offered, not whatever appeared since.
* **Running from a clone** (``pip install -e .``, for development): the check
  counts commits the checkout is behind its upstream, and updating is
  ``git pull --ff-only`` plus a reinstall. A pip upgrade here would replace
  the editable checkout with a plain copy, silently cutting the working tree
  out of what runs.

Everything is best-effort: offline, no upstream, a pre-release installed, an
unreadable response — the check returns ``None`` and the banner stays hidden.
This is a startup nicety, never something that should block opening a
project.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PYPI_JSON_URL = "https://pypi.org/pypi/segfix/json"


@dataclass
class UpdateStatus:
    """An update worth offering. ``source`` is ``"pypi"`` (``latest`` is the
    release to install) or ``"git"`` (``commits_behind`` of ``repo_root``)."""

    source: str
    latest: str | None = None
    commits_behind: int = 0
    repo_root: Path | None = None

    def describe(self) -> str:
        """The banner text for the startup dialog."""
        if self.source == "pypi":
            return f"Update available: segfix {self.latest}."
        n = self.commits_behind
        return f"Update available ({n} commit{'s' if n != 1 else ''} behind)."


# -- which kind of install is this? -------------------------------------------
def _repo_root(package_dir: Path | None = None) -> Path | None:
    """The segfix checkout this package runs from, or ``None`` if it isn't
    running from one.

    Asking git for "the repo this directory is in" is not enough on its own:
    a PyPI install into a virtualenv that happens to sit inside some *other*
    git project (a ``.venv`` in your own analysis repo, say) is "in a repo"
    too — and treating that as segfix's checkout would offer to ``git pull``
    and ``pip install -e`` somebody else's project. So the package also has
    to be exactly where a segfix checkout keeps it, ``<repo>/src/segfix``.
    """
    here = (package_dir or Path(__file__).parent).resolve()
    if here.parent.name != "src":
        return None  # installed layout (site-packages): no need to ask git
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here, capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    root = Path(out.stdout.strip()).resolve()
    if here != root / "src" / "segfix":
        return None
    return root


def checkout_version() -> str | None:
    """``git describe`` for the checkout segfix runs from — e.g. ``0.3.2`` on
    a release tag, ``0.3.2-5-gabc1234`` a few commits past it, with a
    ``-dirty`` suffix when the working tree has uncommitted changes.
    ``None`` when segfix isn't running from a git checkout.

    ``segfix.__version__`` comes from the installed distribution's metadata,
    frozen at ``pip install`` time — for an editable git install that goes
    stale the moment ``pyproject.toml`` is bumped or the branch moves, until
    the next reinstall. This tracks the working tree instead, which is what
    actually runs.
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
    return checkout_version() or _installed_version()


def _installed_version() -> str:
    from . import __version__

    return __version__


# -- PyPI ---------------------------------------------------------------------
def release_tuple(version: str) -> tuple[int, ...] | None:
    """``(0, 6, 0)`` for ``"0.6.0"``; ``None`` for anything that isn't a
    plain final release — pre-releases, dev builds (``0.0.0.dev0`` is what an
    uninstalled checkout reports) and local versions are never offered as an
    update, nor compared against as the installed one."""
    parts = version.strip().split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def is_newer(candidate: str, installed: str) -> bool:
    """Whether release ``candidate`` is newer than ``installed``; False when
    either isn't a plain release (see :func:`release_tuple`)."""
    a, b = release_tuple(candidate), release_tuple(installed)
    if a is None or b is None:
        return False
    width = max(len(a), len(b))  # "0.6" == "0.6.0"
    return a + (0,) * (width - len(a)) > b + (0,) * (width - len(b))


def latest_release(data: dict) -> str | None:
    """The newest installable final release in a PyPI JSON API response.

    Picked from ``releases`` rather than trusting ``info.version`` alone, so
    a yanked release (withdrawn after upload) or one with no files is never
    offered; ``info.version`` is the fallback if ``releases`` is absent.
    """
    best: tuple[tuple[int, ...], str] | None = None
    for version, files in (data.get("releases") or {}).items():
        key = release_tuple(version)
        if key is None or not files or all(f.get("yanked") for f in files):
            continue
        if best is None or key > best[0]:
            best = (key, version)
    if best is not None:
        return best[1]
    fallback = (data.get("info") or {}).get("version")
    return fallback if fallback and release_tuple(fallback) else None


def _fetch_pypi_json(timeout: float) -> dict:
    import urllib.request

    req = urllib.request.Request(
        PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": f"segfix/{_installed_version()} update-check",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _check_pypi(timeout: float) -> UpdateStatus | None:
    installed = _installed_version()
    if release_tuple(installed) is None:
        return None  # a dev/pre-release build: nothing sensible to compare
    try:
        latest = latest_release(_fetch_pypi_json(timeout))
    except Exception:
        # Offline, DNS, TLS, a proxy page instead of JSON, an unexpected
        # shape... any of them just means "no update to offer" — this runs
        # on every startup and must never be the reason segfix won't open.
        return None
    if latest is None or not is_newer(latest, installed):
        return None
    return UpdateStatus(source="pypi", latest=latest)


# -- git ----------------------------------------------------------------------
def _check_git(root: Path, timeout: float) -> UpdateStatus | None:
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
    return UpdateStatus(source="git", commits_behind=behind, repo_root=root)


# -- entry points -------------------------------------------------------------
def check_for_update(timeout: float = 8.0) -> UpdateStatus | None:
    """An update worth offering, or ``None``: a newer PyPI release for an
    installed copy, new upstream commits for a checkout (see the module
    docstring). Never raises; meant to be called from a background thread
    on every startup."""
    root = _repo_root()
    if root is not None:
        return _check_git(root, timeout)
    return _check_pypi(timeout)


def apply_update(status: UpdateStatus) -> str:
    """Install the update ``status`` describes. Returns the tools' stdout for
    display; raises ``subprocess.CalledProcessError`` if a step fails, so the
    caller can show the user what went wrong.

    PyPI: pip into the running interpreter (``sys.executable -m pip``, so a
    conda env or virtualenv upgrades itself, not whatever ``pip`` is first on
    PATH). Git: ``--ff-only`` refuses rather than merges/rebases if the local
    checkout has diverged (e.g. someone hacking on it directly) — silently
    rewriting history out from under a working copy would be worse than just
    failing.
    """
    if status.source == "pypi":
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             f"segfix=={status.latest}"],
            capture_output=True, text=True, timeout=600, check=True,
        )
        return install.stdout
    pull = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=status.repo_root, capture_output=True, text=True, timeout=60,
        check=True,
    )
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        cwd=status.repo_root, capture_output=True, text=True, timeout=600,
        check=True,
    )
    return pull.stdout + install.stdout
