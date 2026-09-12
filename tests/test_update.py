"""The updater: PyPI releases for an installed copy, git for a checkout.

No network and no real installs: PyPI's response and subprocess are faked.
The one real git call is in the wrong-repo tests, against a throwaway repo in
tmp_path, because "is this directory segfix's checkout?" is exactly the
question those tests are about.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from segfix import update
from segfix.update import UpdateStatus, is_newer, latest_release, release_tuple


def _files(yanked=False):
    return [{"filename": "segfix.whl", "yanked": yanked}]


# -- version handling ----------------------------------------------------------
@pytest.mark.parametrize("version, expected", [
    ("0.6.0", (0, 6, 0)),
    ("1.10.2", (1, 10, 2)),
    ("0.0.0.dev0", None),   # what an uninstalled checkout reports
    ("0.7.0rc1", None),
    ("0.7.0b2", None),
    ("0.6.0+local", None),
    ("", None),
])
def test_release_tuple_accepts_only_plain_releases(version, expected):
    assert release_tuple(version) == expected


def test_is_newer_compares_numerically_not_as_text():
    assert is_newer("0.10.0", "0.9.0")   # "0.10.0" < "0.9.0" as strings
    assert is_newer("0.6.1", "0.6.0")
    assert not is_newer("0.6.0", "0.6.0")
    assert not is_newer("0.5.0", "0.6.0")
    assert not is_newer("0.6", "0.6.0")  # same release, different spelling
    assert not is_newer("0.7.0", "0.0.0.dev0")  # never offer to a dev build


def test_latest_release_skips_yanked_prereleases_and_empty_releases():
    data = {
        "info": {"version": "0.9.0"},
        "releases": {
            "0.5.0": _files(),
            "0.6.0": _files(),
            "0.10.0rc1": _files(),        # pre-release
            "0.9.0": _files(yanked=True),  # withdrawn after upload
            "0.8.0": [],                   # no files to install
        },
    }
    assert latest_release(data) == "0.6.0"


def test_latest_release_falls_back_to_info_version():
    assert latest_release({"info": {"version": "0.6.0"}}) == "0.6.0"
    assert latest_release({"info": {"version": "0.7.0rc1"}}) is None
    assert latest_release({}) is None


# -- the PyPI check ---------------------------------------------------------
@pytest.fixture
def installed_from_pypi(monkeypatch):
    """No checkout, version 0.6.0 installed; returns a setter for what the
    fake PyPI answers."""
    monkeypatch.setattr(update, "_repo_root", lambda *a: None)
    monkeypatch.setattr(update, "_installed_version", lambda: "0.6.0")
    answer = {}

    def fetch(timeout):
        if isinstance(answer.get("value"), Exception):
            raise answer["value"]
        return answer["value"]

    monkeypatch.setattr(update, "_fetch_pypi_json", fetch)
    return lambda value: answer.__setitem__("value", value)


def test_a_newer_pypi_release_is_offered(installed_from_pypi):
    installed_from_pypi({"releases": {"0.6.0": _files(), "0.7.0": _files()}})
    status = update.check_for_update()
    assert status == UpdateStatus(source="pypi", latest="0.7.0")
    assert status.describe() == "Update available: segfix 0.7.0."


def test_nothing_is_offered_when_already_on_the_latest(installed_from_pypi):
    installed_from_pypi({"releases": {"0.5.0": _files(), "0.6.0": _files()}})
    assert update.check_for_update() is None


def test_offline_or_garbage_means_no_offer_not_an_exception(installed_from_pypi):
    import urllib.error

    installed_from_pypi(urllib.error.URLError("no route to host"))
    assert update.check_for_update() is None
    installed_from_pypi(ValueError("proxy login page, not JSON"))
    assert update.check_for_update() is None
    installed_from_pypi({"unexpected": "shape"})
    assert update.check_for_update() is None


def test_a_dev_build_is_never_offered_a_release(monkeypatch):
    monkeypatch.setattr(update, "_repo_root", lambda *a: None)
    monkeypatch.setattr(update, "_installed_version", lambda: "0.0.0.dev0")
    monkeypatch.setattr(
        update, "_fetch_pypi_json",
        lambda timeout: pytest.fail("should not even ask PyPI"),
    )
    assert update.check_for_update() is None


def test_a_checkout_follows_git_not_pypi(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_repo_root", lambda *a: tmp_path)
    monkeypatch.setattr(
        update, "_fetch_pypi_json",
        lambda timeout: pytest.fail("a checkout must not ask PyPI"),
    )
    monkeypatch.setattr(
        update, "_check_git",
        lambda root, timeout: UpdateStatus(
            source="git", commits_behind=3, repo_root=root
        ),
    )
    status = update.check_for_update()
    assert status.source == "git" and status.repo_root == tmp_path
    assert status.describe() == "Update available (3 commits behind)."


# -- applying -----------------------------------------------------------------
@pytest.fixture
def recorded_runs(monkeypatch):
    runs = []

    def fake_run(cmd, **kwargs):
        runs.append((cmd, kwargs.get("cwd")))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    return runs


def test_pypi_update_pins_the_offered_version_in_this_interpreter(recorded_runs):
    update.apply_update(UpdateStatus(source="pypi", latest="0.7.0"))
    assert recorded_runs == [(
        [sys.executable, "-m", "pip", "install", "--upgrade", "segfix==0.7.0"],
        None,
    )]


def test_git_update_pulls_then_reinstalls_editable(recorded_runs, tmp_path):
    update.apply_update(
        UpdateStatus(source="git", commits_behind=2, repo_root=tmp_path)
    )
    assert [cmd for cmd, _ in recorded_runs] == [
        ["git", "pull", "--ff-only"],
        [sys.executable, "-m", "pip", "install", "-e", "."],
    ]
    assert all(cwd == tmp_path for _, cwd in recorded_runs)


# -- is this segfix's own checkout? ------------------------------------------
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git_repo(path):
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


@needs_git
def test_a_segfix_checkout_is_recognised(tmp_path):
    repo = _git_repo(tmp_path / "segfix")
    pkg = repo / "src" / "segfix"
    pkg.mkdir(parents=True)
    assert update._repo_root(pkg) == repo.resolve()


@needs_git
def test_a_pypi_install_inside_someone_elses_repo_is_not_a_checkout(tmp_path):
    """The hazard: a venv living inside the user's own git project. Git
    happily reports that project as "the repo" -- the updater must not take
    it for segfix's checkout and offer to pull and reinstall it."""
    repo = _git_repo(tmp_path / "my-analysis")
    pkg = repo / ".venv" / "lib" / "python3.11" / "site-packages" / "segfix"
    pkg.mkdir(parents=True)
    assert update._repo_root(pkg) is None


@needs_git
def test_a_src_folder_in_someone_elses_repo_is_not_a_checkout(tmp_path):
    """Same, for the one layout the cheap pre-check can't rule out."""
    repo = _git_repo(tmp_path / "other")
    pkg = repo / "vendor" / "src" / "segfix"
    pkg.mkdir(parents=True)
    assert update._repo_root(pkg) is None
