"""segfix — fix instance segmentation of tree point clouds."""

from importlib.metadata import PackageNotFoundError, version

from .model import NOISE, UNASSIGNED, PointCloud

__all__ = ["PointCloud", "UNASSIGNED", "NOISE"]

try:
    # Single source of truth is pyproject.toml; read it back off the
    # installed distribution rather than repeating the number here, where
    # it silently goes stale the first time a release bumps only one of them.
    __version__ = version("segfix")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"
