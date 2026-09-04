"""LassoTool path recording: a real bug on Windows, where mouse-move events
arrive far more frequently (and far less coalesced) than on Linux/X11 --
without decimation, that turned an ordinary drag into a path of thousands of
near-duplicate vertices, and _finish()'s polygon test is O(vertices x
loaded points), so completing the lasso could block the app for seconds.

Lightweight Qt test (no GL rendering needed, unlike test_edl-style checks):
LassoTool only touches ``view.native`` while recording a path, so a bare
QWidget stands in for the view. Same offscreen-platform default as
test_shift_ui.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from segfix.lasso import LassoTool  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _event(x, y, button=None):
    return types.SimpleNamespace(pos=(x, y), button=button, modifiers=(), handled=False)


def _armed_tool():
    # LassoTool.__init__ stashes view.canvas but _on_press/_on_move (all this
    # exercises) never touch it -- set_armed()/_connect() are what would.
    view = types.SimpleNamespace(native=QWidget(), canvas=types.SimpleNamespace())
    tool = LassoTool(view, on_select=lambda *a, **k: None)
    tool._armed = True  # bypass set_armed(): that needs view.canvas/status too
    return tool


def test_dense_high_frequency_moves_are_decimated():
    """The Windows scenario: hundreds of mouse-move events for a drag that
    barely covers any screen distance."""
    tool = _armed_tool()
    tool._on_press(_event(0.0, 0.0, button=1))
    for i in range(1, 1501):
        tool._on_move(_event(i * 0.1, 0.0))  # 150px of travel, 1500 events
    assert len(tool._path) < 100, (
        f"expected the path to stay small, got {len(tool._path)} vertices"
    )


def test_normal_drag_still_records_real_movement():
    """A drag that actually covers ground must still produce an outline with
    plenty of vertices -- decimation must not flatten a real lasso."""
    tool = _armed_tool()
    tool._on_press(_event(0.0, 0.0, button=1))
    for i in range(1, 51):
        tool._on_move(_event(i * 20.0, 0.0))  # 50 moves, 20px apart
    assert len(tool._path) >= 45, (
        f"expected most large steps to be recorded, got {len(tool._path)}"
    )


def test_sub_threshold_move_does_not_grow_the_path():
    tool = _armed_tool()
    tool._on_press(_event(10.0, 10.0, button=1))
    before = len(tool._path)
    tool._on_move(_event(10.0 + LassoTool.MIN_SEGMENT_PX / 2, 10.0))
    assert len(tool._path) == before


def test_move_past_threshold_grows_the_path():
    tool = _armed_tool()
    tool._on_press(_event(10.0, 10.0, button=1))
    before = len(tool._path)
    tool._on_move(_event(10.0 + LassoTool.MIN_SEGMENT_PX * 2, 10.0))
    assert len(tool._path) == before + 1
