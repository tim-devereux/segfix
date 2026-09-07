"""ProgressWindow: the bar the long open and save drive.

Like the other Qt tests here, this runs on PyQt6's bundled "offscreen"
platform plugin, with no real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.progress_ui import ProgressWindow, progress_window  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def test_report_moves_the_bar_and_names_the_stage():
    win = ProgressWindow("Opening cloud", "plot.las")
    win.show()

    win.report("Downsampling…", 0.25)
    assert win.stage_label.text() == "Downsampling…"
    quarter = win.bar.value()
    assert quarter == pytest.approx(win.bar.maximum() * 0.25, abs=1)

    win.report("Indexing trees…", 0.8)
    assert win.bar.value() > quarter
    win.close()


def test_report_clamps_a_fraction_outside_the_bar():
    """A phase weight that doesn't quite add up must not throw or wrap the
    bar round to empty."""
    win = ProgressWindow("Opening cloud")
    win.report("over", 1.4)
    assert win.bar.value() == win.bar.maximum()
    win.report("under", -0.2)
    assert win.bar.value() == win.bar.minimum()
    win.close()


def test_pause_hides_the_window_so_a_prompt_can_be_answered():
    win = ProgressWindow("Opening cloud")
    win.show()
    assert win.isVisible()

    win.pause()
    assert not win.isVisible()
    win.resume()
    assert win.isVisible()
    win.close()


def test_context_manager_closes_the_window_even_when_the_work_raises():
    """A failed load pops an error dialog; the bar must not still be sitting
    in front of it."""
    captured = {}
    with pytest.raises(ValueError):
        with progress_window(None, "Opening cloud", "broken.las") as win:
            captured["win"] = win
            win.report("Reading coordinates…", 0.1)
            raise ValueError("bad file")

    assert not captured["win"].isVisible()
