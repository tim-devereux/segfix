"""GlobalShiftDialog layout: regression coverage for a real bug — the shift
spinboxes and the Apply/Keep buttons visually overlapped when the dialog rode
on QMessageBox's private grid layout instead of an explicit one of its own.

The only Qt-dependent test in the suite; needs a QApplication, which PyQt6's
bundled "offscreen" platform plugin provides with no real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtCore import QPoint  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.shift_ui import GlobalShiftDialog  # noqa: E402

_MINS = np.array([204277.15, 7223248.32, -0.37])
_MAXS = np.array([204302.76, 7223264.30, 14.04])
_SUGGESTED = np.array([-204277.0, -7223248.0, 0.0])


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _laid_out(dialog):
    """Force a real layout pass so widget geometries are meaningful."""
    dialog.show()
    dialog.resize(500, dialog.sizeHint().height())
    QApplication.processEvents()
    dialog.layout().activate()
    QApplication.processEvents()
    return dialog


def test_shift_controls_do_not_overlap_the_button_row():
    dlg = _laid_out(GlobalShiftDialog(_MINS, _MAXS, _SUGGESTED))

    def top_in_dialog(widget):
        return widget.mapTo(dlg, QPoint(0, 0)).y()

    spin_bottom = max(top_in_dialog(s) + s.height() for s in dlg.spins)
    button_top = top_in_dialog(dlg.button_box)
    assert button_top >= spin_bottom, (
        f"button box (top={button_top}) overlaps the shift spinboxes "
        f"(bottom={spin_bottom})"
    )


def test_apply_button_returns_the_edited_shift():
    dlg = _laid_out(GlobalShiftDialog(_MINS, _MAXS, _SUGGESTED))
    dlg.spins[0].setValue(-100.0)
    dlg.apply_btn.click()
    assert dlg.shift() == (-100.0, _SUGGESTED[1], _SUGGESTED[2])


def test_keep_button_declines_the_shift():
    dlg = _laid_out(GlobalShiftDialog(_MINS, _MAXS, _SUGGESTED))
    dlg.keep_btn.click()
    assert dlg.shift() is None
