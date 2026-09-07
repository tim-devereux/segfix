"""DownsampleDialog: the two answers it can give back to the catalog.

Like :mod:`tests.test_shift_ui`, this needs a QApplication, which PyQt6's
bundled "offscreen" platform plugin provides with no real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.density import DENSE_SPACING  # noqa: E402
from segfix.density_ui import DownsampleDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _dialog():
    dlg = DownsampleDialog(0.006, 128_000_000, DENSE_SPACING)
    dlg.show()
    QApplication.processEvents()
    return dlg


def test_downsample_button_returns_the_edited_voxel_size():
    dlg = _dialog()
    assert dlg.spin.value() == pytest.approx(DENSE_SPACING)
    dlg.spin.setValue(0.05)
    dlg.downsample_btn.click()
    assert dlg.voxel() == pytest.approx(0.05)


def test_keep_full_resolution_declines_downsampling():
    dlg = _dialog()
    dlg.keep_btn.click()
    assert dlg.voxel() is None
