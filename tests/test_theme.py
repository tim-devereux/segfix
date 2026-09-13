"""Theme switching reaches widgets that carry their own stylesheet.

Regression coverage for a real bug: the top bar's highlighted mode buttons
(Lasso, Cluster, Draw ...) each have a ``QPushButton:checked`` stylesheet,
and after a theme switch they kept the *previous* theme's colours -- dark
buttons in the light theme, light ones after switching back -- while plain
buttons beside them switched correctly.

Offscreen Qt, like the other widget tests; no GL needed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget  # noqa: E402

from segfix import theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _background(button) -> str:
    QApplication.processEvents()
    img = button.grab().toImage()
    return img.pixelColor(6, img.height() // 2).name()


def test_styled_buttons_follow_a_theme_switch_in_both_directions(monkeypatch):
    monkeypatch.setattr(theme, "save", lambda mode: None)  # leave QSettings alone
    app = QApplication.instance()
    host = QWidget()
    row = QHBoxLayout(host)
    plain = QPushButton("Slab…")
    styled = QPushButton("Lasso (L)")
    styled.setCheckable(True)
    # The same shape of stylesheet the real mode buttons carry: it only
    # styles the checked state, so an unchecked button draws in the palette.
    styled.setStyleSheet(
        "QPushButton:checked { background: #7a6a20; color: #ffe066; }"
    )
    row.addWidget(plain)
    row.addWidget(styled)
    host.show()

    theme.apply(app, "dark")
    assert _background(styled) == _background(plain)
    for mode in ("light", "dark", "light", "dark"):
        theme.set_mode(app, mode)
        assert _background(styled) == _background(plain), (
            f"stylesheet button kept the previous theme after switching to {mode}"
        )
    host.close()
