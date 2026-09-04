"""CloudCompare-style "Global Shift" prompt.

Point clouds georeferenced in a global system (UTM, State Plane, ...) can
have coordinates in the millions while the tree structure that matters is
sub-metre. ``segfix`` stores coordinates as float32 for the same reasons
CloudCompare does (memory, GPU upload), and float32 only carries ~7
significant digits — a UTM northing around 7,000,000 already only keeps
~1m precision once cast down. The fix, same as CloudCompare's, is a one-time
"global shift" applied on load: subtract (add, here) a large round offset so
the working coordinates stay small and full sub-metre precision survives the
cast. See :mod:`segfix.treecatalog` for where this is actually applied —
this module is just the dialog that asks.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QStyle,
    QVBoxLayout,
)


class GlobalShiftDialog(QDialog):
    """Reports the cloud's coordinate range and offers an editable shift.

    A plain ``QDialog`` with its own explicit layout — earlier this rode on
    ``QMessageBox``'s internal (private, undocumented) grid layout to get a
    free icon and button styling, but inserting an extra row into that grid
    landed on top of the button row instead of above it, so the shift
    controls and the buttons overlapped. Building the layout by hand avoids
    depending on QMessageBox's internals at all.
    """

    def __init__(self, mins: np.ndarray, maxs: np.ndarray,
                 suggested: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Large coordinates")
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(40, 40)
        )
        icon_label.setFixedWidth(48)
        top.addWidget(icon_label, 0)
        span = maxs - mins
        text = QLabel(
            "This cloud's coordinates are large, up to "
            f"{float(np.abs(np.concatenate([mins, maxs])).max()):,.1f}, "
            "which can lose sub-metre precision once stored as the 32-bit "
            "floats segfix (and the GPU) use.\n\n"
            f"Extent: {span[0]:,.2f} × {span[1]:,.2f} × {span[2]:,.2f} m\n\n"
            "Apply a global shift to bring the cloud near the origin? The "
            "shift is remembered for this session only; nothing is written "
            "back to the file."
        )
        text.setWordWrap(True)
        top.addWidget(text, 1)
        layout.addLayout(top)

        layout.addWidget(QLabel("Shift to apply (added to every coordinate):"))
        row = QHBoxLayout()
        self.spins = []
        for label, value in zip("XYZ", suggested.tolist()):
            row.addWidget(QLabel(f"{label}:"))
            spin = QDoubleSpinBox()
            spin.setRange(-1.0e9, 1.0e9)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setMinimumWidth(110)
            row.addWidget(spin)
            self.spins.append(spin)
        layout.addLayout(row)

        self.button_box = QDialogButtonBox()
        self.apply_btn = self.button_box.addButton(
            "Apply Shift", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.keep_btn = self.button_box.addButton(
            "Keep Original Coordinates", QDialogButtonBox.ButtonRole.RejectRole
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.apply_btn.setDefault(True)
        layout.addWidget(self.button_box)

        self.setMinimumWidth(440)

    def shift(self) -> tuple[float, float, float] | None:
        """The chosen shift, or ``None`` if the user declined it. Call after
        ``exec()``."""
        if self.result() != QDialog.DialogCode.Accepted:
            return None
        return tuple(s.value() for s in self.spins)


def prompt_global_shift(
    parent, mins: np.ndarray, maxs: np.ndarray, suggested: np.ndarray
) -> tuple[float, float, float] | None:
    """Show :class:`GlobalShiftDialog` and return the chosen shift, or
    ``None`` to load the cloud with its original coordinates.

    Matches :data:`segfix.treecatalog.ShiftPrompt` — pass this (bound to a
    parent window) straight through as ``open_catalog``'s ``shift_prompt``.
    """
    dlg = GlobalShiftDialog(mins, maxs, suggested, parent)
    dlg.exec()
    return dlg.shift()
