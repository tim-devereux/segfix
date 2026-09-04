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
from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout


class GlobalShiftDialog(QMessageBox):
    """Reports the cloud's coordinate range and offers an editable shift.

    A ``QMessageBox`` subclass (not a plain ``QDialog``) so it gets a
    sensible icon and default "Yes"-shaped button styling for free; the shift
    spinboxes are inserted into its layout alongside the usual text.
    """

    def __init__(self, mins: np.ndarray, maxs: np.ndarray,
                 suggested: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Large coordinates")
        self.setIcon(QMessageBox.Icon.Warning)
        span = maxs - mins
        self.setText(
            "This cloud's coordinates are large — up to "
            f"{float(np.abs(np.concatenate([mins, maxs])).max()):,.1f} — "
            "which can lose sub-metre precision once stored as the 32-bit "
            "floats segfix (and the GPU) use.\n\n"
            f"Extent: {span[0]:,.2f} × {span[1]:,.2f} × {span[2]:,.2f} m\n\n"
            "Apply a global shift to bring the cloud near the origin? The "
            "shift is remembered for this session only — nothing is written "
            "back to the file."
        )

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
        holder = QVBoxLayout()
        holder.addWidget(QLabel("Shift to apply (added to every coordinate):"))
        holder.addLayout(row)
        # QMessageBox lays its own contents out in a QGridLayout; append our
        # extra row beneath the standard text/icon area and above the buttons.
        grid = self.layout()
        grid.addLayout(holder, grid.rowCount() - 1, 0, 1, grid.columnCount())

        self.apply_btn = self.addButton("Apply Shift", QMessageBox.ButtonRole.AcceptRole)
        self.keep_btn = self.addButton(
            "Keep Original Coordinates", QMessageBox.ButtonRole.RejectRole
        )
        self.setDefaultButton(self.apply_btn)

    def shift(self) -> tuple[float, float, float] | None:
        """The chosen shift, or ``None`` if the user declined it. Call after
        ``exec()``."""
        if self.clickedButton() is not self.apply_btn:
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
