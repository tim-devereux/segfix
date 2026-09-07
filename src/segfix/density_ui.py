"""The "very dense cloud" prompt: review decimated, or at full resolution.

A cloud with a point every few millimetres carries far more detail than
re-labelling a tree needs, and pays for it in memory, GPU upload and lasso
latency on every single interaction. When :mod:`segfix.density` measures a
spacing finer than :data:`~segfix.density.DENSE_SPACING`, this dialog offers
to work on one point per voxel instead.

Nothing is thrown away: the decimation is a working-set choice for the
session, and :meth:`segfix.treecatalog._BaseCatalog.save` interpolates the
edited labels back onto every full-resolution point in the file before
writing. Same shape as :mod:`segfix.shift_ui`, which asks its own
load-time question the same way.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QStyle,
    QVBoxLayout,
)


class DownsampleDialog(QDialog):
    """Reports the measured spacing and offers an editable voxel size."""

    def __init__(self, spacing: float, n_points: int, suggested: float,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dense point cloud")
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
            .pixmap(40, 40)
        )
        icon_label.setFixedWidth(48)
        top.addWidget(icon_label, 0)
        text = QLabel(
            f"This cloud has {n_points:,} points about {spacing * 100:.1f} cm "
            "apart. That is finer than segfix needs to fix a segmentation, "
            "and it slows down loading, drawing and lassoing.\n\n"
            "Downsample to one point per voxel for this session? Your edits "
            "are interpolated back onto every original point when you save, "
            "so the saved file keeps all of its points and all of its fields."
        )
        text.setWordWrap(True)
        top.addWidget(text, 1)
        layout.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel("Voxel size:"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.001, 5.0)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.005)
        self.spin.setSuffix(" m")
        self.spin.setValue(float(suggested))
        self.spin.setMinimumWidth(110)
        row.addWidget(self.spin)
        row.addStretch(1)
        layout.addLayout(row)

        self.button_box = QDialogButtonBox()
        self.downsample_btn = self.button_box.addButton(
            "Downsample", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.keep_btn = self.button_box.addButton(
            "Keep Full Resolution", QDialogButtonBox.ButtonRole.RejectRole
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.downsample_btn.setDefault(True)
        layout.addWidget(self.button_box)

        self.setMinimumWidth(460)

    def voxel(self) -> float | None:
        """The chosen voxel size, or ``None`` if the user kept every point.
        Call after ``exec()``."""
        if self.result() != QDialog.DialogCode.Accepted:
            return None
        return float(self.spin.value())


def prompt_downsample(
    parent, spacing: float, n_points: int, suggested: float
) -> float | None:
    """Show :class:`DownsampleDialog` and return the chosen voxel size, or
    ``None`` to review the cloud at full resolution.

    Matches :data:`segfix.treecatalog.DensityPrompt` — pass this (bound to a
    parent window) straight through as ``open_catalog``'s ``density_prompt``.
    """
    dlg = DownsampleDialog(spacing, n_points, suggested, parent)
    dlg.exec()
    return dlg.voxel()
