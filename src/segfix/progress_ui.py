"""A progress window for the two operations that block long enough to look
like a freeze: opening a cloud, and saving one.

Opening a big plot is tens of seconds of numpy — decoding coordinates and
labels, measuring density, decimating, sorting the label index (about 22
seconds all told for a 39-million-point PLY) — and until now all of it
happened behind a single status-bar line, with a maximized window that
painted nothing. This shows what the load is actually doing and how far
through it is.

The work runs on the GUI thread, so the bar advances *between* phases rather
than smoothly through them: a phase that is one numpy call cannot repaint
while it runs, whoever asks. Naming each phase is what makes the wait
legible — "Downsampling" sitting there for twelve seconds reads as work,
where a frozen window reads as a crash.
"""

from __future__ import annotations

from contextlib import contextmanager

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

_BAR_STEPS = 1000  # bar resolution; fractions are 0..1


class ProgressWindow(QDialog):
    """Title, a line of detail, and a bar. No cancel button.

    Cancelling is deliberately not offered: the phases are single numpy
    calls, so a click could only ever be noticed once the phase it
    interrupted had finished anyway — a button that does nothing for twelve
    seconds is worse than no button.
    """

    def __init__(self, title: str, detail: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # No close button: there is nothing sensible to do with a half-built
        # catalog, and the window closes itself the moment the work is done.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.detail_label = QLabel(detail)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: gray;")
        layout.addWidget(self.detail_label)

        self.stage_label = QLabel("Starting…")
        layout.addWidget(self.stage_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, _BAR_STEPS)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.setMinimumWidth(420)

    def report(self, message: str, fraction: float) -> None:
        """Show ``message`` at ``fraction`` (0..1) and paint it now.

        ``processEvents`` rather than a return to the event loop, because the
        caller is a blocking load: control does not come back here until the
        whole thing is finished, so anything merely queued would never be
        drawn. See :func:`segfix.viewer.busy`, which does the same for the
        status bar.
        """
        self.stage_label.setText(message)
        self.bar.setValue(int(max(0.0, min(1.0, fraction)) * _BAR_STEPS))
        QApplication.processEvents()

    # Called by prompts that have to interrupt the work to ask a question
    # (the global-shift and downsample dialogs both run from inside the
    # load): a progress bar frozen at 10% behind a question looks stuck.
    def pause(self) -> None:
        self.hide()
        QApplication.processEvents()

    def resume(self) -> None:
        self.show()
        QApplication.processEvents()


@contextmanager
def progress_window(parent, title: str, detail: str = ""):
    """Show a :class:`ProgressWindow` for the duration of the block.

    Yields the window itself, whose ``report(message, fraction)`` matches
    :data:`segfix.treecatalog.ProgressFn` — pass it straight to
    ``open_catalog`` or ``save``. Closes on the way out, including when the
    work raises, so a failed load can't leave the bar on screen in front of
    the error dialog.
    """
    win = ProgressWindow(title, detail, parent)
    win.show()
    QApplication.processEvents()
    try:
        yield win
    finally:
        win.close()
        QApplication.processEvents()
