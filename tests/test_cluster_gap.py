"""The cluster tool's adjustable gap, and its live re-apply.

One tree in two blobs, 5cm apart, sampled every 1cm: at the default gap
(4x spacing = 4cm) a click takes one blob; loosen it past 5cm and the same
click takes both. The view is a stand-in, as in test_lasso.py -- the
controller and ClusterTool only need its coords/selection/status, never GL.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import numpy as np
import pytest

pytest.importorskip("qtpy")

from segfix.model import PointCloud  # noqa: E402
from segfix.widgets import (  # noqa: E402
    CLUSTER_GAP_FACTORS,
    DEFAULT_CLUSTER_GAP_FACTOR,
    SegFixController,
)

_SPACING = 0.01
_BLOB = 20  # points per blob
_A = np.column_stack([np.arange(_BLOB) * _SPACING, np.zeros(_BLOB), np.zeros(_BLOB)])
# Blob B starts 5cm past A's last point: bridged at 6x spacing, not at 4x.
_B = _A + [(_BLOB - 1) * _SPACING + 0.05, 0.0, 0.0]
_OTHER = np.array([[5.0, 5.0, 5.0]])  # a different tree, far away

A_IDX = set(range(_BLOB))
B_IDX = set(range(_BLOB, 2 * _BLOB))
OTHER_IDX = 2 * _BLOB


def _controller(factor=4.0):
    """A controller over the two-blob tree. ``factor`` pins the gap (4x by
    default: at the 1x default even blob A falls apart, since a median
    spacing only links about half the pairs); ``None`` leaves the default."""
    coords = np.vstack([_A, _B, _OTHER]).astype(np.float32)
    labels = np.array([1] * (2 * _BLOB) + [2], dtype=np.int32)
    cloud = PointCloud(coords=coords, labels=labels)
    view = types.SimpleNamespace(
        canvas=types.SimpleNamespace(),
        native=None,
        coords=coords,
        shown=np.ones(len(coords), dtype=bool),
        selected=set(),
        status="",
        pick_point=lambda xy: 0,  # every click lands on blob A's first point
    )
    ctrl = SegFixController(view, cloud)
    # Bypass set_armed(): it would hook real canvas events.
    ctrl.cluster._armed = True
    if factor is not None:
        ctrl.set_cluster_gap_factor(factor)
    return ctrl, view


def _click(ctrl, additive=False, xy=(10.0, 10.0)):
    ctrl.cluster._select_at(xy, additive)


def _step_looser(ctrl):
    """What the panel wires on_repeat to (step_cluster_gap(1) moving the
    slider), minus the slider: one notch up CLUSTER_GAP_FACTORS."""
    def step():
        i = CLUSTER_GAP_FACTORS.index(ctrl.cluster_gap_factor)
        ctrl.set_cluster_gap_factor(
            CLUSTER_GAP_FACTORS[min(i + 1, len(CLUSTER_GAP_FACTORS) - 1)]
        )
    return step


def test_the_default_gap_is_one_point_spacing():
    ctrl, _ = _controller(factor=None)
    assert ctrl.cluster_gap_factor == DEFAULT_CLUSTER_GAP_FACTOR == 1.0
    assert ctrl.cluster_gap == pytest.approx(_SPACING, rel=0.05)


def test_the_default_first_click_is_a_small_seed():
    """What 1x means in practice: linking only pairs no further apart than
    the *median* spacing leaves most of a blob unconnected, so the first
    click takes a sliver and the repeat clicks do the growing."""
    ctrl, view = _controller(factor=None)
    _click(ctrl)
    assert 0 < len(view.selected) < _BLOB


def test_at_4x_a_click_takes_one_blob():
    ctrl, view = _controller()
    _click(ctrl)
    assert view.selected == A_IDX


def test_loosening_the_gap_grows_the_last_click_live():
    ctrl, view = _controller()
    _click(ctrl)

    assert ctrl.set_cluster_gap_factor(6.0) is True
    # No second click: the patch already on screen took in blob B.
    assert view.selected == A_IDX | B_IDX


def test_tightening_again_really_shrinks_it():
    """reapply() rebuilds from the pre-click selection, not the current
    one -- otherwise a patch could grow but never shrink back."""
    ctrl, view = _controller()
    _click(ctrl)
    ctrl.set_cluster_gap_factor(6.0)
    ctrl.set_cluster_gap_factor(3.0)
    assert view.selected == A_IDX


def test_reapply_keeps_what_a_shift_click_was_added_to():
    ctrl, view = _controller()
    view.selected = {OTHER_IDX}
    _click(ctrl, additive=True)
    assert view.selected == A_IDX | {OTHER_IDX}

    ctrl.set_cluster_gap_factor(6.0)
    assert view.selected == A_IDX | B_IDX | {OTHER_IDX}
    ctrl.set_cluster_gap_factor(4.0)
    assert view.selected == A_IDX | {OTHER_IDX}


def test_changing_the_gap_with_no_click_leaves_the_selection_alone():
    ctrl, view = _controller()
    view.selected = {3, 4}
    assert ctrl.set_cluster_gap_factor(8.0) is False
    assert view.selected == {3, 4}
    assert ctrl.cluster_gap == pytest.approx(8 * _SPACING, rel=0.05)


def test_setting_the_same_gap_is_a_no_op():
    ctrl, view = _controller()
    _click(ctrl)
    view.selected = {7}  # if it re-ran, this would be overwritten
    assert ctrl.set_cluster_gap_factor(4.0) is False
    assert view.selected == {7}


def test_the_gap_setting_survives_loading_another_cloud():
    """It's a preference, like the lasso mode -- but the measured spacing
    belongs to the old points and must be re-measured."""
    ctrl, view = _controller()
    ctrl.set_cluster_gap_factor(8.0)
    _ = ctrl.cluster_gap  # measure on the first cloud

    coarse = (np.vstack([_A, _B, _OTHER]) * 3).astype(np.float32)  # 3cm spacing
    view.coords = coarse
    # set_cloud disarms and re-arms an armed tool through real canvas events,
    # which the stand-in view doesn't have; arming isn't what's under test.
    ctrl.cluster._armed = False
    ctrl.set_cloud(PointCloud(coords=coarse, labels=ctrl.cloud.labels.copy()))

    assert ctrl.cluster_gap_factor == 8.0
    assert ctrl.cluster_gap == pytest.approx(8 * 3 * _SPACING, rel=0.05)


def test_clicking_the_same_spot_again_loosens_the_gap_a_step():
    """A repeat click is the slider's move, not a level of its own: the gap
    goes up one notch (4x -> 6x) and the patch grows to bridge blob B."""
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    _click(ctrl)
    assert view.selected == A_IDX

    _click(ctrl)
    assert ctrl.cluster_gap_factor == 6.0
    assert view.selected == A_IDX | B_IDX


def test_a_repeat_click_keeps_its_seed_and_does_not_pick_again():
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    picks = []
    view.pick_point = lambda xy: picks.append(xy) or 0

    _click(ctrl)
    _click(ctrl)
    _click(ctrl)
    assert len(picks) == 1  # only the first click picked a point
    assert ctrl.cluster_gap_factor == 8.0  # 4 -> 6 -> 8


def test_a_repeat_click_keeps_what_a_shift_click_was_added_to():
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    view.selected = {OTHER_IDX}
    _click(ctrl, additive=True)
    _click(ctrl)  # a plain repeat click still just loosens
    assert view.selected == A_IDX | B_IDX | {OTHER_IDX}


def test_a_click_somewhere_else_starts_afresh_instead_of_loosening():
    ctrl, view = _controller()
    stepped = []
    ctrl.cluster.on_repeat = lambda: stepped.append(True)
    _click(ctrl)
    _click(ctrl, xy=(500.0, 500.0))  # far past CHAIN_PIXELS
    assert stepped == []
    assert ctrl.cluster_gap_factor == 4.0


def test_reset_goes_back_to_the_default_and_keeps_the_selection():
    """Switching Cluster off: the gap resets, but the patch the clicks grew
    stays selected for A/N/U -- it must not be re-run at 1x and shrunk."""
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    _click(ctrl)
    _click(ctrl)  # 4x -> 6x: both blobs
    assert view.selected == A_IDX | B_IDX

    ctrl.reset_cluster_gap()
    assert ctrl.cluster_gap_factor == DEFAULT_CLUSTER_GAP_FACTOR
    assert view.selected == A_IDX | B_IDX


def test_reset_ends_the_click_sequence():
    """After a reset the same spot is a fresh first click at the default,
    not a repeat that loosens from where the old sequence left off."""
    ctrl, view = _controller()
    stepped = []
    ctrl.cluster.on_repeat = lambda: stepped.append(True)
    _click(ctrl)
    ctrl.reset_cluster_gap()
    _click(ctrl)
    assert stepped == []
    assert len(view.selected) < _BLOB  # a 1x seed, not blob A


def test_gap_steps_are_ordered_and_include_the_default():
    assert list(CLUSTER_GAP_FACTORS) == sorted(CLUSTER_GAP_FACTORS)
    assert DEFAULT_CLUSTER_GAP_FACTOR in CLUSTER_GAP_FACTORS
