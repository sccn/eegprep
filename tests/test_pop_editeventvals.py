"""Tests for pop_editeventvals latency display/edit symmetry."""

import os
from copy import deepcopy

import numpy as np
import pytest

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_editeventvals import (
    pop_editeventvals,
    pop_editeventvals_dialog_spec,
    _display_event_value,
    _display_field_label,
)


def _epoched_eeg():
    """Epoched dataset: 2 trials, 50 points/epoch, srate 100, xmin -0.1 s."""
    return {
        "data": np.zeros((2, 50, 2), dtype=float),
        "nbchan": 2,
        "pnts": 50,
        "trials": 2,
        "srate": 100.0,
        "xmin": -0.1,
        "xmax": 0.39,
        "times": np.arange(50, dtype=float),
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "chaninfo": {},
        # latencies stored in points across concatenated epochs (1-based)
        "event": [
            {"type": "stim", "latency": 15.0, "duration": 0.0, "epoch": 1},
            {"type": "resp", "latency": 70.0, "duration": 0.0, "epoch": 2},
        ],
        "urevent": [],
        "epoch": [{}, {}],
        "reject": {},
        "stats": {},
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
        "history": "",
        "saved": "no",
    }


def test_epoched_latency_display_matches_ms_label():
    """The epoched latency dialog label says ms, so the displayed value must be ms."""
    eeg = _epoched_eeg()
    assert _display_field_label(eeg, "latency") == "Latency (ms)"
    # point 15 in epoch 1 with xmin -0.1 s -> (15-1)/100 - 0.1 = 0.04 s = 40 ms
    assert _display_event_value(eeg, eeg["event"][0], "latency") == 40.0
    # the dialog spec exposes the same ms value, not the raw stored point count
    spec = pop_editeventvals_dialog_spec(eeg)
    latency_edit = next(control for control in spec.controls if control.tag == "field_latency")
    assert latency_edit.value == 40.0


def test_epoched_latency_edit_is_symmetric_with_display():
    """Editing an epoched latency to its displayed ms value leaves the stored point unchanged."""
    eeg = _epoched_eeg()
    displayed = _display_event_value(eeg, eeg["event"][0], "latency")
    out = pop_editeventvals(deepcopy(eeg), "changefield", [1, "latency", displayed])
    assert out["event"][0]["latency"] == pytest.approx(eeg["event"][0]["latency"])


def test_run_gui_submits_change_for_navigated_event_index():
    """A fake renderer returning eventnum=2 produces a changefield for event 2."""
    eeg = _epoched_eeg()

    class Renderer:
        def run(self, spec, initial_values=None):
            # Simulate the navigation handler having loaded event 2's display
            # values into the field_* widgets, with the user editing 'type'.
            controls = controls_by_tag(spec)
            nav = controls["next1"].callback.params
            result = {tag: ctrl.value for tag, ctrl in controls.items()}
            result.update(nav["field_displays"][1])
            result["eventnum"] = "2"
            result["field_type"] = "renamed"
            return result

    out, com = pop_editeventvals(deepcopy(eeg), gui=True, renderer=Renderer(), return_com=True)
    assert out["event"][0]["type"] == eeg["event"][0]["type"]
    assert out["event"][1]["type"] == "renamed"
    assert "'changefield', {2 'type' 'renamed'}" in com


def test_qt_renderer_navigation_updates_eventnum_and_fields():
    """Clicking next1 advances eventnum and rewrites field_* widgets (#228)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from eegprep.functions.guifunc import qt as qt_module

    if qt_module.QtCore is None or qt_module.QtWidgets is None:
        pytest.skip("PySide6 Qt libraries unavailable in this environment")
    QtDialogRenderer = qt_module.QtDialogRenderer

    eeg = _epoched_eeg()
    renderer = QtDialogRenderer()
    spec = pop_editeventvals_dialog_spec(eeg)
    _app, dialog, widgets = renderer.build_dialog(spec)
    try:
        assert widgets["eventnum"].text() == "1"
        widgets["next1"].click()
        assert widgets["eventnum"].text() == "2"
        # field_type widget now reflects event #2's display value.
        assert widgets["field_type"].text() == str(eeg["event"][1]["type"])
        # Going past the end clamps at the last event.
        widgets["next10"].click()
        assert widgets["eventnum"].text() == str(len(eeg["event"]))
        widgets["back10"].click()
        assert widgets["eventnum"].text() == "1"
    finally:
        dialog.close()
