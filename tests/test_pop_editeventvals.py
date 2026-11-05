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


def _continuous_events(count=5):
    return {
        "data": np.zeros((1, 40), dtype=float),
        "nbchan": 1,
        "pnts": 40,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.39,
        "times": np.arange(40, dtype=float),
        "chanlocs": [{"labels": "Cz"}],
        "chaninfo": {},
        "event": [{"type": f"e{index}", "latency": float(index)} for index in range(1, count + 1)],
        "urevent": [],
        "epoch": [],
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


def _qt_event_dialog(eeg):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from eegprep.functions.guifunc import qt as qt_module

    if qt_module.QtCore is None or qt_module.QtWidgets is None:
        pytest.skip("PySide6 Qt libraries unavailable in this environment")
    renderer = qt_module.QtDialogRenderer()
    _app, dialog, widgets = renderer.build_dialog(pop_editeventvals_dialog_spec(eeg))
    ok = dialog.findChild(qt_module.QtWidgets.QPushButton, "ok")
    return renderer, dialog, widgets, ok


def _drive_event_dialog(eeg, edit):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from eegprep.functions.guifunc import qt as qt_module

    if qt_module.QtCore is None or qt_module.QtWidgets is None:
        pytest.skip("PySide6 Qt libraries unavailable in this environment")

    class Renderer:
        def run(self, spec, initial_values=None):
            renderer = qt_module.QtDialogRenderer()
            _app, dialog, widgets = renderer.build_dialog(spec)
            try:
                edit(dialog, widgets, qt_module)
                dialog.findChild(qt_module.QtWidgets.QPushButton, "ok").click()
                return {tag: renderer._read_widget(widget) for tag, widget in widgets.items()}
            finally:
                dialog.close()

    return pop_editeventvals(deepcopy(eeg), gui=True, renderer=Renderer(), return_com=True)


def test_event_number_enter_loads_that_event_and_ok_keeps_the_edit():
    """Enter on the event number shows that event; Ok writes the edit to it."""
    eeg = _continuous_events()

    def edit(_dialog, widgets, _qt):
        widgets["eventnum"].setText("5")
        widgets["eventnum"].editingFinished.emit()
        assert widgets["eventnum"].text() == "5"
        assert widgets["field_type"].text() == "e5"
        assert widgets["field_latency"].text() == str(_display_event_value(eeg, eeg["event"][4], "latency"))
        widgets["field_type"].setText("zz")

    out, command = _drive_event_dialog(eeg, edit)
    assert out["event"][0]["type"] == "e1"
    assert out["event"][4]["type"] == "zz"
    assert out["event"][4]["latency"] == eeg["event"][4]["latency"]
    assert "zz" in command


def test_leaving_an_event_keeps_the_edit_for_ok():
    """Arrow navigation applies the pending edit before showing the next event."""
    eeg = _continuous_events()

    def edit(_dialog, widgets, _qt):
        widgets["eventnum"].setText("5")
        widgets["eventnum"].editingFinished.emit()
        widgets["field_type"].setText("yy")
        widgets["back1"].click()
        assert widgets["eventnum"].text() == "4"
        assert widgets["field_type"].text() == "e4"
        widgets["field_type"].setText("qq")
        widgets["next1"].click()
        assert widgets["eventnum"].text() == "5"
        assert widgets["field_type"].text() == "yy"

    out, command = _drive_event_dialog(eeg, edit)
    assert out["event"][3]["type"] == "qq"
    assert out["event"][4]["type"] == "yy"
    assert out["event"][0]["type"] == "e1"
    assert "'yy'" in command
    assert "'qq'" in command


def test_ok_applies_a_pending_edit_before_a_typed_event_number():
    """Ok follows the typed event number after saving the event still on screen."""
    eeg = _continuous_events()

    def edit(_dialog, widgets, _qt):
        widgets["field_type"].setText("zz")
        widgets["eventnum"].setText("5")

    out, command = _drive_event_dialog(eeg, edit)
    assert out["event"][0]["type"] == "zz"
    assert out["event"][4]["type"] == "e5"
    assert "zz" in command


def test_event_number_out_of_range_is_clamped_and_non_numeric_is_restored():
    eeg = _continuous_events()
    _renderer, dialog, widgets, _ok = _qt_event_dialog(eeg)
    try:
        widgets["eventnum"].setText("999")
        widgets["eventnum"].editingFinished.emit()
        assert widgets["eventnum"].text() == "5"
        assert widgets["field_type"].text() == "e5"
        widgets["field_type"].setText("zz")
        widgets["eventnum"].setText("abc")
        widgets["eventnum"].editingFinished.emit()
        assert widgets["eventnum"].text() == "5"
        assert widgets["field_type"].text() == "zz"
    finally:
        dialog.close()


def test_run_gui_rejects_an_event_number_outside_the_list():
    eeg = _continuous_events()

    class Renderer:
        def run(self, spec, initial_values=None):
            result = {control.tag: control.value for control in spec.controls if control.tag}
            result["eventnum"] = "999"
            result["field_type"] = "zz"
            return result

    with pytest.raises(ValueError, match=r"pop_editeventvals: event number 999 is outside 1 to 5"):
        pop_editeventvals(deepcopy(eeg), gui=True, renderer=Renderer())
    assert eeg["event"][0]["type"] == "e1"
