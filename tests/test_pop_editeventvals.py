"""Tests for pop_editeventvals latency display/edit symmetry."""

from copy import deepcopy

import numpy as np
import pytest

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


def test_epoched_latency_edit_round_trips_through_display():
    """Writing a new ms latency then reading it back yields the same ms value."""
    eeg = _epoched_eeg()
    new_ms = 60.0
    out = pop_editeventvals(deepcopy(eeg), "changefield", [1, "latency", new_ms])
    assert _display_event_value(out, out["event"][0], "latency") == new_ms
