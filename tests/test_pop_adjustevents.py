import copy
import os
import unittest
import warnings

import numpy as np

from eegprep.eeg_options import EEG_OPTIONS
from eegprep.eeglabcompat import get_eeglab
from eegprep.gui import controls_by_tag
from eegprep.gui.specs import pop_adjustevents_dialog_spec
from eegprep.pop_adjustevents import pop_adjustevents


def _make_eeg(events=None, trials=1):
    if events is None:
        events = [
            {"type": "stim", "latency": 1.0},
            {"type": "resp", "latency": 10.5},
            {"type": "stim", "latency": 20.0},
        ]
    data = np.zeros((2, 40, trials)) if trials > 1 else np.zeros((2, 40))
    return {
        "data": data,
        "nbchan": 2,
        "pnts": 40,
        "trials": trials,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.39,
        "event": copy.deepcopy(events),
    }


def _events(EEG):
    return list(EEG["event"])


class TestPopAdjustEvents(unittest.TestCase):
    def test_add_samples_to_all_events(self):
        EEG_out, com = pop_adjustevents(_make_eeg(), addsamples=2.5)

        latencies = [event["latency"] for event in _events(EEG_out)]
        np.testing.assert_allclose(latencies, [3.5, 13.0, 22.5])
        self.assertEqual(com, "[EEG,com] = pop_adjustevents(EEG, 'addsamples', 2.5);")

    def test_add_ms_to_selected_event_type(self):
        EEG_out, com = pop_adjustevents(_make_eeg(), addms=50, eventtypes="stim")

        latencies = [event["latency"] for event in _events(EEG_out)]
        np.testing.assert_allclose(latencies, [6.0, 10.5, 25.0])
        self.assertIn("'eventtypes', {'stim'}", com)

    def test_quoted_event_type_list(self):
        EEG_out, _ = pop_adjustevents(_make_eeg(), addsamples=-1, eventtypes="'stim' 'resp'")

        latencies = [event["latency"] for event in _events(EEG_out)]
        np.testing.assert_allclose(latencies, [0.0, 9.5, 19.0])

    def test_missing_event_type_warns_then_errors(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "requested not found"):
                pop_adjustevents(_make_eeg(), addsamples=1, eventtypes=["missing"])

        self.assertEqual(len(caught), 1)
        self.assertIn("missing", str(caught[0].message))

    def test_requires_latency_shift(self):
        with self.assertRaisesRegex(ValueError, "specify addms or addsamples"):
            pop_adjustevents(_make_eeg())

    def test_force_off_rejects_boundary_events(self):
        EEG = _make_eeg(events=[{"type": "boundary", "latency": 4.0}])

        with self.assertRaisesRegex(ValueError, "boundary events"):
            pop_adjustevents(EEG, addsamples=1, force="off")

    def test_force_auto_command_line_allows_boundary_events(self):
        EEG = _make_eeg(events=[{"type": "boundary", "latency": 4.0}])
        EEG_out, com = pop_adjustevents(EEG, addsamples=1)

        self.assertEqual(_events(EEG_out)[0]["latency"], 5.0)
        self.assertNotIn("'force'", com)

    def test_force_off_rejects_epoched_data(self):
        with self.assertRaisesRegex(ValueError, "data epochs"):
            pop_adjustevents(_make_eeg(trials=2), addsamples=1, force="off")

    def test_numeric_boundary99_honors_eeglab_option(self):
        old_value = EEG_OPTIONS["option_boundary99"]
        EEG_OPTIONS["option_boundary99"] = 1
        try:
            EEG = _make_eeg(events=[{"type": 99, "latency": 4.0}])
            with self.assertRaisesRegex(ValueError, "boundary events"):
                pop_adjustevents(EEG, addsamples=1, force="off")
        finally:
            EEG_OPTIONS["option_boundary99"] = old_value

    def test_gui_renderer_decodes_options(self):
        class FakeRenderer:
            def run(self, spec, *, initial_values=None):
                self.spec = spec
                return {"events": "stim", "edit_time": "10", "edit_samples": "1", "force": True}

        renderer = FakeRenderer()
        EEG_out, com = pop_adjustevents(_make_eeg(), gui=True, renderer=renderer)

        self.assertEqual(renderer.spec.title, "Adjust event latencies - pop_adjustevents()")
        latencies = [event["latency"] for event in _events(EEG_out)]
        np.testing.assert_allclose(latencies, [2.0, 10.5, 21.0])
        self.assertIn("'force', 'on'", com)


class TestPopAdjustEventsGuiSpec(unittest.TestCase):
    def test_spec_matches_eeglab_dialog_shape(self):
        spec = pop_adjustevents_dialog_spec(100.0, event_types=["stim", "resp"])
        controls = controls_by_tag(spec)

        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_adjustevents.m")
        self.assertEqual(spec.geometry, ([1, 0.7, 0.5], [1, 0.7, 0.5], [1, 0.7, 0.5], 1))
        self.assertEqual(controls["events"].style, "edit")
        self.assertEqual(controls["edit_time"].callback.name, "sync_time_to_samples")
        self.assertEqual(controls["edit_samples"].callback.name, "sync_samples_to_time")
        self.assertEqual(controls["force"].string, "Force adjustment even when boundaries are present")


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopAdjustEventsParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")

    def test_add_samples_matches_eeglab(self):
        EEG = _make_eeg()
        EEG_py, _ = pop_adjustevents(copy.deepcopy(EEG), addsamples=2.5, force="on")
        EEG_mat, _ = self.eeglab.pop_adjustevents(copy.deepcopy(EEG), "addsamples", 2.5, "force", "on", nout=2)

        py_latencies = [event["latency"] for event in _events(EEG_py)]
        mat_latencies = [event["latency"] for event in _events(EEG_mat)]
        np.testing.assert_allclose(py_latencies, mat_latencies)
