import unittest

import numpy as np

from eegprep import pop_adjustevents
from eegprep.eeg_options import EEG_OPTIONS


def demo_eeg():
    return {
        "data": np.zeros((1, 1000), dtype=np.float32),
        "nbchan": 1,
        "pnts": 1000,
        "trials": 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 3.996,
        "event": [
            {"type": "stim", "latency": 100.0, "duration": 0.0},
            {"type": "resp", "latency": 350.0, "duration": 0.0},
            {"type": "stim", "latency": 700.0, "duration": 0.0},
        ],
    }


def events(eeg):
    value = eeg["event"]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class PopAdjustEventsTests(unittest.TestCase):
    def test_addsamples_shifts_all_events(self):
        eeg = demo_eeg()

        out, com = pop_adjustevents(eeg, "addsamples", 10, return_com=True)

        self.assertEqual([event["latency"] for event in events(out)], [110.0, 360.0, 710.0])
        self.assertEqual([event["latency"] for event in eeg["event"]], [100.0, 350.0, 700.0])
        self.assertIn("'addsamples', 10", com)

    def test_force_key_value_arg_is_not_treated_as_duplicate(self):
        out = pop_adjustevents(demo_eeg(), "addsamples", 1, "force", "on")

        self.assertEqual([event["latency"] for event in events(out)], [101.0, 351.0, 701.0])

    def test_addms_uses_sampling_rate(self):
        out = pop_adjustevents(demo_eeg(), addms=20)

        self.assertEqual([event["latency"] for event in events(out)], [105.0, 355.0, 705.0])

    def test_eventtypes_filter_selected_events(self):
        out = pop_adjustevents(demo_eeg(), addsamples=-5, eventtypes=["stim"])

        self.assertEqual([event["latency"] for event in events(out)], [95.0, 350.0, 695.0])

    def test_eventtype_string_is_parsed(self):
        out = pop_adjustevents(demo_eeg(), addsamples=2, eventtypes="'stim' 'resp'")

        self.assertEqual([event["latency"] for event in events(out)], [102.0, 352.0, 702.0])

    def test_missing_eventtype_errors(self):
        with self.assertRaisesRegex(ValueError, "requested not found"):
            pop_adjustevents(demo_eeg(), addsamples=1, eventtypes=["missing"])

    def test_requires_one_latency_shift_source(self):
        with self.assertRaisesRegex(ValueError, "specify a number of samples or ms"):
            pop_adjustevents(demo_eeg(), eventtypes=["stim"])
        with self.assertRaisesRegex(ValueError, "either addms or addsamples"):
            pop_adjustevents(demo_eeg(), addms=10, addsamples=1)

    def test_force_off_rejects_epoched_data(self):
        eeg = demo_eeg()
        eeg["trials"] = 2

        with self.assertRaisesRegex(ValueError, "data epochs"):
            pop_adjustevents(eeg, addsamples=1, force="off")

    def test_cli_default_force_auto_allows_epoched_data_like_eeglab(self):
        eeg = demo_eeg()
        eeg["trials"] = 2

        out = pop_adjustevents(eeg, addsamples=1)

        self.assertEqual([event["latency"] for event in events(out)], [101.0, 351.0, 701.0])

    def test_force_off_rejects_boundary_events(self):
        eeg = demo_eeg()
        eeg["event"].append({"type": "boundary", "latency": 800.0, "duration": 1.0})

        with self.assertRaisesRegex(ValueError, "boundary events"):
            pop_adjustevents(eeg, addsamples=1, force="off")

    def test_cli_default_force_auto_allows_boundary_events_like_eeglab(self):
        eeg = demo_eeg()
        eeg["event"].append({"type": "boundary", "latency": 800.0, "duration": 1.0})

        out = pop_adjustevents(eeg, addsamples=1)

        self.assertEqual(
            [event["latency"] for event in events(out)],
            [101.0, 351.0, 701.0, 801.0],
        )

    def test_numeric_boundary99_respects_option(self):
        old_value = EEG_OPTIONS["option_boundary99"]
        try:
            EEG_OPTIONS["option_boundary99"] = 1
            eeg = demo_eeg()
            eeg["event"].append({"type": 99, "latency": 800.0, "duration": 1.0})

            with self.assertRaisesRegex(ValueError, "boundary events"):
                pop_adjustevents(eeg, addsamples=1, force="off")
        finally:
            EEG_OPTIONS["option_boundary99"] = old_value

    def test_gui_path_uses_renderer_values_and_force_off_default(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                self.spec = spec
                return {"events": "stim", "edit_time": "20", "edit_samples": "", "force": False}

        renderer = Renderer()
        eeg = demo_eeg()
        out = pop_adjustevents(eeg, gui=True, renderer=renderer)

        self.assertEqual(renderer.spec.title, "Adjust event latencies - pop_adjustevents()")
        self.assertEqual([event["latency"] for event in events(out)], [105.0, 350.0, 705.0])

    def test_gui_path_prioritizes_time_when_callback_syncs_samples(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"events": "stim", "edit_time": "20", "edit_samples": "5000", "force": False}

        out, com = pop_adjustevents(demo_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual([event["latency"] for event in events(out)], [105.0, 350.0, 705.0])
        self.assertIn("'addms', 20", com)
        self.assertNotIn("'addsamples'", com)

    def test_gui_cancel_returns_original_dataset(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return None

        eeg = demo_eeg()
        out = pop_adjustevents(eeg, gui=True, renderer=Renderer())

        self.assertIs(out, eeg)
        self.assertEqual(out, eeg)


if __name__ == "__main__":
    unittest.main()
