import unittest

import numpy as np

from eegprep import pop_adjustevents
from tests.eeglab_tests import assert_matlab_equal, eeglab_test


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

    def test_gui_path_prioritizes_time_when_callback_syncs_samples(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"events": "stim", "edit_time": "20", "edit_samples": "5000", "force": False}

        out, com = pop_adjustevents(demo_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual([event["latency"] for event in events(out)], [105.0, 350.0, 705.0])
        self.assertIn("'addms', 20", com)
        self.assertNotIn("'addsamples'", com)


@eeglab_test(
    "unittesting_popfunc/pop_adjustevents/popfunc_pop_adjustevents_wrapperTest.m",
    "test_test_pop_adjustevents1",
)
def test_reference_adjustevents_epoched(eeglab_backend, eeglab_suite_root, subtests):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    _reference_adjustevents(eeglab_backend, eeg, subtests)


@eeglab_test(
    "unittesting_popfunc/pop_adjustevents/popfunc_pop_adjustevents_wrapperTest.m",
    "test_test_pop_adjustevents2",
)
def test_reference_adjustevents_continuous(eeglab_backend, eeglab_suite_root, subtests):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    _reference_adjustevents(eeglab_backend, eeg, subtests)


def _reference_adjustevents(eeglab_backend, eeg, subtests):
    event_types = eeg["event"]["type"][0].tolist()
    rt_indices = [index for index, kind in enumerate(event_types) if "rt" in kind]
    latencies = np.concatenate(eeg["event"]["latency"][0].tolist(), axis=1)
    for parameter, amount in (("addms", 20.0), ("addsamples", 30.0)):
        for selected in (False, True):
            # The source catches each case's failure and reports after all four.
            with subtests.test(parameter=parameter, selected=selected):
                options = ("eventtypes", np.array([["rt"]], dtype=object)) if selected else ()
                result, _ = eeglab_backend("pop_adjustevents", eeg, parameter, amount, *options, nargout=2)
                actual = np.concatenate(result["event"]["latency"][0].tolist(), axis=1)
                expected = latencies
                if selected:
                    actual, expected = actual[:, rt_indices], expected[:, rt_indices]
                shift = 20.0 / 1000.0 * eeg["srate"] if parameter == "addms" else 30.0
                # Retain source subtraction before exact equality: adding to the
                # oracle instead would change floating-point rounding behavior.
                assert_matlab_equal(actual - shift, expected)


if __name__ == "__main__":
    unittest.main()
