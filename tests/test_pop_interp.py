import unittest
from unittest.mock import patch

import numpy as np

from eegprep import pop_interp
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import eeglab_test


def _eeg(n_channels=8, n_points=50, trials=1):
    data_shape = (n_channels, n_points) if trials == 1 else (n_channels, n_points, trials)
    rng = np.random.default_rng(123)
    eeg = {
        "data": rng.normal(size=data_shape),
        "nbchan": n_channels,
        "pnts": n_points,
        "trials": trials,
        "srate": 100,
        "xmin": 0,
        "xmax": (n_points - 1) / 100,
        "chanlocs": [],
        "chaninfo": {},
        "epoch": [] if trials == 1 else [{} for _ in range(trials)],
    }
    for index in range(n_channels):
        angle = 2 * np.pi * index / n_channels
        x = np.cos(angle)
        y = np.sin(angle)
        theta = np.degrees(np.arctan2(x, y))
        radius = 0.5
        eeg["chanlocs"].append(
            {
                "labels": f"Ch{index + 1}",
                "X": x,
                "Y": y,
                "Z": 0.5,
                "theta": theta,
                "radius": radius,
            }
        )
    return eeg


@eeglab_test("unittesting_popfunc/pop_interp/popfunc_pop_interp_wrapperTest.m", "test_test_pop_interp")
def test_reference_pop_interp_original_locations_and_recording(eeglab_backend, eeglab_suite_root):
    sample_directory = eeglab_suite_root / "eeglab/sample_data"
    eeg = eeglab_backend("pop_loadset", str(sample_directory / "eeglab_data.set"))
    eeg["chanlocs"] = eeglab_backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        np.array([[str(sample_directory / "eeglab_chan32.locs"), "filetype", ""]], dtype=object),
        "shrink",
        -0.1,
    )
    eeg["pnts"] = 1000.0
    eeg["data"] = eeg["data"][:, :1000]
    eeg = eeglab_backend("eeg_checkset", eeg)
    for method in ("spherical", "invdist"):
        eeglab_backend("pop_interp", eeg, np.arange(1.0, 17.0)[None, :], method)
        eeglab_backend("pop_interp", eeg, np.empty((0, 0)), method)


def test_pop_interp_current_suite_sample_channel_workflows():
    eeg = pop_loadset("sample_data/eeglab_data.set")
    eeg["data"] = eeg["data"][:, :1000]
    eeg["pnts"] = 1000

    for method in ("spherical", "invdist"):
        interpolated = pop_interp(eeg, list(range(16)), method)
        unchanged = pop_interp(eeg, [], method)

        assert interpolated["data"].shape == eeg["data"].shape
        assert np.isfinite(interpolated["data"]).all()
        np.testing.assert_array_equal(interpolated["data"][16:], eeg["data"][16:])
        np.testing.assert_array_equal(unchanged["data"], eeg["data"])


class PopInterpTests(unittest.TestCase):
    def test_command_line_supports_planar_and_spacetime_methods_from_eeglab_doc(self):
        eeg = _eeg()

        planar = pop_interp(eeg, [0], "invdist")
        spacetime = pop_interp(eeg, [0], "spacetime")

        self.assertEqual(planar["data"].shape, eeg["data"].shape)
        self.assertEqual(spacetime["data"].shape, eeg["data"].shape)
        self.assertTrue(np.all(np.isfinite(planar["data"])))
        self.assertTrue(np.all(np.isfinite(spacetime["data"])))

    def test_interpolating_removed_chanloc_removes_it_from_removedchans(self):
        eeg = _eeg()
        removed = {
            "labels": "M1",
            "X": 0.0,
            "Y": -1.0,
            "Z": 0.5,
            "theta": 180.0,
            "radius": 0.5,
        }
        eeg["chaninfo"]["removedchans"] = [removed.copy()]

        out = pop_interp(eeg, [removed], "spherical")

        self.assertEqual(out["nbchan"], eeg["nbchan"] + 1)
        self.assertEqual(out["chanlocs"][-1]["labels"], "M1")
        self.assertEqual(out["chaninfo"]["removedchans"], [])

    def test_gui_path_uses_selection_userdata_for_history_and_processing(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                self.spec = spec
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 1, "timerange": ""}

        eeg = _eeg()
        renderer = Renderer()

        out, com = pop_interp(eeg, gui=True, renderer=renderer, return_com=True)

        self.assertEqual(renderer.spec.title, "Interpolate channel(s) -- pop_interp()")
        self.assertEqual(com, "EEG = pop_interp(EEG, [1], 'spherical');")
        self.assertFalse(np.array_equal(out["data"][0], eeg["data"][0]))

    def test_continuous_gui_second_method_matches_eeglab_planar_mapping(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 2, "timerange": ""}

        _out, com = pop_interp(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_interp(EEG, [1], 'invdist');")

    def test_epoched_gui_second_method_matches_eeglab_kang_mapping(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 2}

        _out, com = pop_interp(_eeg(trials=2), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_interp(EEG, [1], 'sphericalKang');")

    def test_gui_history_includes_user_entered_time_range(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 1, "timerange": "0 0.2"}

        _out, com = pop_interp(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_interp(EEG, [1], 'spherical', [0 0.2]);")


class PopInterpGuiSpecTests(unittest.TestCase):
    def test_epoched_dialog_spec_accepts_numpy_epoch_arrays(self):
        eeg = _eeg(trials=2)
        eeg["epoch"] = np.asarray([{"event": 1}], dtype=object)

        spec = pop_interp_dialog_spec(eeg)
        controls = controls_by_tag(spec)

        self.assertEqual(controls["method"].string, "Spherical|spherical(Kang et al.)|Planar (slow)")
        self.assertNotIn("timerange", controls)

    def test_interp_datchan_callback_keeps_zero_based_data_and_one_based_history(self):
        target = _Target()
        params = {
            "source": "datchan",
            "chanlocs": ({"labels": "Ch1"}, {"labels": "Ch2"}, {"labels": "Ch3"}),
        }

        with patch("eegprep.functions.guifunc.qt.pop_chansel", return_value=([1, 3], "Ch1 Ch3", ["Ch1", "Ch3"])):
            QtDialogRenderer._select_interp_channels(None, target, params)

        self.assertEqual(target.text, "Ch1 Ch3")
        self.assertEqual(QtDialogRenderer._read_widget(target), {"chans": [0, 2], "chanstr": "[1 3]"})


class _Target:
    def __init__(self):
        self._properties = {}

    def setText(self, text):
        self.text = text

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


if __name__ == "__main__":
    unittest.main()
