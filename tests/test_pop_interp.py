import unittest
from unittest.mock import patch

import numpy as np

from eegprep import pop_interp
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec


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


class PopInterpTests(unittest.TestCase):
    def test_no_input_returns_like_eeglab_help_path(self):
        self.assertIsNone(pop_interp())

    def test_command_line_empty_selection_matches_eeglab_no_history_command(self):
        eeg = _eeg()

        out, com = pop_interp(eeg, [], return_com=True)

        np.testing.assert_array_equal(out["data"], eeg["data"])
        self.assertEqual(com, "")

    def test_command_line_defaults_to_spherical(self):
        eeg = _eeg()

        out = pop_interp(eeg, [0])

        self.assertEqual(out["data"].shape, eeg["data"].shape)
        self.assertFalse(np.array_equal(out["data"][0], eeg["data"][0]))

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

    def test_continuous_gui_second_method_matches_eeglab_source_mapping(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 2, "timerange": ""}

        _out, com = pop_interp(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_interp(EEG, [1], 'sphericalKang');")

    def test_gui_blank_time_range_matches_eeglab_full_range_gui_behavior(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 1, "timerange": ""}

        eeg = _eeg()

        gui_out = pop_interp(eeg, gui=True, renderer=Renderer())
        expected = pop_interp(eeg, [0], "spherical", [eeg["xmin"], eeg["xmax"]])

        np.testing.assert_allclose(gui_out["data"], expected["data"], rtol=1e-7, atol=1e-7)

    def test_gui_time_range_validation_matches_eeglab_errors(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"chanlist": {"chans": [0], "chanstr": "[1]"}, "method": 1, "timerange": "0 99"}

        with self.assertRaisesRegex(ValueError, "upper data limits"):
            pop_interp(_eeg(), gui=True, renderer=Renderer())


class PopInterpGuiSpecTests(unittest.TestCase):
    def test_continuous_dialog_spec_matches_eeglab_control_order(self):
        eeg = _eeg()
        eeg["chaninfo"]["removedchans"] = [{"labels": "M1"}]

        spec = pop_interp_dialog_spec(eeg)

        self.assertEqual(spec.title, "Interpolate channel(s) -- pop_interp()")
        self.assertEqual(spec.function_name, "pop_interp")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_interp.m")
        self.assertEqual(spec.help_text, "pophelp('pop_interp')")
        self.assertEqual(
            [(control.style, control.string, control.tag, control.enabled) for control in spec.controls],
            [
                ("text", "What channel(s) do you want to interpolate", None, True),
                ("text", "none selected", "chanlist", True),
                ("pushbutton", "Select from removed channels", "interp_nondatchan", True),
                ("pushbutton", "Select from data channels", "interp_datchan", True),
                ("pushbutton", "Use specific channels of other dataset", "interp_selectchan", True),
                ("pushbutton", "Use all channels from other dataset", "interp_uselist", True),
                ("spacer", "", None, True),
                ("text", "Interpolation method", None, True),
                ("popupmenu", "Spherical|Planar (slow)", "method", True),
                ("spacer", "", None, True),
                ("text", "Time range [min max] (s)", None, True),
                ("edit", "", "timerange", True),
                ("spacer", "", None, True),
                ("text", "Note: for group level analysis, interpolate in STUDY", None, True),
            ],
        )

    def test_epoched_dialog_spec_includes_kang_method_like_eeglab(self):
        spec = pop_interp_dialog_spec(_eeg(trials=2))
        controls = controls_by_tag(spec)

        self.assertEqual(controls["method"].string, "Spherical|spherical(Kang et al.)|Planar (slow)")
        self.assertNotIn("timerange", controls)

    def test_dialog_callbacks_keep_matlab_metadata(self):
        controls = controls_by_tag(pop_interp_dialog_spec(_eeg()))

        self.assertEqual(controls["interp_datchan"].callback.name, "select_interp_channels")
        self.assertEqual(controls["interp_datchan"].callback.params["source"], "datchan")
        self.assertEqual(controls["interp_datchan"].callback.params["button"], "interp_datchan")
        self.assertIn("pop_interp('datchan'", controls["interp_datchan"].callback.matlab_callback)

    def test_other_dataset_callbacks_store_chanlocs_without_copying_data(self):
        other = _eeg()
        controls = controls_by_tag(pop_interp_dialog_spec(_eeg(), alleeg=[other]))

        alleeg = controls["interp_uselist"].callback.params["alleeg"]

        self.assertEqual(tuple(alleeg[0]), ("chanlocs",))
        self.assertEqual(alleeg[0]["chanlocs"][0]["labels"], "Ch1")

    def test_renderer_stores_interp_selection_as_inputgui_userdata(self):
        class Target:
            def __init__(self):
                self._properties = {}

            def setText(self, text):
                self.text = text

            def setProperty(self, name, value):
                self._properties[name] = value

            def property(self, name):
                return self._properties.get(name)

        target = Target()

        QtDialogRenderer._store_interp_selection(target, [0, 2], "[1 3]", "Ch1 Ch3")

        self.assertEqual(target.text, "Ch1 Ch3")
        self.assertEqual(QtDialogRenderer._read_widget(target), {"chans": [0, 2], "chanstr": "[1 3]"})

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

    def test_interp_removed_callback_uses_removedchans_history_expression(self):
        target = _Target()
        removed = ({"labels": "M1"}, {"labels": "M2"})
        params = {"source": "nondatchan", "removedchans": removed}

        with patch("eegprep.functions.guifunc.qt.pop_chansel", return_value=([1, 2], "M1 M2", ["M1", "M2"])):
            QtDialogRenderer._select_interp_channels(None, target, params)

        self.assertEqual(QtDialogRenderer._read_widget(target), {
            "chans": [{"labels": "M1"}, {"labels": "M2"}],
            "chanstr": "EEG.chaninfo.removedchans([1 2])",
        })

    def test_interp_selectchan_callback_uses_alleeg_subset_history_expression(self):
        target = _Target()
        params = {
            "source": "selectchan",
            "chanlocs": ({"labels": "Ch1"},),
            "alleeg": ({"chanlocs": ({"labels": "Ch1"}, {"labels": "Pz"})},),
        }

        with (
            patch("eegprep.functions.guifunc.qt._require_qt", return_value=(object(), _FakeQtWidgets)),
            patch("eegprep.functions.guifunc.qt.pop_chansel", return_value=([2], "Pz", ["Pz"])),
        ):
            QtDialogRenderer._select_interp_channels(None, target, params)

        self.assertEqual(QtDialogRenderer._read_widget(target), {
            "chans": [{"labels": "Pz"}],
            "chanstr": "ALLEEG(1).chanlocs([2])",
        })

    def test_interp_uselist_callback_uses_alleeg_full_history_expression(self):
        target = _Target()
        params = {
            "source": "uselist",
            "chanlocs": ({"labels": "Ch1"},),
            "alleeg": ({"chanlocs": ({"labels": "Pz"}, {"labels": "Oz"})},),
        }

        with patch("eegprep.functions.guifunc.qt._require_qt", return_value=(object(), _FakeQtWidgets)):
            QtDialogRenderer._select_interp_channels(None, target, params)

        self.assertEqual(QtDialogRenderer._read_widget(target), {
            "chans": [{"labels": "Pz"}, {"labels": "Oz"}],
            "chanstr": "ALLEEG(1).chanlocs",
        })


class _Target:
    def __init__(self):
        self._properties = {}

    def setText(self, text):
        self.text = text

    def setProperty(self, name, value):
        self._properties[name] = value

    def property(self, name):
        return self._properties.get(name)


class _FakeInputDialog:
    @staticmethod
    def getInt(*args, **kwargs):
        return 1, True


class _FakeMessageBox:
    @staticmethod
    def warning(*args, **kwargs):
        return None


class _FakeQtWidgets:
    QInputDialog = _FakeInputDialog
    QMessageBox = _FakeMessageBox


if __name__ == "__main__":
    unittest.main()
