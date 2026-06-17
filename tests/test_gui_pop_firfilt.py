from __future__ import annotations

import unittest

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc import qt as qt_renderer
from eegprep.functions.guifunc.qt import QtDialogRenderer, _firpm_order_shape
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt, pop_eegfilt_dialog_spec
from eegprep.plugins.firfilt.pop_eegfiltnew import pop_eegfiltnew, pop_eegfiltnew_dialog_spec
from eegprep.plugins.firfilt.pop_firma import pop_firma, pop_firma_dialog_spec
from eegprep.plugins.firfilt.pop_firpm import pop_firpm, pop_firpm_dialog_spec
from eegprep.plugins.firfilt.pop_firpmord import pop_firpmord, pop_firpmord_dialog_spec
from eegprep.plugins.firfilt.pop_firws import pop_firws, pop_firws_dialog_spec
from eegprep.plugins.firfilt.pop_firwsord import pop_firwsord_dialog_spec
from eegprep.plugins.firfilt.pop_kaiserbeta import pop_kaiserbeta_dialog_spec
from eegprep.plugins.firfilt.pop_xfirws import pop_xfirws_dialog_spec


def _eeg():
    rng = np.random.default_rng(10)
    return {
        "data": rng.standard_normal((3, 600)),
        "nbchan": 3,
        "pnts": 600,
        "trials": 1,
        "srate": 200,
        "xmin": 0,
        "xmax": 2.995,
        "times": np.arange(600) / 200,
        "event": [],
        "urevent": [],
        "chanlocs": [
            {"labels": "Cz", "type": "EEG"},
            {"labels": "Pz", "type": "EEG"},
            {"labels": "EOG", "type": "EOG"},
        ],
    }


class PopFirfiltGuiTests(unittest.TestCase):
    def test_pop_eegfiltnew_dialog_matches_eeglab_sections(self):
        spec = pop_eegfiltnew_dialog_spec(_eeg())

        self.assertEqual(spec.title, "Filter the data -- pop_eegfiltnew()")
        self.assertEqual(spec.function_name, "pop_eegfiltnew")
        self.assertEqual(spec.eeglab_source, "plugins/firfilt/pop_eegfiltnew.m")
        self.assertEqual(spec.help_text, "pophelp('pop_eegfiltnew')")
        labels = [(control.style, control.string, control.tag) for control in spec.controls]
        self.assertIn(("text", "Lower edge of the frequency pass band (Hz)", None), labels)
        self.assertIn(("text", "Higher edge of the frequency pass band (Hz)", None), labels)
        self.assertIn(("checkbox", "Notch filter the data instead of pass band", "revfilt"), labels)
        self.assertIn(("text", "Channel type(s)", None), labels)
        self.assertIn(("text", "OR channel labels or indices", None), labels)

    def test_pop_eegfiltnew_dialog_accepts_numpy_chanlocs(self):
        eeg = _eeg()
        eeg["chanlocs"] = np.asarray(eeg["chanlocs"], dtype=object)

        controls = controls_by_tag(pop_eegfiltnew_dialog_spec(eeg))

        self.assertEqual(controls["chantype_button"].callback.params["channels"], ["EEG", "EOG"])
        self.assertEqual(controls["channels_button"].callback.params["channels"], ["Cz", "Pz", "EOG"])

    def test_pop_eegfiltnew_gui_result_filters_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "locutoff": "",
                    "hicutoff": "30",
                    "filtorder": "80",
                    "revfilt": False,
                    "minphase": False,
                    "plotfreqz": False,
                    "chantype": "",
                    "channels": "2",
                    "usefftfilt": True,
                }

        out, command = pop_eegfiltnew(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 600))
        self.assertEqual(
            command,
            "EEG = pop_eegfiltnew(EEG, 'hicutoff', 30, 'filtorder', 80, 'usefftfilt', 1, 'channels', [2]);",
        )

    def test_pop_eegfiltnew_gui_accepts_channel_labels(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "locutoff": "",
                    "hicutoff": "30",
                    "filtorder": "80",
                    "revfilt": False,
                    "minphase": False,
                    "plotfreqz": False,
                    "chantype": "",
                    "channels": "Cz Pz",
                    "usefftfilt": False,
                }

        eeg = _eeg()
        before = eeg["data"].copy()
        out, command = pop_eegfiltnew(eeg, gui=True, renderer=Renderer(), return_com=True)

        self.assertFalse(np.allclose(out["data"][:2], before[:2]))
        np.testing.assert_allclose(out["data"][2], before[2])
        self.assertIn("'channels', {'Cz' 'Pz'}", command)

    def test_pop_eegfiltnew_accepts_numpy_chanlocs_for_channel_type_filtering(self):
        eeg = _eeg()
        eeg["chanlocs"] = np.asarray(eeg["chanlocs"], dtype=object)
        before = eeg["data"].copy()

        out, command = pop_eegfiltnew(
            eeg,
            "hicutoff",
            30,
            "filtorder",
            80,
            "chantype",
            ["EOG"],
            return_com=True,
        )

        np.testing.assert_allclose(out["data"][:2], before[:2])
        self.assertFalse(np.allclose(out["data"][2], before[2]))
        self.assertIn("'chantype', {'EOG'}", command)

    def test_legacy_pop_eegfilt_dialog_and_gui_result(self):
        spec = pop_eegfilt_dialog_spec(_eeg())

        self.assertEqual(spec.title, "Filter the data -- pop_eegfilt()")
        self.assertEqual(spec.function_name, "pop_eegfilt")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_eegfilt.m")

        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "locutoff": "1",
                    "hicutoff": "40",
                    "filtorder": "100",
                    "revfilt": False,
                    "usefft": False,
                    "plotfreqz": False,
                    "causal": False,
                    "firtype": True,
                }

        out, command = pop_eegfilt(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 600))
        self.assertEqual(command, "EEG = pop_eegfilt( EEG, 1, 40, [100], [0], 0, 0, 'fir1', 0);")

    def test_firfilt_plugin_dialog_specs_are_eeglab_labeled(self):
        specs = [
            pop_firws_dialog_spec(_eeg()),
            pop_firpm_dialog_spec(_eeg()),
            pop_firma_dialog_spec(_eeg()),
        ]

        self.assertEqual([spec.function_name for spec in specs], ["pop_firws", "pop_firpm", "pop_firma"])
        self.assertEqual(
            [spec.eeglab_source for spec in specs],
            ["plugins/firfilt/pop_firws.m", "plugins/firfilt/pop_firpm.m", "plugins/firfilt/pop_firma.m"],
        )

    def test_firfilt_dialog_buttons_have_live_callbacks(self):
        for spec in (pop_firws_dialog_spec(_eeg()), pop_firpm_dialog_spec(_eeg()), pop_firma_dialog_spec(_eeg())):
            for control in spec.controls:
                if control.style == "pushbutton" and control.string in {"Estimate", "Plot filter responses"}:
                    self.assertIsNotNone(control.callback)
                    if control.tag != "wargpush":
                        self.assertTrue(control.enabled)

    def test_firfilt_order_dialog_specs_are_eeglab_labeled(self):
        specs = [
            pop_kaiserbeta_dialog_spec(),
            pop_firwsord_dialog_spec(),
            pop_firpmord_dialog_spec(),
            pop_xfirws_dialog_spec(),
        ]

        self.assertEqual(
            [spec.function_name for spec in specs],
            ["pop_kaiserbeta", "pop_firwsord", "pop_firpmord", "pop_xfirws"],
        )
        self.assertEqual(
            [spec.eeglab_source for spec in specs],
            [
                "plugins/firfilt/pop_kaiserbeta.m",
                "plugins/firfilt/pop_firwsord.m",
                "plugins/firfilt/pop_firpmord.m",
                "plugins/firfilt/pop_xfirws.m",
            ],
        )
        kaiser_controls = {(control.style, control.string, control.tag) for control in specs[0].controls}
        self.assertIn(("text", "Max passband deviation/ripple:", None), kaiser_controls)
        self.assertEqual(specs[1].help_text, "pophelp('pop_firwsord')")
        firpm_controls = controls_by_tag(specs[2])
        self.assertIn("rp", firpm_controls)
        self.assertIn("rs", firpm_controls)
        self.assertNotIn("f", firpm_controls)
        self.assertNotIn("a", firpm_controls)

    def test_qt_renderer_stateless_helpers_have_module_ownership(self):
        self.assertIs(QtDialogRenderer._read_widget, qt_renderer._read_widget)
        self.assertIs(QtDialogRenderer._validation_message, qt_renderer._validation_message)

    def test_firpm_estimate_order_shape_uses_paired_edges_for_single_cutoff_filters(self):
        highpass_edges, highpass_amplitudes = _firpm_order_shape([8], 4, "highpass", 200)
        lowpass_edges, lowpass_amplitudes = _firpm_order_shape([30], 4, "lowpass", 200)

        self.assertEqual(highpass_edges, [0.0, 6.0, 10.0, 100.0])
        self.assertEqual(highpass_amplitudes, [0, 1])
        self.assertEqual(lowpass_edges, [0.0, 28.0, 32.0, 100.0])
        self.assertEqual(lowpass_amplitudes, [1, 0])
        self.assertGreater(pop_firpmord(highpass_edges, highpass_amplitudes, [0.001, 0.01], 200)[0], 0)
        self.assertGreater(pop_firpmord(lowpass_edges, lowpass_amplitudes, [0.01, 0.001], 200)[0], 0)

    def test_firws_gui_result_filters_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "fcutoff": "8 30",
                    "ftype": 1,
                    "wtype": 3,
                    "warg": "",
                    "forder": "120",
                    "minphase": False,
                    "usefftfilt": False,
                }

        out, command = pop_firws(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 600))
        self.assertEqual(
            command, "EEG = pop_firws(EEG, 'fcutoff', [8 30], 'forder', 120, 'ftype', 'bandpass', 'wtype', 'hamming');"
        )

    def test_firpm_gui_result_filters_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "fcutoff": "8 30",
                    "ftrans": "4",
                    "ftype": 1,
                    "wtpass": "",
                    "wtstop": "",
                    "forder": "120",
                }

        out, command = pop_firpm(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 600))
        self.assertEqual(
            command, "EEG = pop_firpm(EEG, 'fcutoff', [8 30], 'ftrans', 4, 'ftype', 'bandpass', 'forder', 120);"
        )

    def test_firma_gui_result_filters_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"forder": "10"}

        out, command = pop_firma(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 600))
        self.assertEqual(command, "EEG = pop_firma(EEG, 'forder', 10);")


if __name__ == "__main__":
    unittest.main()
