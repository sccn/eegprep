import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_runica import pop_runica, pop_runica_dialog_spec, pop_runica_gui_options


def _eeg():
    return {
        "data": np.arange(80, dtype=np.float64).reshape(4, 20),
        "nbchan": 4,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "chanlocs": [
            {"labels": "Fz", "type": "EEG"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "HEOG", "type": "EOG"},
            {"labels": "VEOG", "type": "EOG"},
        ],
    }


class PopRunicaGuiTests(unittest.TestCase):
    def test_gui_channel_callbacks_ignore_empty_numpy_type_fields_like_eeglab(self):
        eeg = _eeg()
        eeg["chanlocs"] = [
            {"labels": "Fz", "type": np.array([], dtype=object)},
            {"labels": np.array("Cz", dtype=object), "type": "EEG"},
            {"labels": "HEOG", "type": np.array(["EOG"], dtype=object)},
            {"labels": "VEOG", "type": ""},
        ]

        controls = controls_by_tag(pop_runica_dialog_spec(eeg))

        self.assertEqual(controls["type_button"].callback.params["channels"], ("EEG", "EOG"))
        self.assertEqual(controls["chan_button"].callback.params["channels"], ("Fz", "Cz", "HEOG", "VEOG"))

    def test_gui_with_empty_numpy_channel_types_preserves_runica_options(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icatype": 1, "params": "'extended', 1, 'maxsteps', 2", "reorder": True, "chantype": ""}

        eeg = _eeg()
        eeg["chanlocs"][0]["type"] = np.array([], dtype=object)
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated) as runica:
            _out, com = pop_runica(eeg, gui=True, renderer=Renderer(), return_com=True)

        np.testing.assert_array_equal(runica.call_args.args[0]["data"], eeg["data"])
        self.assertEqual(runica.call_args.kwargs["extended"], 1)
        self.assertEqual(runica.call_args.kwargs["maxsteps"], 2)
        self.assertNotIn("lrate", runica.call_args.kwargs)
        self.assertEqual(
            com,
            "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 2, 'interrupt', 'on');",
        )

    def test_gui_options_do_not_inject_interrupt_for_non_runica_algorithms(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icatype": 4, "params": "'maxiter', 7", "reorder": True, "chantype": ""}

        options = pop_runica_gui_options(_eeg(), renderer=Renderer())

        self.assertIsNotNone(options)
        assert options is not None
        self.assertEqual(options["icatype"], "picard")
        self.assertEqual(options["options"], {"maxiter": 7})
        self.assertNotIn("interrupt", options["options"])

    def test_selectamica_forces_gui_with_amica_defaults(self):
        seen = {}
        amicaout = str(Path.cwd() / "amicaout")

        class Renderer:
            def run(self, spec, initial_values=None):
                seen["initial_values"] = initial_values
                return {
                    "icatype": initial_values["icatype"],
                    "params": initial_values["params"],
                    "reorder": True,
                    "chantype": "",
                }

        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_amica", return_value=updated) as amica:
            out, com = pop_runica(eeg, "selectamica", renderer=Renderer(), return_com=True)

        self.assertEqual(seen["initial_values"]["icatype"], 3)
        self.assertEqual(seen["initial_values"]["params"], f"'outdir', '{amicaout}'")
        amica.assert_called_once()
        self.assertEqual(amica.call_args.kwargs["outdir"], amicaout)
        self.assertNotIn("qsub", amica.call_args.kwargs)
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(com, f"EEG = pop_runica(EEG, 'icatype', 'runamica15', 'outdir', '{amicaout}');")

    def test_gui_numeric_chanind_keeps_one_based_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icatype": 1, "params": "'extended', 1, 'maxsteps', 2", "reorder": True, "chantype": "1 2"}

        eeg = _eeg()
        updated = dict(
            eeg,
            data=eeg["data"][:2],
            nbchan=2,
            chanlocs=eeg["chanlocs"][:2],
            icaweights=np.eye(2),
            icasphere=np.eye(2),
            icawinv=np.eye(2),
            icaact=np.zeros((2, 20, 1)),
        )
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated):
            out, com = pop_runica(eeg, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["icaweights"].shape, (2, 2))
        np.testing.assert_array_equal(out["icachansind"], np.array([0, 1]))
        self.assertEqual(
            com,
            "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 2, 'interrupt', 'on', 'chanind', [1 2]);",
        )

    def test_sample_data_pop_runica_does_not_surface_finite_matmul_warnings(self):
        eeg = pop_loadset("sample_data/eeglab_data.set")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            out, com = pop_runica(eeg, extended=1, maxsteps=1, return_com=True)

        self.assertIn("'maxsteps', 1", com)
        self.assertTrue(np.isfinite(out["icaweights"]).all())
        self.assertTrue(np.isfinite(out["icasphere"]).all())
        self.assertTrue(np.isfinite(out["icaact"]).all())
        self.assertFalse([warning for warning in captured if "matmul" in str(warning.message)])

    def test_gui_dataset_selection_routes_to_dataset_argument(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "icatype": 1,
                    "params": "'extended', 1",
                    "reorder": True,
                    "chantype": "",
                    "dataset": [2],
                    "concatenate": False,
                    "concatcond": False,
                }

        first = dict(_eeg(), setname="first")
        second = dict(_eeg(), setname="second")
        updated = dict(
            second, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1))
        )

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated) as runica:
            out, com = pop_runica([first, second], gui=True, renderer=Renderer(), return_com=True)

        runica.assert_called_once()
        self.assertIs(out[0], first)
        self.assertEqual(out[1]["icaweights"].shape, (4, 4))
        self.assertIn("'dataset', [2]", com)

    def test_picard_algorithm_routes_to_eeg_picard(self):
        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_picard", return_value=updated) as picard:
            out, com = pop_runica(
                eeg,
                icatype="picard",
                options={"maxiter": 7, "mode": "standard", "seed": 3},
                return_com=True,
            )

        picard.assert_called_once()
        self.assertEqual(picard.call_args.kwargs["max_iter"], 7)
        self.assertEqual(picard.call_args.kwargs["random_state"], 3)
        self.assertFalse(picard.call_args.kwargs["ortho"])
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(
            com,
            "EEG = pop_runica(EEG, 'icatype', 'picard', 'maxiter', 7, 'mode', 'standard', 'seed', 3);",
        )

    def test_existing_ica_is_saved_and_iclabel_removed_before_recompute(self):
        eeg = dict(
            _eeg(),
            icaweights=np.eye(4),
            icasphere=np.eye(4),
            icachansind=np.arange(4),
            etc={"ic_classification": {"ICLabel": {"version": "default"}}},
        )
        updated = dict(
            eeg, icaweights=np.eye(4) * 2, icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1))
        )

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated):
            out = pop_runica(eeg, options={"extended": 1})

        self.assertNotIn("ic_classification", out["etc"])
        np.testing.assert_array_equal(out["etc"]["oldicaweights"][0], np.eye(4))

    def test_qt_renderer_reads_listbox_as_one_based_row(self):
        class ListboxWidget:
            def property(self, name):
                return None

            def currentRow(self):
                return 1

            def currentIndex(self):
                raise AssertionError("QListWidget currentIndex returns a QModelIndex")

        self.assertEqual(QtDialogRenderer._read_widget(ListboxWidget()), 2)

    def test_qt_renderer_reads_multiselect_listbox_as_one_based_rows(self):
        class Index:
            def __init__(self, row):
                self._row = row

            def row(self):
                return self._row

        class ListboxWidget:
            def property(self, name):
                return True if name == "eegprep_multiselect" else None

            def selectedIndexes(self):
                return [Index(2), Index(0)]

            def currentRow(self):
                raise AssertionError("multi-select listboxes should read selected indexes")

        self.assertEqual(QtDialogRenderer._read_widget(ListboxWidget()), [1, 3])

    def test_qt_renderer_defaults_to_all_selected_dataset_rows(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")
        renderer = QtDialogRenderer()
        spec = pop_runica_dialog_spec([dict(_eeg(), setname="first"), dict(_eeg(), setname="second")])
        _app, dialog, widgets = renderer.build_dialog(spec)

        try:
            self.assertEqual(QtDialogRenderer._read_widget(widgets["dataset"]), [1, 2])
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
