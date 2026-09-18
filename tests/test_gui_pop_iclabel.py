import asyncio
import unittest
from unittest import mock

import numpy as np

import eegprep.plugins.ICLabel.pop_iclabel as pop_iclabel_module
import eegprep.functions.guifunc.menu_actions as menu_actions_module
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel, pop_iclabel_async, pop_iclabel_dialog_spec


def _eeg(setname="demo"):
    return {
        "data": np.zeros((2, 20), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "xmin": 0.0,
        "xmax": 0.19,
        "times": np.arange(20) / 100,
        "event": [],
        "urevent": [],
        "epoch": [],
        "chanlocs": [],
        "chaninfo": {},
        "setname": setname,
        "history": "",
        "ref": "",
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
        "icawinv": np.eye(2),
        "icachansind": np.arange(2),
        "etc": {},
    }


class PopIclabelGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_iclabel_prompt(self):
        spec = pop_iclabel_dialog_spec()

        self.assertEqual(spec.title, "ICLabel")
        self.assertEqual(spec.function_name, "pop_iclabel")
        self.assertEqual(spec.eeglab_source, "plugins/ICLabel/pop_iclabel.m")
        self.assertIsNone(spec.help_text)
        self.assertFalse(spec.show_help_button)
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls],
            [
                ("text", "Select which icversion of ICLabel to use:", None),
                ("popupmenu", "Default (recommended)|Lite|Beta", "icversion"),
            ],
        )

    def test_gui_result_runs_iclabel_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icversion": 1}

        eeg = _eeg()
        updated = dict(eeg, etc={"ic_classification": {"ICLabel": {"version": "default"}}})
        with mock.patch("eegprep.plugins.ICLabel.pop_iclabel.iclabel", return_value=updated) as classify:
            out, com = pop_iclabel(eeg, gui=True, renderer=Renderer(), return_com=True)

        classify.assert_called_once_with(eeg, algorithm="default", engine=None)
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "default")
        self.assertEqual(com, "EEG = pop_iclabel(EEG, 'default');")

    def test_python_engine_rejects_unbundled_lite_and_beta_networks(self):
        eeg = _eeg()

        with self.assertRaisesRegex(NotImplementedError, "standalone Python ICLabel only ships the default network"):
            pop_iclabel(eeg, "lite")

    def test_matlab_engine_can_request_lite_network(self):
        eeg = _eeg()
        updated = dict(eeg, etc={"ic_classification": {"ICLabel": {"version": "lite"}}})

        with mock.patch("eegprep.plugins.ICLabel.pop_iclabel.iclabel", return_value=updated) as classify:
            out, com = pop_iclabel(eeg, "lite", engine="matlab", return_com=True)

        classify.assert_called_once_with(eeg, algorithm="lite", engine="matlab")
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "lite")
        self.assertEqual(com, "EEG = pop_iclabel(EEG, 'lite');")

    def test_missing_ica_raises_clear_error(self):
        eeg = dict(_eeg(), icaweights=np.array([]))

        with self.assertRaisesRegex(ValueError, "requires an ICA decomposition"):
            pop_iclabel(eeg, "default")

    def test_async_result_has_replayable_history_command(self):
        async def run():
            updated = dict(_eeg(), etc={"ic_classification": {"ICLabel": {"version": "default"}}})
            with mock.patch.object(pop_iclabel_module, "iclabel_async", new=mock.AsyncMock(return_value=updated)):
                return await pop_iclabel_async(_eeg(), "default", gui=False, return_com=True)

        out, command = asyncio.run(run())

        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "default")
        self.assertEqual(command, "EEG = await pop_iclabel_async(EEG, 'default');")

    def test_async_list_recursion_preserves_input_order(self):
        first = _eeg()
        second = _eeg()
        updated_first = dict(first, setname="first")
        updated_second = dict(second, setname="second")

        async def run():
            with mock.patch.object(
                pop_iclabel_module,
                "iclabel_async",
                new=mock.AsyncMock(side_effect=[updated_first, updated_second]),
            ):
                return await pop_iclabel_async([first, second], "default", gui=False, return_com=True)

        output, command = asyncio.run(run())

        self.assertEqual([item["setname"] for item in output], ["first", "second"])
        self.assertEqual(command, "EEG = await pop_iclabel_async(EEG, 'default');")

    def test_sync_entry_point_fails_fast_under_emscripten(self):
        with mock.patch.object(pop_iclabel_module, "_IS_EMSCRIPTEN", True):
            with self.assertRaisesRegex(RuntimeError, r'or await pop_iclabel_async\(\.\.\.\)'):
                pop_iclabel(None)

    def test_emscripten_gui_dispatch_awaits_and_commits_to_original_slot(self):
        session = EEGPrepSession()
        session.store_current(_eeg(), new=True)
        updated = dict(session.EEG, setname="classified")
        command = "EEG = await pop_iclabel_async(EEG, 'default');"

        async def classify(selection, *, renderer=None, return_com=False):
            self.assertIs(selection, session.EEG)
            self.assertIsNone(renderer)
            return (updated, command) if return_com else updated

        dispatcher = MenuActionDispatcher(session)
        with (
            mock.patch.object(menu_actions_module, "_IS_EMSCRIPTEN", True),
            mock.patch.object(pop_iclabel_module, "pop_iclabel_async", side_effect=classify),
        ):
            asyncio.run(dispatcher.dispatch_gui("pop_iclabel"))

        self.assertEqual(session.EEG["setname"], "classified")
        self.assertEqual(session.CURRENTSET, [1])
        self.assertEqual(session.ALLCOM, [command])

    def test_emscripten_gui_dispatch_discards_stale_result(self):
        session = EEGPrepSession()
        session.store_current(_eeg("first"), new=True)
        original = session.ALLEEG[0]

        async def classify(_selection, *, renderer=None, return_com=False):
            session.store_current(_eeg("second"), new=True)
            return (dict(original, setname="classified"), "EEG = await pop_iclabel_async(EEG, 'default');")

        dispatcher = MenuActionDispatcher(session)
        with (
            mock.patch.object(menu_actions_module, "_IS_EMSCRIPTEN", True),
            mock.patch.object(pop_iclabel_module, "pop_iclabel_async", side_effect=classify),
        ):
            with self.assertRaisesRegex(RuntimeError, "session changed"):
                asyncio.run(dispatcher.dispatch_gui("pop_iclabel"))

        self.assertIs(session.ALLEEG[0], original)
        self.assertEqual(session.ALLEEG[1]["setname"], "second")
        self.assertEqual(session.ALLCOM, [])


if __name__ == "__main__":
    unittest.main()
