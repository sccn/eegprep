import asyncio
import unittest
from unittest import mock

import numpy as np

import eegprep.plugins.ICLabel.pop_iclabel as pop_iclabel_module
import eegprep.functions.guifunc.menu_actions as menu_actions_module
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel, pop_iclabel_async


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

        dispatcher = MenuActionDispatcher(session)

        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            async def classify(_selection, *, renderer=None, return_com=False):
                started.set()
                await release.wait()
                return (dict(original, setname="classified"), "EEG = await pop_iclabel_async(EEG, 'default');")

            with (
                mock.patch.object(menu_actions_module, "_IS_EMSCRIPTEN", True),
                mock.patch.object(pop_iclabel_module, "pop_iclabel_async", side_effect=classify),
            ):
                task = asyncio.create_task(dispatcher.dispatch_gui("pop_iclabel"))
                await started.wait()
                session.mark_current_saved()
                session.store_current(_eeg("second"), new=True)
                release.set()
                with self.assertRaisesRegex(RuntimeError, "discarded"):
                    await task

        asyncio.run(scenario())

        self.assertIs(session.ALLEEG[0], original)
        self.assertEqual(session.ALLEEG[0]["saved"], "yes")
        self.assertEqual(session.ALLEEG[1]["setname"], "second")
        self.assertEqual(session.ALLCOM, [])


if __name__ == "__main__":
    unittest.main()
