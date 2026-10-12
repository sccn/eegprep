import asyncio
import unittest
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, _apply_save_metadata
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.plugins.ICLabel.iclabel import iclabel
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel, pop_iclabel_async
from tests.fixtures import gate_iclabel_backend, load_sample_ica_eeg

ICLABEL_ASYNC_COMMAND = "EEG = await pop_iclabel_async(EEG, 'default');"


def _eeg():
    return {
        "data": np.zeros((2, 20), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
        "icawinv": np.eye(2),
        "icachansind": np.arange(2),
        "etc": {},
    }


class _Renderer:
    """Headless ICLabel version dialog that answers like a user."""

    def __init__(self, values):
        self.values = values

    def run(self, spec, initial_values=None):
        return self.values


class PopIclabelGuiTests(unittest.TestCase):
    def test_gui_result_runs_iclabel_and_returns_history(self):
        eeg = _eeg()
        updated = dict(eeg, etc={"ic_classification": {"ICLabel": {"version": "default"}}})
        with mock.patch("eegprep.plugins.ICLabel.pop_iclabel.iclabel", return_value=updated) as classify:
            out, com = pop_iclabel(eeg, gui=True, renderer=_Renderer({"icversion": 1}), return_com=True)

        classify.assert_called_once_with(eeg, algorithm="default", engine=None)
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "default")
        self.assertEqual(com, "EEG = pop_iclabel(EEG, 'default');")


@pytest.fixture(scope="module")
def sample_ica_eeg():
    return load_sample_ica_eeg()


def _classifications(eeg):
    return eeg["etc"]["ic_classification"]["ICLabel"]["classifications"]


def _session(sample_ica_eeg, *setnames):
    session = EEGPrepSession()
    for setname in setnames:
        session.store_current(dict(sample_ica_eeg, setname=setname), new=True)
    return session


def test_pop_iclabel_async_returns_the_sync_classification_and_replayable_history(sample_ica_eeg):
    expected = _classifications(iclabel(sample_ica_eeg))

    output, command = asyncio.run(pop_iclabel_async(sample_ica_eeg, "default", gui=False, return_com=True))

    np.testing.assert_array_equal(_classifications(output), expected)
    assert command == ICLABEL_ASYNC_COMMAND
    assert "ic_classification" not in sample_ica_eeg["etc"]


def test_pop_iclabel_async_classifies_a_dataset_list_in_input_order(sample_ica_eeg):
    datasets = [dict(sample_ica_eeg, setname="first"), dict(sample_ica_eeg, setname="second")]

    output, command = asyncio.run(pop_iclabel_async(datasets, "default", gui=False, return_com=True))

    assert [dataset["setname"] for dataset in output] == ["first", "second"]
    assert all(_classifications(dataset).shape == (32, 7) for dataset in output)
    assert command == ICLABEL_ASYNC_COMMAND


def test_gui_async_iclabel_action_stores_the_real_classification(sample_ica_eeg):
    session = _session(sample_ica_eeg, "sample")
    expected = _classifications(iclabel(session.EEG))

    asyncio.run(
        MenuActionDispatcher(session).dispatch_gui_async("pop_iclabel", None, renderer=_Renderer({"icversion": 1}))
    )

    np.testing.assert_array_equal(_classifications(session.EEG), expected)
    assert session.EEG is session.ALLEEG[0]
    assert session.ALLCOM == [ICLABEL_ASYNC_COMMAND]
    assert session.CURRENTSET == [1]


def test_gui_async_iclabel_action_cancelled_dialog_changes_nothing(sample_ica_eeg):
    session = _session(sample_ica_eeg, "sample")
    token = session.dataset_state_token()

    asyncio.run(MenuActionDispatcher(session).dispatch_gui_async("pop_iclabel", None, renderer=_Renderer(None)))

    assert session.dataset_state_token() == token
    assert session.ALLCOM == []
    assert "ic_classification" not in session.EEG["etc"]


def _run_during_gui_iclabel(session, backend, during):
    """Start the GUI ICLabel action, call ``during`` while inference is suspended, then finish it."""
    dispatcher = MenuActionDispatcher(session)

    async def scenario():
        task = asyncio.create_task(
            dispatcher.dispatch_gui_async("pop_iclabel", None, renderer=_Renderer({"icversion": 1}))
        )
        await backend.started.wait()
        try:
            during(session)
        finally:
            backend.release.set()
        await task

    asyncio.run(scenario())


def _edit_data(session):
    session.EEG["data"][0, 0] = 99.0


def _putmask(session):
    np.putmask(session.EEG["data"], session.EEG["data"] > 0, 0.0)


def _store_new(session):
    session.store_current(dict(session.EEG, setname="stored as new"), new=True)


@pytest.mark.parametrize(
    ("during", "message", "edit_survives"),
    [
        (_edit_data, "the selected dataset was modified", lambda session: session.EEG["data"][0, 0] == 99.0),
        (_putmask, "the selected dataset was modified", lambda session: not (session.EEG["data"] > 0).any()),
        (
            lambda session: session.retrieve(1),
            "the dataset selection changed",
            lambda session: session.CURRENTSET == [1],
        ),
        (_store_new, "the dataset selection changed", lambda session: session.EEG["setname"] == "stored as new"),
        (
            lambda session: _apply_save_metadata(session.EEG, "/data/first_save.set"),
            "the selected dataset was modified",
            lambda session: session.EEG["filename"] == "first_save.set",
        ),
    ],
    ids=["data-edit", "putmask", "retrieve", "store-new", "first-save-names-the-file"],
)
def test_gui_edit_during_async_iclabel_discards_the_result_and_keeps_the_edit(
    sample_ica_eeg, monkeypatch, during, message, edit_survives
):
    session = _session(sample_ica_eeg, "first", "second")
    backend = gate_iclabel_backend(monkeypatch)

    with pytest.raises(RuntimeError, match=f"ICLabel result discarded because {message}"):
        _run_during_gui_iclabel(session, backend, during)

    assert edit_survives(session)
    assert ICLABEL_ASYNC_COMMAND not in session.ALLCOM
    assert all("ic_classification" not in dataset["etc"] for dataset in session.ALLEEG)
    assert "ic_classification" not in session.EEG["etc"]


@pytest.mark.parametrize(
    "during",
    [
        lambda session: session.notify_changed(),
        lambda session: session.add_history("pop_eegplot(EEG, 1, 1, 1);"),
        lambda session: session.mark_current_saved(),
        lambda session: session.ALLEEG[0].update(setname="unselected edit"),
    ],
    ids=["refresh", "plot-history", "marks-saved", "unselected-dataset"],
)
def test_gui_benign_actions_during_async_iclabel_keep_the_result(sample_ica_eeg, monkeypatch, during):
    session = _session(sample_ica_eeg, "first", "second")
    backend = gate_iclabel_backend(monkeypatch)

    _run_during_gui_iclabel(session, backend, during)

    assert session.CURRENTSET == [2]
    assert _classifications(session.EEG).shape == (32, 7)
    assert session.ALLCOM[-1] == ICLABEL_ASYNC_COMMAND


@pytest.mark.parametrize("version", ["lite", "beta"])
def test_browser_requests_for_other_networks_do_not_advise_unavailable_engines(monkeypatch, version):
    """Under Emscripten the MATLAB and Octave engines do not exist, so the error must not point at them."""
    monkeypatch.setattr("eegprep.plugins.ICLabel.iclabel.IS_EMSCRIPTEN", True)

    with pytest.raises(NotImplementedError, match="available in the browser") as excinfo:
        asyncio.run(pop_iclabel_async(_eeg(), version, gui=False))

    assert "engine='matlab'" not in str(excinfo.value)


def test_gui_second_iclabel_dispatch_while_the_first_is_pending_is_discarded(sample_ica_eeg, monkeypatch):
    session = _session(sample_ica_eeg, "sample")
    backend = gate_iclabel_backend(monkeypatch)
    dispatcher = MenuActionDispatcher(session)

    async def scenario():
        runs = [
            asyncio.create_task(
                dispatcher.dispatch_gui_async("pop_iclabel", None, renderer=_Renderer({"icversion": 1}))
            )
            for _ in range(2)
        ]
        while backend.calls < 2:
            await asyncio.sleep(0)
        backend.release.set()
        return await asyncio.gather(*runs, return_exceptions=True)

    first, second = asyncio.run(scenario())

    assert first is None
    assert isinstance(second, RuntimeError)
    assert "concurrent ICLabel call" in str(second)
    assert session.ALLCOM == [ICLABEL_ASYNC_COMMAND]
    assert _classifications(session.EEG).shape == (32, 7)


if __name__ == "__main__":
    unittest.main()
