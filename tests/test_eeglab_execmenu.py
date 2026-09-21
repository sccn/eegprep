"""Ports of the current EEGLAB ``eeglab_execmenu`` wrapper test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import eegprep
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.eeglab_execmenu import eeglab_execmenu
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.eeglab_tests import eeglab_test
from tests.fixtures import SAMPLE_DATASET_PATH


EEGLAB_EXECMENU_WRAPPER = "unittesting_adminfunc/eeglab_execmenu/adminfunc_eeglab_execmenu_wrapperTest.m"


def _workflow_calls(save_file: Path) -> list[tuple[str, str, list[object]]]:
    data = np.random.default_rng(37).random((10, 1000))
    montage = Path(eegprep.__file__).resolve().parent / "resources/headplot/Standard-10-5-Cap385.sfp"
    return [
        (
            "From ASCII/float file or MATLAB array",
            "pop_importdata",
            ["dataformat", "array", "nbchan", 0, "data", data, "srate", 100],
        ),
        ("Save current dataset as", "pop_saveset", [save_file]),
        ("Dataset info", "pop_editset", ["subject", "test2"]),
        ("Resave current dataset(s)", "pop_saveset", ["savemode", "resave"]),
        ("Load existing dataset", "pop_loadset", [SAMPLE_DATASET_PATH]),
        ("Event values", "pop_editeventvals", ["changefield", [1, "position", 3]]),
        ("About this dataset", "pop_comments", [["EEGLAB Tutorial Dataset", "test"]]),
        ("Channel locations", "pop_chanedit", ["lookup", montage]),
        ("Change sampling rate", "pop_resample", [64]),
    ]


def _run_workflow(calls: list[tuple[str, str, list[object]]]) -> EEGPrepSession:
    session = EEGPrepSession()
    for label, function, parameters in calls:
        command = eeglab_execmenu(label, function, parameters, session=session)
        assert command == session.LASTCOM
        assert session.ALLCOM[-1] == command
        assert session.CURRENTSET == [1]
        assert session.EEG is session.ALLEEG[0]
    return session


@eeglab_test(EEGLAB_EXECMENU_WRAPPER, "test_i_pass_general")
def test_current_eeglab_execmenu_workflow_updates_session_and_replays_deterministically(tmp_path: Path):
    calls = _workflow_calls(tmp_path / "test.set")
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    try:
        import_command = eeglab_execmenu(*calls[0], session=session)
        assert np.asarray(session.EEG["data"]).shape == (10, 1000)
        assert session.EEG["srate"] == 100
        assert import_command.startswith("EEG = pop_importdata(")

        eeglab_execmenu(*calls[1], session=session)
        assert (tmp_path / "test.set").is_file()
        assert session.EEG["filename"] == "test.set"

        eeglab_execmenu(*calls[2], session=session)
        assert session.EEG["subject"] == "test2"

        eeglab_execmenu(*calls[3], session=session)
        assert session.LASTCOM == "EEG = pop_saveset(EEG, 'savemode', 'resave');"

        eeglab_execmenu(*calls[4], session=session)
        assert np.asarray(session.EEG["data"]).shape == (32, 30504)
        assert len(session.ALLEEG) == 1

        eeglab_execmenu(*calls[5], session=session)
        assert session.EEG["event"][0]["position"] == 3

        eeglab_execmenu(*calls[6], session=session)
        assert session.EEG["comments"] == "EEGLAB Tutorial Dataset\ntest"

        eeglab_execmenu(*calls[7], session=session)
        assert session.EEG["chanlocs"][0]["labels"] == "FPz"
        assert session.EEG["chanlocs"][0]["urchan"] == 0
        assert np.isfinite(session.EEG["chanlocs"][0]["X"])
        assert "theta" not in session.EEG["chanlocs"][1]

        eeglab_execmenu(*calls[8], session=session)
        assert session.EEG["srate"] == 64
        assert np.asarray(session.EEG["data"]).shape == (32, 15252)
        assert session.EEG["event"][0]["position"] == 3
        assert session.EEG["comments"] == "EEGLAB Tutorial Dataset\ntest"
        assert len(session.ALLCOM) == len(calls)
        assert workspace.namespace["EEG"] is session.EEG
        assert workspace.namespace["ALLEEG"] is session.ALLEEG
        assert workspace.namespace["CURRENTSET"] == 1
        assert workspace.namespace["LASTCOM"] == session.LASTCOM
        assert workspace.namespace["ALLCOM"] is session.ALLCOM
    finally:
        workspace.close()

    replayed = _run_workflow(calls)
    np.testing.assert_array_equal(replayed.EEG["data"], session.EEG["data"])
    np.testing.assert_array_equal(
        [event["latency"] for event in replayed.EEG["event"]],
        [event["latency"] for event in session.EEG["event"]],
    )
    assert [event["type"] for event in replayed.EEG["event"]] == [event["type"] for event in session.EEG["event"]]
    assert replayed.EEG["event"][0]["position"] == session.EEG["event"][0]["position"] == 3
    assert replayed.EEG["comments"] == session.EEG["comments"]
    assert [chan["labels"] for chan in replayed.EEG["chanlocs"]] == [chan["labels"] for chan in session.EEG["chanlocs"]]
    np.testing.assert_allclose(
        [replayed.EEG["chanlocs"][0][field] for field in ("X", "Y", "Z")],
        [session.EEG["chanlocs"][0][field] for field in ("X", "Y", "Z")],
    )
    assert replayed.ALLCOM == session.ALLCOM


def test_execmenu_resave_uses_the_registered_action_default(tmp_path: Path):
    calls = _workflow_calls(tmp_path / "resave-default.set")
    session = EEGPrepSession()
    eeglab_execmenu(*calls[0], session=session)
    eeglab_execmenu(*calls[1], session=session)

    command = eeglab_execmenu("Resave current dataset(s)", "pop_saveset", None, session=session)

    assert command == "EEG = pop_saveset(EEG, 'savemode', 'resave');"
    assert session.EEG["saved"] == "yes"


def test_execmenu_accepts_explicit_keyword_parameters_without_cross_session_state():
    first = EEGPrepSession()
    second = EEGPrepSession()

    command = eeglab_execmenu(
        "From ASCII/float file or MATLAB array",
        "pop_importdata",
        {"data": np.ones((2, 12)), "dataformat": "array", "srate": 50},
        session=first,
    )

    assert first.CURRENTSET == [1]
    assert np.asarray(first.EEG["data"]).shape == (2, 12)
    assert "'srate', 50" in command
    assert second.CURRENTSET == []
    assert second.ALLCOM == []


@pytest.mark.parametrize(
    ("label", "function", "message"),
    [
        ("Not a menu", "pop_loadset", "Could not find menu"),
        ("Dataset info", "pop_resample", "registered for pop_editset"),
        ("File", "pop_loadset", "not an executable menu item"),
    ],
)
def test_execmenu_rejects_unregistered_label_function_pairs(label: str, function: str, message: str):
    session = EEGPrepSession()

    with pytest.raises(ValueError, match=message):
        eeglab_execmenu(label, function, [], session=session)

    assert session.CURRENTSET == []
    assert session.ALLCOM == []


def test_execmenu_rejects_registered_workflows_without_a_safe_parameterized_dispatcher():
    session = EEGPrepSession()

    with pytest.raises(NotImplementedError, match="pop_select"):
        eeglab_execmenu("Select data", "pop_select", ["point", [1, 10]], session=session)

    assert session.CURRENTSET == []
    assert session.ALLCOM == []


def test_execmenu_rejects_parameters_that_could_enable_a_dialog():
    session = EEGPrepSession()

    with pytest.raises(ValueError, match="controls parameter.*gui"):
        eeglab_execmenu(
            "From ASCII/float file or MATLAB array",
            "pop_importdata",
            {"data": np.ones((1, 4)), "gui": True},
            session=session,
        )


def test_eeglab_execmenu_is_public():
    assert eegprep.eeglab_execmenu is eeglab_execmenu
