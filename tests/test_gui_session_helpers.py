from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.adminfunc.eegh import eegh
from eegprep.functions.adminfunc import eeglab as eeglab_module
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.adminfunc.pop_delset import pop_delset
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_newset import pop_newset


def _eeg(*, name: str = "demo", saved: str = "no", xmin: float = 0.0, trials: int = 1) -> dict:
    pnts = 4
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": name,
            "filename": f"{name}.set",
            "filepath": "/tmp",
            "data": np.zeros((1, pnts, trials), dtype=np.float32) if trials > 1 else np.zeros((1, pnts), dtype=np.float32),
            "nbchan": 1,
            "pnts": pnts,
            "trials": trials,
            "srate": 100,
            "xmin": xmin,
            "xmax": xmin + (pnts - 1) / 100,
            "times": np.arange(pnts, dtype=float),
            "chanlocs": [{"labels": "Cz", "theta": 0.0, "radius": 0.0, "ref": "common"}],
            "event": np.array([{"type": "stim", "latency": 1}], dtype=object),
            "icaweights": np.eye(1),
            "icasphere": np.eye(1),
            "icawinv": np.eye(1),
            "icachansind": np.arange(1),
            "saved": saved,
        }
    )
    return eeg


def test_eeg_emptyset_includes_eeglab_reference_defaults():
    eeg = eeg_emptyset()

    assert eeg["ref"] == "common"
    assert eeg["urchanlocs"] == []
    assert eeg["run"] == []
    assert eeg["eventdescription"] == []
    assert eeg["epochdescription"] == []
    assert eeg["datfile"] == ""
    assert eeg["roi"] == {}


def test_eeg_store_appends_modified_dataset_as_unsaved():
    alleeg, checked, index = eeg_store([], _eeg(saved="no"), 0)

    assert index == 1
    assert checked["saved"] == "no"
    assert alleeg[0]["saved"] == "no"


def test_eeg_store_preserves_justloaded_dataset_as_saved():
    alleeg, checked, index = eeg_store([], _eeg(saved="justloaded"), 0)

    assert index == 1
    assert checked["saved"] == "yes"
    assert alleeg[0]["saved"] == "yes"


def test_eeg_store_keeps_loaded_saved_dataset_saved_when_appending():
    alleeg, checked, index = eeg_store([], _eeg(saved="yes"), 0)

    assert index == 1
    assert checked["saved"] == "yes"
    assert alleeg[0]["saved"] == "yes"


def test_eeg_store_handles_multiple_eeg_inputs_with_one_based_indices():
    alleeg, current, indices = eeg_store([], [_eeg(name="first"), _eeg(name="second")], [0, 0])

    assert indices == [1, 2]
    assert [eeg["setname"] for eeg in current] == ["first", "second"]
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second"]


def test_eeg_store_replaces_existing_one_based_slot():
    existing = [_eeg(name="first", saved="yes"), _eeg(name="second", saved="yes")]

    alleeg, checked, index = eeg_store(existing, _eeg(name="replacement", saved="no"), 2)

    assert index == 2
    assert checked["setname"] == "replacement"
    assert checked["saved"] == "no"
    assert [eeg["setname"] for eeg in alleeg] == ["first", "replacement"]


def test_eeg_store_appends_when_index_omitted_or_none():
    alleeg, checked, index = eeg_store(None, _eeg(name="first"), None)

    assert index == 1
    assert checked["setname"] == "first"
    assert alleeg[0]["setname"] == "first"


def test_eeg_store_rejects_mismatched_multiple_indices():
    with pytest.raises(ValueError, match="Length of EEG list"):
        eeg_store([], [_eeg(name="first"), _eeg(name="second")], [1])


def test_eeg_store_rejects_non_positive_explicit_index():
    with pytest.raises(ValueError, match="1-based"):
        eeg_store([], _eeg(), -1)


def test_eeg_retrieve_returns_deepcopy_and_one_based_index():
    source = _eeg(name="source")
    selected, alleeg, current = eeg_retrieve([source], 1)

    selected["setname"] = "changed"

    assert current == 1
    assert alleeg[0]["setname"] == "source"


def test_eeg_retrieve_handles_multiple_indices_and_empty_slots():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), {}, _eeg(name="third")], [1, 2, 3])

    assert current == [1, 2, 3]
    assert [eeg["setname"] for eeg in selected] == ["first", "", "third"]
    assert selected[1]["ref"] == "common"


def test_eeg_retrieve_accepts_tuple_indices():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), _eeg(name="second")], (2,))

    assert current == [2]
    assert [eeg["setname"] for eeg in selected] == ["second"]


def test_eeg_retrieve_rejects_zero_and_missing_indices():
    with pytest.raises(ValueError, match="1-based"):
        eeg_retrieve([_eeg()], 0)
    with pytest.raises(IndexError, match="No dataset"):
        eeg_retrieve([_eeg()], 2)


def test_pop_newset_stores_setname_and_retrieves_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(), 0, "setname", "renamed")

    assert current_set == 1
    assert current["setname"] == "renamed"
    assert alleeg[0]["setname"] == "renamed"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"

    _alleeg, retrieved, retrieved_set, retrieve_command = pop_newset(alleeg, current, current_set, "retrieve", 1)

    assert retrieved_set == 1
    assert retrieved["setname"] == "renamed"
    assert retrieve_command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 1);"


def test_pop_newset_rejects_unknown_keyword_options():
    with pytest.raises(ValueError, match="Unsupported pop_newset"):
        pop_newset([], _eeg(), 0, unknown=True)


def test_pop_newset_empty_retrieve_option_stores_current_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(name="stored"), 0, "retrieve", [])

    assert current_set == 1
    assert current["setname"] == "stored"
    assert alleeg[0]["setname"] == "stored"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"


def test_pop_delset_rejects_non_positive_indices():
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], -1)
    with pytest.raises(ValueError, match="1-based"):
        pop_delset([_eeg()], 0)


def test_pop_delset_deletes_unique_indices_in_reverse_order():
    alleeg, command = pop_delset([_eeg(name="first"), _eeg(name="second"), _eeg(name="third")], [1, 3, 3])

    assert [eeg["setname"] for eeg in alleeg] == ["second"]
    assert command == "ALLEEG = pop_delset( ALLEEG, [1, 3, 3] );"


def test_eegh_records_only_non_empty_commands():
    history = []

    assert eegh(" EEG = pop_reref(EEG); ", history) == "EEG = pop_reref(EEG);"
    assert eegh("", history) == ""
    assert eegh(None, history) == ""
    assert history == ["EEG = pop_reref(EEG);"]


def test_session_treats_nonzero_xmin_single_trial_data_as_continuous():
    session = EEGPrepSession()
    session.EEG = _eeg(xmin=1.5, trials=1)

    assert session.menu_statuses() == {"continuous_dataset"}


def test_session_dataset_summaries_include_empty_dataset_structs():
    session = EEGPrepSession()
    session.ALLEEG = [eeg_emptyset(), {}, _eeg(name="real")]
    session.CURRENTSET = [1]

    assert session.dataset_summaries() == [
        (1, "Dataset 1:(no dataset name)", True),
        (3, "Dataset 3:real", False),
    ]


def test_show_help_missing_resource_raises_clear_error_not_coming_soon():
    dispatcher = MenuActionDispatcher(EEGPrepSession())

    with (
        mock.patch(
            "eegprep.functions.guifunc.menu_actions.pophelp",
            side_effect=FileNotFoundError("missing packaged help"),
        ),
        mock.patch.object(dispatcher, "show_coming_soon") as coming_soon,
        pytest.raises(FileNotFoundError, match="missing packaged help"),
    ):
        dispatcher.dispatch("help:missing")

    coming_soon.assert_not_called()


def test_dispatch_gui_reraises_headless_errors_and_logs_traceback(caplog):
    dispatcher = MenuActionDispatcher(EEGPrepSession())

    with (
        mock.patch.object(dispatcher, "dispatch", side_effect=RuntimeError("boom")),
        caplog.at_level(logging.ERROR, logger="eegprep.functions.guifunc.menu_actions"),
        pytest.raises(RuntimeError, match="boom"),
    ):
        dispatcher.dispatch_gui("pop_reref")

    assert "EEGPrep GUI menu action failed: pop_reref" in caplog.text


def test_eeglab_versions_and_nogui_entry_points():
    import eegprep

    session = EEGPrepSession()

    assert eeglab_module.eeglab("versions") == eegprep.__version__
    assert eeglab_module.eeglab("nogui", session=session, show=False) is session


def test_eeglab_full_mode_builds_window_without_showing():
    session = EEGPrepSession()
    window = mock.Mock()

    with mock.patch.object(eeglab_module, "build_main_window", return_value=window) as build:
        returned = eeglab_module.eeglab("full", session=session, show=False, include_plugins=False)

    assert returned is window
    build.assert_called_once_with(session, all_menus=True, include_plugins=False)
    window.show.assert_not_called()
    window.exec.assert_not_called()


def test_eeglab_show_and_block_paths():
    session = EEGPrepSession()
    window = mock.Mock()
    window.exec.return_value = 7

    with mock.patch.object(eeglab_module, "build_main_window", return_value=window):
        assert eeglab_module.eeglab(session=session, show=True) is window
        assert eeglab_module.eeglab(session=session, block=True) == 7

    assert window.show.call_count == 1
    assert window.exec.call_count == 1


def test_eeglab_main_parses_nogui_and_full_plugin_options():
    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--nogui"]) == 0
        eeglab.assert_called_once_with("nogui", show=False)

    with mock.patch.object(eeglab_module, "eeglab") as eeglab:
        assert eeglab_module.main(["--full", "--no-plugins"]) == 0
        eeglab.assert_called_once_with("full", block=True, include_plugins=False)
