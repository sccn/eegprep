from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.select_multiple_datasets import select_multiple_datasets
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit
from eegprep.functions.popfunc.pop_copyset import pop_copyset
from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield
from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals
from eegprep.functions.popfunc.pop_fileio_brainvision_mat import pop_fileio_brainvision_mat
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_rmdat import pop_rmdat
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent
from tests.fixtures import SAMPLE_DATASET_PATH, matlab_engine_available

EEGLAB_REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "eegprep" / "eeglab"


def eeglab_reference_available() -> bool:
    return (
        (EEGLAB_REFERENCE_ROOT / "functions" / "popfunc" / "pop_selectevent.m").exists()
        and (EEGLAB_REFERENCE_ROOT / "functions" / "popfunc" / "pop_mergeset.m").exists()
        and (EEGLAB_REFERENCE_ROOT / "plugins" / "clean_rawdata" / "private").is_dir()
    )


def _eeg(setname: str = "demo") -> dict:
    return {
        "setname": setname,
        "filename": "",
        "filepath": "",
        "subject": "",
        "condition": "",
        "group": "",
        "session": "",
        "comments": "",
        "data": np.arange(200, dtype=np.float64).reshape(2, 100),
        "nbchan": 2,
        "pnts": 100,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.99,
        "times": np.arange(100, dtype=float),
        "ref": "common",
        "chanlocs": [
            {"labels": "Cz", "theta": 0.0, "radius": 0.25, "X": 0.0, "Y": 1.0, "Z": 0.0},
            {"labels": "Pz", "theta": 180.0, "radius": 0.30, "X": 0.0, "Y": -1.0, "Z": 0.0},
        ],
        "urchanlocs": [],
        "chaninfo": {},
        "event": [
            {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 1},
            {"type": "resp", "latency": 50.0, "duration": 0.0, "urevent": 2},
        ],
        "urevent": [
            {"type": "stim", "latency": 10.0, "duration": 0.0},
            {"type": "resp", "latency": 50.0, "duration": 0.0},
        ],
        "epoch": [],
        "eventdescription": {},
        "reject": {},
        "stats": {},
        "specdata": {},
        "specicaact": {},
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
        "history": "",
        "saved": "no",
    }


def _assert_python_echo_is_parseable(command: str) -> None:
    ast.parse(_console_python_command(command))


def test_pop_editeventfield_adds_renames_deletes_fields_and_updates_urevent():
    eeg = _eeg()

    out, command = pop_editeventfield(eeg, "condition", ["target", "button"], return_com=True)

    assert eeg["event"][0].get("condition") is None
    assert [event["condition"] for event in out["event"]] == ["target", "button"]
    assert out["urevent"][0]["condition"] == "target"
    assert "pop_editeventfield" in command
    _assert_python_echo_is_parseable(command)

    renamed = pop_editeventfield(out, "rename", "condition->trialtype")
    assert "condition" not in renamed["event"][0]
    assert renamed["event"][0]["trialtype"] == "target"

    deleted = pop_editeventfield(renamed, "trialtype", [])
    assert "trialtype" not in deleted["event"][0]


def test_pop_editeventvals_change_insert_delete_and_sort_events():
    eeg = _eeg()

    out, command = pop_editeventvals(eeg, "changefield", [1, "type", "target"], return_com=True)
    assert out["event"][0]["type"] == "target"
    assert out["urevent"][0]["type"] == "target"
    _assert_python_echo_is_parseable(command)

    inserted = pop_editeventvals(out, "insert", [2, "new", 25.0, 0.0, 3])
    assert len(inserted["event"]) == 3
    assert inserted["event"][1]["type"] == "new"

    sorted_out = pop_editeventvals(inserted, "sort", ["latency"])
    assert [event["latency"] for event in sorted_out["event"]] == sorted(
        [event["latency"] for event in sorted_out["event"]]
    )

    deleted = pop_editeventvals(sorted_out, "delete", [2])
    assert len(deleted["event"]) == 2


def test_pop_editeventvals_insert_preserves_existing_urevent_links():
    eeg = _eeg()
    eeg["event"][0]["urevent"] = 2
    eeg["event"][1]["urevent"] = 1
    eeg["urevent"] = [
        {"type": "resp", "latency": 50.0, "duration": 0.0},
        {"type": "stim", "latency": 10.0, "duration": 0.0},
    ]

    inserted = pop_editeventvals(eeg, "insert", [2, "new", 25.0, 0.0, 99])

    assert [event["urevent"] for event in inserted["event"]] == [2, 3, 1]
    assert [event["type"] for event in inserted["urevent"]] == ["resp", "stim", "new"]


def test_pop_selectevent_matches_nonnumeric_event_types_without_selecting_every_string():
    eeg = _eeg()
    eeg["event"][0]["type"] = "target"

    out, selected = pop_selectevent(eeg, "type", "target", "deleteevents", "on")

    assert selected == [1]
    assert [event["type"] for event in out["event"]] == ["target"]


def test_pop_selectevent_renames_selected_types_and_keeps_old_type_field():
    out, command = pop_selectevent(
        _eeg(),
        "type",
        "stim",
        "renametype",
        "target",
        "oldtypefield",
        "oldtype",
        return_com=True,
    )

    assert out["event"][0]["type"] == "target"
    assert out["event"][0]["oldtype"] == "stim"
    assert out["event"][1]["type"] == "resp"
    _assert_python_echo_is_parseable(command)


def test_pop_selectevent_renames_events_before_epoched_trial_selection():
    eeg = _eeg()
    eeg["data"] = eeg["data"].reshape(2, 50, 2)
    eeg["pnts"] = 50
    eeg["trials"] = 2
    eeg["times"] = np.arange(50, dtype=float)
    eeg["event"] = [
        {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 1, "epoch": 1},
        {"type": "resp", "latency": 60.0, "duration": 0.0, "urevent": 2, "epoch": 2},
    ]

    out, selected = pop_selectevent(
        eeg,
        "type",
        "stim",
        "renametype",
        "target",
        "deleteepochs",
        "on",
        "deleteevents",
        "off",
    )

    assert selected == [1]
    assert out["trials"] == 1
    assert any(event["type"] == "target" for event in out["event"])


def test_pop_selectevent_keeps_numeric_boundary_when_deleting_continuous_events():
    eeg = _eeg()
    eeg["event"].insert(1, {"type": -1, "latency": 25.0, "duration": 0.0, "urevent": 3})
    eeg["urevent"].append({"type": -1, "latency": 25.0, "duration": 0.0})

    out, selected = pop_selectevent(eeg, "type", "stim", "deleteevents", "on")

    assert selected == [1, 2]
    assert [event["type"] for event in out["event"]] == ["stim", -1]


def test_pop_rmdat_removes_or_keeps_continuous_windows_around_events():
    eeg = _eeg()

    removed, command = pop_rmdat(eeg, ["stim"], [-0.01, 0.01], 1, return_com=True)
    kept = pop_rmdat(eeg, ["stim"], [-0.01, 0.01], 0)

    assert removed["pnts"] < eeg["pnts"]
    assert kept["pnts"] < eeg["pnts"]
    assert kept["pnts"] < removed["pnts"]
    _assert_python_echo_is_parseable(command)


def test_pop_rmdat_matches_sorted_event_behavior_when_events_are_unsorted():
    unsorted_eeg = _eeg()
    unsorted_eeg["event"] = [
        {"type": "stim", "latency": 50.0, "duration": 0.0, "urevent": 2},
        {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 1},
    ]
    sorted_eeg = deepcopy(unsorted_eeg)
    sorted_eeg["event"] = list(reversed(unsorted_eeg["event"]))

    unsorted_out = pop_rmdat(unsorted_eeg, ["stim"], [-0.01, 0.01], 1)
    sorted_out = pop_rmdat(sorted_eeg, ["stim"], [-0.01, 0.01], 1)

    assert unsorted_out["pnts"] == sorted_out["pnts"]
    assert np.array_equal(unsorted_out["data"], sorted_out["data"])


def test_pop_chanedit_changes_fields_converts_coordinates_and_round_trips_files(tmp_path):
    eeg = _eeg()
    loc_file = tmp_path / "locs.ced"

    out, command = pop_chanedit(
        eeg,
        "changefield",
        [1, "labels", "Fz"],
        "convert",
        "cart2all",
        "save",
        loc_file,
        return_com=True,
    )

    assert out["chanlocs"][0]["labels"] == "Fz"
    assert out["chanlocs"][0]["sph_radius"] == pytest.approx(1.0)
    assert loc_file.exists()
    _assert_python_echo_is_parseable(command)

    loaded = pop_chanedit(eeg, "load", loc_file)
    assert loaded["chanlocs"][0]["labels"] == "Fz"


def test_pop_chanedit_applies_same_edit_to_selected_datasets():
    first = _eeg("first")
    second = _eeg("second")

    outputs, command = pop_chanedit([first, second], "changefield", [1, "type", "EEG"], return_com=True)

    assert [output["chanlocs"][0]["type"] for output in outputs] == ["EEG", "EEG"]
    _assert_python_echo_is_parseable(command)


def test_pop_chanedit_gui_unchanged_submission_does_not_emit_history():
    class UnchangedRenderer:
        def run(self, spec, initial_values=None):
            return {control.tag: control.value for control in spec.controls if control.tag}

    eeg = _eeg()

    output, command = pop_chanedit(eeg, gui=True, renderer=UnchangedRenderer(), return_com=True)

    assert output["chanlocs"][0]["labels"] == eeg["chanlocs"][0]["labels"]
    assert command == ""


def test_pop_chanedit_reads_comma_delimited_ced_with_comments(tmp_path):
    loc_file = tmp_path / "locs.ced"
    loc_file.write_text("% exported by EEGLAB\nlabels,X,Y,Z\nFz,0,1,0\nCz,0,0,1\n", encoding="utf-8")

    loaded = pop_chanedit(_eeg(), "load", loc_file)

    assert [chan["labels"] for chan in loaded["chanlocs"]] == ["Fz", "Cz"]
    assert loaded["chanlocs"][0]["X"] == 0


def test_pop_copyset_uses_one_based_indices_and_preserves_source_order():
    first = _eeg("first")
    second = _eeg("second")

    alleeg, eeg, current_set, command = pop_copyset([first, second], 1, 3, return_com=True)

    assert len(alleeg) == 3
    assert current_set == 3
    assert eeg["setname"] == "first"
    assert alleeg[0]["setname"] == "first"
    assert alleeg[2]["setname"] == "first"
    assert "LASTCOM" in command
    _assert_python_echo_is_parseable(command)


def test_pop_mergeset_continuous_offsets_events_and_inserts_boundary():
    first = _eeg("first")
    second = _eeg("second")
    second["event"][0]["latency"] = 5.0

    merged, command = pop_mergeset([first, second], [1, 2], return_com=True)

    assert merged["pnts"] == first["pnts"] + second["pnts"]
    assert any(event["type"] == "boundary" and event["latency"] == first["pnts"] + 0.5 for event in merged["event"])
    assert any(event["type"] == "stim" and event["latency"] == first["pnts"] + 5.0 for event in merged["event"])
    assert merged["icaweights"].size == 0
    _assert_python_echo_is_parseable(command)


def test_pop_mergeset_gui_uses_selected_indices_as_defaults():
    first = _eeg("first")
    second = _eeg("second")
    seen = {}

    class MergeRenderer:
        def run(self, spec, initial_values=None):
            controls = {control.tag: control.value for control in spec.controls if control.tag}
            seen["indices"] = controls["indices"]
            return controls

    merged, command = pop_mergeset([first, second], [2, 1], gui=True, renderer=MergeRenderer(), return_com=True)

    assert seen["indices"] == "2 1"
    assert merged["pnts"] == first["pnts"] + second["pnts"]
    assert command == "EEG = pop_mergeset( ALLEEG, [2 1], 0);"


def test_pop_fileio_brainvision_mat_delegates_to_fileio_mat_import(monkeypatch, tmp_path):
    mat_path = tmp_path / "brainvision.mat"
    mat_path.write_bytes(b"placeholder")
    imported = _eeg("brainvision")
    imported["history"] = "EEG = pop_fileio('brainvision.mat');"

    def fake_pop_fileio(filename, *, return_com=False, **kwargs):
        assert filename == mat_path
        assert return_com is True
        assert kwargs == {"dataformat": "matlab"}
        return imported, "EEG = pop_fileio('brainvision.mat');"

    monkeypatch.setattr(
        "eegprep.functions.popfunc.pop_fileio_brainvision_mat.pop_fileio",
        fake_pop_fileio,
    )

    out, command = pop_fileio_brainvision_mat(mat_path, dataformat="matlab", return_com=True)

    assert out is imported
    assert command == f"EEG = pop_fileio_brainvision_mat('{mat_path.as_posix()}');"
    assert out["history"] == f"EEG = pop_fileio('brainvision.mat');\n{command}"
    _assert_python_echo_is_parseable(command)


def test_pop_fileio_brainvision_mat_escapes_quote_paths(monkeypatch, tmp_path):
    mat_path = tmp_path / "brain'vision.mat"
    mat_path.write_bytes(b"placeholder")
    imported = _eeg("brainvision")

    monkeypatch.setattr(
        "eegprep.functions.popfunc.pop_fileio_brainvision_mat.pop_fileio",
        lambda filename, *, return_com=False, **kwargs: (imported, ""),
    )

    _out, command = pop_fileio_brainvision_mat(mat_path, return_com=True)

    assert "brain''vision.mat" in command
    _assert_python_echo_is_parseable(command)


def test_pop_fileio_brainvision_mat_rejects_non_mat_files(tmp_path):
    with pytest.raises(ValueError, match="\\.mat"):
        pop_fileio_brainvision_mat(tmp_path / "recording.vhdr")


def test_select_multiple_datasets_preserves_order_and_updates_session_history_contract():
    session = EEGPrepSession()
    session.store_current(_eeg("first"), new=True)
    session.store_current(_eeg("second"), new=True)

    eeg, command = select_multiple_datasets(session, [2, 1], return_com=True)

    assert [item["setname"] for item in eeg] == ["second", "first"]
    assert session.CURRENTSET == [2, 1]
    assert session.current_set_value() == [2, 1]
    assert "pop_newset" in command
    _assert_python_echo_is_parseable(command)


def test_select_multiple_datasets_gui_uses_pop_chansel_style_positions():
    session = EEGPrepSession()
    session.store_current(_eeg("first"), new=True)
    session.store_current(_eeg("second"), new=True)
    seen = {}

    def chooser(labels, **kwargs):
        seen["labels"] = labels
        seen["withindex"] = kwargs["withindex"]
        return [2, 1], "Dataset 2:second Dataset 1:first", labels

    eeg, command = select_multiple_datasets(session, gui=True, renderer=chooser, return_com=True)

    assert seen["labels"] == ["Dataset 1:first", "Dataset 2:second"]
    assert seen["withindex"] == [1, 2]
    assert [item["setname"] for item in eeg] == ["second", "first"]
    assert session.CURRENTSET == [2, 1]
    assert "retrieve', [2 1]" in command


def test_phase1b_pop_functions_accept_sample_data_eeglab_dataset():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)
    edited = pop_editeventfield(eeg, "phase1bflag", "yes", "indices", [1])
    changed = pop_editeventvals(edited, "changefield", [1, "phase1bflag", "changed"])
    selected, selected_events = pop_selectevent(changed, "event", [1])
    channeled = pop_chanedit(selected, "changefield", [1, "type", "EEG"])
    copied_alleeg, copied, current_set, _command = pop_copyset([channeled], 1, 2, return_com=True)
    merged = pop_mergeset(copied_alleeg, [1, 2])

    assert changed["event"][0]["phase1bflag"] == "changed"
    assert selected_events == [1]
    assert channeled["chanlocs"][0]["type"] == "EEG"
    assert copied["setname"] == channeled["setname"]
    assert current_set == 2
    assert merged["nbchan"] == channeled["nbchan"]
    assert merged["pnts"] == channeled["pnts"] * 2


def test_phase1b_gui_cancel_paths_return_original_dataset_without_history():
    class CancelRenderer:
        def run(self, spec, initial_values=None):
            return None

    eeg = _eeg()
    for function in (pop_editeventfield, pop_editeventvals, pop_chanedit, pop_selectevent, pop_rmdat):
        result = function(deepcopy(eeg), gui=True, renderer=CancelRenderer(), return_com=True)
        assert result[1] == ""


@pytest.mark.matlab
@pytest.mark.skipif(
    not (matlab_engine_available() and eeglab_reference_available()),
    reason="MATLAB engine or EEGLAB reference not available",
)
def test_pop_selectevent_matches_eeglab_for_basic_event_type_selection():
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

    eeg = _eeg()
    py_out, py_selected = pop_selectevent(eeg, "type", "stim", "deleteevents", "on")
    matlab_out = get_eeglab("MAT").pop_selectevent(eeg, "type", "stim", "deleteevents", "on")

    assert py_selected == [1]
    assert [event["type"] for event in py_out["event"]] == [event["type"] for event in matlab_out["event"]]
    assert np.allclose(py_out["data"], matlab_out["data"])


@pytest.mark.matlab
@pytest.mark.skipif(
    not (matlab_engine_available() and eeglab_reference_available()),
    reason="MATLAB engine or EEGLAB reference not available",
)
def test_pop_mergeset_matches_eeglab_for_continuous_event_offsets():
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

    first = _eeg("first")
    second = _eeg("second")
    second["event"][0]["latency"] = 5.0
    py_out = pop_mergeset(first, second)
    matlab_out = get_eeglab("MAT").pop_mergeset(first, second)

    assert py_out["pnts"] == matlab_out["pnts"]
    assert [event["type"] for event in py_out["event"]] == [event["type"] for event in matlab_out["event"]]
    assert np.allclose(
        [float(event["latency"]) for event in py_out["event"]],
        [float(event["latency"]) for event in matlab_out["event"]],
    )
