from __future__ import annotations

import ast
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.select_multiple_datasets import select_multiple_datasets
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit, pop_chanedit_dialog_spec
from eegprep.functions.popfunc.pop_copyset import pop_copyset
from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield
from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_rmdat import pop_rmdat
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent
from tests.eeglab_tests import assert_matlab_equal, eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near, matlab_field_concat
from tests.fixtures import SAMPLE_DATASET_PATH, matlab_engine_available


def _reference_selection_eeg(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(
        setname="selection regression",
        nbchan=1.0,
        srate=1000.0,
        pnts=10.0,
        trials=4.0,
        xmin=0.0,
        xmax=0.009,
        data=np.arange(1, 41, dtype=np.float32).reshape((1, 10, 4), order="F"),
    )
    events = []
    for trial in range(1, 5):
        event_type = "target" if trial % 2 else "other"
        events.extend(
            [
                (event_type, float((trial - 1) * 10 + 3), float(trial)),
                ("distractor", float((trial - 1) * 10 + 7), float(trial)),
            ]
        )
    eeg["event"] = np.array([events], dtype=[("type", object), ("latency", object), ("epoch", object)])
    return eeglab_backend("eeg_checkset", eeg, "eventconsistency")


def _reference_dataset_row(*datasets):
    fields = list(datasets[0])
    return np.array(
        [[tuple(eeg[field] for field in fields) for eeg in datasets]], dtype=[(field, object) for field in fields]
    )


@eeglab_test("regression_tests/t_pop_selectevent.m", "testRetainsMatchingEpochs")
def test_reference_selectevent_retains_matching_epochs(eeglab_backend):
    eeg = _reference_selection_eeg(eeglab_backend)
    selected = eeglab_backend("pop_selectevent", eeg, "type", "target", "deleteepochs", "on")
    np.testing.assert_array_equal(selected["trials"], np.array([[2.0]]), strict=True)
    np.testing.assert_array_equal(selected["data"], eeg["data"][:, :, [0, 2]], strict=True)
    assert selected["event"].size == 4
    np.testing.assert_array_equal(
        matlab_field_concat(selected["event"], "epoch"), np.array([[1.0, 1.0, 2.0, 2.0]]), strict=True
    )


@eeglab_test("regression_tests/t_pop_selectevent.m", "testEmptySelectionErrorsByDefault")
def test_reference_selectevent_empty_selection_error(eeglab_backend, request):
    args = (_reference_selection_eeg(eeglab_backend), "type", "absent", "deleteepochs", "on")
    if request.config.getoption("--eeglab-backend") == "matlab":
        raised, identifier = eeglab_backend("eegprep_test_error_identifier", "pop_selectevent", *args, nargout=2)
        assert raised.item()
        assert np.asarray(identifier).size == 0
    else:
        # The source requires an error with MATLAB identifier ''. Python has no
        # identifier analogue, and the source specifies no Python exception type.
        with pytest.raises(Exception):
            eeglab_backend("pop_selectevent", *args)


@eeglab_test("regression_tests/t_pop_selectevent.m", "testEmptySelectionAllowed")
def test_reference_selectevent_empty_selection_allowed(eeglab_backend):
    selected = eeglab_backend(
        "pop_selectevent",
        _reference_selection_eeg(eeglab_backend),
        "type",
        "absent",
        "deleteepochs",
        "on",
        "erroronempty",
        "off",
    )
    assert selected["data"].size == 0
    assert selected["event"].size == 0


@eeglab_test("regression_tests/t_pop_selectevent.m", "testInverseEpochSelection")
def test_reference_selectevent_inverse_epochs(eeglab_backend):
    eeg = _reference_selection_eeg(eeglab_backend)
    selected = eeglab_backend("pop_selectevent", eeg, "type", "target", "deleteepochs", "on", "invertepochs", "on")
    np.testing.assert_array_equal(selected["data"], eeg["data"][:, :, [1, 3]], strict=True)
    np.testing.assert_array_equal(selected["trials"], np.array([[2.0]]), strict=True)


@eeglab_test("regression_tests/t_pop_selectevent.m", "testDeleteUnselectedEvents")
def test_reference_selectevent_delete_unselected_events(eeglab_backend):
    selected = eeglab_backend(
        "pop_selectevent",
        _reference_selection_eeg(eeglab_backend),
        "type",
        "target",
        "deleteepochs",
        "on",
        "deleteevents",
        "on",
    )
    assert_matlab_equal(
        selected["event"]["type"].reshape((1, -1), order="F"), np.array([["target", "target"]], dtype=object)
    )
    np.testing.assert_array_equal(matlab_field_concat(selected["event"], "epoch"), np.array([[1.0, 2.0]]), strict=True)


@eeglab_test("regression_tests/t_pop_selectevent.m", "testKeepEpochsWhenDeletingOnlyEvents")
def test_reference_selectevent_keep_epochs_delete_events(eeglab_backend):
    eeg = _reference_selection_eeg(eeglab_backend)
    selected = eeglab_backend("pop_selectevent", eeg, "type", "target", "deleteepochs", "off", "deleteevents", "on")
    np.testing.assert_array_equal(selected["data"], eeg["data"], strict=True)
    assert selected["event"].size == 2


@eeglab_test("regression_tests/t_pop_selectevent.m", "testExplicitErrorOptionWithNonemptySelection")
def test_reference_selectevent_explicit_error_nonempty_selection(eeglab_backend):
    eeg = _reference_selection_eeg(eeglab_backend)
    selected = eeglab_backend("pop_selectevent", eeg, "type", "target", "deleteepochs", "on", "erroronempty", "on")
    np.testing.assert_array_equal(selected["data"], eeg["data"][:, :, [0, 2]], strict=True)


@eeglab_test("regression_tests/t_pop_selectevent.m", "testDatasetArray")
def test_reference_selectevent_dataset_array(eeglab_backend):
    eeg = _reference_selection_eeg(eeglab_backend)
    selected = eeglab_backend(
        "pop_selectevent", _reference_dataset_row(eeg, eeg), "type", "target", "deleteepochs", "on"
    )
    assert selected.size == 2
    np.testing.assert_array_equal(matlab_field_concat(selected, "trials"), np.array([[2.0, 2.0]]), strict=True)
    np.testing.assert_array_equal(selected["data"].ravel(order="F")[0], eeg["data"][:, :, [0, 2]], strict=True)
    np.testing.assert_array_equal(selected["data"].ravel(order="F")[1], eeg["data"][:, :, [0, 2]], strict=True)


@eeglab_test("unittesting_popfunc/pop_copyset/popfunc_pop_copyset_wrapperTest.m", "test_test_pop_copyset")
def test_reference_copyset_recorded_dataset_calls(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    alleeg, _, _ = eeglab_backend("pop_copyset", _reference_dataset_row(eeg, eeg), 1.0, 2.0, nargout=3)
    eeglab_backend("pop_copyset", alleeg, 1.0, 1.0, nargout=3)


@eeglab_test("unittesting_popfunc/pop_copyset/popfunc_pop_copyset_wrapperTest.m", "test_pass_set_out")
def test_reference_copyset_original_set_out(eeglab_backend, eeglab_suite_root):
    directory = eeglab_suite_root / "unittesting_popfunc/pop_copyset"
    first = eeglab_backend("pop_loadset", str(directory / "test.set"), "")
    second = eeglab_backend("pop_loadset", str(directory / "test_2.set"), "")
    alleeg, eeg, current, _ = eeglab_backend("pop_copyset", _reference_dataset_row(first, second), 2.0, 1.0, nargout=4)
    for index in range(alleeg.size):
        position = np.unravel_index(index, alleeg.shape, order="F")
        alleeg["history"][position] = ""
        alleeg["saved"][position] = ""
    eeg["history"] = eeg["saved"] = ""
    first = {field: alleeg[field].ravel(order="F")[0] for field in alleeg.dtype.names}
    second = {field: alleeg[field].ravel(order="F")[1] for field in alleeg.dtype.names}
    assert_matlab_struct_near(second, first)
    selected = {field: alleeg[field].ravel(order="F")[int(current.item()) - 1] for field in alleeg.dtype.names}
    assert_matlab_struct_near(selected, eeg)
    assert current.item() == 1


def eeglab_reference_available() -> bool:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]).expanduser())
    candidates.append(repo_root / "src" / "eegprep" / "eeglab")
    return any(
        (candidate / "functions" / "popfunc" / "pop_selectevent.m").exists()
        and (candidate / "functions" / "popfunc" / "pop_mergeset.m").exists()
        and (candidate / "plugins" / "clean_rawdata" / "private").is_dir()
        for candidate in candidates
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
            {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 0},
            {"type": "resp", "latency": 50.0, "duration": 0.0, "urevent": 1},
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


def test_pop_editeventfield_updates_urevent_pointed_to_by_loaded_event():
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))
    values = [f"tag{index}" for index in range(len(eeg["event"]))]

    out = pop_editeventfield(eeg, "tag", values)

    for event in out["event"]:
        assert out["urevent"][event["urevent"]]["tag"] == event["tag"]


def test_pop_editeventvals_append_continues_zero_based_urevent_numbering():
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))
    n_events = len(eeg["event"])
    n_urevents = len(eeg["urevent"])

    # Value list follows the dataset's field order: type, position, latency.
    out = pop_editeventvals(eeg, "append", [n_events, "new", 2, 100.0])

    assert len(out["event"]) == n_events + 1
    new_event = next(event for event in out["event"] if event["type"] == "new")
    assert new_event["urevent"] == n_urevents
    assert len(out["urevent"]) == n_urevents + 1
    assert out["urevent"][new_event["urevent"]]["type"] == "new"


def test_pop_editeventvals_insert_preserves_existing_urevent_links():
    eeg = _eeg()
    eeg["event"][0]["urevent"] = 1
    eeg["event"][1]["urevent"] = 0
    eeg["urevent"] = [
        {"type": "resp", "latency": 50.0, "duration": 0.0},
        {"type": "stim", "latency": 10.0, "duration": 0.0},
    ]

    inserted = pop_editeventvals(eeg, "insert", [2, "new", 25.0, 0.0, 99])

    assert [event["urevent"] for event in inserted["event"]] == [1, 2, 0]
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
        {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 0, "epoch": 1},
        {"type": "resp", "latency": 60.0, "duration": 0.0, "urevent": 1, "epoch": 2},
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
    eeg["event"].insert(1, {"type": -99, "latency": 25.0, "duration": 0.0, "urevent": 2})
    eeg["urevent"].append({"type": -99, "latency": 25.0, "duration": 0.0})
    old = EEG_OPTIONS["option_boundary99"]
    EEG_OPTIONS["option_boundary99"] = 1
    try:
        out, selected = pop_selectevent(eeg, "type", "stim", "deleteevents", "on")
    finally:
        EEG_OPTIONS["option_boundary99"] = old

    assert selected == [1, 2]
    assert [event["type"] for event in out["event"]] == ["stim", -99]


def test_pop_rmdat_removes_or_keeps_continuous_windows_around_events():
    eeg = _eeg()

    removed, command = pop_rmdat(eeg, ["stim"], [-0.01, 0.01], 1, return_com=True)
    kept = pop_rmdat(eeg, ["stim"], [-0.01, 0.01], 0)

    assert removed["pnts"] < eeg["pnts"]
    assert kept["pnts"] < eeg["pnts"]
    assert kept["pnts"] < removed["pnts"]
    _assert_python_echo_is_parseable(command)


def test_pop_rmdat_uses_numeric_boundary99_to_limit_windows():
    eeg = _eeg()
    eeg["event"].insert(1, {"type": -99, "latency": 15.0, "duration": 0.0, "urevent": 2})
    old = EEG_OPTIONS["option_boundary99"]
    try:
        EEG_OPTIONS["option_boundary99"] = 1
        limited = pop_rmdat(eeg, ["stim"], [-0.05, 0.1], 1)
        EEG_OPTIONS["option_boundary99"] = 0
        unlimited = pop_rmdat(eeg, ["stim"], [-0.05, 0.1], 1)
    finally:
        EEG_OPTIONS["option_boundary99"] = old

    assert limited["pnts"] > unlimited["pnts"]


def test_pop_rmdat_matches_sorted_event_behavior_when_events_are_unsorted():
    unsorted_eeg = _eeg()
    unsorted_eeg["event"] = [
        {"type": "stim", "latency": 50.0, "duration": 0.0, "urevent": 1},
        {"type": "stim", "latency": 10.0, "duration": 0.0, "urevent": 0},
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


def test_pop_chanedit_gui_submits_change_for_navigated_channel():
    class Renderer:
        def run(self, spec, initial_values=None):
            controls = controls_by_tag(spec)
            nav = controls["next1"].callback.params
            result = {tag: control.value for tag, control in controls.items()}
            result.update(nav["field_displays"][1])
            result["channel"] = "2"
            result["field_labels"] = "Oz"
            return result

    eeg = _eeg()

    output, command = pop_chanedit(eeg, gui=True, renderer=Renderer(), return_com=True)

    assert output["chanlocs"][0]["labels"] == "Cz"
    assert output["chanlocs"][1]["labels"] == "Oz"
    assert "'changefield', [2 'labels' 'Oz']" in command


def test_qt_renderer_navigation_updates_channel_and_fields():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from eegprep.functions.guifunc import qt as qt_module

    if qt_module.QtCore is None or qt_module.QtWidgets is None:
        pytest.skip("PySide6 Qt libraries unavailable in this environment")
    QtDialogRenderer = qt_module.QtDialogRenderer

    eeg = _eeg()
    renderer = QtDialogRenderer()
    spec = pop_chanedit_dialog_spec(eeg)
    _app, dialog, widgets = renderer.build_dialog(spec)
    try:
        assert widgets["channel"].text() == "1"
        widgets["next1"].click()
        assert widgets["channel"].text() == "2"
        assert widgets["field_labels"].text() == "Pz"
        widgets["next10"].click()
        assert widgets["channel"].text() == str(len(eeg["chanlocs"]))
        widgets["back10"].click()
        assert widgets["channel"].text() == "1"
        assert widgets["field_labels"].text() == "Cz"
    finally:
        dialog.close()


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


def test_pop_mergeset_default_indices_skip_deleted_slots():
    from eegprep.functions.popfunc.pop_mergeset import _default_indices_text

    # Deleted datasets leave empty ALLEEG slots; the dialog must offer real dataset numbers.
    assert _default_indices_text([_eeg("first"), {}, _eeg("third")], None) == "1 3"
    assert _default_indices_text([{}, _eeg("second")], None) == "2"
    assert _default_indices_text([_eeg("only"), {}], None) == "1"
