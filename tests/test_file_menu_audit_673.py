"""Locks for the Nov 23, 2026 File-menu and data I/O audit."""

from __future__ import annotations

import ast
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions_dialog_spec
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc._file_io import mne_raw_to_eeg
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_chanevent import pop_chanevent, pop_chanevent_dialog_spec
from eegprep.functions.popfunc.pop_expevents import pop_expevents
from eegprep.functions.popfunc.pop_export import pop_export
from eegprep.functions.popfunc.pop_fileio import pop_fileio_dialog_spec
from eegprep.functions.popfunc.pop_loadcnt import _eeg_event
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_runscript import pop_runscript
from eegprep.functions.popfunc.pop_saveh import pop_saveh
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.plugins.EEG_BIDS.bids_tools import validate_bids
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


def _continuous() -> dict:
    eeg = eeg_emptyset()
    eeg["data"] = np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=float)
    eeg["nbchan"] = 1
    eeg["pnts"] = 10
    eeg["trials"] = 1
    eeg["srate"] = 100
    eeg["xmin"] = 0
    eeg["xmax"] = 0.09
    return eeg


def test_pop_chanevent_threshold_on_sample_channel_is_numeric_and_leading():
    eeg = pop_loadset(SAMPLE)
    imported = pop_chanevent(eeg, 2, oper="X>50", edge="leading", delchan="off", delevent="on")
    types = [event["type"] for event in imported["event"]]
    assert len(imported["event"]) == 143
    assert set(types) == {1}


def test_pop_chanevent_dialog_defaults_match_the_eeglab_dialog():
    controls = controls_by_tag(pop_chanevent_dialog_spec())
    assert controls["chan"].value == ""
    assert controls["edge"].value == 1
    assert controls["edgelen"].value == "0"


def test_pop_chanevent_rejects_an_unsafe_expression():
    with pytest.raises(ValueError, match="pop_chanevent: preprocessing"):
        pop_chanevent(_continuous(), 1, oper="__import__('os')")


def test_pop_saveset_round_trips_multiline_comments_and_keeps_the_original_on_failure(tmp_path):
    eeg = _continuous()
    eeg["setname"] = "comments"
    eeg["comments"] = "first line\nsecond line"
    path = tmp_path / "comments.set"
    pop_saveset(eeg, str(path))
    loaded = pop_loadset(path)
    assert loaded["comments"] == "first line\nsecond line"

    original = path.read_bytes()
    broken = pop_loadset(path)
    broken["etc"] = {"not_matlab": {1, 2}}
    with pytest.raises(ValueError, match="pop_saveset: could not write"):
        pop_saveset(broken, str(path))
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_epoch_field_is_an_empty_array_and_the_dataset_saves(tmp_path):
    eeg = _continuous()
    eeg["data"] = np.ones((1, 4, 2))
    eeg["pnts"] = 4
    eeg["trials"] = 2
    eeg["event"] = [{"type": "a", "latency": 1, "epoch": 1}, {"type": "b", "latency": 5, "epoch": 2}]
    checked = eeg_checkset(eeg, "eventconsistency")
    assert all(isinstance(row.get("eventtype"), np.ndarray) or "eventtype" in row for row in checked["epoch"])
    path = tmp_path / "epochs.set"
    pop_saveset(checked, str(path))
    assert path.is_file()


def test_pop_export_epoched_values_are_column_major_with_lf(tmp_path):
    eeg = eeg_emptyset()
    data = np.arange(12, dtype=float).reshape(2, 3, 2)
    eeg.update({"data": data, "nbchan": 2, "pnts": 3, "trials": 2, "srate": 100, "xmin": 0, "xmax": 0.02})
    target = tmp_path / "export.txt"
    pop_export(eeg, str(target), time="off", elec="off", transpose="off")
    text = target.read_bytes()
    assert b"\r\n" not in text
    rows = list(csv.reader(text.decode().splitlines(), delimiter="\t"))
    flat = np.reshape(data, (2, 6), order="F")
    assert [float(value) for value in rows[0]] == list(flat[0])


def test_pop_expevents_uses_lf(tmp_path):
    eeg = _continuous()
    eeg["event"] = [{"type": "stim", "latency": 2}]
    target = tmp_path / "events.txt"
    pop_expevents(eeg, str(target))
    assert b"\r\n" not in target.read_bytes()


def test_cnt_event_latency_is_one_based_and_has_no_duration():
    event = _eeg_event(
        {"stimtype": 1, "accept_ev1": 0, "offset": 10, "keypad_accept": 0, "keyboard": 0},
        keystroke=False,
    )
    assert event == {"type": 1, "latency": 11.0}
    boundary = _eeg_event(
        {"stimtype": 0, "accept_ev1": 11, "offset": 1, "keypad_accept": 0, "keyboard": 0},
        keystroke=False,
    )
    assert boundary["type"] == "boundary"
    assert "duration" not in boundary
    assert boundary["latency"] == 2.0


def test_pop_savestudy_writes_matlab_v6_and_still_reads_json(tmp_path):
    eeg = _continuous()
    eeg["subject"] = "S01"
    study, alleeg = pop_study(None, [eeg], name="Audit")
    saved, _datasets, command = pop_savestudy(study, alleeg, filename="audit.study", filepath=tmp_path, return_com=True)
    path = tmp_path / "audit.study"
    assert path.read_bytes().startswith(b"MATLAB 5.0 MAT-file")
    assert command.startswith("[STUDY ALLEEG] = pop_savestudy(")
    loaded, _alleeg = pop_loadstudy(path, load_datasets="off")
    assert loaded["name"] == "Audit"
    json_study, _json_datasets, _json_command = pop_savestudy(
        study, alleeg, filename="audit.json", filepath=tmp_path, fileformat="json", return_com=True
    )
    assert json_study["filename"].endswith(".json")
    reloaded, _ignored = pop_loadstudy(tmp_path / "audit.json", load_datasets="off")
    assert reloaded["name"] == "Audit"


def test_pop_saveset_refuses_to_resave_inside_a_raw_bids_folder(tmp_path):
    root = tmp_path / "bids"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "raw", "BIDSVersion": "1.9.0"}', encoding="utf-8")
    eeg = _continuous()
    target = eeg_dir / "sub-01_task-rest_eeg.set"
    pop_saveset(eeg, str(target))
    eeg["filename"] = target.name
    eeg["filepath"] = str(target.parent)
    with pytest.raises(ValueError, match="pop_saveset: refusing to overwrite raw BIDS file"):
        pop_saveset(eeg, savemode="resave")
    derivative = root / "derivatives" / "eeglab" / "sub-01" / "eeg" / "copy.set"
    pop_saveset(eeg, str(derivative))
    assert derivative.is_file()


def test_export_bids_omits_run_for_one_dataset_and_counts_eog(tmp_path):
    eeg = _continuous()
    eeg["subject"] = "S1"
    eeg["chanlocs"] = [{"labels": "Cz"}, {"labels": "EOG1"}]
    eeg["data"] = np.zeros((2, 10))
    eeg["nbchan"] = 2
    pop_exportbids([eeg], tmp_path / "out", task="rest")
    names = [path.name for path in (tmp_path / "out").rglob("*")]
    assert any(name.endswith("_eeg.set") and "_run-" not in name for name in names)
    assert any(name.endswith("_coordsystem.json") and "_run-" not in name and "_task-" not in name for name in names)
    eeg_json = next((tmp_path / "out").rglob("*_eeg.json"))
    import json

    payload = json.loads(eeg_json.read_text(encoding="utf-8"))
    assert payload["EEGChannelCount"] == 1
    assert payload["EOGChannelCount"] == 1
    assert (tmp_path / "out" / "participants.json").is_file()
    assert next((tmp_path / "out").rglob("*_events.json")).is_file()


def test_validate_bids_names_a_run_coordsystem_file(tmp_path):
    (tmp_path / "dataset_description.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README").write_text("readme\n", encoding="utf-8")
    bad = tmp_path / "sub-S1_run-01_coordsystem.json"
    bad.write_text("{}", encoding="utf-8")
    report = validate_bids(tmp_path)
    assert any("sub-S1_run-01_coordsystem.json" in error and "validate_bids" in error for error in report["errors"])


def test_pop_saveh_dataset_history_names_eeg_history(tmp_path):
    command = pop_saveh("EEG = pop_loadset('filename', 'a.set');", "hist.m", tmp_path)
    assert command.startswith("pop_saveh(EEG.history,")
    session = pop_saveh(["EEG = pop_reref(EEG, []);"], "session.m", tmp_path)
    assert session.startswith("pop_saveh(ALLCOM,")


def test_pop_runscript_runs_a_matlab_assignment_and_names_an_unsupported_line(tmp_path):
    script = tmp_path / "history.m"
    script.write_text("x = 1;\n", encoding="utf-8")
    namespace: dict = {}
    pop_runscript(script, namespace)
    assert namespace["x"] == 1
    broken = tmp_path / "broken.m"
    broken.write_text("this is not a command\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pop_runscript: broken.m line 1"):
        pop_runscript(broken, {})


def test_fileio_dialog_offers_an_explicit_data_format():
    controls = controls_by_tag(pop_fileio_dialog_spec())
    assert "auto" in controls["dataformat"].string


def test_empty_fileio_annotation_is_named_boundary():
    mne = pytest.importorskip("mne")
    info = mne.create_info(["Cz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((1, 20)), info, verbose="error")
    raw.set_annotations(mne.Annotations([0.05], [0.0], [""]))
    eeg = mne_raw_to_eeg(raw, setname="boundary")
    assert eeg["event"][0]["type"] == "boundary"


def test_option_single_tooltip_says_datasets_stay_float64():
    spec = pop_editoptions_dialog_spec(dict(EEG_OPTIONS, option_showadvanced=1))
    control = controls_by_tag(spec)["option_single"]
    assert "float64" in (control.tooltip or "")


def test_task_info_reaches_the_exported_description_and_eeg_json(tmp_path):
    import json

    eeg = _continuous()
    eeg["subject"] = "01"
    from eegprep.plugins.EEG_BIDS.bids_tools import pop_taskinfo

    pop_taskinfo(eeg, Name="Audit set", TaskName="oddball", PowerLineFrequency="60")
    root = Path(pop_exportbids([eeg], tmp_path / "bids", task="oddball"))
    description = json.loads((root / "dataset_description.json").read_text(encoding="utf-8"))
    sidecar = json.loads(next(root.rglob("*_eeg.json")).read_text(encoding="utf-8"))
    assert description["Name"] == "Audit set"
    assert sidecar["PowerLineFrequency"] == 60


def test_participant_rows_and_event_fields_are_written_on_export(tmp_path):
    import json

    from eegprep.plugins.EEG_BIDS.bids_tools import pop_eventinfo, pop_participantinfo

    first, second = _continuous(), _continuous()
    first["subject"] = "001"
    second["subject"] = "002"
    pop_participantinfo(
        {"datasetinfo": []},
        datasets=[first, second],
        rows=[
            {"participant_id": "sub-001", "Gender": "M", "Age": "31"},
            {"participant_id": "sub-002", "Gender": "F", "Age": "28"},
        ],
        column_info={"Age": {"Description": "Age at recording", "Units": "years"}},
    )
    pop_eventinfo(
        first,
        datasets=[first, second],
        event_fields=[{"bids_field": "value", "Description": "stimulus code", "Levels": "1=rare"}],
    )
    root = Path(pop_exportbids([first, second], tmp_path / "bids", task="oddball"))
    text = (root / "participants.tsv").read_text(encoding="utf-8")
    assert "sub-001" in text and "sub-002" in text
    assert "\tM\t" in text and "\tF\t" in text
    participants = json.loads((root / "participants.json").read_text(encoding="utf-8"))
    assert participants["Age"]["Description"] == "Age at recording"
    events = json.loads(next(root.rglob("*_events.json")).read_text(encoding="utf-8"))
    assert events["value"]["Description"] == "stimulus code"
    assert events["value"]["Levels"] == {"1": "rare"}


def test_bids_exporter_welcome_cancel_writes_nothing_and_grid_exports(tmp_path):
    from eegprep.plugins.EEG_BIDS.bids_tools import bids_exporter

    class Cancel:
        def run(self, spec, initial_values=None):
            return None

    assert bids_exporter(renderer=Cancel()) == ("", "")
    source = tmp_path / "raw.set"
    pop_saveset(_continuous(), source)
    answers = [
        {},
        {"output": str(tmp_path / "out"), "file1": str(source), "subject1": "07", "task1": "oddball"},
    ]

    class Grid:
        def run(self, spec, initial_values=None):
            return answers.pop(0)

    output, command = bids_exporter(renderer=Grid())
    assert Path(output, "dataset_description.json").is_file()
    assert "sub-07" in command or (Path(output) / "sub-07").is_dir() or any(Path(output).rglob("sub-07*"))


def test_empty_session_menu_opens_the_bids_wizard(monkeypatch):
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    seen: dict = {}

    def fake(EEG=None, output_dir=None, **kwargs):
        seen["EEG"] = EEG
        seen.update(kwargs)
        return "", ""

    monkeypatch.setattr("eegprep.plugins.EEG_BIDS.bids_tools.bids_exporter", fake)
    MenuActionDispatcher(EEGPrepSession()).dispatch("bids_exporter")
    assert seen["EEG"] is None
    assert seen["gui"] is True


def test_studyerp_cancel_changes_nothing_and_ok_builds():
    from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp

    class Cancel:
        def run(self, spec, initial_values=None):
            return None

    study, datasets, command = pop_studyerp(gui=True, renderer=Cancel(), return_com=True)
    assert study is None and datasets == [] and command == ""
    answers = [
        {"cond": "1", "subjects": "1"},
        {"study_name": "ERP audit", "cond1": "rare", "set1_1": str(SAMPLE)},
    ]

    class Grid:
        def run(self, spec, initial_values=None):
            return answers.pop(0)

    study, datasets, command = pop_studyerp(gui=True, renderer=Grid(), return_com=True)
    assert study["name"] == "ERP audit"
    assert datasets
    info = study["datasetinfo"][0]
    assert str(info["subject"]) == "S01"
    assert str(info["condition"]) == "rare"
    assert "std_editset" in command


def test_studywizard_cancel_changes_nothing_and_ok_applies_condition():
    from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard

    class Cancel:
        def run(self, spec, initial_values=None):
            return None

    study, datasets, command = pop_studywizard([str(SAMPLE)], gui=True, renderer=Cancel(), return_com=True)
    assert study is None and datasets == [] and command == ""

    class Grid:
        def run(self, spec, initial_values=None):
            return {"name": "Wizard", "file1": str(SAMPLE), "subject1": "S3", "condition1": "left"}

    study, datasets, command = pop_studywizard([str(SAMPLE)], gui=True, renderer=Grid(), return_com=True)
    assert study["name"] == "Wizard"
    assert datasets
    info = study["datasetinfo"][0]
    assert str(info["subject"]) == "S3"
    assert str(info["condition"]) == "left"
    assert "std_editset" in command
    assert "'condition'" in command
    assert "'left'" in command


def test_dataset_history_dialog_starts_in_the_dataset_folder(tmp_path):
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    captured: dict = {}

    class QFileDialog:
        @staticmethod
        def getSaveFileName(_parent, _caption, directory, _filter, **_kwargs):
            captured["directory"] = directory
            return "", ""

    session = EEGPrepSession()
    eeg = _continuous()
    eeg["filepath"] = str(tmp_path)
    eeg["history"] = "EEG = pop_loadset('filename', 'a.set');"
    session.store_current(eeg, new=True)
    widgets = type("FakeQt", (), {"QFileDialog": QFileDialog})
    dispatcher = MenuActionDispatcher(session)
    from unittest.mock import patch

    with patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=widgets):
        dispatcher.dispatch("pop_saveh:dataset")
    assert captured["directory"] == str(tmp_path / "eegprephist.m")


def test_empty_extension_catalog_explains_pip_packages():
    from eegprep.functions.adminfunc.plugin_menu import _catalog_notice

    notice = _catalog_notice({"catalog_source": "extension_catalog.json", "catalog_entry_count": 0})
    assert "pip packages" in notice


@pytest.mark.gui
@pytest.mark.modal_dialogs
def test_participant_table_dialog_stores_each_row(qapp):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QTableWidget

    from eegprep.plugins.EEG_BIDS.bids_tools import pop_participantinfo

    first, second = _continuous(), _continuous()
    first["subject"] = "001"
    second["subject"] = "002"

    def fill():
        dialog = qapp.activeModalWidget()
        if dialog is None:
            return
        table = dialog.findChild(QTableWidget, "bids-table")
        table.item(0, 1).setText("M")
        table.item(1, 1).setText("F")
        dialog.accept()

    timer = QTimer()
    timer.timeout.connect(fill)
    timer.start(20)
    try:
        _updated, command = pop_participantinfo({"datasetinfo": []}, gui=True, datasets=[first, second])
    finally:
        timer.stop()
    assert first["etc"]["bids"]["participant"]["Gender"] == "M"
    assert second["etc"]["bids"]["participant"]["Gender"] == "F"
    assert command.startswith("[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY,")
    assert "'rows', 2" not in command
    assert "Gender" in command
    assert "rows" not in first["etc"]["bids"]["participant"]


def test_mff_segmentation_follows_categories_not_epochs_xml(tmp_path):
    from eegprep.functions.popfunc.pop_mffimport import _looks_segmented

    folder = tmp_path / "continuous.mff"
    folder.mkdir()
    (folder / "epochs.xml").write_text("<epochs/>", encoding="utf-8")
    assert not _looks_segmented(folder)
    (folder / "categories.xml").write_text("<categories/>", encoding="utf-8")
    assert _looks_segmented(folder)


def test_pop_loadeeg_parses_subset_text_into_replayable_history():
    from eegprep.functions.popfunc.pop_loadeeg import _ranges

    ranges = _ranges({"precision": 1, "trials": "1:3", "types": "1 2", "channels": "", "responses": ""})
    assert ranges["triallist"] == [1, 2, 3]
    assert ranges["typerange"] == [1, 2]


def test_derivative_dataset_can_be_resaved(tmp_path):
    root = tmp_path / "derivatives" / "eegprep"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "prep", "DatasetType": "derivative"}', encoding="utf-8")
    eeg = _continuous()
    target = eeg_dir / "x.set"
    pop_saveset(eeg, target)
    eeg["saved"] = "no"
    pop_saveset(eeg, target, savemode="resave")
    assert target.is_file()


def test_pop_savestudy_replay_keeps_alleeg_as_datasets(tmp_path):
    from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
    from eegprep.functions.guifunc.session import EEGPrepSession

    eeg = _continuous()
    eeg["subject"] = "S01"
    study, alleeg = pop_study(None, [eeg], name="Audit")
    _saved, datasets, command = pop_savestudy(study, alleeg, filename="audit.study", filepath=tmp_path, return_com=True)
    session = EEGPrepSession()
    session.set_study(study, datasets)
    workspace = EEGPrepConsoleWorkspace(session)
    exec(_console_python_command(command), workspace.namespace)  # noqa: S102
    replayed = workspace.namespace["ALLEEG"]
    assert isinstance(replayed, list)
    assert replayed and isinstance(replayed[0], dict)


def test_event_info_reopen_keeps_levels_longname_and_units():
    from eegprep.plugins.EEG_BIDS.bids_tools import _event_rows, pop_eventinfo

    eeg = _continuous()
    fields = [
        {
            "bids_field": "value",
            "Description": "stimulus code",
            "Levels": "1=rare",
            "LongName": "Stimulus",
            "Units": "code",
        }
    ]
    pop_eventinfo(eeg, event_fields=fields)
    reopened = _event_rows(eeg["etc"]["bids"]["event"])
    value = next(row for row in reopened if row["bids_field"] == "value")
    assert value["Levels"] == "1=rare"
    assert value["LongName"] == "Stimulus"
    assert value["Units"] == "code"
    pop_eventinfo(eeg, event_fields=reopened)
    stored = eeg["etc"]["bids"]["event"]["value"]
    assert stored["Levels"] == {"1": "rare"}
    assert stored["LongName"] == "Stimulus"
    assert stored["Units"] == "code"


def test_bids_wizard_sends_edf_to_pop_biosig(tmp_path, monkeypatch):
    from eegprep.plugins.EEG_BIDS.bids_tools import bids_exporter

    path = tmp_path / "recording.edf"
    path.write_bytes(b"not a real edf")
    seen = {}

    def fake(filename, **_kwargs):
        seen["filename"] = filename
        eeg = _continuous()
        eeg["subject"] = "01"
        return eeg

    monkeypatch.setattr("eegprep.functions.popfunc.pop_biosig.pop_biosig", fake)
    output, _command = bids_exporter(
        entries=[{"file": str(path), "subject": "01", "task": "oddball"}],
        output_dir=tmp_path / "out",
    )
    assert seen["filename"] == str(path)
    assert Path(output, "dataset_description.json").is_file()


def test_matlab_study_history_is_console_replayable():
    from eegprep.functions.adminfunc.console import _console_python_command

    command = "[STUDY ALLEEG] = std_editset(STUDY, ALLEEG, 'name', 'Audit', 'updatedat', 'on');"
    python = _console_python_command(command)
    ast.parse(python)
    assert python.startswith("STUDY, ALLEEG = std_editset(")


def test_chanevent_expression_matches_matlab_round_bitand_and_chaining():
    from eegprep.functions.popfunc.pop_chanevent import _apply_oper

    rounded = _apply_oper(np.array([0.5, 1.5, 2.5, -2.5]), "round(X)")
    assert list(rounded) == [1.0, 2.0, 3.0, -3.0]
    masked = _apply_oper(np.array([1.0, 2.0]), "bitand(X, 1)")
    assert list(masked) == [1.0, 0.0]
    original = np.array([-3.2, -2.6, 0.2, 4.0])
    for expression in ("round(X)==-3", "round(X)>=-3", "round(X)<=-3", "round(X)~=-3"):
        compared = np.asarray(_apply_oper(original, expression))
        assert compared.shape == original.shape
        assert not np.array_equal(compared, original)
    with pytest.raises(ValueError, match="pop_chanevent") as bitand_error:
        _apply_oper(np.array([-1.0, 1.5]), "bitand(X, 1)")
    assert "round" in str(bitand_error.value).lower() or "absolute" in str(bitand_error.value).lower()
    chained = np.asarray(_apply_oper(np.array([0.0, 2.0, 4.0]), "1<X<3"), dtype=bool)
    assert chained[2]
    inverted = np.asarray(_apply_oper(np.array([0.0, 1.0]), "~(X>0)"), dtype=bool)
    assert list(inverted) == [True, False]
    with pytest.raises(ValueError, match="not_a_call") as caught:
        _apply_oper(np.array([1.0]), "not_a_call(X)")
    assert "ast." not in str(caught.value)
    spec = pop_chanevent_dialog_spec()
    hint = next(control for control in spec.controls if control.string == "Optional. Ex: X>3")
    row = spec.geometry[spec.controls.index(hint) // 3]
    assert row[2] > 1


def test_pop_loadeeg_history_is_positional_and_events_follow_latency(tmp_path, monkeypatch):
    from eegprep.functions.popfunc.pop_loadeeg import pop_loadeeg

    path = tmp_path / "epochs.eeg"
    path.write_bytes(b"")

    def fake(_path, **_ranges):
        signal = np.zeros((1, 8))
        return signal, [1, 1], [1, 1], np.array([0.5, 0.01]), [0, 0], ["Cz"], 4, 2, 100.0, -0.1, 0.03

    monkeypatch.setattr("eegprep.functions.popfunc.pop_loadeeg.loadeeg", fake)
    eeg, command = pop_loadeeg(
        path.name,
        str(path.parent),
        [],
        [1],
        [],
        [],
        return_com=True,
    )
    latencies = [float(event["latency"]) for event in eeg["event"]]
    assert latencies == sorted(latencies)
    assert "'triallist'" not in command
    assert "'chanlist'" not in command
    folder = str(path.parent)
    if not folder.endswith(("/", "\\")):
        folder += os.sep
    assert command.startswith(f"EEG = pop_loadeeg('{path.name}', '{folder}'")


def test_segmented_mff_splits_categories_without_mocking_mne(tmp_path):
    mne = pytest.importorskip("mne")
    from eegprep.functions.popfunc.pop_mffimport import _epochs_from_categories, pop_mffimport

    assert not hasattr(mne, "read_epochs_egi")
    srate = 250.0
    info = mne.create_info(257, srate, "eeg")
    raw = mne.io.RawArray(np.ones((257, 600)), info, verbose="error")
    folder = tmp_path / "testmff.mff"
    folder.mkdir()
    (folder / "epochs.xml").write_text(
        "<epochs><epoch><beginTime>0</beginTime><endTime>1200000000</endTime></epoch>"
        "<epoch><beginTime>1200000000</beginTime><endTime>2400000000</endTime></epoch></epochs>",
        encoding="utf-8",
    )
    (folder / "categories.xml").write_text(
        "<categories><cat><name>target</name><segments>"
        "<segment><beginTime>0</beginTime><endTime>1200000000</endTime>"
        "<eventBegin>202000000</eventBegin></segment>"
        "<segment><beginTime>1200000000</beginTime><endTime>2400000000</endTime>"
        "<eventBegin>1402000000</eventBegin></segment>"
        "</segments></cat></categories>",
        encoding="utf-8",
    )
    (folder / "info.xml").write_text(
        "<fileInfo><recordTime>2016-01-01T00:00:00+00:00</recordTime></fileInfo>", encoding="utf-8"
    )
    (folder / "Events_stm+.xml").write_text(
        "<eventTrack><name>stm+</name><trackType>STIM</trackType>"
        "<event><beginTime>2016-01-01T00:00:00.203960000+00:00</beginTime>"
        "<relativeBeginTime>203960000</relativeBeginTime><duration>1000000</duration>"
        "<code>stm+</code><label>stm+</label><description></description>"
        "<keys><key><keyCode>cel#</keyCode><data>1</data><dataType>long</dataType></key></keys></event>"
        "<event><relativeBeginTime>1406000000</relativeBeginTime><duration>1000000</duration>"
        "<code>resp</code><label>resp</label></event>"
        "</eventTrack>",
        encoding="utf-8",
    )
    eeg = _epochs_from_categories(raw, folder, "code")
    assert eeg["data"].shape == (257, 300, 2)
    assert float(np.max(eeg["data"])) == pytest.approx(1e6)
    assert eeg["xmin"] == pytest.approx(-0.202)
    assert [event["type"] for event in eeg["event"]] == ["stm+", "resp"]
    assert eeg["event"][0]["latency"] == pytest.approx(0.20396 * 250)
    assert eeg["event"][0]["mffkey_cel"] == "1"
    assert eeg["event"][0]["duration"] == pytest.approx(0.001 * 250)
    seconds = tmp_path / "seconds.mff"
    seconds.mkdir()
    (seconds / "categories.xml").write_text(
        "<categories><cat><name>target</name><segments>"
        "<segment><beginTime unit='s'>0</beginTime><endTime unit='s'>1.2</endTime>"
        "<eventBegin unit='s'>0.202</eventBegin></segment>"
        "</segments></cat></categories>",
        encoding="utf-8",
    )
    declared = _epochs_from_categories(raw, seconds, "code")
    assert declared["data"].shape[1] == 300
    assert declared["xmin"] == pytest.approx(-0.202)
    assert declared["event"] == []
    broken = tmp_path / "broken.mff"
    broken.mkdir()
    (broken / "categories.xml").write_text("<categories/>", encoding="utf-8")
    with pytest.raises(Exception) as caught:
        pop_mffimport(broken)
    assert not isinstance(caught.value, AttributeError)


def test_importbids_history_and_sessionless_derivative_path(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_importbids import _history_command, _save_derivatives

    eeg = _continuous()
    eeg["subject"] = "001"
    eeg["session"] = 1
    eeg["filename"] = "sub-001_task-rest_eeg.set"
    eeg["filepath"] = str(tmp_path / "sub-001" / "eeg")
    _save_derivatives([eeg], tmp_path, tmp_path / "derivatives" / "eeglab")
    assert (tmp_path / "derivatives" / "eeglab" / "sub-001" / "eeg" / "sub-001_task-rest_eeg.set").is_file()
    assert not (tmp_path / "derivatives" / "eeglab" / "sub-001" / "ses-1").exists()
    command = _history_command(tmp_path, {}, return_study=True)
    assert command.startswith("[STUDY ALLEEG] = pop_importbids(")
    assert "return_study" not in command


def test_pop_savestudy_strips_measures_resaves_json_as_mat_and_falls_back_to_v73(tmp_path, monkeypatch):
    eeg = _continuous()
    eeg["subject"] = "S01"
    study, alleeg = pop_study(None, [eeg], name="Audit")
    study["cluster"] = [{"name": "Parent", "sets": [[1]], "comps": [1], "specdata": np.ones((2, 3))}]
    saved, datasets, command = pop_savestudy(study, alleeg, filename="audit.study", filepath=tmp_path)
    assert command.startswith("[STUDY ALLEEG] = pop_savestudy(")
    assert datasets[0]["subject"] == "S01"
    assert saved["cluster"][0]["specdata"].shape == (2, 3)
    loaded, _alleeg = pop_loadstudy(tmp_path / "audit.study", load_datasets="off")
    assert "specdata" not in loaded["cluster"][0]
    json_path = tmp_path / "legacy.study"
    json_path.write_text('{"name": "legacy"}', encoding="utf-8")
    study["filename"] = json_path.name
    study["filepath"] = str(tmp_path)
    pop_savestudy(study, alleeg, savemode="resave")
    assert json_path.read_bytes().startswith(b"MATLAB")

    def too_big(*_args, **_kwargs):
        raise OverflowError("Matrix too large to save with Matlab 5 format")

    monkeypatch.setattr("eegprep.functions.studyfunc.pop_savestudy.scipy.io.savemat", too_big)
    pop_savestudy(study, alleeg, filename="huge.study", filepath=tmp_path)
    huge = tmp_path / "huge.study"
    assert huge.read_bytes().startswith(b"MATLAB 7.3")
    reloaded, _ignored = pop_loadstudy(huge, load_datasets="off")
    assert reloaded["name"] == "Audit"


def test_study_commands_stay_replayable_and_studyerp_is_condition_major(tmp_path):
    from eegprep.functions.studyfunc._study_utils import matlab_flat_commands
    from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp, pop_studyerp_grid_dialog_spec

    rendered = str(matlab_flat_commands([{"index": 1, "load": "a.set", "subject": "S01"}]))
    assert rendered.startswith("{{'index'")
    assert "''index''" not in rendered
    spec = pop_studyerp_grid_dialog_spec(2, 2)
    buttons = [control for control in spec.controls if control.style == "pushbutton"]
    assert len(buttons) == 4
    assert {control.string for control in buttons} == {"..."}
    copies = []
    for name in ("a1", "a2", "a3", "a4"):
        destination = tmp_path / f"{name}.set"
        shutil.copy(SAMPLE, destination)
        shutil.copy(SAMPLE.with_suffix(".fdt"), tmp_path / "eeglab_data.fdt")
        copies.append(destination)
    answers = [
        {"cond": "2", "subjects": "2"},
        {
            "study_name": "ERP",
            "cond1": "A",
            "cond2": "B",
            "set1_1": str(copies[0]),
            "set2_1": str(copies[1]),
            "set1_2": str(copies[2]),
            "set2_2": str(copies[3]),
        },
    ]

    class Grid:
        def run(self, spec, initial_values=None):
            return answers.pop(0)

    study, datasets, command = pop_studyerp(gui=True, renderer=Grid(), return_com=True)
    import re

    indexes = [int(value) for value in re.findall(r"'index', (\d+)", command)]
    assert indexes == [2, 4, 3, 5]
    assert len(datasets) == 4
    identities = [(info["subject"], info["condition"], Path(info["filename"]).name) for info in study["datasetinfo"]]
    assert identities == [
        ("S01", "A", "a1.set"),
        ("S01", "B", "a2.set"),
        ("S02", "A", "a3.set"),
        ("S02", "B", "a4.set"),
    ]
    with pytest.raises(ValueError, match="Duplicate dataset") as duplicate:
        pop_studyerp(
            gui=True,
            renderer=type(
                "Same",
                (),
                {
                    "run": staticmethod(
                        lambda spec, initial_values=None: (
                            {"cond": "1", "subjects": "2"}
                            if "cond" in {control.tag for control in spec.controls}
                            else {"set1_1": str(copies[0]), "set1_2": str(copies[0])}
                        )
                    )
                },
            )(),
            return_com=True,
        )
    assert "Copy and rename" in str(duplicate.value)


def test_coordsystem_acq_entity_is_allowed(tmp_path):
    folder = tmp_path / "bids"
    (folder / "sub-001" / "eeg").mkdir(parents=True)
    (folder / "dataset_description.json").write_text('{"Name": "acq"}', encoding="utf-8")
    (folder / "sub-001_acq-high_coordsystem.json").write_text("{}", encoding="utf-8")
    result = validate_bids(folder)
    assert not any("acq-" in error for error in result["errors"])


def test_bids_wizard_keeps_a_task_on_every_file(tmp_path):
    from eegprep.plugins.EEG_BIDS.bids_tools import bids_exporter, bids_exporter_grid_spec

    spec = bids_exporter_grid_spec(row_count=6)
    assert any(control.tag == "file6" for control in spec.controls)
    first = tmp_path / "one.set"
    second = tmp_path / "two.set"
    pop_saveset(_continuous(), first)
    pop_saveset(_continuous(), second)
    output, _command = bids_exporter(
        entries=[
            {"file": str(first), "subject": "001", "task": "rest"},
            {"file": str(second), "subject": "002", "task": "oddball"},
        ],
        output_dir=tmp_path / "out",
    )
    root = Path(output)
    assert (root / "sub-001" / "eeg" / "sub-001_task-rest_eeg.set").is_file()
    assert (root / "sub-002" / "eeg" / "sub-002_task-oddball_eeg.set").is_file()
    assert (root / "task-rest_events.json").is_file()
    assert (root / "task-oddball_events.json").is_file()
    assert not (root / "sub-001" / "eeg" / "task-rest_events.json").exists()
    assert not list(root.rglob("sub-*_events.json"))


def test_export_uses_channel_type_and_prompts_for_power_line(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

    eeg = _continuous()
    eeg["subject"] = "001"
    eeg["data"] = np.zeros((2, 10))
    eeg["nbchan"] = 2
    eeg["chanlocs"] = [{"labels": "Cz", "type": "EOG"}, {"labels": "Pz", "type": "EEG"}]
    out = tmp_path / "bids"

    class Power:
        def run(self, spec, initial_values=None):
            return {"PowerLineFrequency": 2}

    pop_exportbids(eeg, out, task="rest", renderer=Power())
    channels = (out / "sub-001" / "eeg" / "sub-001_task-rest_channels.tsv").read_text(encoding="utf-8")
    assert "Cz\tEOG\tuV" in channels
    sidecar = (out / "sub-001" / "eeg" / "sub-001_task-rest_eeg.json").read_text(encoding="utf-8")
    assert '"PowerLineFrequency": 60' in sidecar


def test_quit_asks_before_discarding_unsaved_datasets():
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, _require_qt_widgets
    from eegprep.functions.guifunc.session import EEGPrepSession

    session = EEGPrepSession()
    eeg = _continuous()
    eeg["saved"] = "no"
    session.ALLEEG = [eeg]
    prompts = []

    class Widgets:
        class QMessageBox:
            Yes = 1
            No = 2
            Cancel = 3

            @staticmethod
            def question(_parent, _title, text, *_args, **_kwargs):
                prompts.append(text)
                return 3

    dispatcher = MenuActionDispatcher(session)
    assert dispatcher._last_data_folder == ""
    original = _require_qt_widgets
    try:
        import eegprep.functions.guifunc.menu_actions as menu_actions

        menu_actions._require_qt_widgets = lambda: Widgets
        closed = []

        class Parent:
            def close(self):
                closed.append(True)

        dispatcher.dispatch("quit", Parent())
    finally:
        menu_actions._require_qt_widgets = original
    assert prompts == ["Some datasets are unsaved. Save them before quitting?"]
    assert closed == []


def test_pop_runscript_runs_nothing_when_a_line_is_unsupported(tmp_path):
    script = tmp_path / "history.m"
    script.write_text("EEG = 1;\nthis is not a supported command (\n", encoding="utf-8")
    namespace: dict = {}
    with pytest.raises(ValueError, match="line 2"):
        pop_runscript(script, namespace)
    assert namespace == {}


def test_pop_fileio_honours_dataformat_from_the_popup(tmp_path):
    from eegprep.functions.popfunc.pop_fileio import pop_fileio

    source = tmp_path / "recording.set"
    pop_saveset(_continuous(), source)
    renamed = tmp_path / "recording.bin"
    renamed.write_bytes(source.read_bytes())
    loaded = pop_fileio(renamed, dataformat="eeglab")
    assert loaded["nbchan"] == 1
    array = tmp_path / "values.mat"
    import scipy.io

    scipy.io.savemat(array, {"data": np.arange(8).reshape(2, 4)})
    imported = pop_fileio(array, dataformat="matlab-array")
    assert imported["data"].shape[0] == 2


def test_real_testmff_matches_eeglab_epochs():
    mne = pytest.importorskip("mne")
    from eegprep.functions.popfunc.pop_mffimport import pop_mffimport

    candidates = []
    if os.environ.get("EEGPREP_TESTMFF"):
        candidates.append(Path(os.environ["EEGPREP_TESTMFF"]))
    candidates.append(Path(__file__).resolve().parents[1] / "sample_data" / "testmff.mff")
    folder = next((path for path in candidates if path.is_dir()), None)
    if folder is None:
        pytest.skip("testmff.mff is not available")
    eeg = pop_mffimport(folder, typefield="code")
    assert eeg["data"].shape == (257, 300, 2)
    assert float(np.max(np.abs(eeg["data"]))) == pytest.approx(19.75, rel=0.02)
    assert eeg["xmin"] == pytest.approx(-0.202, abs=1e-3)
    assert len(eeg["event"]) == 10
    latencies = [float(event["latency"]) for event in eeg["event"]]
    for expected in (50.99, 51.5, 242.2, 246.7, 350.7):
        assert any(abs(found - expected) < 0.05 for found in latencies)
    found_types = {str(event["type"]).lower() for event in eeg["event"]}
    for name in ("stm+", "lrnd", "ulrn", "resp", "fbk+"):
        assert name in found_types
    assert mne is not None


def test_cluster_sets_round_trip_as_a_numeric_matrix(tmp_path):
    import scipy.io

    eeg = _continuous()
    eeg["subject"] = "S01"
    study, alleeg = pop_study(None, [eeg], name="Clusters")
    study["cluster"] = [
        {"name": "One", "sets": [[1], [2]], "comps": [1]},
        {"name": "Two", "sets": [[1, 2], [3, 4]], "comps": [1, 2]},
    ]
    pop_savestudy(study, alleeg, filename="clusters.study", filepath=tmp_path)
    loaded = scipy.io.loadmat(tmp_path / "clusters.study", squeeze_me=False, struct_as_record=False, appendmat=False)
    clusters = loaded["STUDY"][0, 0].cluster
    sets = [clusters[0, index].sets for index in range(clusters.shape[1])]
    assert sets[0].dtype.kind == "f" and sets[0].ndim == 2 and sets[0].shape == (2, 1)
    assert sets[1].dtype.kind == "f" and sets[1].ndim == 2 and sets[1].shape == (2, 2)


def test_v73_study_with_two_datasets_and_a_design_reloads(tmp_path, monkeypatch):
    import h5py

    first = _continuous()
    second = _continuous()
    first["subject"] = "S01"
    first["filename"] = "s01.set"
    second["subject"] = "S02"
    second["filename"] = "s02.set"
    study, alleeg = pop_study(None, [first, second], name="Pair")
    study["design"] = [{"name": "Within", "variable": [{"label": "condition", "value": "A"}]}]
    study["cluster"] = [{"name": "Parent", "sets": [[1, 2]], "comps": [1], "child": []}]

    def too_big(*_args, **_kwargs):
        raise OverflowError("Matrix too large to save with Matlab 5 format")

    monkeypatch.setattr("eegprep.functions.studyfunc.pop_savestudy.scipy.io.savemat", too_big)
    pop_savestudy(study, alleeg, filename="pair.study", filepath=tmp_path)
    path = tmp_path / "pair.study"
    header = path.read_bytes()[:128]
    assert header.startswith(b"MATLAB 7.3")
    assert header[124:128] == b"\x00\x01IM"
    with h5py.File(path, "r") as handle:
        assert "struct" in str(handle["STUDY"].attrs["MATLAB_class"])
        assert "#refs#" in handle
    reloaded, datasets = pop_loadstudy(path, load_datasets="off")
    assert reloaded["name"] == "Pair"
    assert len(reloaded["datasetinfo"]) == 2
    assert reloaded["design"][0]["name"] == "Within"
    assert datasets == []


def test_scripted_power_line_uses_the_eeg_number(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

    eeg = _continuous()
    eeg["subject"] = "001"
    eeg["etc"] = {"bids": {"task": {"PowerLineFrequency": "50"}}}
    out = tmp_path / "bids"
    pop_exportbids(eeg, out, task="rest")
    sidecar = (out / "sub-001" / "eeg" / "sub-001_task-rest_eeg.json").read_text(encoding="utf-8")
    assert '"PowerLineFrequency": 50' in sidecar


def test_participantinfo_key_value_history_includes_study():
    from eegprep.plugins.EEG_BIDS.bids_tools import pop_participantinfo

    _eeg, command = pop_participantinfo({"data": []}, participant_id="sub-01")
    assert command.startswith("[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY, 'participant_id'")


def test_export_wizard_pages_ten_rows(qapp):
    from PySide6.QtWidgets import QLabel, QPushButton

    from eegprep.plugins.EEG_BIDS.bids_tools import _export_grid_dialog, bids_exporter_grid_spec

    spec = bids_exporter_grid_spec()
    assert (
        sum(
            bool(control.tag) and control.tag.startswith("file") and control.tag[4:].isdigit()
            for control in spec.controls
        )
        == 10
    )
    assert {control.string for control in spec.controls if control.style == "text"} >= {
        "Raw dataset filename",
        "subject",
        "Page 1",
    }
    assert sum(control.string == "..." for control in spec.controls) == 10
    assert any(control.string == "<" for control in spec.controls)
    dialog = _export_grid_dialog()
    dialog.show()
    qapp.processEvents()
    headers = {label.text() for label in dialog.findChildren(QLabel)}
    assert {"Raw dataset filename", "subject", "session", "run", "condition", "group", "task"} <= headers
    assert len([button for button in dialog.findChildren(QPushButton) if button.text() == "..."]) == 10
    dialog.edits[0]["file"].setText("a.set")
    dialog.shift(1)
    assert dialog.page_label.text() == "Page 2"
    dialog.edits[0]["file"].setText("b.set")
    dialog.save_page()
    assert dialog.state["rows"][0]["file"] == "a.set"
    assert dialog.state["rows"][10]["file"] == "b.set"
    dialog.close()


def test_chanevent_labels_fit_their_column(qapp):
    from PySide6.QtWidgets import QLabel

    from eegprep.functions.guifunc.qt import QtDialogRenderer

    _app, dialog, _widgets = QtDialogRenderer().build_dialog(pop_chanevent_dialog_spec())
    dialog.show()
    qapp.processEvents()
    expected = {
        "Preprocessing transform (data='X')",
        "Transitions to extract? (up|down)",
        "Transition length (1=perfect edges)",
        "Assign duration to each events?",
    }
    found = {label.text(): label for label in dialog.findChildren(QLabel)}
    for text in expected:
        label = found[text]
        assert label.width() >= label.sizeHint().width()
    dialog.close()


def test_studyerp_grid_sizes_to_its_contents(qapp):
    from PySide6.QtWidgets import QLabel, QScrollArea

    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.functions.studyfunc.pop_studyerp import pop_studyerp_grid_dialog_spec

    _app, dialog, _widgets = QtDialogRenderer().build_dialog(pop_studyerp_grid_dialog_spec(2, 2))
    dialog.show()
    qapp.processEvents()
    assert dialog.findChildren(QScrollArea) == []
    hint = next(
        label for label in dialog.findChildren(QLabel) if label.text().startswith("When using more than 1 condition")
    )
    assert hint.width() >= hint.sizeHint().width()
    assert dialog.height() >= hint.y() + hint.sizeHint().height()
    dialog.close()


def test_size_mb_rounds_half_up(monkeypatch):
    from eegprep.functions.guifunc.main_window import _size_mb

    monkeypatch.setattr("eegprep.functions.guifunc.main_window._memory_bytes", lambda _value, _seen: 1_250_000)
    assert _size_mb({}) == "1.3"
