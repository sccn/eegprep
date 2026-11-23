"""Locks for the Nov 23, 2026 File-menu and data I/O audit."""

from __future__ import annotations

import ast
import csv
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
    assert event == {"type": 1, "latency": 10.0}
    boundary = _eeg_event(
        {"stimtype": 0, "accept_ev1": 11, "offset": 1, "keypad_accept": 0, "keyboard": 0},
        keystroke=False,
    )
    assert boundary["type"] == "boundary"
    assert "duration" not in boundary
    assert boundary["latency"] == 1.0


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
    assert sidecar["PowerLineFrequency"] == "60"


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
    assert "pop_studywizard" in command


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
    assert "pop_participantinfo" in command


def test_mff_segmentation_follows_categories_not_epochs_xml(tmp_path):
    from eegprep.functions.popfunc.pop_mffimport import _epoch_events, _looks_segmented

    folder = tmp_path / "continuous.mff"
    folder.mkdir()
    (folder / "epochs.xml").write_text("<epochs/>", encoding="utf-8")
    assert not _looks_segmented(folder)
    (folder / "categories.xml").write_text("<categories/>", encoding="utf-8")
    assert _looks_segmented(folder)

    class _Epochs:
        times = np.array([-0.1, 0.0, 0.1])
        events = np.array([[5000, 0, 1]])

    events = _epoch_events(_Epochs(), "code")
    assert events[0]["latency"] == 2.0


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
