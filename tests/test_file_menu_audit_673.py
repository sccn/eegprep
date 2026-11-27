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
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
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
    saved, _datasets, command = pop_savestudy(study, alleeg, filename="audit.study", filepath=tmp_path)
    path = tmp_path / "audit.study"
    assert path.read_bytes().startswith(b"MATLAB 5.0 MAT-file")
    assert command.startswith("[STUDY ALLEEG] = pop_savestudy(")
    loaded, _alleeg = pop_loadstudy(path, load_datasets="off")
    assert loaded["name"] == "Audit"
    json_study, _json_datasets, _json_command = pop_savestudy(
        study, alleeg, filename="audit.json", filepath=tmp_path, fileformat="json"
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


def test_wizard_dialogs_open_in_the_qt_renderer(qapp):
    from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea

    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.functions.studyfunc.pop_studywizard import pop_studywizard_dialog_spec
    from eegprep.plugins.EEG_BIDS.bids_tools import bids_exporter_grid_spec, bids_exporter_welcome_spec

    renderer = QtDialogRenderer()
    _app, study_dialog, study_widgets = renderer.build_dialog(pop_studywizard_dialog_spec(["kept.set"], "Study"))
    study_dialog.show()
    qapp.processEvents()
    study_labels = {label.text() for label in study_dialog.findChildren(QLabel)}
    assert "browse" in study_labels
    assert "1" in study_labels
    assert "Clear" not in study_labels
    assert study_widgets["clear1"].text() == "Clear"
    assert study_widgets["file1"].text() == "kept.set"
    _assert_clear_buttons_are_visible(study_dialog, study_widgets)
    study_widgets["clear1"].click()
    qapp.processEvents()
    assert study_widgets["file1"].text() == ""
    study_dialog.close()

    files = [f"dataset_{index}.set" for index in range(1, 11)]
    _app, wide_dialog, wide_widgets = renderer.build_dialog(pop_studywizard_dialog_spec(files, "Study"))
    wide_dialog.show()
    qapp.processEvents()
    _assert_clear_buttons_are_visible(wide_dialog, wide_widgets)
    _assert_clear_column_is_visible(wide_dialog, wide_widgets["clear10"])
    assert wide_dialog.findChild(QScrollArea).verticalScrollBar().maximum() == 0
    wide_dialog.close()

    many = [f"dataset_{index}.set" for index in range(1, 41)]
    _app, tall_dialog, tall_widgets = renderer.build_dialog(pop_studywizard_dialog_spec(many, "Study"))
    tall_dialog.show()
    qapp.processEvents()
    assert tall_dialog.width() <= 1280
    assert tall_dialog.height() <= 800
    ok = next(button for button in tall_dialog.findChildren(QPushButton) if button.objectName() == "ok")
    assert tall_dialog.rect().contains(ok.mapTo(tall_dialog, ok.rect().topLeft()))
    assert tall_dialog.rect().contains(ok.mapTo(tall_dialog, ok.rect().bottomRight()))
    _assert_clear_column_is_visible(tall_dialog, tall_widgets["clear1"])
    scroll = tall_dialog.findChild(QScrollArea)
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    qapp.processEvents()
    _assert_clear_column_is_visible(tall_dialog, tall_widgets["clear40"])
    assert tall_dialog.rect().contains(ok.mapTo(tall_dialog, ok.rect().bottomRight()))
    tall_dialog.close()

    _app, grid_dialog, grid_widgets = renderer.build_dialog(bids_exporter_grid_spec())
    grid_dialog.show()
    qapp.processEvents()
    grid_labels = {label.text() for label in grid_dialog.findChildren(QLabel)}
    assert "browse" in grid_labels
    assert {str(index) for index in range(1, 11)} <= grid_labels
    assert "Clear" not in grid_labels
    assert grid_widgets["clear1"].text() == "Clear"
    grid_widgets["file1"].setText("a.set")
    grid_widgets["clear1"].click()
    qapp.processEvents()
    assert grid_widgets["file1"].text() == ""
    grid_dialog.close()

    _app, welcome, _widgets = renderer.build_dialog(bids_exporter_welcome_spec())
    welcome.show()
    qapp.processEvents()
    buttons = {button.text() for button in welcome.findChildren(QPushButton)}
    assert "Continue" in buttons
    assert "Ok" not in buttons
    continue_button = next(button for button in welcome.findChildren(QPushButton) if button.text() == "Continue")
    assert continue_button.contentsRect().width() >= continue_button.fontMetrics().horizontalAdvance("Continue")
    welcome.close()


def test_dialog_buttons_grow_past_button_size_when_the_label_is_wider(qapp):
    from PySide6.QtWidgets import QPushButton

    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec

    spec = DialogSpec(
        title="wide buttons",
        controls=(ControlSpec("text", "Export"),),
        geometry=((1,),),
        function_name="wide_buttons",
        eeglab_source="test",
        button_size=(110, 20),
        ok_label="Continue",
        help_text="Help for the export.",
        extra_stylesheet="QPushButton { font-size: 28px; }",
    )
    _app, dialog, _widgets = QtDialogRenderer().build_dialog(spec)
    dialog.show()
    qapp.processEvents()
    for name in ("ok", "cancel", "help"):
        button = dialog.findChild(QPushButton, name)
        assert button.width() >= 110
        assert button.height() >= 20
        assert button.contentsRect().width() >= button.fontMetrics().horizontalAdvance(button.text())
    assert dialog.findChild(QPushButton, "ok").width() > 110
    dialog.close()


def _assert_clear_buttons_are_visible(dialog, widgets) -> None:
    assert dialog.width() <= 1280
    assert dialog.height() <= 800
    _assert_clear_column_is_visible(dialog, widgets["clear1"])


def _assert_clear_column_is_visible(dialog, button) -> None:
    from PySide6.QtWidgets import QScrollArea

    scroll = dialog.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBar().maximum() == 0
    viewport = scroll.viewport()
    top_left = button.mapTo(viewport, button.rect().topLeft())
    bottom_right = button.mapTo(viewport, button.rect().bottomRight())
    assert viewport.rect().contains(top_left)
    assert viewport.rect().contains(bottom_right)


def _dialog_row(spec, control):
    index = spec.controls.index(control)
    cursor = 0
    for row in spec.geometry:
        width = len(row)
        if cursor <= index < cursor + width:
            return row
        cursor += width
    raise AssertionError(control.string)


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
    assert command == "[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY);"
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
    _saved, datasets, command = pop_savestudy(study, alleeg, filename="audit.study", filepath=tmp_path)
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
    assert _dialog_row(spec, hint) == (1.5, 1, 1)
    delete = next(control for control in spec.controls if control.string == "Delete event channel(s)?")
    assert _dialog_row(spec, delete) == (1.5, 0.21, 1)
    assert spec.controls[spec.controls.index(delete) + 2].string.strip() == "(set = yes)"
    assert sum(len(row) == 1 for row in spec.geometry) == 2


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
        "<epochs><epoch><beginTime>0</beginTime><endTime>1200000</endTime></epoch>"
        "<epoch><beginTime>1200000</beginTime><endTime>2400000</endTime></epoch></epochs>",
        encoding="utf-8",
    )
    (folder / "categories.xml").write_text(
        "<categories><cat><name>target</name><segments>"
        "<segment status=\"kept\"><beginTime>0</beginTime><endTime>1200000</endTime>"
        "<eventBegin>202000</eventBegin><eventEnd>202000</eventEnd>"
        "<keys><key><keyCode>#seg</keyCode><data>15</data><dataType>long</dataType></key>"
        "<key><keyCode>subj</keyCode><data>S01</data><dataType>person</dataType></key></keys></segment>"
        "<segment><beginTime>1200000</beginTime><endTime>2400000</endTime>"
        "<eventBegin>1402000</eventBegin><eventEnd>1402000</eventEnd></segment>"
        "</segments></cat></categories>",
        encoding="utf-8",
    )
    (folder / "info.xml").write_text(
        "<fileInfo><recordTime>2016-01-01T00:00:00+00:00</recordTime></fileInfo>", encoding="utf-8"
    )
    (folder / "Events_stm+.xml").write_text(
        "<eventTrack><name>stm+</name><trackType>STIM</trackType>"
        "<event><beginTime>2016-01-01T00:00:00.203960000+00:00</beginTime>"
        "<relativeBeginTime>203960</relativeBeginTime><duration>1000</duration>"
        "<code>stm+</code><label>stm+</label><description></description>"
        "<keys><key><keyCode>cel#</keyCode><data>1</data><dataType>long</dataType></key></keys></event>"
        "<event><relativeBeginTime>1406000</relativeBeginTime><duration>1000</duration>"
        "<code>resp</code><label>resp</label></event>"
        "</eventTrack>",
        encoding="utf-8",
    )
    eeg = _epochs_from_categories(raw, folder, "code")
    assert eeg["data"].shape == (257, 300, 2)
    assert float(np.max(eeg["data"])) == pytest.approx(1e6)
    assert eeg["xmin"] == pytest.approx(-0.202)
    assert [event["type"] for event in eeg["event"]] == ["stm+", "target", "resp", "target"]
    assert eeg["event"][0]["latency"] == pytest.approx(0.20396 * 250)
    assert eeg["event"][0]["mffkey_cel"] == "1"
    assert eeg["event"][0]["mffkeys"] == "[cel#: 1]"
    assert eeg["event"][0]["classid"] == "EVNT"
    assert eeg["event"][0]["status"] == ""
    assert eeg["event"][1]["status"] == "kept"
    assert eeg["event"][1]["classid"] == ""
    assert eeg["event"][1]["mffkeys"] == "[#seg: 15, subj: S01]"
    assert eeg["event"][1]["mffkeysbackup"].startswith("struct('code',{'#seg','subj'}")
    assert eeg["event"][0]["duration"] == pytest.approx(1000)
    assert len(eeg["epoch"]) == 2
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
    assert declared["event"][0]["latency"] == pytest.approx(0.202 * 250 + 1)
    assert [event["type"] for event in declared["event"]] == ["target"]
    assert declared["event"][0]["classid"] == ""
    assert declared["event"][0]["mffkeysbackup"] == "struct([])"
    broken = tmp_path / "broken.mff"
    broken.mkdir()
    (broken / "categories.xml").write_text("<categories/>", encoding="utf-8")
    with pytest.raises(Exception) as caught:
        pop_mffimport(broken)
    assert not isinstance(caught.value, AttributeError)


def test_importbids_dialog_lists_event_columns_and_aligns_filters_like_eeglab(qapp, tmp_path):
    from eegprep.functions.guifunc.qt import QtDialogRenderer
    from eegprep.plugins.EEG_BIDS.pop_importbids import _event_columns, pop_importbids_dialog_spec

    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    for run, header in (("1", "onset\tduration\tvalue\tfeedback"), ("2", "onset\tduration\tevent_code\tvalue")):
        (eeg_dir / f"sub-01_task-x_run-{run}_eeg.set").write_bytes(b"")
        (eeg_dir / f"sub-01_task-x_run-{run}_events.tsv").write_text(header + "\n0\t0\ta\tb\n")
    # A UTF-8 BOM (Excel export) must not turn "onset" into a selectable column.
    (eeg_dir / "sub-01_task-x_run-3_events.tsv").write_text("onset\tduration\tvalue\n", encoding="utf-8-sig")
    fields = _event_columns(tmp_path)
    assert fields == ("value", "event_code", "feedback")
    _app, dialog, widgets = QtDialogRenderer().build_dialog(pop_importbids_dialog_spec(tmp_path, fields))
    dialog.show()
    qapp.processEvents()
    filters = [widgets[tag] for tag in ("bidstaskstr", "bidsrunsstr", "bidssubjectsstr")]
    assert len({widget.mapTo(dialog, widget.rect().topLeft()).x() for widget in filters}) == 1
    assert not widgets["bidsrunsstr"].isEnabled()
    assert widgets["bidsrunsstr"].height() < 2 * widgets["bidstaskstr"].height()
    dialog.close()


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
    blocked = tmp_path / "readonly-mount"
    blocked.write_text("x", encoding="utf-8")
    with pytest.raises(OSError, match="pop_importbids") as caught:
        _save_derivatives([eeg], tmp_path, blocked)
    assert "writable folder" in str(caught.value)
    assert "DataLad" in str(caught.value)
    assert "OpenNeuro" in str(caught.value)


def test_read_only_folder_keeps_the_wrapped_save_messages(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_importbids import _save_derivatives

    if os.name == "nt" or getattr(os, "geteuid", lambda: 0)() == 0:
        pytest.skip("directory mode 0o555 is not enforced here")
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o555)
    leaf = tmp_path / "bids-out" / "sub-001" / "eeg"
    leaf.mkdir(parents=True)
    os.chmod(leaf, 0o555)
    try:
        eeg = _continuous()
        with pytest.raises(ValueError, match="pop_saveset: could not write") as saved:
            pop_saveset(eeg, str(locked / "x.set"))
        assert "The original file was left unchanged" in str(saved.value)
        assert "Choose a writable .set path" in str(saved.value)

        study, alleeg = pop_study(None, [eeg], name="Audit")
        with pytest.raises(ValueError, match="pop_savestudy: could not write") as studied:
            pop_savestudy(study, alleeg, filename=str(locked / "audit.study"))
        assert "Choose a writable .study path" in str(studied.value)
        assert "Permission denied" in str(studied.value)
        assert str(locked / "audit.study") in str(studied.value)
        assert "Reader needs file name" not in str(studied.value)

        eeg["subject"] = "001"
        eeg["filename"] = "sub-001_task-rest_eeg.set"
        eeg["filepath"] = str(tmp_path / "sub-001" / "eeg")
        with pytest.raises(OSError, match="pop_importbids") as imported:
            _save_derivatives([eeg], tmp_path, tmp_path / "bids-out")
        message = str(imported.value)
        assert "writable folder" in message
        assert isinstance(imported.value.__cause__, ValueError)
        assert isinstance(imported.value.__cause__.__cause__, OSError)
    finally:
        os.chmod(locked, 0o755)
        os.chmod(leaf, 0o755)


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


def test_export_writes_na_for_empty_channel_coordinates(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

    eeg = pop_loadset(SAMPLE.with_name("eeglab_data_epochs_ica.set"))
    eeg["subject"] = "S01"
    root = Path(pop_exportbids(eeg, tmp_path / "bids", task="rest"))
    rows = list(
        csv.reader((root / "sub-S01" / "eeg" / "sub-S01_electrodes.tsv").open(encoding="utf-8"), delimiter="\t")
    )
    by_name = {row[0]: row for row in rows[1:]}
    assert by_name["EOG1"] == ["EOG1", "n/a", "n/a", "n/a"]
    assert by_name["EOG2"] == ["EOG2", "n/a", "n/a", "n/a"]
    assert by_name["FPz"][1:] != ["n/a", "n/a", "n/a"]


def test_electrode_coordinates_use_six_decimals_including_negative_zero(tmp_path):
    from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids

    eeg = _continuous()
    eeg["subject"] = "01"
    eeg["nbchan"] = 3
    eeg["data"] = np.zeros((3, eeg["pnts"]))
    eeg["chanlocs"] = [
        {"labels": "Cz", "X": -0.0, "Y": 1.5, "Z": -2.25},
        {"labels": "Pz", "X": 84.98123361344625, "Y": 0.0, "Z": -1.7860385037488253},
        {"labels": "EOG1", "X": np.empty((1, 0)), "Y": np.empty((1, 0)), "Z": np.empty((1, 0))},
    ]
    root = Path(pop_exportbids(eeg, tmp_path / "bids", task="rest"))
    eeg_dir = root / "sub-01" / "eeg"
    assert (eeg_dir / "sub-01_electrodes.tsv").read_bytes() == (
        b"name\tx\ty\tz\nCz\t-0.000000\t1.500000\t-2.250000\nPz\t84.981234\t0.000000\t-1.786039\nEOG1\tn/a\tn/a\tn/a\n"
    )
    assert (eeg_dir / "sub-01_task-rest_channels.tsv").read_bytes() == (
        b"name\ttype\tunits\nCz\tEEG\tuV\nPz\tEEG\tuV\nEOG1\tEOG\tuV\n"
    )
    assert (eeg_dir / "sub-01_task-rest_events.tsv").read_bytes() == b"onset\tduration\tsample\tvalue\n"
    assert (root / "participants.tsv").read_bytes() == b"participant_id\nsub-01\n"
    text_files = [path for path in root.rglob("*") if path.suffix in {".tsv", ".json"} or path.name == "README"]
    assert text_files
    for path in text_files:
        assert b"\r" not in path.read_bytes()


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


def test_save_study_keeps_resaved_datasets_in_the_session(tmp_path):
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    eeg = _continuous()
    pop_saveset(eeg, tmp_path / "one.set")
    eeg["saved"] = "no"
    study, alleeg = pop_study(None, [eeg], name="Audit")
    study, alleeg, _command = pop_savestudy(study, alleeg, filename=tmp_path / "audit.study")
    alleeg[0]["saved"] = "no"
    session = EEGPrepSession()
    session.STUDY = study
    session.ALLEEG = alleeg
    session.EEG = alleeg[0]
    session.CURRENTSET = [1]
    prompts = []

    class Widgets:
        class QMessageBox:
            Yes = 1
            No = 2
            Cancel = 3

            @staticmethod
            def question(*_args, **_kwargs):
                prompts.append("question")
                return 1

    dispatcher = MenuActionDispatcher(session)
    import eegprep.functions.guifunc.menu_actions as menu_actions

    original = menu_actions._require_qt_widgets
    menu_actions._require_qt_widgets = lambda: Widgets
    try:
        dispatcher.dispatch("pop_savestudy:resave", object())
        saved = session.ALLEEG[0]
        assert saved["saved"] == "yes"
        assert saved["filename"] == "one.set"
        assert Path(saved["filepath"]) == tmp_path
        dispatcher.dispatch("pop_savestudy:resave", object())
    finally:
        menu_actions._require_qt_widgets = original
    assert prompts == ["question"]


def test_quit_yes_leaves_a_raw_bids_dataset_unchanged(tmp_path):
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    root = tmp_path / "bids"
    eeg_dir = root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        '{"Name": "raw", "BIDSVersion": "1.8.0", "DatasetType": "raw"}\n',
        encoding="utf-8",
    )
    set_path = eeg_dir / "sub-01_task-rest_eeg.set"
    eeg = _continuous()
    pop_saveset(eeg, set_path)
    eeg["saved"] = "no"
    before = set_path.read_bytes()
    modified = set_path.stat().st_mtime_ns
    session = EEGPrepSession()
    session.ALLEEG = [eeg]
    warnings = []

    class Widgets:
        class QMessageBox:
            Yes = 1
            No = 2
            Cancel = 3

            @staticmethod
            def question(*_args, **_kwargs):
                return 1

            @staticmethod
            def warning(_parent, _title, text):
                warnings.append(text)

    dispatcher = MenuActionDispatcher(session)
    import eegprep.functions.guifunc.menu_actions as menu_actions

    original_require = menu_actions._require_qt_widgets
    original_widgets = menu_actions._qt_widgets
    menu_actions._require_qt_widgets = lambda: Widgets
    menu_actions._qt_widgets = lambda: Widgets
    closed = []

    class Parent:
        def close(self):
            closed.append(True)

    try:
        dispatcher.dispatch("quit", Parent())
    finally:
        menu_actions._require_qt_widgets = original_require
        menu_actions._qt_widgets = original_widgets
    assert set_path.read_bytes() == before
    assert set_path.stat().st_mtime_ns == modified
    assert warnings and "refusing to overwrite raw BIDS" in warnings[0]
    assert closed == []


@pytest.fixture
def raw_bids_set(tmp_path):
    eeg = _continuous()
    eeg.update(setname="Raw recording", subject="01")
    root = Path(pop_exportbids([eeg], tmp_path / "bids", task="rest"))
    return next(root.rglob("*_eeg.set"))


@pytest.mark.parametrize("entry_point", ["script", "menu"])
@pytest.mark.parametrize("raw_saved", ["yes", "justloaded"])
def test_save_study_resaves_only_the_modified_derivative(tmp_path, raw_bids_set, raw_saved, entry_point, monkeypatch):
    raw = pop_loadset(raw_bids_set)
    raw["saved"] = raw_saved
    before = raw_bids_set.read_bytes()
    modified = raw_bids_set.stat().st_mtime_ns
    derivative_path = tmp_path / "bids" / "derivatives" / "eeglab" / "edited.set"
    derivative = _continuous()
    pop_saveset(derivative, derivative_path)
    derivative.update(comments="edited derivative", saved="no")
    study, alleeg = pop_study(None, [raw, derivative], name="Mixed study")
    study_path = tmp_path / "mixed.study"

    if entry_point == "menu":
        box = pytest.importorskip("PySide6.QtWidgets").QMessageBox
        monkeypatch.setattr(box, "question", lambda *_args: box.Yes)
        study.update(filename=study_path.name, filepath=str(tmp_path))
        session = EEGPrepSession(STUDY=study, ALLEEG=alleeg, CURRENTSTUDY=1)
        MenuActionDispatcher(session).dispatch("pop_savestudy:resave", parent=object())
        datasets = session.ALLEEG
    else:
        _, datasets, _ = pop_savestudy(study, alleeg, filename=study_path, resavedatasets="on")

    assert study_path.is_file()
    assert pop_loadset(derivative_path)["comments"] == "edited derivative"
    assert raw_bids_set.read_bytes() == before
    assert raw_bids_set.stat().st_mtime_ns == modified
    assert [eeg["saved"] for eeg in datasets] == [raw_saved, "yes"]


def test_pop_savestudy_refuses_to_resave_a_raw_bids_dataset(tmp_path, raw_bids_set):
    eeg = pop_loadset(raw_bids_set)
    eeg["saved"] = "no"
    before = raw_bids_set.read_bytes()
    modified = raw_bids_set.stat().st_mtime_ns
    study, alleeg = pop_study(None, [eeg], name="Raw")
    with pytest.raises(ValueError, match="pop_saveset: refusing to overwrite raw BIDS"):
        pop_savestudy(
            study,
            alleeg,
            filename=tmp_path / "saved.study",
            resavedatasets="on",
        )
    assert raw_bids_set.read_bytes() == before
    assert raw_bids_set.stat().st_mtime_ns == modified


def test_resave_keeps_a_live_memmap_view(tmp_path):
    from eegprep.functions.adminfunc.storage import memmap_fdt

    eeg = _continuous()
    path = tmp_path / "mapped.set"
    pop_saveset(eeg, path, savemode="twofiles")
    eeg["data"] = memmap_fdt(path.with_suffix(".fdt"), eeg)
    pop_saveset(eeg, path, savemode="twofiles")
    reloaded = pop_loadset(path)
    np.testing.assert_array_equal(np.asarray(reloaded["data"]), np.asarray(eeg["data"]))
    view = np.asarray(eeg["data"])[0]
    expected = np.array(view, copy=True)
    if os.name == "nt":
        with pytest.raises(ValueError, match="plot or the console"):
            pop_saveset(eeg, path, savemode="twofiles")
        np.testing.assert_array_equal(view, expected)
        return
    pop_saveset(eeg, path, savemode="twofiles")
    np.testing.assert_array_equal(view, expected)
    view[0] = expected[0] + 5
    reloaded = pop_loadset(path)
    assert float(np.asarray(reloaded["data"])[0, 0]) == float(expected[0])
    assert float(view[0]) == float(expected[0]) + 5


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
    candidates.append(Path(__file__).resolve().parent / "fixtures" / "testmff.mff")
    candidates.append(Path(__file__).resolve().parents[1] / "sample_data" / "test_data" / "testmff.mff")
    folder = next((path for path in candidates if path.is_dir()), None)
    if folder is None:
        pytest.fail("tests/fixtures/testmff.mff is not in the tree")
    eeg = pop_mffimport(folder, typefield="code")
    assert eeg["data"].shape == (257, 300, 2)
    assert float(np.max(np.abs(eeg["data"]))) == pytest.approx(19.75, rel=0.02)
    assert eeg["xmin"] == pytest.approx(-0.202, abs=1e-3)
    assert len(eeg["event"]) == 10
    latencies = [float(event["latency"]) for event in eeg["event"]]
    for expected in (50.99, 51.5, 242.2, 246.7, 350.7):
        assert any(abs(found - expected) < 0.05 for found in latencies)
    assert [str(event["type"]) for event in eeg["event"][:3]] == ["stm+", "LRND", "LRND"]
    found_types = {str(event["type"]).lower() for event in eeg["event"]}
    for name in ("stm+", "lrnd", "ulrn", "resp", "fbk+"):
        assert name in found_types
    stimulus = [event for event in eeg["event"] if str(event["type"]).lower() == "stm+"]
    assert stimulus
    latency = stimulus[0]["latency"]
    assert latency == 50.99, f"exact microseconds, not EEGLAB mff_decodetime 50.990568 (got {latency})"
    assert stimulus[0]["begintime"] == "2015-07-01T16:24:33.845920-07:00"
    assert stimulus[0]["classid"] == "EVNT"
    assert stimulus[0]["relativebegintime"] == 0
    assert stimulus[0]["mffkeys"] == "[cel#: 1, obs#: 3, pos#: 3, argu: 0]"
    category = next(event for event in eeg["event"] if "begintime" not in event and str(event["type"]) == "LRND")
    assert category["classid"] == ""
    assert category["mffkeys"].startswith("[#seg: 15, subj: ")
    assert category["mffkeysbackup"].startswith("struct('code'")
    assert stimulus[0]["status"] == ""
    assert stimulus[0]["mffkeysbackup"].startswith("struct('code',{'cel#'")
    assert stimulus[0]["urevent"] == 1
    assert all(float(event["duration"]) == pytest.approx(1000) for event in stimulus)
    assert len(eeg["epoch"]) == eeg["trials"] == 2
    assert all("event" in epoch and "eventlatency" in epoch for epoch in eeg["epoch"])
    first_latency = eeg["epoch"][0]["eventlatency"]
    first_value = float(np.asarray(first_latency).reshape(-1)[0])
    assert first_value == -2.042
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
    study["cluster"] = [
        {"name": "Parent", "sets": [[1], [2]], "comps": [1], "child": []},
        {"name": "Wide", "sets": [[1, 2], [3, 4]], "comps": [1, 2]},
    ]

    def too_big(*_args, **_kwargs):
        raise OverflowError("Matrix too large to save with Matlab 5 format")

    monkeypatch.setattr("eegprep.functions.studyfunc.pop_savestudy.scipy.io.savemat", too_big)
    pop_savestudy(study, alleeg, filename="pair.study", filepath=tmp_path)
    path = tmp_path / "pair.study"
    header = path.read_bytes()[:128]
    assert header.startswith(b"MATLAB 7.3")
    assert header[124:128] == b"\x00\x02IM"
    with h5py.File(path, "r") as handle:
        assert "struct" in str(handle["STUDY"].attrs["MATLAB_class"])
        assert "#refs#" in handle
    reloaded, datasets = pop_loadstudy(path, load_datasets="off")
    assert reloaded["name"] == "Pair"
    assert len(reloaded["datasetinfo"]) == 2
    assert reloaded["design"][0]["name"] == "Within"
    by_name = {cluster["name"]: cluster for cluster in reloaded["cluster"]}
    assert np.asarray(by_name["Parent"]["sets"]).shape == (2, 1)
    assert np.asarray(by_name["Wide"]["sets"]).shape == (2, 2)


def test_v73_char_and_empty_arrays_match_matlab_hdf5(tmp_path):
    import h5py

    from eegprep.functions.popfunc.pop_loadset_h5 import _read_value
    from eegprep.functions.studyfunc.pop_savestudy import _Mat73Writer, _write_mat73_value

    path = tmp_path / "types.h5"
    writer = _Mat73Writer()
    with h5py.File(path, "w") as handle:
        _write_mat73_value(handle, "hello", "hello", handle, writer)
        _write_mat73_value(handle, "blank", "", handle, writer)
        _write_mat73_value(handle, "nodouble", np.zeros((0, 0)), handle, writer)
        _write_mat73_value(handle, "onedim", np.array([]), handle, writer)
        _write_mat73_value(handle, "nocell", np.empty((0, 0), dtype=object), handle, writer)
        _write_mat73_value(handle, "flag", True, handle, writer)
        _write_mat73_value(handle, "box", {"a": 1.0}, handle, writer)
    with h5py.File(path, "r") as handle:
        hello = handle["hello"]
        _assert_fixed_matlab_class(hello, "char")
        assert hello.dtype == np.dtype("<u2")
        assert hello[()].reshape(-1).tolist() == [104, 101, 108, 108, 111]
        assert int(hello.attrs["MATLAB_int_decode"]) == 2
        assert hello.attrs["MATLAB_int_decode"].dtype == np.dtype("int32")
        blank = handle["blank"]
        _assert_fixed_matlab_class(blank, "char")
        assert blank.dtype == np.dtype("uint64")
        assert blank[()].reshape(-1).tolist() == [0, 0]
        assert int(blank.attrs["MATLAB_empty"]) == 1
        assert blank.attrs["MATLAB_empty"].dtype == np.dtype("uint8")
        assert "MATLAB_int_decode" not in blank.attrs
        nodouble = handle["nodouble"]
        _assert_fixed_matlab_class(nodouble, "double")
        assert nodouble.dtype == np.dtype("uint64")
        assert nodouble[()].reshape(-1).tolist() == [0, 0]
        assert int(nodouble.attrs["MATLAB_empty"]) == 1
        onedim = handle["onedim"]
        _assert_fixed_matlab_class(onedim, "double")
        assert onedim[()].reshape(-1).tolist() == [0, 0]
        assert int(onedim.attrs["MATLAB_empty"]) == 1
        assert "MATLAB_int_decode" not in onedim.attrs
        nocell = handle["nocell"]
        _assert_fixed_matlab_class(nocell, "cell")
        assert nocell.dtype == np.dtype("uint64")
        assert nocell[()].reshape(-1).tolist() == [0, 0]
        assert int(nocell.attrs["MATLAB_empty"]) == 1
        _assert_fixed_matlab_class(handle["flag"], "logical")
        _assert_fixed_matlab_class(handle["box"], "struct")
        assert _read_value(hello) == "hello"
        assert _read_value(blank) == ""
        assert _read_value(nodouble).shape == (0, 0)
        assert _read_value(nocell).dtype == object
        assert _read_value(nocell).shape == (0, 0)


def _assert_fixed_matlab_class(node, class_name: str) -> None:
    import h5py

    attribute = node.attrs.get_id("MATLAB_class")
    assert h5py.check_dtype(vlen=attribute.dtype) is None
    assert attribute.dtype == np.dtype(f"|S{len(class_name)}")
    assert bytes(node.attrs["MATLAB_class"]) == class_name.encode("ascii")


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

    eeg, command = pop_participantinfo({"data": []}, participant_id="sub-01")
    assert command == "[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY);"
    assert eeg["etc"]["bids"]["participant"]["participant_id"] == "sub-01"
    scripted, scripted_command = pop_participantinfo(
        {"data": []},
        rows=[{"participant_id": "sub-02", "Gender": "F"}],
    )
    assert scripted_command == "[EEG, STUDY, command] = pop_participantinfo(EEG, STUDY);"
    assert scripted["etc"]["bids"]["participant"]["Gender"] == "F"


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
    headers = {control.string for control in spec.controls if control.style == "text"}
    assert {"Raw dataset filename", "browse", "subject", "Page 1"} <= headers
    assert {str(index) for index in range(1, 11)} <= headers
    assert "Clear" not in headers
    assert sum(control.string == "..." for control in spec.controls) == 10
    assert sum(control.style == "pushbutton" and control.string == "Clear" for control in spec.controls) == 10
    assert any(control.string == "<" for control in spec.controls)
    dialog = _export_grid_dialog()
    dialog.show()
    qapp.processEvents()
    headers = {label.text() for label in dialog.findChildren(QLabel)}
    assert {"Raw dataset filename", "browse", "subject", "session", "run", "condition", "group", "task"} <= headers
    assert {str(index) for index in range(1, 11)} <= headers
    assert "Clear" not in headers
    assert len([button for button in dialog.findChildren(QPushButton) if button.text() == "..."]) == 10
    assert len([button for button in dialog.findChildren(QPushButton) if button.text() == "Clear"]) == 10
    dialog.edits[0]["file"].setText("a.set")
    dialog.edits[0]["subject"].setText("S01")
    clear = next(button for button in dialog.findChildren(QPushButton) if button.objectName() == "clear1")
    clear.click()
    assert dialog.edits[0]["file"].text() == ""
    assert dialog.edits[0]["subject"].text() == ""
    named = {button.objectName(): button for button in dialog.findChildren(QPushButton)}
    assert named["help"].x() < named["cancel"].x() < named["ok"].x()
    dialog.edits[0]["file"].setText("a.set")
    dialog.shift(1)
    assert dialog.page_label.text() == "Page 2"
    row_labels = {
        label.objectName(): label.text()
        for label in dialog.findChildren(QLabel)
        if label.objectName().startswith("row")
    }
    assert row_labels["row1"] == "11"
    assert row_labels["row10"] == "20"
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
    assert found["(click to select)"].text() == "(click to select)"
    assert found["(set=yes)"].text() == "(set=yes)"
    assert found["        (set = yes)"].text() == "        (set = yes)"
    layout = dialog.layout()
    assert dialog.height() <= layout.sizeHint().height() + 4
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
