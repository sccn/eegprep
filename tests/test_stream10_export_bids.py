"""Regression tests for release-readiness stream 10 (export and BIDS)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_export import pop_export, pop_export_dialog_spec
from eegprep.functions.popfunc.pop_expevents import pop_expevents
from eegprep.functions.popfunc.pop_expica import pop_expica
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_writeeeg import pop_writeeeg, pop_writeeeg_dialog_spec
from eegprep.plugins.EEG_BIDS.bids_tools import (
    pop_eventinfo,
    pop_eventinfo_dialog_spec,
    pop_participantinfo,
    pop_participantinfo_dialog_spec,
    pop_taskinfo,
    pop_taskinfo_dialog_spec,
    validate_bids,
)
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids, pop_exportbids_dialog_spec
from eegprep.plugins.EEG_BIDS.pop_importbids import pop_importbids, pop_importbids_dialog_spec


class _Renderer:
    def __init__(self, values):
        self.values = values

    def run(self, spec, initial_values=None):
        return None if self.values is None else dict(self.values)


def _eeg(*, pnts=8, srate=100.0, trials=1, events=None, data=None):
    eeg = eeg_emptyset()
    if data is None:
        data = np.arange(2 * pnts * trials, dtype=float).reshape(2, pnts, trials)
        if trials == 1:
            data = data[:, :, 0]
    eeg.update(
        {
            "setname": "stream10",
            "subject": "01",
            "nbchan": int(np.asarray(data).shape[0]),
            "pnts": pnts,
            "trials": trials,
            "srate": srate,
            "xmin": 0.0,
            "xmax": (pnts - 1) / srate,
            "data": data,
            "chanlocs": [{"labels": "Fz", "X": 0.1, "Y": 0.2, "Z": 0.3}, {"labels": "Cz"}],
            "event": events or [],
            "ref": "common",
        }
    )
    return eeg


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_pop_export_dialog_matches_eeglab_defaults_and_decimal_places(tmp_path):
    spec = pop_export_dialog_spec()
    controls = controls_by_tag(spec)
    labels = [control.string for control in spec.controls]

    assert spec.title == "Export data - pop_export()"
    assert "Number of significant digits to output:" in labels
    assert controls["precision"].string == "4"
    assert controls["transpose"].value is True
    assert controls["timeunit"].string == "1E-3"
    assert controls["time"].value is True

    eeg = _eeg(data=np.array([[1.23456789, -2.5], [0.0, 1.0]]))
    eeg["pnts"] = 2
    dialog_file = tmp_path / "dialog.txt"
    command = pop_export(
        eeg,
        renderer=_Renderer(
            {
                "filename": str(dialog_file),
                "ica": False,
                "erp": False,
                "transpose": True,
                "elec": True,
                "separator": False,
                "time": False,
                "timeunit": "1E-3",
                "precision": "4",
                "expr": "",
            }
        ),
    )
    assert _lines(dialog_file)[0] == "Fz\tCz"
    assert _lines(dialog_file)[1] == "1.2346\t0.0000"
    assert "'precision', 4" in command
    assert "'transpose', 'on'" in command

    direct_eeg = _eeg(data=np.array([[1.23456789, -2.5]]))
    direct_eeg["pnts"] = 2
    direct = tmp_path / "direct.txt"
    pop_export(direct_eeg, direct, "precision", 7, "time", "off", "elec", "off", "transpose", "off")
    assert _lines(direct) == ["1.2345679\t-2.5000000"]

    spaced = tmp_path / "rows.txt"
    pop_export(direct_eeg, spaced, "precision", 7, "transpose", "off", "time", "on", "elec", "on")
    assert _lines(spaced)[0].startswith(" Time\t")


def test_pop_export_rejects_a_blank_file_name_and_a_bad_time_unit(tmp_path):
    eeg = _eeg()
    with pytest.raises(ValueError, match="pop_export: File name required"):
        pop_export(
            eeg,
            renderer=_Renderer({"filename": "", "precision": "7", "timeunit": "1E-3", "elec": True, "time": True}),
        )
    with pytest.raises(ValueError, match="pop_export: time unit 'abc'"):
        pop_export(eeg, tmp_path / "ignored.txt", "timeunit", "abc")


def test_pop_export_ica_header_uses_component_numbers(tmp_path):
    eeg = _eeg(data=np.array([[1.0, 2.0], [3.0, 4.0]]))
    eeg["pnts"] = 2
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icaact"] = np.array([[0.25, 0.5], [0.75, 1.25]])
    path = tmp_path / "ica.txt"
    pop_export(eeg, path, "ica", "on", "transpose", "on", "time", "off", "elec", "on", "precision", 1)
    assert _lines(path)[0] == "1\t2"


def test_pop_expica_writes_save_ascii_width(tmp_path):
    eeg = _eeg()
    eeg["icaweights"] = np.array([[1.0, -0.5], [0.25, 2.0]])
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.array([[3.0], [4.0]])
    weights = tmp_path / "weights.txt"
    inverse = tmp_path / "inverse.txt"
    pop_expica(eeg, weights)
    pop_expica(eeg, inverse, "inv")
    text = weights.read_text(encoding="utf-8")
    assert "\t" not in text
    assert text.splitlines()[0] == f"{1.0:16.7e}{-0.5:16.7e}"
    assert inverse.read_text(encoding="utf-8").splitlines() == [f"{3.0:16.7e}", f"{4.0:16.7e}"]
    with pytest.raises(ValueError, match="pop_expica: matrix must be"):
        pop_expica(eeg, weights, "nope")


def test_pop_expevents_matches_eeg_eventtable_columns(tmp_path):
    eeg = _eeg(
        events=[
            {"type": "square", "latency": 11.0, "urevent": 0, "position": 3},
            {"type": "rt", "latency": 20.0, "urevent": 1},
        ]
    )
    path = tmp_path / "events.txt"
    command = pop_expevents(eeg, path)
    rows = list(csv.reader(path.open(encoding="utf-8"), delimiter="\t"))
    assert rows[0] == ["number", "type", "latency", "urevent", "position"]
    assert "duration" not in rows[0]
    assert rows[1][0] == "1.000000"
    assert rows[1][rows[0].index("urevent")] == "1.000000"
    assert rows[2][rows[0].index("position")] == ""
    assert "'samples'" in command

    seconds = tmp_path / "seconds.txt"
    pop_expevents(eeg, seconds, "seconds")
    second_rows = list(csv.reader(seconds.open(encoding="utf-8"), delimiter="\t"))
    latency = float(second_rows[1][second_rows[0].index("latency")])
    assert latency == pytest.approx(11.0 / eeg["srate"])
    with pytest.raises(ValueError, match="pop_expevents: unit must be"):
        pop_expevents(eeg, path, "ms")


def test_pop_writeeeg_keeps_every_sample_and_annotation(tmp_path):
    srate = 200.0
    pnts = 2112
    rng = np.random.default_rng(0)
    data = rng.normal(0, 20, size=(4, pnts))
    events = [
        {"type": f"e{index}", "latency": float(latency), "duration": 0.0}
        for index, latency in enumerate(np.linspace(1, pnts, 17))
    ]
    eeg = _eeg(pnts=pnts, srate=srate, data=data, events=events)
    eeg["chanlocs"] = [{"labels": f"C{index}"} for index in range(4)]
    eeg["nbchan"] = 4
    path = tmp_path / "recording.edf"
    command = pop_writeeeg(eeg, path, "TYPE", "EDF")

    reader = pyedflib.EdfReader(str(path))
    try:
        assert reader.getNSamples()[0] == pnts
        assert reader.file_duration == pytest.approx(pnts / srate)
        onsets, _durations, texts = reader.readAnnotations()
        quantum = (reader.getPhysicalMaximum(0) - reader.getPhysicalMinimum(0)) / (
            reader.getDigitalMaximum(0) - reader.getDigitalMinimum(0)
        )
        restored = reader.readSignal(0)
    finally:
        reader.close()
    assert len(onsets) == 17
    assert list(texts)[0] == "e0"
    assert list(texts)[-1] == "e16"
    assert np.max(np.abs(restored - data[0])) <= quantum
    assert command == f"pop_writeeeg(EEG, '{path.as_posix()}', 'TYPE', 'EDF');"
    assert not command.startswith("LASTCOM")

    spec = pop_writeeeg_dialog_spec(eeg)
    assert controls_by_tag(spec)["format"].string == "EDF|GDF|BDF"
    epoched = _eeg(trials=2, pnts=10)
    assert (
        pop_writeeeg(epoched, renderer=_Renderer({"filename": "epochs.edf", "format": 1, "concatenate": False})) == ""
    )
    written = tmp_path / "epochs.edf"
    pop_writeeeg(epoched, renderer=_Renderer({"filename": str(written), "format": 1, "concatenate": True}))
    reader = pyedflib.EdfReader(str(written))
    try:
        assert reader.getNSamples()[0] == 20
    finally:
        reader.close()


def test_pop_writeeeg_warns_for_bdf_and_rejects_an_impossible_length(tmp_path):
    eeg = _eeg(pnts=4, srate=2000.0, data=np.ones((1, 4)))
    eeg["nbchan"] = 1
    eeg["chanlocs"] = [{"labels": "Fz"}]
    with pytest.warns(UserWarning, match="pop_writeeeg: BDF"):
        pop_writeeeg(eeg, renderer=_Renderer({"filename": str(tmp_path / "short.bdf"), "format": 3}))
    one = _eeg(pnts=1, srate=2000.0, data=np.ones((1, 1)))
    one["nbchan"] = 1
    with pytest.raises(ValueError, match="pop_writeeeg: 1 samples at 2000"):
        pop_writeeeg(one, tmp_path / "nope.edf")


def test_bids_import_prefers_the_value_column(tmp_path):
    eeg = _eeg(events=[{"type": "stimulus", "latency": 1.0, "duration": 0.0}, {"type": "response", "latency": 50.0}])
    root = Path(pop_exportbids(eeg, tmp_path / "bids", task="oddball"))
    events_path = next(root.rglob("*_events.tsv"))
    events_path.write_text(
        "onset\tduration\tsample\tvalue\ttrial_type\n"
        "0\t0\t0\tstimulus\toddball\n"
        "0.49\tn/a\t49\tresponse\tstandard\n"
        "0.5\t0\t50\tnan\tnan\n",
        encoding="utf-8",
    )
    imported = pop_importbids(root)
    assert [event["type"] for event in imported["event"][:2]] == ["stimulus", "response"]
    assert imported["event"][2]["type"] == ""
    assert imported["etc"]["event_column"] == "value"

    spec = pop_importbids_dialog_spec(root)
    assert controls_by_tag(spec)["typefield"].string.split("|")[0] == "value"
    assert controls_by_tag(spec)["chanlocs"].value is True
    study, datasets, command = pop_importbids(
        root,
        gui=True,
        return_study=True,
        return_com=True,
        renderer=_Renderer(
            {
                "studyName": "Oddball",
                "chanlocs": True,
                "events": True,
                "typefield": 1,
                "bidstask": False,
                "bidssessions": False,
                "bidsruns": False,
                "bidssubjects": False,
                "folder": str(tmp_path / "derivatives" / "eeglab"),
            }
        ),
    )
    assert study["name"] == "Oddball"
    assert study["datasetinfo"]
    assert len(datasets) == 1
    assert command.startswith("[STUDY, ALLEEG] = pop_importbids(")
    assert "'eventtype', 'value'" in command
    assert "'bidsevent', 'on'" in command
    assert "derivatives/eeglab" in command
    assert any("listboxes" in note for note in spec.known_differences)
    assert not (tmp_path / "derivatives" / "eeglab").exists()
    assert pop_importbids(root, gui=True, renderer=_Renderer(None), return_com=True) == (None, "")


def test_imported_bids_dataset_can_be_saved(tmp_path):
    root = pop_exportbids(_eeg(), tmp_path / "bids")
    imported = pop_importbids(root)
    chaninfo = imported["chaninfo"]
    assert chaninfo["plotrad"] is not None
    assert chaninfo["shrink"] is not None
    saved = tmp_path / "saved.set"
    pop_saveset(imported, saved)
    loaded = pop_loadset(saved)
    assert loaded["data"].shape == imported["data"].shape


def test_pop_exportbids_writes_required_sidecars_and_value_column(tmp_path):
    eeg = _eeg(events=[{"type": 1.0, "latency": 6.0, "duration": 0.0, "trial_type": "oddball"}])
    root = Path(pop_exportbids(eeg, tmp_path / "bids", task="oddball"))
    prefix = root / "sub-01" / "eeg" / "sub-01_task-oddball_run-01"
    assert (root / "README").read_text(encoding="utf-8").startswith("# ")
    sidecar = (prefix.parent / f"{prefix.name}_eeg.json").read_text(encoding="utf-8")
    assert '"TaskName": "oddball"' in sidecar
    assert '"PowerLineFrequency": "n/a"' in sidecar
    electrodes = prefix.parent / f"{prefix.name}_electrodes.tsv"
    assert electrodes.exists()
    assert "task-oddball" in electrodes.name
    rows = list(csv.reader((prefix.parent / f"{prefix.name}_events.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert rows[0][:4] == ["onset", "duration", "sample", "value"]
    assert rows[1][rows[0].index("value")] == "1"
    assert rows[1][rows[0].index("sample")] == "5"
    assert rows[1][rows[0].index("trial_type")] == "oddball"
    assert validate_bids(root)["errors"] == []
    assert not (root / "eegprep_bids_validation.json").exists()

    (root / "README").unlink()
    report = validate_bids(root)
    assert any("validate_bids" in error and "README" in error for error in report["errors"])
    with pytest.raises(ValueError, match="pop_exportbids: BIDS export can only export EEGLAB studies"):
        pop_exportbids(eeg, tmp_path / "needs-study", require_study=True)
    with pytest.raises(ValueError, match="pop_exportbids: task name"):
        pop_exportbids(eeg, tmp_path / "bad-task", task="my task")

    wizard = pop_exportbids_dialog_spec()
    labels = [control.string for control in wizard.controls]
    assert "Output folder:" in labels
    assert "Edit task & EEG info" in labels
    assert "Edit participants" in labels
    assert "Edit event info" in labels
    exported, command = pop_exportbids(
        eeg,
        gui=True,
        return_com=True,
        renderer=_Renderer(
            {
                "outputfolder": str(tmp_path / "wizard"),
                "license": "Creative Common 0 (CC0)",
                "changes": "",
                "newids": False,
                "taskname": "oddball",
            }
        ),
    )
    assert command.startswith("LASTCOM = pop_exportbids")
    assert Path(exported, "dataset_description.json").exists()


def test_bids_metadata_dialogs_store_structured_fields():
    eeg = _eeg()
    task_labels = [control.string for control in pop_taskinfo_dialog_spec().controls]
    for label in (
        "Dataset name",
        "Task name (no space)",
        "README",
        "Participant task description",
        "Participant instructions",
        "Authors",
        "References and links",
        "Task-relevant Cognitive Atlas term",
        "Task-relevant CogPO term",
        "Institution",
        "Department",
        "Institution location",
        "Cap manufacturer",
        "Cap model",
        "EEG reference location",
        "EEG ground electrode location",
        "EEG montage system (10-20, 10-10, custom)",
        "EEG amplifier maker",
        "EEG amplifier model",
        "EEG amplifier serial #",
        "EEG acquisition software version",
        "Hardware filters",
        "Software filters",
        "Line frequency (Hz)",
    ):
        assert label in task_labels
    participant_labels = [control.string for control in pop_participantinfo_dialog_spec().controls]
    assert participant_labels[1] == "Subject code"
    event_labels = [control.string for control in pop_eventinfo_dialog_spec().controls]
    for label in ("onset", "duration", "sample", "value"):
        assert label in event_labels

    updated, command = pop_taskinfo(eeg, TaskName="sample-task")
    assert updated["etc"]["bids"]["task"]["TaskName"] == "sample-task"
    assert "pop_taskinfo" in command
    updated, command = pop_taskinfo(
        eeg,
        gui=True,
        renderer=_Renderer({"TaskName": "visual", "EEGReference": "average", "PowerLineFrequency": 3}),
    )
    assert command
    assert updated["etc"]["bids"]["task"]["TaskName"] == "visual"
    assert updated["etc"]["bids"]["task"]["PowerLineFrequency"] == "60"
    with pytest.raises(ValueError, match="pop_participantinfo: No subject ID found"):
        pop_participantinfo(eeg, gui=True, renderer=_Renderer({"participant_id": ""}))
    updated, _command = pop_eventinfo(eeg, gui=True, renderer=_Renderer({"value": "stimulus code", "onset": ""}))
    assert updated["etc"]["bids"]["event"] == {"value": "stimulus code"}


def test_export_menu_uses_dialogs_and_requires_a_study(monkeypatch):
    session = EEGPrepSession()
    session.store_current(_eeg())
    dispatcher = MenuActionDispatcher(session)
    with pytest.raises(ValueError, match="pop_exportbids: BIDS export can only export EEGLAB studies"):
        dispatcher.dispatch("pop_exportbids")

    seen = {}

    def _record(kind):
        def inner(*args, **kwargs):
            seen[kind] = args[1] if len(args) > 1 else kwargs.get("filename")
            return f"{kind}();"

        return inner

    monkeypatch.setattr("eegprep.functions.popfunc.pop_export.pop_export", _record("pop_export"))
    monkeypatch.setattr("eegprep.functions.popfunc.pop_writeeeg.pop_writeeeg", _record("pop_writeeeg"))
    dispatcher.dispatch("pop_export")
    dispatcher.dispatch("pop_writeeeg")
    assert seen == {"pop_export": None, "pop_writeeeg": None}


def test_qt_export_dialogs_build(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from eegprep.functions.guifunc import qt as qt_module

    if qt_module.QtCore is None or qt_module.QtWidgets is None:
        pytest.skip("PySide6 Qt libraries unavailable in this environment")
    QtDialogRenderer = qt_module.QtDialogRenderer

    renderer = QtDialogRenderer()
    eeg = _eeg()
    for spec in (
        pop_export_dialog_spec(),
        pop_writeeeg_dialog_spec(eeg),
        pop_exportbids_dialog_spec(),
        pop_importbids_dialog_spec(tmp_path),
        pop_taskinfo_dialog_spec(),
        pop_participantinfo_dialog_spec(),
        pop_eventinfo_dialog_spec(),
    ):
        _app, dialog, widgets = renderer.build_dialog(spec)
        try:
            assert dialog.windowTitle() == spec.title
            assert widgets
        finally:
            dialog.close()


def test_reference_pop_export_decimal_places(eeglab_backend, eeglab_working_directory):
    data = np.array([[1.23456789, -2.5]], dtype=float)
    eeg = {"data": data, "nbchan": 1, "pnts": 2, "trials": 1, "srate": 100.0, "xmin": 0.0, "xmax": 0.01}
    python_path = eeglab_working_directory / "python_export.txt"
    pop_export(eeg, python_path, "precision", 7, "time", "off", "elec", "off", "transpose", "off")
    matlab_path = eeglab_working_directory / "matlab_export.txt"
    eeglab_backend(
        "pop_export",
        eeg,
        str(matlab_path),
        "precision",
        7,
        "time",
        "off",
        "elec",
        "off",
        "transpose",
        "off",
    )
    assert python_path.read_text(encoding="utf-8") == matlab_path.read_text(encoding="utf-8")


def test_reference_pop_expica_save_ascii(eeglab_backend, eeglab_working_directory):
    eeg = {"icaweights": np.array([[1.0, -0.5], [0.25, 2.0]]), "icasphere": np.eye(2)}
    python_path = eeglab_working_directory / "python_ica.txt"
    pop_expica(eeg, python_path)
    matlab_path = eeglab_working_directory / "matlab_ica.txt"
    eeglab_backend("pop_expica", eeg, "weights", str(matlab_path))
    assert python_path.read_text(encoding="utf-8") == matlab_path.read_text(encoding="utf-8")


def test_reference_pop_expevents_table(eeglab_backend, eeglab_working_directory):
    eeg = {
        "srate": 100.0,
        "event": [{"type": "square", "latency": 11.0, "urevent": 0}],
    }
    python_path = eeglab_working_directory / "python_events.txt"
    pop_expevents(eeg, python_path)
    matlab_path = eeglab_working_directory / "matlab_events.txt"
    matlab_eeg = {
        "srate": 100.0,
        "event": [{"type": "square", "latency": 11.0, "urevent": 1.0}],
    }
    eeglab_backend("pop_expevents", matlab_eeg, str(matlab_path), "samples")
    assert python_path.read_text(encoding="utf-8") == matlab_path.read_text(encoding="utf-8")


def test_reference_bids_importeventfile_uses_value(eeglab_backend, tmp_path):
    present = int(np.asarray(eeglab_backend("exist", "bids_importeventfile", "file")).reshape(-1)[0])
    if present == 0:
        pytest.skip("EEG-BIDS bids_importeventfile is not on the MATLAB path")
    events = tmp_path / "events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\tvalue\n0\t0\toddball\tstimulus\n1\t0\tstandard\tresponse\n",
        encoding="utf-8",
    )
    eeg = {"srate": 100.0, "trials": 1, "pnts": 200, "data": np.zeros((1, 200)), "event": []}
    imported = eeglab_backend("bids_importeventfile", eeg, str(events))
    event = imported["event"]
    if isinstance(event, dict):
        types = list(np.asarray(event["type"]).reshape(-1))
    else:
        types = [item["type"] for item in np.asarray(event).reshape(-1)]
    assert [str(item) for item in types] == ["stimulus", "response"]


def test_reference_pop_writeeeg_edf_length(eeglab_backend, eeglab_working_directory):
    present = int(np.asarray(eeglab_backend("exist", "sopen", "file")).reshape(-1)[0])
    if present == 0:
        pytest.skip("Biosig sopen is not on the MATLAB path (pop_writeeeg.m:49)")
    pnts = 2112
    eeg = {
        "data": np.zeros((1, pnts)),
        "srate": 200.0,
        "nbchan": 1,
        "pnts": pnts,
        "trials": 1,
        "event": [{"type": f"e{index}", "latency": float(index + 1)} for index in range(17)],
        "chanlocs": [{"labels": "Fz"}],
    }
    matlab_path = eeglab_working_directory / "matlab_recording.edf"
    eeglab_backend("pop_writeeeg", eeg, str(matlab_path), "TYPE", "EDF")
    reader = pyedflib.EdfReader(str(matlab_path))
    try:
        assert reader.getNSamples()[0] == pnts
        assert len(reader.readAnnotations()[0]) == 17
    finally:
        reader.close()
