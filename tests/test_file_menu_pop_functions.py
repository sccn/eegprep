import math

import numpy as np

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc.pop_chanevent import pop_chanevent
from eegprep.functions.popfunc.pop_expevents import pop_expevents
from eegprep.functions.popfunc.pop_expica import pop_expica
from eegprep.functions.popfunc.pop_export import pop_export
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_importepoch import pop_importepoch
from eegprep.functions.popfunc.pop_importevent import pop_importevent
from eegprep.functions.popfunc.pop_runscript import pop_runscript
from eegprep.functions.popfunc.pop_saveh import pop_saveh
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.plugins.EEG_BIDS.bids_tools import pop_eventinfo, pop_participantinfo, pop_taskinfo, validate_bids
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids


def _eeg(epoched=False):
    data = np.arange(12, dtype=float).reshape(2, 6)
    if epoched:
        data = np.arange(24, dtype=float).reshape(2, 6, 2)
    eeg = eeg_from_data(data, srate=100, setname="demo", chanlocs=[{"labels": "Cz"}, {"labels": "Pz"}])
    eeg["event"] = [{"type": "stim", "latency": 2, "duration": 1}]
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)
    eeg["icachansind"] = np.arange(2)
    return eeg


def test_pop_importdata_imports_ascii_data(tmp_path):
    data_file = tmp_path / "data.tsv"
    np.savetxt(data_file, np.array([[1, 2, 3], [4, 5, 6]]), delimiter="\t")

    eeg, command = pop_importdata("data", data_file, "srate", 250, return_com=True)

    assert eeg["data"].shape == (2, 3)
    assert eeg["srate"] == 250
    assert eeg["setname"] == "data"
    assert "pop_importdata" in command


def test_pop_importdata_imports_in_memory_array():
    eeg, command = pop_importdata("data", np.array([[1, 2, 3], [4, 5, 6]]), "srate", 250, return_com=True)

    assert eeg["data"].shape == (2, 3)
    assert eeg["srate"] == 250
    assert "pop_importdata" in command


def test_pop_fileio_uses_importdata_for_text_arrays(tmp_path):
    data_file = tmp_path / "data.csv"
    np.savetxt(data_file, np.array([[1, 2, 3], [4, 5, 6]]), delimiter=",")

    eeg, command = pop_fileio(data_file, return_com=True)

    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 3
    assert command == f"EEG = pop_fileio('{data_file}');"


def test_pop_importevent_replaces_and_appends_events(tmp_path):
    events_file = tmp_path / "events.tsv"
    events_file.write_text("type\tlatency\tduration\nstim\t1\t0\nresp\t4\t1\n", encoding="utf-8")
    eeg = _eeg()

    replaced, command = pop_importevent(eeg, "event", events_file, "timeunit", math.nan, return_com=True)
    appended = pop_importevent(eeg, "event", events_file, "timeunit", math.nan, "append", "yes")

    assert [event["type"] for event in replaced["event"]] == ["stim", "resp"]
    assert [event["latency"] for event in replaced["event"]] == [1, 4]
    assert len(appended["event"]) == 3
    assert "pop_importevent" in command


def test_pop_importepoch_requires_epoch_count_match(tmp_path):
    epoch_file = tmp_path / "epochs.tsv"
    epoch_file.write_text("condition\nrare\nfrequent\n", encoding="utf-8")
    eeg = _eeg(epoched=True)

    imported, command = pop_importepoch(eeg, epoch_file, return_com=True)

    assert [epoch["condition"] for epoch in imported["epoch"]] == ["rare", "frequent"]
    assert imported["event"].size == 0
    assert "pop_importepoch" in command


def test_pop_chanevent_extracts_edges_without_deleting_channel():
    eeg = eeg_from_data(np.array([[0, 0, 1, 1, 0, 0]], dtype=float), srate=100)

    imported, command = pop_chanevent(eeg, 1, "edge", "both", "delchan", "off", return_com=True)

    assert [event["latency"] for event in imported["event"]] == [2, 5]
    assert imported["nbchan"] == 1
    assert "pop_chanevent" in command


def test_pop_chanevent_rejects_out_of_range_channels():
    eeg = eeg_from_data(np.array([[0, 1, 0]], dtype=float), srate=100)

    try:
        pop_chanevent(eeg, 0)
    except ValueError as exc:
        assert "1-based" in str(exc)
    else:
        raise AssertionError("pop_chanevent accepted a zero channel index")


def test_text_exports_write_data_ica_and_events(tmp_path):
    eeg = _eeg()
    data_file = tmp_path / "data.tsv"
    weights_file = tmp_path / "weights.tsv"
    events_file = tmp_path / "events.tsv"

    data_command = pop_export(eeg, data_file, "transpose", "on")
    weights_command = pop_expica(eeg, weights_file, "weights")
    events_command = pop_expevents(eeg, events_file)

    assert data_file.read_text(encoding="utf-8").splitlines()[0].startswith("Time\tCz\tPz")
    assert weights_file.exists()
    assert events_file.read_text(encoding="utf-8").splitlines()[0] == "duration\tlatency\ttype"
    assert "pop_export" in data_command
    assert "pop_expica" in weights_command
    assert str(weights_file) in weights_command
    assert "pop_expevents" in events_command


def test_history_script_save_and_python_run(tmp_path):
    history_file = tmp_path / "hist.m"
    script_file = tmp_path / "script.py"
    script_file.write_text("EEG['setname'] = 'scripted'\n", encoding="utf-8")
    namespace = {"EEG": _eeg()}

    command = pop_saveh(["first;", "second;"], history_file.name, history_file.parent)
    run_command = pop_runscript(script_file, namespace)

    assert "second;" in history_file.read_text(encoding="utf-8").splitlines()[2]
    assert namespace["EEG"]["setname"] == "scripted"
    assert "pop_saveh" in command
    assert "pop_runscript" in run_command


def test_study_save_load_and_bids_metadata(tmp_path):
    study, alleeg, command = pop_study(None, [_eeg()], name="demo study")
    study, task_command = pop_taskinfo(study, TaskName="oddball")
    study, participant_command = pop_participantinfo(study, participant_id="sub-01")
    study, event_command = pop_eventinfo(study, trial_type="stimulus")
    saved, save_command = pop_savestudy(study, filename=tmp_path / "demo.study")
    loaded, loaded_alleeg, load_command = pop_loadstudy(tmp_path / "demo.study")

    assert alleeg[0]["setname"] == "demo"
    assert saved["filename"] == "demo.study"
    assert loaded["etc"]["bids"]["task"]["TaskName"] == "oddball"
    assert loaded_alleeg == []
    assert "pop_study" in command
    assert "pop_taskinfo" in task_command
    assert "pop_participantinfo" in participant_command
    assert "pop_eventinfo" in event_command
    assert "pop_savestudy" in save_command
    assert "demo.study" in save_command
    assert "pop_loadstudy" in load_command


def test_pop_exportbids_and_validate_bids_write_minimal_dataset(tmp_path):
    root, command = pop_exportbids(_eeg(), tmp_path / "bids", return_com=True)
    report = validate_bids(root)

    assert (tmp_path / "bids" / "dataset_description.json").exists()
    assert list((tmp_path / "bids" / "sub-01" / "eeg").glob("*_eeg.set"))
    assert report["errors"] == []
    assert "pop_exportbids" in command
