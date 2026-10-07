import ast
import math
from pathlib import PurePath

import numpy as np
import pytest

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc.pop_chanevent import pop_chanevent
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.popfunc.pop_importevent import pop_importevent
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.plugins.EEG_BIDS.bids_tools import pop_eventinfo, pop_participantinfo, pop_taskinfo
from tests.fixtures import SAMPLE_DATASET_PATH


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


def _matlab_string(value):
    if isinstance(value, PurePath):
        value = value.as_posix()
    return "'" + str(value).replace("'", "''") + "'"


def test_eeg_from_data_raises_on_ambiguous_tall_array():
    # A tall channel-major array (more channels than samples) must not be
    # silently transposed; orientation has to be stated via nbchan.
    with pytest.raises(ValueError, match="orientation"):
        eeg_from_data(np.zeros((256, 100)), srate=100)


def test_eeg_from_data_loads_tall_channel_major_with_nbchan():
    eeg = eeg_from_data(np.arange(256 * 100, dtype=float).reshape(256, 100), srate=100, nbchan=256)

    assert eeg["nbchan"] == 256
    assert eeg["pnts"] == 100
    assert eeg["data"].shape == (256, 100)


def test_pop_fileio_uses_importdata_for_text_arrays(tmp_path):
    data_file = tmp_path / "data's.csv"
    np.savetxt(data_file, np.array([[1, 2, 3], [4, 5, 6]]), delimiter=",")

    eeg, command = pop_fileio(data_file, return_com=True)

    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 3
    assert command == f"EEG = pop_fileio({_matlab_string(data_file)});"


def test_pop_fileio_imports_plain_mat_data_array(tmp_path):
    import scipy.io

    data_file = tmp_path / "raw.mat"
    scipy.io.savemat(data_file, {"data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})

    eeg = pop_fileio(data_file)

    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 3


def test_pop_fileio_does_not_silently_fall_back_for_eeglab_mat(tmp_path):
    import scipy.io

    # An EEGLAB dataset .mat whose data sidecar is missing must surface the load
    # failure, not be silently re-imported as a raw MATLAB data array.
    dataset = tmp_path / "broken.mat"
    scipy.io.savemat(
        dataset,
        {
            "data": "missing.fdt",
            "datfile": "missing.fdt",
            "nbchan": 4,
            "srate": 100,
            "pnts": 100,
            "trials": 1,
            "xmin": 0.0,
            "xmax": 1.0,
            "setname": "x",
            "chanlocs": np.array([]),
            "event": np.array([]),
            "icachansind": np.array([]),
        },
    )

    with pytest.raises(FileNotFoundError):
        pop_fileio(dataset)


def test_pop_importevent_append_rebuilds_sorted_urevent_pointers(tmp_path):
    events_file = tmp_path / "events.tsv"
    events_file.write_text("type\tlatency\tduration\nnew1\t5\t0\nnew2\t9\t0\n", encoding="utf-8")
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))
    original = [(event["type"], event["latency"]) for event in eeg["event"]]
    n_urevents = len(eeg["urevent"])

    appended = pop_importevent(eeg, "event", events_file, "timeunit", math.nan, "append", "yes")

    kept = [(e["type"], e["latency"]) for e in appended["event"] if e["type"] not in {"new1", "new2"}]
    assert kept == original
    assert len(appended["urevent"]) == n_urevents + 2
    for event in appended["event"]:
        assert appended["urevent"][event["urevent"]]["type"] == event["type"]
        assert appended["urevent"][event["urevent"]]["latency"] == event["latency"]


def test_pop_chanevent_preserves_existing_urevents_when_appending():
    eeg = eeg_from_data(np.array([[0, 0, 1, 1, 0, 0]], dtype=float), srate=100)
    eeg["event"] = [{"type": "old", "latency": 1, "urevent": 0}]
    eeg["urevent"] = [{"type": "old", "latency": 1}]

    imported = pop_chanevent(eeg, 1, "edge", "both", "delchan", "off", "delevent", "off")
    events = [dict(event) for event in imported["event"]]
    urevents = [dict(event) for event in imported["urevent"]]

    assert [event["latency"] for event in events] == [1, 2, 5]
    assert [event["urevent"] for event in events] == [0, 1, 2]
    assert [urevents[event["urevent"]]["latency"] for event in events] == [1, 2, 5]
    assert urevents == [
        {"type": "old", "latency": 1},
        {"type": "chan1", "latency": 2},
        {"type": "chan1", "latency": 5},
    ]


def test_pop_chanevent_rebuilds_urevents_when_replacing_events():
    eeg = eeg_from_data(np.array([[0, 0, 1, 1, 0, 0]], dtype=float), srate=100)
    eeg["event"] = [{"type": "old", "latency": 1, "urevent": 0}]
    eeg["urevent"] = [{"type": "old", "latency": 1}]

    imported = pop_chanevent(eeg, 1, "edge", "both", "delchan", "off", "delevent", "on")
    events = [dict(event) for event in imported["event"]]
    urevents = [dict(event) for event in imported["urevent"]]

    assert [event["latency"] for event in events] == [2, 5]
    assert [event["urevent"] for event in events] == [0, 1]
    assert urevents == [
        {"type": "chan1", "latency": 2},
        {"type": "chan1", "latency": 5},
    ]


def test_study_save_load_and_bids_metadata(tmp_path):
    study, alleeg, command = pop_study(None, [_eeg()], name="demo study", return_com=True)
    study, task_command = pop_taskinfo(study, TaskName="odd'ball")
    study, participant_command = pop_participantinfo(study, participant_id="sub-01")
    study, event_command = pop_eventinfo(study, trial_type="stimulus")
    study_file = tmp_path / "demo's.study"
    saved, save_command = pop_savestudy(study, filename=study_file, return_com=True)
    loaded, loaded_alleeg, load_command = pop_loadstudy(study_file, load_datasets=False, return_com=True)

    assert alleeg[0]["setname"] == "demo"
    assert saved["filename"] == "demo's.study"
    assert loaded["etc"]["bids"]["task"]["TaskName"] == "odd'ball"
    assert loaded_alleeg == []
    assert "pop_study" in command
    assert "'odd''ball'" in task_command
    assert "pop_participantinfo" in participant_command
    assert "pop_eventinfo" in event_command
    assert "pop_savestudy" in save_command
    assert "demo's.study" in save_command
    ast.parse(save_command)
    assert study_file.name in load_command
    assert "load_datasets=False" in load_command
    ast.parse(load_command)
    assert f"filepath={str(study_file.parent)!r}" in load_command
    ast.parse(load_command)
