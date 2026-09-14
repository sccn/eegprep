"""Generated-data ports of current EEGLAB tutorial workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.popfunc.pop_comments import pop_comments
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_newset import pop_newset
from eegprep.functions.popfunc.pop_resample import pop_resample
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_maketrialinfo import std_maketrialinfo
from tests.eeglab_tests import eeglab_test
from tests.fixtures import create_test_eeg


TUTORIAL_WRAPPER = "unittesting_tutorial/tutorial_wrapperTest.m"


def _continuous_tutorial_eeg() -> dict:
    eeg = create_test_eeg(n_channels=4, n_samples=1024, srate=128.0)
    seconds = np.arange(eeg["pnts"], dtype=float) / eeg["srate"]
    eeg["data"] = np.stack(
        [
            np.sin(2 * np.pi * 6 * seconds) + 0.15 * np.sin(2 * np.pi * 0.2 * seconds),
            np.sin(2 * np.pi * 8 * seconds + 0.2),
            np.sin(2 * np.pi * 10 * seconds + 0.4),
            np.sin(2 * np.pi * 12 * seconds + 0.6),
        ]
    )
    eeg["times"] = seconds * 1000.0
    eeg["event"] = [
        {"type": "square", "latency": 257.0, "duration": 0.0, "urevent": 1},
        {"type": "rt", "latency": 289.0, "duration": 0.0, "urevent": 2},
        {"type": "square", "latency": 641.0, "duration": 0.0, "urevent": 3},
        {"type": "rt", "latency": 681.0, "duration": 0.0, "urevent": 4},
    ]
    eeg["urevent"] = [dict(event) for event in eeg["event"]]
    eeg["setname"] = "Generated continuous tutorial EEG"
    return eeg


@eeglab_test(TUTORIAL_WRAPPER, "test_eeglab_history")
def test_eeglab_history_tutorial_runs_as_replayable_generated_data_pipeline():
    eeg = _continuous_tutorial_eeg()
    alleeg, eeg, currentset = eeg_store([], eeg)

    eeg, filter_command = pop_eegfilt(eeg, 1, 0, 128, return_com=True)
    alleeg, eeg, currentset, newset_command = pop_newset(
        alleeg,
        eeg,
        currentset,
        setname="filtered Continuous EEG Data",
    )
    eeg, reref_command = pop_reref(eeg, [], return_com=True)
    eeg, comment_command = pop_comments(
        eeg,
        "",
        "Dataset was highpass filtered at 1 Hz and rereferenced.",
        True,
        return_com=True,
    )
    eeg, epoch_command = pop_epoch(
        eeg,
        ["square"],
        [-1, 2],
        newname="Continuous EEG Data epochs",
        return_com=True,
    )
    alleeg, eeg, currentset, overwrite_command = pop_newset(
        alleeg,
        eeg,
        currentset,
        setname="Continuous EEG Data epochs",
        overwrite="on",
    )
    eeg, baseline_command = pop_rmbase(eeg, [-1000, 0], return_com=True)
    alleeg, eeg, currentset = eeg_store(alleeg, eeg, currentset)
    resampled, resample_command = pop_resample(eeg, 64, return_com=True)
    figures, topoplot_command = pop_topoplot(
        eeg,
        1,
        [0, 200, 400],
        "Topographic plot",
        [1, 3],
        0,
        electrodes="on",
        plot="off",
        return_com=True,
    )

    assert eeg["trials"] == 2
    assert eeg["data"].shape == (4, eeg["pnts"], 2)
    baseline = np.asarray(eeg["times"]) <= 0
    np.testing.assert_allclose(np.mean(eeg["data"][:, baseline, :], axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.mean(eeg["data"], axis=0), 0.0, atol=1e-10)
    assert resampled["srate"] == 64.0
    assert resampled["pnts"] < eeg["pnts"]
    assert currentset == 2
    assert len(alleeg) == 2
    assert len(figures) == 1
    assert "highpass filtered" in eeg["comments"]
    for command in (
        filter_command,
        newset_command,
        reref_command,
        comment_command,
        epoch_command,
        overwrite_command,
        baseline_command,
        resample_command,
        topoplot_command,
    ):
        assert command
    plt.close(figures[0])


@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_single_dataset")
def test_event_processing_single_dataset_shifts_events_and_adds_time_locked_cues():
    eeg = _continuous_tutorial_eeg()
    original_count = len(eeg["event"])
    original_square_latencies = [event["latency"] for event in eeg["event"] if event["type"] == "square"]

    for event in eeg["event"]:
        event["latency"] += 10
    for event in list(eeg["event"]):
        if event["type"] != "square":
            continue
        cue = dict(event)
        cue["latency"] = event["latency"] - 0.1 * eeg["srate"]
        cue["type"] = "cue"
        eeg["event"].append(cue)
    eeg = eeg_checkset(eeg, "eventconsistency")

    events = list(eeg["event"])
    shifted_squares = [event["latency"] for event in events if event["type"] == "square"]
    cue_latencies = [event["latency"] for event in events if event["type"] == "cue"]
    assert len(events) == original_count + 2
    np.testing.assert_allclose(shifted_squares, np.asarray(original_square_latencies) + 10)
    np.testing.assert_allclose(cue_latencies, np.asarray(shifted_squares) - 0.1 * eeg["srate"])
    assert [event["latency"] for event in events] == sorted(event["latency"] for event in events)


@eeglab_test(TUTORIAL_WRAPPER, "test_event_processing_study")
def test_event_processing_study_exposes_derived_reaction_time_as_a_design_variable(tmp_path: Path):
    eeg = create_test_eeg(n_channels=4, n_samples=64, n_trials=2, srate=128.0)
    eeg["setname"] = "generated_rtevents"
    eeg["subject"] = "S01"
    eeg["epoch"] = []
    eeg["event"] = [
        {"type": "square", "latency": 17.0, "epoch": 1},
        {"type": "rt", "latency": 37.0, "epoch": 1},
        {"type": "square", "latency": 81.0, "epoch": 2},
        {"type": "rt", "latency": 113.0, "epoch": 2},
    ]
    for current, following in zip(eeg["event"], eeg["event"][1:]):
        if current["type"] == "square" and following["type"] == "rt" and current["epoch"] == following["epoch"]:
            current["rt"] = (following["latency"] - current["latency"]) / eeg["srate"] * 1000.0
    dataset_path = tmp_path / "generated_rtevents.set"
    pop_saveset(eeg, dataset_path)

    study, alleeg = std_editset(
        None,
        None,
        commands=[["index", 1, "load", dataset_path, "subject", "S01"]],
        updatedat="off",
    )
    study, trialinfo = std_maketrialinfo(study, alleeg)
    rt_values = [row["rt"] for row in trialinfo[0]]
    study = std_makedesign(
        study,
        alleeg,
        1,
        name="Reaction time",
        variable1="rt",
        values1=rt_values,
        vartype1="continuous",
    )

    np.testing.assert_allclose(rt_values, [156.25, 250.0])
    assert study["design"][0]["variable"][0]["label"] == "rt"
    assert study["design"][0]["variable"][0]["vartype"] == "continuous"
    assert study["datasetinfo"][0]["trialinfo"] == trialinfo[0]
