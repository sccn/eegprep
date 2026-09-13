"""Behavioral ports of current EEGLAB dataset-workflow wrapper tests."""

from __future__ import annotations

from copy import deepcopy

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from eegprep.functions.adminfunc.pop_delset import pop_delset
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.importevent import importevent
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit
from eegprep.functions.popfunc.pop_editset import pop_editset
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_eventstat import event_values, pop_eventstat
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_newset import pop_newset
from eegprep.functions.popfunc.pop_rmdat import pop_rmdat
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent
from eegprep.functions.popfunc.pop_signalstat import pop_signalstat
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp
from tests.eeglab_tests import eeglab_test


def _continuous_eeg(name: str = "continuous", *, nbchan: int = 4, pnts: int = 240) -> dict:
    srate = 100.0
    samples = np.arange(pnts, dtype=float)
    data = np.vstack(
        [np.sin(2 * np.pi * (channel + 1) * samples / 53) + channel + samples / pnts for channel in range(nbchan)]
    )
    latencies = [31.0, 81.0, 151.0, 211.0]
    types = ["square", "rt", "square", "rt"]
    events = [
        {"type": event_type, "latency": latency, "position": index % 2 + 1, "urevent": index}
        for index, (event_type, latency) in enumerate(zip(types, latencies))
    ]
    chanlocs = [
        {
            "labels": f"Ch{index + 1}",
            "theta": float(index * 90),
            "radius": 0.3,
            "X": float(np.cos(index * np.pi / 2)),
            "Y": float(np.sin(index * np.pi / 2)),
            "Z": 0.0,
            "type": "EEG",
        }
        for index in range(nbchan)
    ]
    return {
        "setname": name,
        "filename": "",
        "filepath": "",
        "subject": "S01",
        "condition": "rest",
        "group": "control",
        "session": 1,
        "run": 1,
        "comments": "",
        "data": data,
        "nbchan": nbchan,
        "pnts": pnts,
        "trials": 1,
        "srate": srate,
        "xmin": 0.0,
        "xmax": (pnts - 1) / srate,
        "times": samples / srate * 1000,
        "ref": "common",
        "chanlocs": chanlocs,
        "urchanlocs": deepcopy(chanlocs),
        "chaninfo": {},
        "event": events,
        "urevent": [{key: value for key, value in event.items() if key != "urevent"} for event in events],
        "epoch": [],
        "eventdescription": {},
        "epochdescription": {},
        "reject": {},
        "stats": {},
        "specdata": {},
        "specicaact": {},
        "icaweights": np.eye(nbchan),
        "icasphere": np.eye(nbchan),
        "icawinv": np.eye(nbchan),
        "icaact": data.copy(),
        "icachansind": np.arange(nbchan),
        "history": "",
        "saved": "no",
        "etc": {},
    }


def _epoched_eeg(name: str = "epoched") -> dict:
    eeg = _continuous_eeg(name, pnts=200)
    eeg["data"] = np.asarray(eeg["data"]).reshape(4, 50, 4, order="F")
    eeg["pnts"] = 50
    eeg["trials"] = 4
    eeg["xmin"] = -0.1
    eeg["xmax"] = 0.39
    eeg["times"] = np.linspace(-100, 390, 50)
    eeg["event"] = []
    eeg["urevent"] = []
    eeg["epoch"] = []
    for trial in range(1, 5):
        event = {
            "type": "square" if trial % 2 else "rt",
            "latency": (trial - 1) * 50 + 11.0,
            "epoch": trial,
            "position": 1 if trial in {1, 3} else 2,
            "urevent": trial - 1,
        }
        eeg["event"].append(event)
        eeg["urevent"].append({key: value for key, value in event.items() if key not in {"epoch", "urevent"}})
        eeg["epoch"].append({"condition": "odd" if trial % 2 else "even"})
    eeg["icaact"] = eeg["data"].copy()
    return eeg


@eeglab_test("unittesting_popfunc/importevent/popfunc_importevent_wrapperTest.m", "test_pass_general")
def test_current_importevent_reads_named_fields_and_skips_header(tmp_path):
    event_file = tmp_path / "testevent.txt"
    event_file.write_text("type latency code\ntest 400 3\n", encoding="utf-8")

    events = importevent(
        event_file,
        [],
        250,
        "fields",
        ["type", "latency", "code"],
        "skipline",
        1,
    )

    assert events == [{"type": "test", "latency": 100001.0, "code": 3, "init_index": 1, "init_time": 400.0}]


@eeglab_test("unittesting_popfunc/importevent/popfunc_importevent_wrapperTest.m", "test_test_latency")
def test_current_importevent_seconds_and_sample_positions_have_identical_boundaries():
    seconds = importevent(
        [[0, "Experiment begins"], [49, "Experiment ends"]],
        [],
        1,
        "fields",
        ["latency", "type"],
        "timeunit",
        1,
    )
    samples = importevent(
        [[1, "Experiment begins"], [50, "Experiment ends"]],
        [],
        1,
        "fields",
        ["latency", "type"],
        "timeunit",
        np.nan,
    )

    assert [event["latency"] for event in seconds] == [1.0, 50.0]
    assert [event["latency"] for event in samples] == [1.0, 50.0]


@eeglab_test("unittesting_popfunc/pop_chanedit/popfunc_pop_chanedit_wrapperTest.m", "test_test_pop_chanedit")
def test_current_pop_chanedit_loads_locations_and_records_display_shrink(tmp_path):
    location_file = tmp_path / "eeglab_chan32.locs"
    location_file.write_text("1 0 0.25 Fz\n2 180 0.3 Pz\n", encoding="utf-8")

    chanlocs = pop_chanedit([], "load", [location_file, "filetype", ""], "shrink", -0.1)

    assert [channel["labels"] for channel in chanlocs] == ["Fz", "Pz"]
    assert chanlocs[0]["shrink"] == pytest.approx(-0.1)
    assert chanlocs[0]["radius"] == pytest.approx(0.25)


@eeglab_test("unittesting_popfunc/pop_delset/popfunc_pop_delset_wrapperTest.m", "test_test_pop_delset")
def test_current_pop_delset_blanks_selected_dataset_without_renumbering_following_sets():
    first = _continuous_eeg("first")
    second = _continuous_eeg("second")

    alleeg, command = pop_delset([first, second], [1])

    assert alleeg[0] == {}
    assert alleeg[1]["setname"] == "second"
    assert command == "ALLEEG = pop_delset( ALLEEG, [1] );"


@eeglab_test("unittesting_popfunc/pop_editset/popfunc_pop_editset_wrapperTest.m", "test_test_pop_editset")
def test_current_pop_editset_builds_a_consistent_epoched_dataset_from_arrays():
    data = np.arange(72, dtype=float).reshape(3, 24)
    chanlocs = [{"labels": "Fz"}, {"labels": "Cz"}, {"labels": "Pz"}]

    output = pop_editset(
        eeg_emptyset(),
        "setname",
        "UnitTesting",
        "data",
        data,
        "dataformat",
        "array",
        "subject",
        "S01",
        "condition",
        "testing",
        "group",
        "control",
        "session",
        1,
        "chanlocs",
        chanlocs,
        "pnts",
        8,
        "srate",
        200,
        "xmin",
        -0.1,
        "ref",
        "common",
        "icaweights",
        np.eye(3),
        "icasphere",
        np.eye(3),
        "comments",
        "fully exercised replacement for the commented MATLAB test",
    )

    assert output["data"].shape == (3, 8, 3)
    assert (output["nbchan"], output["pnts"], output["trials"]) == (3, 8, 3)
    assert output["times"][[0, -1]].tolist() == pytest.approx([-100.0, -65.0])
    assert output["icaact"].shape == (3, 8, 3)
    assert [channel["labels"] for channel in output["chanlocs"]] == ["Fz", "Cz", "Pz"]


@eeglab_test("unittesting_popfunc/pop_eegfilt/popfunc_pop_eegfilt_wrapperTest.m", "test_test_pop_eegfilt")
def test_current_pop_eegfilt_default_highpass_attenuates_sub_cutoff_signal():
    srate = 100.0
    pnts = 4000
    time = np.arange(pnts) / srate
    below_cutoff = 2 * np.sin(2 * np.pi * 0.2 * time)
    passband = np.sin(2 * np.pi * 10 * time)
    eeg = _continuous_eeg(pnts=pnts)
    eeg["data"] = np.vstack([below_cutoff + passband, 0.5 * below_cutoff + 2 * passband, below_cutoff, passband])
    eeg["icaact"] = np.ones_like(eeg["data"])
    input_data = eeg["data"].copy()
    input_events = deepcopy(eeg["event"])

    output, command = pop_eegfilt(eeg, 1, 0, [], [0], return_com=True)

    frequencies = np.fft.rfftfreq(pnts, 1 / srate)
    low_bin = int(np.argmin(np.abs(frequencies - 0.2)))
    pass_bin = int(np.argmin(np.abs(frequencies - 10)))
    input_spectrum = np.abs(np.fft.rfft(input_data[0]))
    output_spectrum = np.abs(np.fft.rfft(output["data"][0]))
    assert output_spectrum[low_bin] < input_spectrum[low_bin] * 0.01
    assert output_spectrum[pass_bin] > input_spectrum[pass_bin] * 0.95
    assert output["data"].shape == input_data.shape
    assert output["event"] == input_events
    assert output["icaact"].size == 0
    assert output["saved"] == "no"
    np.testing.assert_array_equal(eeg["data"], input_data)
    assert command == "EEG = pop_eegfilt( EEG, 1, 0, [], [0], 0, 0, 'firls', 0);"


@eeglab_test("unittesting_popfunc/pop_eventstat/popfunc_pop_eventstat_wrapperTest.m", "test_test_pop_eventstat")
def test_current_pop_eventstat_filters_type_and_epoch_relative_latency_numerically():
    continuous = _continuous_eeg()
    epoched = _epoched_eeg()

    all_result = pop_eventstat(continuous, "latency", "", [], 5, plot="off")
    typed_result = pop_eventstat(continuous, "latency", "rt", [500, 2200], 80, plot="off")
    epoch_result = pop_eventstat(epoched, "latency", "square", [-1, 1], 5, plot="off")

    assert all_result.mean == pytest.approx(np.mean([31, 81, 151, 211]))
    assert typed_result.mean == pytest.approx(np.mean([81, 211]))
    np.testing.assert_array_equal(event_values(epoched, "latency", type="square", latrange=[-1, 1]), [11, 111])
    assert epoch_result.mean == pytest.approx(61.0)
    assert typed_result.trimmed_indices.size == 0
    assert np.isnan(typed_result.trimmed_mean)
    plt.close(all_result.figure)
    plt.close(typed_result.figure)
    plt.close(epoch_result.figure)


@eeglab_test("unittesting_popfunc/pop_mergeset/popfunc_pop_mergeset_wrapperTest.m", "test_test_pop_mergeset")
def test_current_pop_mergeset_handles_continuous_epoched_list_and_pair_forms():
    continuous = [_continuous_eeg(f"continuous-{index}") for index in range(3)]
    merged = pop_mergeset(continuous, [1, 2], 0)
    kept_ica = pop_mergeset(continuous, [1, 2], 1)
    direct = pop_mergeset(continuous[0], continuous[1], 0)
    merged_three = pop_mergeset(continuous, [1, 2, 3], 0)

    assert merged["data"].shape == direct["data"].shape == (4, 480)
    assert merged_three["data"].shape == (4, 720)
    assert sum(event["type"] == "boundary" for event in merged_three["event"]) == 2
    assert merged["icaweights"].size == 0
    assert kept_ica["icaweights"].shape == (4, 4)
    np.testing.assert_allclose(
        kept_ica["icaweights"] @ kept_ica["icasphere"] @ kept_ica["icawinv"],
        np.eye(4),
    )

    epoched = [_epoched_eeg(f"epoched-{index}") for index in range(3)]
    merged_epochs = pop_mergeset(epoched, [1, 2], 0)
    kept_epoch_ica = pop_mergeset(epoched, [1, 2], 1)
    direct_epochs = pop_mergeset(epoched[0], epoched[1], 0)
    merged_three_epochs = pop_mergeset(epoched, [1, 2, 3], 0)

    assert merged_epochs["data"].shape == direct_epochs["data"].shape == (4, 50, 8)
    assert merged_three_epochs["data"].shape == (4, 50, 12)
    assert merged_epochs["trials"] == 8
    assert [event["epoch"] for event in merged_epochs["event"]] == list(range(1, 9))
    assert kept_epoch_ica["icaweights"].shape == (4, 4)
    np.testing.assert_allclose(
        kept_epoch_ica["icaweights"] @ kept_epoch_ica["icasphere"] @ kept_epoch_ica["icawinv"],
        np.eye(4),
    )


@eeglab_test("unittesting_popfunc/pop_newset/popfunc_pop_newset_wrapperTest.m", "test_test_pop_newset")
def test_current_pop_newset_overwrites_appends_saves_and_retrieves(tmp_path):
    original = _epoched_eeg("base")
    alleeg = [deepcopy(original) for _index in range(8)]

    alleeg, current, current_set, _ = pop_newset(
        alleeg, deepcopy(original), 1, "setname", "origin", "comments", "no change", "overwrite", "on"
    )
    assert current_set == 1
    assert alleeg[0]["setname"] == current["setname"] == "origin"

    alleeg, current, current_set, _ = pop_newset(
        alleeg, deepcopy(alleeg[7]), 8, "setname", "new", "comments", "change", "overwrite", "off"
    )
    assert current_set == 9
    assert len(alleeg) == 9

    alleeg, current, current_set, _ = pop_newset(
        alleeg, deepcopy(alleeg[7]), 8, "setname", "replacement", "overwrite", "on"
    )
    assert current_set == 8
    assert alleeg[7]["setname"] == "replacement"

    old_file = tmp_path / "old.set"
    new_file = tmp_path / "new.set"
    alleeg, current, current_set, _ = pop_newset(
        alleeg,
        deepcopy(alleeg[1]),
        2,
        "setname",
        "saved replacement",
        "overwrite",
        "on",
        "saveold",
        old_file,
        "savenew",
        new_file,
    )
    assert old_file.exists() and new_file.exists()
    assert current["saved"] == "yes"

    _alleeg, retrieved, retrieved_set, _ = pop_newset(alleeg, current, current_set, "retrieve", 1)
    assert retrieved_set == 1
    assert retrieved["setname"] == "origin"


@eeglab_test("unittesting_popfunc/pop_rmdat/popfunc_pop_rmdat_wrapperTest.m", "test_test_pop_rmdat")
def test_current_pop_rmdat_keeps_and_removes_each_requested_event_window():
    eeg = _continuous_eeg(pnts=240)

    keep_rt = pop_rmdat(eeg, ["rt"], [-0.1, 0.2], 0)
    remove_rt = pop_rmdat(eeg, ["rt"], [-0.1, 0.2], 1)
    keep_square = pop_rmdat(eeg, ["square"], [-0.1, 0.2], 0)
    keep_clipped = pop_rmdat(eeg, ["rt"], [-10, 200], 0)
    keep_both = pop_rmdat(eeg, ["rt", "square"], [-0.1, 0.2], 0)

    assert 0 < keep_rt["pnts"] < eeg["pnts"]
    assert remove_rt["pnts"] == eeg["pnts"] - keep_rt["pnts"]
    assert keep_square["pnts"] == keep_rt["pnts"]
    assert keep_clipped["pnts"] == eeg["pnts"]
    assert keep_both["pnts"] > keep_rt["pnts"]


@eeglab_test("unittesting_popfunc/pop_runica/popfunc_pop_runica_wrapperTest.m", "test_test_pop_runica")
def test_current_pop_runica_extended_pca_produces_a_valid_reduced_decomposition():
    rng = np.random.default_rng(17)
    sources = np.vstack(
        [
            rng.laplace(size=600),
            rng.uniform(-1, 1, size=600),
            np.sin(np.linspace(0, 35, 600)),
            np.sign(np.sin(np.linspace(0, 21, 600))),
        ]
    )
    mixing = rng.normal(size=(6, 4))
    eeg = _continuous_eeg(nbchan=6, pnts=600)
    with np.errstate(all="ignore"):
        eeg["data"] = mixing @ sources
    eeg["icaweights"] = np.array([])
    eeg["icasphere"] = np.array([])
    eeg["icawinv"] = np.array([])
    eeg["icaact"] = np.array([])
    eeg["icachansind"] = np.array([], dtype=int)

    output = pop_runica(eeg, "icatype", "runica", "extended", 1, "pca", 4, "maxsteps", 96, "seed", 7)

    assert output["icaweights"].shape == (4, 6)
    assert output["icasphere"].shape == (6, 6)
    assert output["icawinv"].shape == (6, 4)
    assert output["icaact"].shape == (4, 600, 1)
    with np.errstate(all="ignore"):
        expected_activations = output["icaweights"] @ output["icasphere"] @ output["data"]
    np.testing.assert_allclose(expected_activations, output["icaact"].reshape(4, 600), rtol=1e-8, atol=1e-8)
    assert output["icachansind"].tolist() == list(range(6))


def _overlapping_epoch_eeg() -> dict:
    eeg = eeg_emptyset()
    eeg.update(
        {
            "srate": 500.0,
            "nbchan": 1,
            "data": np.zeros((1, 2000)),
            "pnts": 2000,
            "trials": 1,
            "xmin": 0.0,
            "xmax": 3.998,
            "times": np.arange(2000) / 500 * 1000,
            "event": [
                {"type": "1", "latency": 201.0, "urevent": 0},
                {"type": "2", "latency": 501.0, "urevent": 1},
            ],
            "urevent": [{"type": "1", "latency": 201.0}, {"type": "2", "latency": 501.0}],
        }
    )
    epoched, _indices = pop_epoch(eeg, ["1", "2"], [-0.2, 1])
    return epoched


@eeglab_test(
    "unittesting_popfunc/pop_selectevent/popfunc_pop_selectevent_wrapperTest.m",
    "test_demo_selectevent_glitch",
)
def test_current_pop_selectevent_handles_overlapping_epochs_and_millisecond_ranges():
    eeg = _overlapping_epoch_eeg()

    latency_selected, _ = pop_selectevent(eeg, "type", "2", "deleteepochs", "on", "latency", "-10<=10")
    all_type_two, _ = pop_selectevent(eeg, "type", "2", "deleteepochs", "on")
    only_type_two, _ = pop_selectevent(eeg, "type", "2", "deleteepochs", "on", "deleteevents", "on")
    only_type_one, _ = pop_selectevent(eeg, "type", "1", "deleteepochs", "on", "deleteevents", "on")

    assert len(latency_selected["event"]) == 1
    assert len(all_type_two["event"]) == 3
    assert len(only_type_two["event"]) == 2
    assert len(only_type_one["event"]) == 1


@eeglab_test("unittesting_popfunc/pop_selectevent/popfunc_pop_selectevent_wrapperTest.m", "test_test_pop_selectevent")
def test_current_pop_selectevent_selects_epochs_using_a_custom_event_field():
    selected, event_indices = pop_selectevent(
        _epoched_eeg(), "position", 1, "deleteevents", "off", "deleteepochs", "on"
    )

    assert selected["trials"] == 2
    assert event_indices == [1, 3]
    assert [event["position"] for event in selected["event"]] == [1, 1]
    np.testing.assert_array_equal(selected["data"], _epoched_eeg()["data"][:, :, [0, 2]])


@eeglab_test("unittesting_popfunc/pop_signalstat/popfunc_pop_signalstat_wrapperTest.m", "test_test_pop_signalstat")
def test_current_pop_signalstat_reports_raw_and_component_statistics_at_multiple_trim_levels():
    eeg = _epoched_eeg()
    data = np.asarray(eeg["data"])

    raw = pop_signalstat(eeg, 1, 2, 5, plot="off")
    component = pop_signalstat(eeg, 0, 2, 5, plot="off")
    heavily_trimmed = pop_signalstat(eeg, 1, 2, 50, plot="off")
    lightly_trimmed = pop_signalstat(eeg, 0, 2, 0.5, plot="off")

    assert raw.mean == pytest.approx(float(np.mean(data[1])))
    assert component.mean == pytest.approx(raw.mean)
    assert heavily_trimmed.trimmed_indices.size < raw.trimmed_indices.size
    assert lightly_trimmed.trimmed_indices.size >= component.trimmed_indices.size
    for result in (raw, component, heavily_trimmed, lightly_trimmed):
        assert np.isfinite(result.matlab_tuple()[:9]).all()
        plt.close(result.figure)


@eeglab_test("unittesting_popfunc/pop_subcomp/popfunc_pop_subcomp_wrapperTest.m", "test_test_pop_subcomp")
def test_current_pop_subcomp_removes_component_three_from_data_and_ica_fields():
    eeg = _epoched_eeg()
    original = np.asarray(eeg["data"]).copy()

    output = pop_subcomp(eeg, [3], 0)

    np.testing.assert_allclose(output["data"][[0, 1, 3]], original[[0, 1, 3]])
    np.testing.assert_allclose(output["data"][2], 0)
    assert output["icaweights"].shape == (3, 4)
    assert output["icawinv"].shape == (4, 3)
    assert output["icaact"].size == 0
    assert output["setname"].endswith("pruned with ICA")
