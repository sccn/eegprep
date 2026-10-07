"""Behavioral ports of current EEGLAB dataset-workflow wrapper tests."""

from __future__ import annotations

from copy import deepcopy
import shutil

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_eegfilt import pop_eegfilt
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_runica import pop_runica
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent
from tests.eeglab_tests import assert_matlab_equal, eeglab_test
from tests.eeglab_tests.assertions import matlab_field_concat


def _reference_dataset_array(eeg, count):
    return np.array([[tuple(eeg.values())] * count], dtype=[(field, object) for field in eeg])


@eeglab_test("unittesting_popfunc/importevent/popfunc_importevent_wrapperTest.m", "test_pass_general")
def test_reference_importevent_original_text_file(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    source = eeglab_suite_root / "unittesting_popfunc/importevent/testevent.txt"
    shutil.copyfile(source, eeglab_working_directory / "testevent.txt")
    events = eeglab_backend(
        "importevent",
        "testevent.txt",
        np.empty((0, 0)),
        250.0,
        "fields",
        np.array([["type", "latency", "code"]], dtype=object),
        "skipline",
        1.0,
    )
    assert isinstance(events, dict) or max(events.shape) == 1


@eeglab_test("unittesting_popfunc/pop_delset/popfunc_pop_delset_wrapperTest.m", "test_test_pop_delset")
def test_reference_delset_original_stored_recordings(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    alleeg, eeg, _ = eeglab_backend("eeg_store", np.empty((0, 0)), eeg, nargout=3)
    alleeg, eeg, _ = eeglab_backend("eeg_store", alleeg, eeg, nargout=3)
    eeglab_backend("pop_delset", alleeg, np.array([[1.0]]))


@eeglab_test("unittesting_popfunc/pop_mergeset/popfunc_pop_mergeset_wrapperTest.m", "test_test_pop_mergeset")
def test_reference_mergeset_all_eight_recorded_data_calls(eeglab_backend, eeglab_suite_root):
    for filename in ("eeglab_data.set", "eeglab_data_epochs_ica.set"):
        eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data" / filename))
        alleeg = eeglab_backend("eeg_checkset", _reference_dataset_array(eeg, 3), "loaddata")
        eeglab_backend("pop_mergeset", alleeg, np.array([[1.0, 2.0]]), 0.0)
        eeglab_backend("pop_mergeset", alleeg, np.array([[1.0, 2.0]]), 1.0)
        first = {field: alleeg[field][0, 0] for field in alleeg.dtype.names}
        second = {field: alleeg[field][0, 1] for field in alleeg.dtype.names}
        eeglab_backend("pop_mergeset", first, second, 0.0)
        eeglab_backend("pop_mergeset", alleeg, np.array([[1.0, 2.0, 3.0]]), 0.0)


@eeglab_test("unittesting_popfunc/pop_newset/popfunc_pop_newset_wrapperTest.m", "test_test_pop_newset")
def test_reference_newset_original_eight_datasets_and_saved_files(
    eeglab_backend, eeglab_suite_root, eeglab_working_directory
):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    alleeg = _reference_dataset_array(eeg, 8)
    alleeg, eeg, _ = eeglab_backend(
        "pop_newset", alleeg, eeg, 1.0, "setname", "origin", "comments", "no change", "overwrite", "on", nargout=3
    )
    for overwrite in ("off", "on"):
        eeg = {field: alleeg[field][0, 7] for field in alleeg.dtype.names}
        alleeg, eeg, _ = eeglab_backend(
            "pop_newset", alleeg, eeg, 8.0, "setname", "new", "comments", "change", "overwrite", overwrite, nargout=3
        )
    eeg = {field: alleeg[field][0, 1] for field in alleeg.dtype.names}
    alleeg, eeg, _ = eeglab_backend(
        "pop_newset",
        alleeg,
        eeg,
        2.0,
        "setname",
        "new",
        "comments",
        "change",
        "overwrite",
        "on",
        "saveold",
        "old.set",
        "savenew",
        "new.set",
        nargout=3,
    )
    for filename in ("old.set", "old.fdt", "new.set", "new.fdt"):
        # MATLAB delete warns, rather than errors, if optional split data is absent.
        (eeglab_working_directory / filename).unlink(missing_ok=True)
    eeg = {field: alleeg[field][0, 2] for field in alleeg.dtype.names}
    eeglab_backend(
        "pop_newset",
        alleeg,
        eeg,
        3.0,
        "setname",
        "new",
        "comments",
        "change",
        "overwrite",
        "on",
        "retrieve",
        1.0,
        nargout=3,
    )


@eeglab_test("unittesting_popfunc/pop_rmdat/popfunc_pop_rmdat_wrapperTest.m", "test_test_pop_rmdat")
def test_reference_rmdat_original_event_windows(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    for kinds, limits, invert in (
        (["rt"], [-1.0, 2.0], 0.0),
        (["rt"], [-1.0, 2.0], 1.0),
        (["square"], [-1.0, 2.0], 0.0),
        (["rt"], [-10.0, 200.0], 0.0),
        (["rt", "square"], [-1.0, 2.0], 0.0),
    ):
        eeglab_backend("pop_rmdat", eeg, np.array([kinds], dtype=object), np.array([limits]), invert)


@eeglab_test("unittesting_popfunc/pop_chanedit/popfunc_pop_chanedit_wrapperTest.m", "test_test_pop_chanedit")
def test_reference_chanedit_original_locations(eeglab_backend, eeglab_suite_root):
    sample_directory = eeglab_suite_root / "eeglab/sample_data"
    eeg = eeglab_backend("pop_loadset", str(sample_directory / "eeglab_data.set"))
    eeg["chanlocs"] = eeglab_backend(
        "pop_chanedit",
        eeg["chanlocs"],
        "load",
        np.array([[str(sample_directory / "eeglab_chan32.locs"), "filetype", ""]], dtype=object),
        "shrink",
        -0.1,
    )


@eeglab_test("unittesting_popfunc/pop_eegfilt/popfunc_pop_eegfilt_wrapperTest.m", "test_test_pop_eegfilt")
def test_reference_eegfilt_original_recording_and_defaults(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("pop_eegfilt", eeg, 1.0, 0.0, np.empty((0, 0)), np.array([[0.0]]))


@eeglab_test("unittesting_popfunc/pop_eventstat/popfunc_pop_eventstat_wrapperTest.m", "test_test_pop_eventstat")
@pytest.mark.gui
def test_reference_eventstat_all_twenty_original_calls(eeglab_backend, eeglab_suite_root, request):
    for filename in ("eeglab_data.set", "eeglab_data_epochs_ica.set"):
        eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data" / filename))
        xmin, xmax = np.asarray(eeg["xmin"]).item(), np.asarray(eeg["xmax"]).item()
        shift = (xmax - xmin) / 10.0
        cases = [("",), ("rt",)]
        for percent in (5.0, 80.0):
            cases.extend(
                (
                    ("", np.empty((0, 0)), percent),
                    ("rt", np.empty((0, 0)), percent),
                    ("rt", 1000.0 * np.array([[xmin, xmax]]), percent),
                    ("rt", 1000.0 * np.array([[xmin + shift, xmax - 2.0 * shift]]), percent),
                )
            )
        for options in cases:
            eeglab_backend("pop_eventstat", eeg, "latency", *options, nargout=11)
            if request.config.getoption("--eeglab-backend") == "matlab":
                eeglab_backend("close", nargout=0)
            else:
                plt.close()


@eeglab_test("unittesting_popfunc/pop_runica/popfunc_pop_runica_wrapperTest.m", "test_test_pop_runica")
@pytest.mark.slow
def test_reference_runica_original_recording_and_defaults(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("pop_runica", eeg, "icatype", "runica", "extended", 1.0, "pca", 4.0)


@eeglab_test(
    "unittesting_popfunc/pop_selectevent/popfunc_pop_selectevent_wrapperTest.m", "test_demo_selectevent_glitch"
)
def test_reference_selectevent_original_overlapping_epoch_workflow(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    eeg["srate"] = 500.0
    eeg["nbchan"] = 1.0
    eeg["data"] = np.zeros((1, 2000))
    eeg["event"] = np.array([[("1", 201.0), ("2", 501.0)]], dtype=[("type", object), ("latency", object)])
    eeg = eeglab_backend("eeg_checkset", eeg)
    eeg = eeglab_backend(
        "pop_epoch", eeg, np.array([["1", "2"]], dtype=object), np.array([[-0.2, 1.0]]), "epochinfo", "yes"
    )
    for kind, options, expected_count in (
        ("2", ("latency", "-10<=10"), 1),
        ("2", (), 3),
        ("2", ("deleteevents", "on"), 2),
        ("1", ("deleteevents", "on"), 1),
    ):
        selected = eeglab_backend("pop_selectevent", eeg, "type", kind, "deleteepochs", "on", *options)
        events = selected["event"]
        count = 1 if isinstance(events, dict) else np.asarray(events).size
        assert count == expected_count


@eeglab_test("unittesting_popfunc/pop_selectevent/popfunc_pop_selectevent_wrapperTest.m", "test_test_pop_selectevent")
def test_reference_selectevent_original_recording_position_field(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("pop_selectevent", eeg, "position", 1.0, "deleteevents", "off", "deleteepochs", "on")


@eeglab_test("unittesting_popfunc/pop_signalstat/popfunc_pop_signalstat_wrapperTest.m", "test_test_pop_signalstat")
@pytest.mark.gui
def test_reference_signalstat_all_four_original_calls(eeglab_backend, eeglab_suite_root, request):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    for arguments in ((1.0, 5.0), (0.0, 5.0), (1.0, 5.0, 50.0), (0.0, 5.0, 0.5)):
        eeglab_backend("pop_signalstat", eeg, *arguments, nargout=11)
        if request.config.getoption("--eeglab-backend") == "matlab":
            eeglab_backend("close", nargout=0)
        else:
            plt.close()


@eeglab_test("unittesting_popfunc/pop_subcomp/popfunc_pop_subcomp_wrapperTest.m", "test_test_pop_subcomp")
def test_reference_subcomp_original_recording(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("pop_subcomp", eeg, np.array([[3.0]]), 0.0)


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


@eeglab_test("unittesting_popfunc/importevent/popfunc_importevent_wrapperTest.m", "test_test_latency")
def test_reference_importevent_original_named_workspace_variables(eeglab_backend, request):
    options = (
        "dataformat",
        "array",
        "nbchan",
        0.0,
        "data",
        "eegdata",
        "setname",
        "importevent_bug",
        "srate",
        1.0,
        "pnts",
        0.0,
        "xmin",
        0.0,
    )
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    if matlab:
        engine = request.getfixturevalue("eeglab_matlab_engine")
        engine.eval("global eegdata; eegdata = rand(2,50);", nargout=0)
        workspace = None
    else:
        workspace = EEGPrepConsoleWorkspace(EEGPrepSession())
        workspace.namespace["eegdata"] = np.random.default_rng().random((2, 50))
    try:
        eeg = (
            eeglab_backend("pop_importdata", *options)
            if workspace is None
            else workspace.namespace["pop_importdata"](*options)
        )
        events = []
        for values, timeunit in (
            ([[0.0, "Experiment begins"], [49.0, "Experiment ends"]], 1.0),
            ([[1.0, "Experiment begins"], [float(np.asarray(eeg["pnts"]).item()), "Experiment ends"]], np.nan),
        ):
            values = np.array(values, dtype=object)
            arguments = (
                np.empty((0, 0)),
                1.0,
                "fields",
                np.array([["latency", "type"]], dtype=object),
                "timeunit",
                timeunit,
            )
            if workspace is None:
                # A tiny transport helper binds the literal caller variable;
                # passing the cell contents directly would drop that workflow.
                result = eeglab_backend("eegprep_test_importevent_caller", values, *arguments)
            else:
                workspace.namespace["myEventValues"] = values
                result = workspace.namespace["importevent"]("myEventValues", *arguments)
            events.append(result)
        first, second = (matlab_field_concat(result, "latency").T for result in events)
        assert_matlab_equal(first, second)
        assert first[0, 0] == 1.0
        assert first[1, 0] == 50.0
    finally:
        if workspace is None:
            engine.eval("clear global eegdata; clear eegdata;", nargout=0)
        else:
            workspace.close()


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


def test_current_pop_selectevent_selects_epochs_using_a_custom_event_field():
    selected, event_indices = pop_selectevent(
        _epoched_eeg(), "position", 1, "deleteevents", "off", "deleteepochs", "on"
    )

    assert selected["trials"] == 2
    assert event_indices == [1, 3]
    assert [event["position"] for event in selected["event"]] == [1, 1]
    np.testing.assert_array_equal(selected["data"], _epoched_eeg()["data"][:, :, [0, 2]])
