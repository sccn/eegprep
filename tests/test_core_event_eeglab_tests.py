"""Ports of current EEGLAB tests for core event and channel helpers."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from eegprep import (
    eeg_addnewevents,
    eeg_amplitudearea,
    eeg_chaninds,
    eeg_context,
    eeg_eegrej,
    eeg_eventhist,
    eeg_eventtypes,
    eeg_getepochevent,
    eeg_insertbound,
    eeg_matchchans,
    eeg_mergechan,
    eeg_mergelocs,
    eeg_timeinterp,
    eeg_urlatency,
)
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import assert_matlab_near, eeglab_test


@eeglab_test(
    "unittesting_popfunc/eeg_amplitudearea/popfunc_eeg_amplitudearea_wrapperTest.m",
    "test_test_eeg_amplitudearea",
)
def test_reference_amplitudearea_preserves_both_epoched_workflows(eeglab_backend):
    datasets = (
        [[[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], [[2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]]],
        [[[1, 4, 13, 16], [2, 5, 14, 17], [3, 6, 15, 18]], [[7, 10, 19, 22], [8, 11, 20, 23], [9, 12, 21, 24]]],
    )
    for data in datasets:
        eeg = eeglab_backend("eeg_emptyset")
        eeg.update(nbchan=2.0, pnts=3.0, trials=4.0, srate=1.0, xmin=0.0, xmax=2.0, data=np.array(data, dtype=float))
        eeg = eeglab_backend("eeg_checkset", eeg)
        # The original source checks successful execution, not an invented numeric oracle.
        eeglab_backend("eeg_amplitudearea", eeg, np.array([[1.0, 2.0]]), 1.0, 0.0, 3.0, nargout=2)


@eeglab_test("unittesting_popfunc/eeg_chaninds/popfunc_eeg_chaninds_wrapperTest.m", "test_test_eeg_chaninds")
def test_reference_chaninds_uses_original_epoched_recording(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    labels = "FPz EOG1 F3 Fz F4 EOG2 FC5 FC1 FC2 FC6 T7 C3 C4 Cz T8 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
    mixed = "T8 CP5 CP1 P3 Pz P4 P8 O1 Oz O2 FPz EOG1 F3 Fz F4 EOG2 FC5 FC1 FC6 Cz".split()
    for selection in (["P7"], labels, mixed):
        eeglab_backend("eeg_chaninds", eeg, np.array([selection], dtype=object))


@eeglab_test("unittesting_popfunc/eeg_eegrej/popfunc_eeg_eegrej_wrapperTest.m", "test_test_eeg_eegrej")
def test_reference_eegrej_preserves_metadata_and_recorded_data_cases(eeglab_backend, eeglab_suite_root):
    data = np.arange(1, 16, dtype=float).reshape(3, 5)
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(nbchan=3.0, pnts=4.0, trials=1.0, srate=1.0, xmin=0.0, xmax=3.0, data=data)
    result = eeglab_backend("eeg_eegrej", eeg, np.empty((0, 0)))
    assert_matlab_near(result["data"], data)
    assert_matlab_near(result["event"], np.empty((0, 0)))
    # The source deliberately includes both consistent and inconsistent pnts metadata.
    for pnts in (5.0, 4.0):
        eeg["pnts"] = pnts
        result = eeglab_backend("eeg_eegrej", eeg, np.array([[2.0, 3.0]]))
        assert_matlab_near(result["data"], data[:, [0, 3, 4]])
        assert set(result["event"]) == {"type", "duration", "latency"}
        assert result["event"]["type"] == "boundary"
        assert_matlab_near(result["event"]["duration"], [[2.0]])
        assert_matlab_near(result["event"]["latency"], [[1.5]])
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("eeg_eegrej", eeg, np.array([[223.0, 572.0], [1.0, 140.0]]))


@eeglab_test("unittesting_popfunc/eeg_eegrej/popfunc_eeg_eegrej_wrapperTest.m", "test_testcase_eegrej")
def test_reference_eegrej_boundary_events_follow_original_regression(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(pnts=10000.0, data=np.zeros((1, 10000)), srate=500.0)
    eeg = eeglab_backend("eeg_checkset", eeg)
    cases = (
        ([999.0, 1000.0, 2000.0, 2001.0], [1000.0, 2000.0], ["mrk1", "boundary", "mrk4"], [999.0, 999.5, 1000.0]),
        ([1.0, 1000.0, 1001.0], [1.0, 1000.0], ["boundary", "mrk3"], [0.5, 1.0]),
        ([8999.0, 9000.0, 10000.0], [9000.0, 10000.0], ["mrk1", "boundary"], None),
    )
    for latencies, interval, expected_types, expected_latencies in cases:
        eeg["event"] = np.array(
            [[(f"mrk{index}", latency) for index, latency in enumerate(latencies, start=1)]],
            dtype=[("type", object), ("latency", object)],
        )
        result = eeglab_backend("eeg_eegrej", eeg, np.array([interval]))
        events = result["event"]
        assert events.shape == (1, len(expected_types))
        assert events["type"][0].tolist() == expected_types
        if expected_latencies is not None:
            np.testing.assert_array_equal([latency.item() for latency in events["latency"][0]], expected_latencies)


@eeglab_test("unittesting_popfunc/eeg_timeinterp/popfunc_eeg_timeinterp_wrapperTest.m", "test_test_eeg_timeinterp")
def test_reference_timeinterp_preserves_original_recording_and_interval(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeg["data"][0, 99:1000] = 0
    eeglab_backend("eeg_timeinterp", eeg, np.arange(100.0, 1001.0)[None, :])


def _reference_merge_locations(eeglab_backend, eeglab_suite_root, function):
    root = eeglab_suite_root / "eeglab"
    locations = eeglab_backend("pop_readlocs", str(root / "sample_data/eeglab_chan32.locs"))
    eeglab_backend(function, locations[:, :1], locations[:, 1:2])
    locations = eeglab_backend("pop_readlocs", str(root / "sample_locs/GSN64v2_0.sfp"))
    eeglab_backend(function, locations[:, :10], locations[:, 4:15])
    first = eeglab_backend("pop_readlocs", str(root / "sample_locs/Standard-10-10-Cap33.locs"))
    second = eeglab_backend("pop_readlocs", str(root / "sample_locs/Standard-10-20-Cap25.locs"), "filetype", "loc")
    eeglab_backend(function, first, second)


@eeglab_test("unittesting_popfunc/eeg_mergechan/popfunc_eeg_mergechan_wrapperTest.m", "test_test_eeg_mergechan")
def test_reference_mergechan_uses_original_montages(eeglab_backend, eeglab_suite_root):
    _reference_merge_locations(eeglab_backend, eeglab_suite_root, "eeg_mergechan")


@eeglab_test("unittesting_popfunc/eeg_mergelocs/popfunc_eeg_mergelocs_wrapperTest.m", "test_test_eeg_mergelocs")
def test_reference_mergelocs_uses_original_montages(eeglab_backend, eeglab_suite_root):
    _reference_merge_locations(eeglab_backend, eeglab_suite_root, "eeg_mergelocs")


def _reference_typed_locations():
    return np.array(
        [[(kind, float(index), float(index + 1), float(index + 2)) for index, kind in enumerate("hello", start=1)]],
        dtype=[("type", object), ("X", object), ("Y", object), ("Z", object)],
    )


def _reference_epoched_events(eeglab_backend, *, duration=False, include_epoch=True):
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(
        nbchan=2.0,
        pnts=3.0,
        trials=3.0,
        srate=1.0,
        xmin=0.0,
        xmax=2.0,
        data=np.array([[[1, 1, 2]] * 3, [[2, 2, 2], [1, 1, 1], [1, 1, 1]]], dtype=float),
    )
    fields = ["type", "position", "latency", "urevent"]
    if duration:
        fields.append("duration")
    if include_epoch:
        fields.append("epoch")
    events = []
    for index, (kind, latency, epoch) in enumerate(
        zip(["square", "square", "rt", "square", "rt"], [2.0, 5.0, 5.3, 8.0, 8.4], [1.0, 2.0, 2.0, 3.0, 3.0]), start=1
    ):
        row = [kind, 2.0 if kind == "square" else np.empty((0, 0)), latency, float(index)]
        if duration:
            row.append(index / 10)
        if include_epoch:
            row.append(epoch)
        events.append(tuple(row))
    eeg["event"] = np.array([events], dtype=[(field, object) for field in fields])
    if include_epoch:
        epochs = []
        for indices, latencies in (([1.0], [1000.0]), ([2.0, 3.0], [1000.0, 1300.0]), ([4.0, 5.0], [1000.0, 1400.0])):
            positions = np.empty((1, len(indices)), dtype=object)
            positions[0, 0] = 2.0
            if len(indices) == 2:
                positions[0, 1] = np.empty((0, 0))
            epochs.append(
                (
                    np.array([indices]),
                    np.array([latencies], dtype=object),
                    positions,
                    np.array([["square"] if len(indices) == 1 else ["square", "rt"]], dtype=object),
                    np.array([indices], dtype=object),
                )
            )
        eeg["epoch"] = np.array(
            [epochs],
            dtype=[
                (field, object) for field in ("event", "eventlatency", "eventposition", "eventtype", "eventurevent")
            ],
        )
    return eeg


def _assert_reference_epoch_values(eeglab_backend, eeg, arguments, expected, all_expected):
    values, all_values = eeglab_backend("eeg_getepochevent", eeg, *arguments, nargout=2)
    assert_matlab_near(values, [expected])
    # The source compares [rallepochval{:}], not the individual cell shapes.
    concatenated = np.concatenate([value for value in all_values.flat if value.size], axis=1)
    assert_matlab_near(concatenated, [all_expected])


@eeglab_test("unittesting_popfunc/eeg_getepochevent/popfunc_eeg_getepochevent_wrapperTest.m", "test_pass_duration")
def test_reference_getepochevent_duration_old_and_new_calls(eeglab_backend):
    eeg = _reference_epoched_events(eeglab_backend, duration=True)
    for arguments in (("type", "rt", "fieldname", "duration"), ("rt", np.empty((0, 0)), "duration")):
        _assert_reference_epoch_values(eeglab_backend, eeg, arguments, [np.nan, 300.0, 500.0], [300.0, 500.0])


@eeglab_test("unittesting_popfunc/eeg_getepochevent/popfunc_eeg_getepochevent_wrapperTest.m", "test_pass_empty_timewin")
def test_reference_getepochevent_default_time_window(eeglab_backend):
    _assert_reference_epoch_values(
        eeglab_backend,
        _reference_epoched_events(eeglab_backend),
        ("type", "rt", "fieldname", "urevent"),
        [np.nan, 3.0, 5.0],
        [3.0, 5.0],
    )


@eeglab_test("unittesting_popfunc/eeg_getepochevent/popfunc_eeg_getepochevent_wrapperTest.m", "test_pass_four_args")
def test_reference_getepochevent_four_argument_window(eeglab_backend):
    _assert_reference_epoch_values(
        eeglab_backend,
        _reference_epoched_events(eeglab_backend),
        ("rt", np.array([[500.0, 1300.0]]), "urevent"),
        [np.nan, 3.0, np.nan],
        [3.0],
    )


@eeglab_test("unittesting_popfunc/eeg_getepochevent/popfunc_eeg_getepochevent_wrapperTest.m", "test_pass_general")
def test_reference_getepochevent_default_latency(eeglab_backend):
    _assert_reference_epoch_values(
        eeglab_backend, _reference_epoched_events(eeglab_backend), ("rt",), [np.nan, 1300.0, 1400.0], [1300.0, 1400.0]
    )


@eeglab_test("unittesting_popfunc/eeg_getepochevent/popfunc_eeg_getepochevent_wrapperTest.m", "test_pass_no_epoch")
def test_reference_getepochevent_without_epoch_field(eeglab_backend):
    _assert_reference_epoch_values(
        eeglab_backend,
        _reference_epoched_events(eeglab_backend, include_epoch=False),
        ("rt",),
        [4300.0, np.nan, np.nan],
        [4300.0, 7400.0],
    )


@eeglab_test("unittesting_popfunc/eeg_insertbound/popfunc_eeg_insertbound_wrapperTest.m", "test_pass_general")
def test_reference_insertbound_original_epoched_events(eeglab_backend):
    eeg = _reference_epoched_events(eeglab_backend)
    eeg["event"]["latency"][0] = [1.0, 3.0, 3.3, 5.0, 5.4]
    eeg["event"]["epoch"][0] = [1.0, 1.0, 2.0, 3.0, 3.0]
    # The source requests both outputs but does not assert their values.
    eeglab_backend("eeg_insertbound", eeg["event"], 12.0, np.array([[2.0, 3.0]]), nargout=2)


@eeglab_test("unittesting_popfunc/eeg_eventtypes/popfunc_eeg_eventtypes_wrapperTest.m", "test_test_eeg_eventtypes")
def test_reference_eventtypes_preserves_inverted_source_condition(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("eeg_emptyset")
    kinds = ["square", "triangle", "circle", "square", "not really a triangle", "triangle", "square", "point"]
    eeg["event"] = np.array([[(kind,) for kind in kinds]], dtype=[("type", object)])
    types, counts = eeglab_backend("eeg_eventtypes", eeg, nargout=2)
    expected = np.array(
        [[kind] for kind in ["circle", "not really a triangle", "point", "square", "triangle"]], dtype=object
    )
    # Upstream rejects equality here (missing ~). Preserve, do not silently repair it.
    if np.array_equal(types, expected):
        with pytest.raises(AssertionError):
            assert_matlab_near(counts, [[1.0], [1.0], [1.0], [3.0], [2.0]])
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("eeg_eventtypes", eeg, nargout=2)


@eeglab_test("unittesting_popfunc/eeg_eventhist/popfunc_eeg_eventhist_wrapperTest.m", "test_test_eeg_eventhist")
def test_reference_eventhist_original_values_and_no_output_plots(request, eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    kinds = ["square", "square", "rt", "square", "rt"]
    eeg["event"] = np.array([[(kind,) for kind in kinds]], dtype=[("type", object)])
    values, counts, bins = eeglab_backend("eeg_eventhist", eeg["event"], "type", nargout=3)
    np.testing.assert_array_equal(values, np.array([[kind] for kind in kinds], dtype=object))
    assert_matlab_near(counts, [[2.0, 3.0]])
    np.testing.assert_array_equal(bins, np.array([["rt"], ["square"]], dtype=object))
    if request.config.getoption("--eeglab-backend") == "python":
        pytest.fail("The source's no-output eeg_eventhist plotting workflow has no Python entry point")
    eeglab_backend("figure", nargout=0)
    eeglab_backend("eeg_eventhist", eeg["event"], "type", 2.0, nargout=0)
    eeglab_backend("close", nargout=0)
    eeg = eeglab_backend("eeg_emptyset")
    eeg["event"] = np.array([[(latency,) for latency in (0.0, 0.0, 0.3, 0.0, 0.4)]], dtype=[("latency", object)])
    # Numeric-output and nested-struct assertions are commented out upstream.
    eeglab_backend("figure", nargout=0)
    eeglab_backend("eeg_eventhist", eeg["event"], "latency", 3.0, nargout=0)
    eeglab_backend("close", nargout=0)


def _reference_context_events(eeglab_backend, epoched):
    source = _reference_epoched_events(eeglab_backend, include_epoch=epoched)
    eeg = eeglab_backend("eeg_emptyset")
    eeg["event"] = source["event"]
    eeg["event"]["latency"][0] = [0.0, 0.0, 0.3, 0.0, 0.4] if epoched else [5.0, 8.0, 8.5, 12.0, 12.4]
    fields = [field for field in eeg["event"].dtype.names if field != "urevent"]
    eeg["urevent"] = np.empty((1, 5), dtype=[(field, object) for field in fields])
    for field in fields:
        eeg["urevent"][field] = eeg["event"][field]
    if epoched:
        eeg["epoch"] = source["epoch"]
        for index, latencies in enumerate(([0.0], [0.0, 0.3], [0.0, 0.4])):
            eeg["epoch"]["eventlatency"][0, index] = np.array([latencies], dtype=object)
    return eeg


@eeglab_test("unittesting_popfunc/eeg_context/popfunc_eeg_context_wrapperTest.m", "test_test_eeg_context")
def test_reference_context_all_six_original_cases(eeglab_backend):
    neighbors = np.array([["square", "rt"]], dtype=object)
    targets = [[1.0, 1.0, np.nan, 1.0], [2.0, 2.0, np.nan, 1.0], [4.0, 4.0, np.nan, 1.0]]
    single = ([[2.0], [3.0], [5.0]], [[1.0], [2.0], [2.0]], [[3000.0], [500.0], [400.0]])
    cases = (
        (
            False,
            ("square", neighbors, np.array([[1.0]]), "type", "all"),
            targets,
            *single,
            [["square"], ["rt"], ["rt"]],
        ),
        (False, ("square", neighbors, np.array([[1.0]]), "type"), targets, *single, [["square"], ["rt"], ["rt"]]),
        (
            True,
            ("square", neighbors, np.array([[1.0]]), "type"),
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 1.0], [4.0, 4.0, 3.0, 1.0]],
            single[0],
            single[1],
            [[0.0], [300.0], [400.0]],
            [["square"], ["rt"], ["rt"]],
        ),
        (
            False,
            ("square", neighbors, np.array([[1.0, 2.0]]), "type"),
            targets,
            [[2.0, 3.0], [3.0, 4.0], [5.0, np.nan]],
            [[1.0, 2.0], [2.0, 1.0], [2.0, np.nan]],
            [[3000.0, 3500.0], [500.0, 4000.0], [400.0, np.nan]],
            [["square", "rt"], ["rt", "square"], ["rt", None]],
        ),
        (
            False,
            ("square", neighbors, np.array([[-2.0, 1.0]]), "type"),
            targets,
            [[np.nan, 2.0], [np.nan, 3.0], [2.0, 5.0]],
            [[np.nan, 1.0], [np.nan, 2.0], [1.0, 2.0]],
            [[np.nan, 3000.0], [np.nan, 500.0], [-4000.0, 400.0]],
            [[None, "square"], [None, "rt"], ["square", "rt"]],
        ),
        (
            False,
            (),
            [[float(index), float(index), np.nan, 1.0] for index in range(1, 6)],
            [[2.0], [3.0], [4.0], [5.0], [np.nan]],
            [[1.0], [1.0], [1.0], [1.0], [np.nan]],
            [[3000.0], [500.0], [3500.0], [400.0], [np.nan]],
            None,
        ),
    )
    for epoched, arguments, expected_targets, indices, kinds, delays, fields in cases:
        actual = eeglab_backend(
            "eeg_context", _reference_context_events(eeglab_backend, epoched), *arguments, nargout=6
        )
        for result, expected in zip(actual[:4], (expected_targets, indices, kinds, delays), strict=True):
            assert_matlab_near(result, expected)
        if fields is None:
            assert_matlab_near(actual[4], np.empty((0, 0)))
            assert_matlab_near(actual[5], np.empty((0, 0)))
            continue
        np.testing.assert_array_equal(actual[4], np.array([["square"]] * 3, dtype=object))
        assert actual[5].shape == (3, len(fields[0]))
        for row, expected_row in zip(actual[5], fields, strict=True):
            for value, expected in zip(row, expected_row, strict=True):
                if expected is None:
                    assert_matlab_near(value, np.empty((0, 0)))
                else:
                    assert value == expected


@eeglab_test("unittesting_popfunc/eeg_matchchans/popfunc_eeg_matchchans_wrapperTest.m", "test_pass_noplot")
def test_reference_matchchans_original_complete_locations(eeglab_backend):
    fields = ["labels", "theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius", "type"]
    dtype = [(field, object) for field in fields]
    small = np.array(
        [
            [
                ("1_1", -90.0, 0.5, 0.0, 1.0, 0.0, 90.0, 0.0, 1.0, "EEG"),
                ("1_2", 90.0, 0.5, 0.0, -1.0, 0.0, -90.0, 0.0, 1.0, "EEG"),
            ]
        ],
        dtype=dtype,
    )
    large = np.array(
        [
            [
                ("2_1", -135.0, 0.5, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 135.0, 0.0, 1.0, "EEG"),
                ("2_2", 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, "EEG"),
                ("2_3", 90.0, 0.5, 0.0, -1.0, 0.0, -90.0, 0.0, 1.0, "EEG"),
            ]
        ],
        dtype=dtype,
    )
    selected, distances, locations = eeglab_backend("eeg_matchchans", large, small, "noplot", nargout=3)
    expected_distance = np.sqrt(2 - np.sqrt(2))
    assert_matlab_near(selected, [[1.0], [3.0]])
    assert_matlab_near(distances, [[expected_distance], [0.0]])
    assert set(locations.dtype.names) == {*fields, "bigchan", "bigdist"}
    # near.m compares concatenated fields of structs, not their individual shapes.
    for field in fields:
        if field in ("labels", "type"):
            assert "".join(locations[field].flat) == "".join(large[field][0, [0, 2]])
        else:
            assert_matlab_near(
                np.concatenate(list(locations[field].flat), axis=1), large[field][:, [0, 2]].astype(float)
            )
    assert_matlab_near(np.concatenate([value for value in locations["bigchan"].flat if value.size], axis=1), [[3.0]])
    assert_matlab_near(np.concatenate(list(locations["bigdist"].flat), axis=1), [[expected_distance, 0.0]])


def _epoched_event_eeg(*, durations: bool = False, include_event_epochs: bool = True) -> dict:
    event_types = ["square", "square", "rt", "square", "rt"]
    latencies = [2, 5, 5.3, 8, 8.4]
    epochs = [1, 2, 2, 3, 3]
    events = []
    for index, (event_type, latency, epoch) in enumerate(zip(event_types, latencies, epochs), start=1):
        event = {
            "type": event_type,
            "position": 2 if event_type == "square" else [],
            "latency": latency,
            "urevent": index - 1,
        }
        if include_event_epochs:
            event["epoch"] = epoch
        if durations:
            event["duration"] = index / 10
        events.append(event)
    return {
        "data": np.zeros((2, 3, 3)),
        "nbchan": 2,
        "pnts": 3,
        "trials": 3,
        "srate": 1,
        "xmin": 0,
        "xmax": 2,
        "event": events,
        "epoch": [
            {"event": [0], "eventlatency": [[1000]], "eventtype": ["square"], "eventurevent": [[0]]},
            {
                "event": [1, 2],
                "eventlatency": [[1000], [1300]],
                "eventtype": ["square", "rt"],
                "eventurevent": [[1], [2]],
            },
            {
                "event": [3, 4],
                "eventlatency": [[1000], [1400]],
                "eventtype": ["square", "rt"],
                "eventurevent": [[3], [4]],
            },
        ],
    }


def _context_eeg(*, epoched: bool = False) -> dict:
    event_types = ["square", "square", "rt", "square", "rt"]
    latencies = [0, 0, 0.3, 0, 0.4] if epoched else [5, 8, 8.5, 12, 12.4]
    epochs = [1, 2, 2, 3, 3]
    events = []
    urevents = []
    for index, (event_type, latency) in enumerate(zip(event_types, latencies), start=1):
        record = {"type": event_type, "position": 2 if event_type == "square" else [], "latency": latency}
        urevent = dict(record)
        event = {**record, "urevent": index - 1}
        if epoched:
            event["epoch"] = epochs[index - 1]
            urevent["epoch"] = epochs[index - 1]
        events.append(event)
        urevents.append(urevent)
    eeg = {"event": events, "urevent": urevents, "srate": 1, "trials": 3 if epoched else 1}
    if epoched:
        eeg["epoch"] = [
            {"event": [0], "eventlatency": [[0]]},
            {"event": [1, 2], "eventlatency": [[0], [0.3]]},
            {"event": [3, 4], "eventlatency": [[0], [0.4]]},
        ]
    return eeg


def _matching_locations() -> tuple[list[dict], list[dict]]:
    small = [
        {"labels": "1_1", "X": 0, "Y": 1, "Z": 0, "sph_radius": 1},
        {"labels": "1_2", "X": 0, "Y": -1, "Z": 0, "sph_radius": 1},
    ]
    big = [
        {"labels": "2_1", "X": -np.sqrt(2) / 2, "Y": np.sqrt(2) / 2, "Z": 0, "sph_radius": 1},
        {"labels": "2_2", "X": 1, "Y": 0, "Z": 0, "sph_radius": 1},
        {"labels": "2_3", "X": 0, "Y": -1, "Z": 0, "sph_radius": 1},
    ]
    return big, small


# The source eeg_addnewevents body is entirely commented out; this is extra coverage.
def test_eeg_addnewevents_current_suite_documented_calls_are_functional():
    eeg = {"event": [], "urevent": []}
    output = eeg_addnewevents(
        eeg,
        [[100, 200], [300, 400, 500]],
        ["type1", "type2"],
        ["field1", "field2"],
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
    )
    assert [event["latency"] for event in output["event"]] == [100, 200, 300, 400, 500]
    assert [event["type"] for event in output["event"]] == ["type1", "type1", "type2", "type2", "type2"]
    assert [event["field1"] for event in output["event"]] == [1, 2, 3, 4, 5]
    assert [event["urevent"] for event in output["event"]] == [0, 1, 2, 3, 4]
    assert all("urevent" not in event for event in output["urevent"])

    existing = {
        "event": [
            {"type": "late-original", "latency": 4, "urevent": 1},
            {"type": "early-original", "latency": 6, "urevent": 0},
        ],
        "urevent": [
            {"type": "early-original", "latency": 2},
            {"type": "late-original", "latency": 10},
        ],
    }
    merged = eeg_addnewevents(existing, [[5]], ["new"])
    assert [event["type"] for event in merged["event"]] == ["late-original", "new", "early-original"]
    assert [event["urevent"] for event in merged["event"]] == [2, 1, 0]
    assert [event["latency"] for event in merged["urevent"]] == [2, 5, 10]
    for event in merged["event"]:
        assert merged["urevent"][event["urevent"]]["type"] == event["type"]


def test_eeg_amplitudearea_current_suite_epoched_cases():
    data = np.zeros((2, 3, 4))
    data[0] = [[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]]
    data[1] = [[2, 2, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]]
    eeg = {"data": data, "times": np.array([0, 1000, 2000]), "xmin": 0, "xmax": 2}
    channels, amplitude = eeg_amplitudearea(eeg, [0, 1], 1, 0, 3)
    np.testing.assert_array_equal(channels, [0, 1])
    np.testing.assert_allclose(amplitude, [1.5, 1.9985], atol=1e-12)

    data[0] = [[1, 4, 13, 16], [2, 5, 14, 17], [3, 6, 15, 18]]
    data[1] = [[7, 10, 19, 22], [8, 11, 20, 23], [9, 12, 21, 24]]
    _, second_amplitude = eeg_amplitudearea(eeg, [0, 1], 1, 0, 3)
    assert np.all(np.isfinite(second_amplitude))
    assert second_amplitude[1] > second_amplitude[0]

    linear = {
        "data": np.array([[[0.0], [1.0], [2.0]]]),
        "times": np.array([0.0, 1.0, 2.0]),
        "xmin": 0,
        "xmax": 0.002,
    }
    _, truncated_interval = eeg_amplitudearea(linear, [0], 1, 0, 1.5)
    np.testing.assert_allclose(truncated_interval, [0.75], atol=1e-12)


def test_eeg_chaninds_current_suite_label_forms():
    labels = "FPz EOG1 F3 Fz F4 EOG2 FC5 FC1 FC2 FC6 T7 C3 C4 Cz T8 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2".split()
    eeg = {"chanlocs": [{"labels": label} for label in labels]}
    assert eeg_chaninds(eeg, ["P7"]) == [19]
    assert eeg_chaninds(eeg, labels) == list(range(32))
    mixed = "T8 CP5 CP1 P3 Pz P4 P8 O1 Oz O2 FPz EOG1 F3 Fz F4 EOG2 FC5 FC1 FC6 Cz".split()
    indices = eeg_chaninds(eeg, mixed)
    assert len(indices) == len(mixed)
    assert {labels[index] for index in indices} == set(mixed)


def test_eeg_context_current_suite_six_context_cases():
    eeg = _context_eeg()
    expected_targets = np.array([[1, 1, np.nan, 1], [2, 2, np.nan, 1], [4, 4, np.nan, 1]])
    result = eeg_context(eeg, "square", ["square", "rt"], [1], "type", "all")
    np.testing.assert_allclose(result[0], expected_targets, equal_nan=True)
    np.testing.assert_allclose(result[1], [[2], [3], [5]], equal_nan=True)
    np.testing.assert_allclose(result[2], [[1], [2], [2]], equal_nan=True)
    np.testing.assert_allclose(result[3], [[3000], [500], [400]], equal_nan=True)
    assert result[4].tolist() == ["square", "square", "square"]
    assert result[5].tolist() == [["square"], ["rt"], ["rt"]]
    repeated = eeg_context(eeg, "square", ["square", "rt"], [1], "type")
    np.testing.assert_allclose(repeated[0], expected_targets, equal_nan=True)
    uppercase_all = eeg_context(eeg, "square", ["square", "rt"], [1], "type", "ALL")
    np.testing.assert_allclose(uppercase_all[0], expected_targets, equal_nan=True)

    epoched = eeg_context(_context_eeg(epoched=True), "square", ["square", "rt"], [1], "type")
    np.testing.assert_allclose(epoched[0][:, 2], [1, 2, 3])
    np.testing.assert_allclose(epoched[3], [[0], [300], [400]])
    two_after = eeg_context(eeg, "square", ["square", "rt"], [1, 2], "type")
    np.testing.assert_allclose(two_after[1], [[2, 3], [3, 4], [5, np.nan]], equal_nan=True)
    around = eeg_context(eeg, "square", ["square", "rt"], [-2, 1], "type")
    np.testing.assert_allclose(around[1], [[np.nan, 2], [np.nan, 3], [2, 5]], equal_nan=True)
    assert around[5].tolist() == [[[], "square"], [[], "rt"], ["square", "rt"]]
    defaults = eeg_context(eeg)
    np.testing.assert_allclose(defaults[1], [[2], [3], [4], [5], [np.nan]], equal_nan=True)
    np.testing.assert_allclose(defaults[3], [[3000], [500], [3500], [400], [np.nan]], equal_nan=True)
    assert defaults[4] == [] and defaults[5] == []

    numeric = eeg_context(eeg, "square", ["square", "rt"], [2], "latency")
    np.testing.assert_allclose(numeric[4], [5, 8, 12])
    np.testing.assert_allclose(numeric[5], [[8.5], [12], [np.nan]], equal_nan=True)
    assert numeric[4].dtype == float and numeric[5].dtype == float

    no_targets = eeg_context(eeg, "missing", ["square"], [1], "latency")
    assert no_targets[4].shape == (0,) and no_targets[5].shape == (0, 1)

    with_boundary = _context_eeg()
    with_boundary["event"].insert(2, {"type": "boundary", "latency": 8.25, "duration": 1})
    with_boundary["urevent"].insert(2, {"type": "boundary", "latency": 8.25, "duration": 1})
    for event in with_boundary["event"]:
        if "urevent" in event and event["urevent"] >= 2:
            event["urevent"] += 1
    boundary_result = eeg_context(with_boundary, "square", ["square", "rt"], [1], "type")
    np.testing.assert_allclose(boundary_result[1], [[2], [np.nan], [6]], equal_nan=True)


def test_eeg_eegrej_current_suite_empty_and_middle_regions():
    data = np.arange(1, 16, dtype=float).reshape(3, 5)
    eeg = {"data": data, "nbchan": 3, "pnts": 5, "trials": 1, "srate": 1, "xmin": 0, "xmax": 4, "event": []}
    unchanged = eeg_eegrej(eeg, [])
    np.testing.assert_array_equal(unchanged["data"], data)
    output = eeg_eegrej(eeg, [[2, 3]])
    np.testing.assert_array_equal(output["data"], data[:, [0, 3, 4]])
    assert output["event"] == [{"type": "boundary", "latency": 1.5, "duration": 2.0}]


def test_eeg_eegrej_current_suite_endpoint_event_regression():
    eeg = {"data": np.zeros((1, 10000)), "pnts": 10000, "srate": 500, "trials": 1, "xmin": 0, "xmax": 19.998}
    cases = [
        (
            [
                {"type": "mrk1", "latency": 999},
                {"type": "mrk2", "latency": 1000},
                {"type": "mrk3", "latency": 2000},
                {"type": "mrk4", "latency": 2001},
            ],
            [1000, 2000],
            ["mrk1", "boundary", "mrk4"],
            [999, 999.5, 1000],
        ),
        (
            [{"type": "mrk1", "latency": 1}, {"type": "mrk2", "latency": 1000}, {"type": "mrk3", "latency": 1001}],
            [1, 1000],
            ["boundary", "mrk3"],
            [0.5, 1],
        ),
        (
            [{"type": "mrk1", "latency": 8999}, {"type": "mrk2", "latency": 9000}, {"type": "mrk3", "latency": 10000}],
            [9000, 10000],
            ["mrk1", "boundary"],
            [8999, 8999.5],
        ),
    ]
    for events, region, expected_types, expected_latencies in cases:
        case = {**eeg, "event": events}
        output = eeg_eegrej(case, [region])
        assert [event["type"] for event in output["event"]] == expected_types
        np.testing.assert_allclose([event["latency"] for event in output["event"]], expected_latencies)


def test_eeg_eventhist_current_suite_string_and_numeric_fields():
    string_events = [{"type": value} for value in ["square", "square", "rt", "square", "rt"]]
    values, counts, labels = eeg_eventhist(string_events, "type")
    assert values == ["square", "square", "rt", "square", "rt"]
    np.testing.assert_array_equal(counts, [2, 3])
    assert labels == ["rt", "square"]
    _, plot_counts, plot_labels = eeg_eventhist(string_events, "type", 2)
    np.testing.assert_array_equal(plot_counts, counts)
    assert plot_labels == labels

    numeric_events = [{"latency": value} for value in [0, 0, 0.3, 0, 0.4]]
    numeric_values, numeric_counts, edges = eeg_eventhist(numeric_events, "latency", 3)
    np.testing.assert_allclose(numeric_values, [0, 0, 0.3, 0, 0.4])
    np.testing.assert_array_equal(numeric_counts, [3, 1, 1])
    np.testing.assert_allclose(edges, [-np.inf, 0.14, 0.3349358869, np.inf])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, singleton_counts, singleton_edges = eeg_eventhist([{"latency": 0.0}], "latency", 3)
    assert singleton_counts.sum() == 1
    assert not np.isnan(singleton_edges).any()

    _, explicit_counts, _ = eeg_eventhist([{"latency": value} for value in [0, 1, 2]], "latency", [0, 1, 2])
    np.testing.assert_array_equal(explicit_counts, [1, 1])


def test_eeg_eventtypes_current_suite_counts_and_order():
    values = ["square", "triangle", "circle", "square", "not really a triangle", "triangle", "square", "point"]
    types, counts = eeg_eventtypes({"event": [{"type": value} for value in values]})
    assert types == ["square", "triangle", "point", "not really a triangle", "circle"]
    assert counts == [3, 2, 1, 1, 1]
    assert eeg_eventtypes({"event": [{"type": 1.0}, {"type": 1}, {"type": 2.5}]}) == (["1", "2.5"], [2, 1])


def test_eeg_getepochevent_current_suite_duration_new_and_old_forms():
    eeg = _epoched_event_eeg(durations=True)
    new_values, new_all = eeg_getepochevent(eeg, "type", "rt", "fieldname", "duration")
    old_values, old_all = eeg_getepochevent(eeg, "rt", [], "duration")
    np.testing.assert_allclose(new_values, [np.nan, 300, 500], equal_nan=True)
    np.testing.assert_allclose(old_values, new_values, equal_nan=True)
    assert new_all == [[], [300], [500]] and old_all == new_all


def test_eeg_getepochevent_current_suite_empty_time_window():
    values, all_values = eeg_getepochevent(_epoched_event_eeg(), "type", "rt", "fieldname", "urevent")
    np.testing.assert_allclose(values, [np.nan, 2, 4], equal_nan=True)
    assert all_values == [[], [2], [4]]

    selected, selected_all = eeg_getepochevent(_epoched_event_eeg(), "type", "rt", "fieldname", "urevent", "trials", 2)
    np.testing.assert_array_equal(selected, [2])
    assert selected_all == [[2]]

    combined, combined_all = eeg_getepochevent(
        [_epoched_event_eeg(), _epoched_event_eeg()],
        "type",
        "rt",
        "trials",
        [[2], [3]],
    )
    np.testing.assert_allclose(combined, [1300, 1400])
    assert combined_all == [[1300], [1400]]


def test_eeg_getepochevent_current_suite_old_four_argument_form():
    values, all_values = eeg_getepochevent(_epoched_event_eeg(), "rt", [500, 1300], "urevent")
    np.testing.assert_allclose(values, [np.nan, 2, np.nan], equal_nan=True)
    assert all_values == [[], [2], []]

    literal_type = _epoched_event_eeg()
    literal_type["event"][0]["type"] = "type"
    type_values, _ = eeg_getepochevent(literal_type, "type", [-1000, 1000])
    np.testing.assert_allclose(type_values, [1000, np.nan, np.nan], equal_nan=True)


def test_eeg_getepochevent_current_suite_default_latency():
    values, all_values = eeg_getepochevent(_epoched_event_eeg(), "rt")
    np.testing.assert_allclose(values, [np.nan, 1300, 1400], equal_nan=True)
    assert all_values == [[], [1300], [1400]]


def test_eeg_getepochevent_current_suite_continuous_fallback():
    eeg = _epoched_event_eeg(include_event_epochs=False)
    values, all_values = eeg_getepochevent(eeg, "rt")
    np.testing.assert_allclose(values, [4300, np.nan, np.nan], equal_nan=True)
    assert all_values == [[4300, 7400], [], []]


def test_eeg_insertbound_current_suite_general_case():
    eeg = _epoched_event_eeg()
    events = copy.deepcopy(eeg["event"])
    for event, latency in zip(events, [1, 3, 3.3, 5, 5.4]):
        event["latency"] = latency
    output, new_indices = eeg_insertbound(events, eeg["pnts"] * (eeg["trials"] + 1), [[2, 3]])
    assert new_indices == [3]
    assert output[3]["type"] == "boundary"
    assert output[3]["latency"] == 1.5
    assert output[3]["duration"] == 2
    np.testing.assert_allclose([event["latency"] for event in output], [1, 1, 1.3, 1.5, 3, 3.4])

    unordered, unordered_indices = eeg_insertbound([{"type": "stim", "latency": 15}], 20, [[10, 12], [2, 4]])
    assert unordered_indices == [0, 1]
    np.testing.assert_allclose([event["latency"] for event in unordered], [1.5, 6.5, 9])
    np.testing.assert_allclose([event["duration"] for event in unordered[:2]], [3, 3])

    rounded, _ = eeg_insertbound([{"type": "stim", "latency": 5}], 20, [[2.5, 3.5]])
    np.testing.assert_allclose([event["latency"] for event in rounded], [2.5, 3])


def _assert_matchchans(option: str | None) -> None:
    big, small = _matching_locations()
    selected, distances, locations = eeg_matchchans(big, small, option)
    assert selected == [0, 2]
    np.testing.assert_allclose(distances, [np.sqrt(2 - np.sqrt(2)), 0])
    assert locations[0]["bigchan"] == 0
    assert locations[0]["bigdist"] == distances[0]
    assert locations[1]["bigchan"] == 2 and locations[1]["bigdist"] == 0


# pass_general.m contains no executable code in the pinned source suite.
def test_eeg_matchchans_current_suite_general_case():
    _assert_matchchans(None)


def test_eeg_matchchans_current_suite_noplot_case():
    _assert_matchchans("noplot")

    big, small = _matching_locations()
    big[0]["X"] = np.nan
    with np.testing.assert_raises_regex(ValueError, "finite coordinates"):
        eeg_matchchans(big, small, "noplot")
    big, small = _matching_locations()
    big[0]["sph_radius"] = 0
    with np.testing.assert_raises_regex(ValueError, "positive spherical radius"):
        eeg_matchchans(big, small, "noplot")


def test_eeg_mergechan_current_suite_three_overlap_shapes():
    first = [{"labels": label} for label in "ABCDEFGHIJ"]
    second = [{"labels": label} for label in "EFGHIJKLMNO"]
    assert [loc["labels"] for loc in eeg_mergechan(first[:1], first[1:2])] == ["A", "B"]
    assert [loc["labels"] for loc in eeg_mergechan(first, second)] == list("ABCDEFGHIJKLMNO")
    subset = [{"labels": label} for label in "ACEGI"]
    assert [loc["labels"] for loc in eeg_mergechan(first, subset)] == list("ABCDEFGHIJ")


def test_eeg_mergelocs_current_suite_three_overlap_shapes():
    first = [{"labels": label} for label in "ABCDEFGHIJ"]
    second = [{"labels": label} for label in "EFGHIJKLMNO"]
    one, warning = eeg_mergelocs(first[:1], first[1:2])
    assert [loc["labels"] for loc in one] == ["A", "B"] and not warning
    overlap, warning = eeg_mergelocs(first, second)
    assert [loc["labels"] for loc in overlap] == list("ABCDEFGHIJKLMNO") and not warning
    subset, warning = eeg_mergelocs(first, [{"labels": label} for label in "ACEGI"])
    assert [loc["labels"] for loc in subset] == list("ABCDEFGHIJ") and not warning


def test_eeg_timeinterp_current_suite_continuous_sample_workflow():
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"
    eeg = pop_loadset(sample)
    original = np.asarray(eeg["data"]).copy()
    eeg["data"][0, 99:1000] = 0
    output = eeg_timeinterp(eeg, np.arange(100, 1001))
    assert output["data"].shape == original.shape
    assert np.any(output["data"][0, 99:1000] != 0)
    assert np.all(np.isfinite(output["data"][:, 99:1000]))
    np.testing.assert_array_equal(output["data"][:, :99], original[:, :99])
    np.testing.assert_array_equal(output["data"][:, 1000:], original[:, 1000:])

    sample_numbers = np.arange(1, 21, dtype=float)
    polynomial = sample_numbers**3 - 2 * sample_numbers**2 + sample_numbers
    polynomial_eeg = {"data": polynomial[np.newaxis, :].copy(), "pnts": 20}
    polynomial_eeg["data"][0, 7:12] = 0
    reconstructed = eeg_timeinterp(polynomial_eeg, [8, 9, 10, 11, 12], interpwin=1)
    np.testing.assert_allclose(reconstructed["data"][0, 7:12], polynomial[7:12], rtol=1e-12)

    selected_electrode = {"data": np.vstack([polynomial, polynomial]), "pnts": 20}
    selected_electrode["data"][:, 7:12] = 0
    selected_electrode = eeg_timeinterp(selected_electrode, [8, 12], elecinds=[0], interpwin=1)
    np.testing.assert_allclose(selected_electrode["data"][0, 7:12], polynomial[7:12], rtol=1e-12)
    np.testing.assert_array_equal(selected_electrode["data"][1, 7:12], np.zeros(5))


def test_eeg_urlatency_current_suite_boundary_durations():
    events = [
        {"type": "boundary", "duration": 2, "latency": 1.5},
        {"type": "boundary", "duration": 3, "latency": 5.5},
        {"type": "boundary", "duration": 1, "latency": 9.5},
    ]
    assert eeg_urlatency(events, 9) == 14
    np.testing.assert_allclose(eeg_urlatency(events, [1, 6, 10]), [1, 11, 16])


def test_eeg_urlatency_current_suite_missing_duration():
    events = [{"type": "boundary", "latency": latency} for latency in [1.5, 5.5, 9.5]]
    assert np.isnan(eeg_urlatency(events, 9))


@eeglab_test("unittesting_popfunc/eeg_urlatency/popfunc_eeg_urlatency_wrapperTest.m", "test_pass_general")
def test_reference_urlatency_boundary_durations(eeglab_backend):
    events = np.array(
        [[("boundary", 2.0, 1.5), ("boundary", 3.0, 5.5), ("boundary", 1.0, 9.5)]],
        dtype=[("type", object), ("duration", object), ("latency", object)],
    )
    assert_matlab_near(eeglab_backend("eeg_urlatency", events, 9.0), [[14.0]])


@eeglab_test("unittesting_popfunc/eeg_urlatency/popfunc_eeg_urlatency_wrapperTest.m", "test_pass_no_duration")
def test_reference_urlatency_missing_duration(eeglab_backend):
    events = np.array(
        [[("boundary", 1.5), ("boundary", 5.5), ("boundary", 9.5)]], dtype=[("type", object), ("latency", object)]
    )
    assert_matlab_near(eeglab_backend("eeg_urlatency", events, 9.0), [[np.nan]])


@eeglab_test("unittesting_popfunc/getchanlist/popfunc_getchanlist_wrapperTest.m", "test_pass_cell")
def test_reference_getchanlist_multiple_types_and_missing_type(eeglab_backend):
    result = eeglab_backend("getchanlist", _reference_typed_locations(), np.array([["e", "f"]], dtype=object))
    assert_matlab_near(result, [[2.0]])


@eeglab_test("unittesting_popfunc/getchanlist/popfunc_getchanlist_wrapperTest.m", "test_pass_general")
def test_reference_getchanlist_case_insensitive_type(eeglab_backend):
    assert_matlab_near(eeglab_backend("getchanlist", _reference_typed_locations(), "L"), [[3.0, 4.0]])


@eeglab_test("unittesting_popfunc/getchanlist/popfunc_getchanlist_wrapperTest.m", "test_pass_no_match")
def test_reference_getchanlist_no_match(eeglab_backend):
    assert eeglab_backend("getchanlist", _reference_typed_locations(), "a").size == 0


@eeglab_test("unittesting_popfunc/getchanlist/popfunc_getchanlist_wrapperTest.m", "test_pass_one_arg")
def test_reference_getchanlist_default_all_channels(eeglab_backend):
    assert_matlab_near(eeglab_backend("getchanlist", _reference_typed_locations()), [[1.0, 2.0, 3.0, 4.0, 5.0]])
