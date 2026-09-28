"""Python counterparts of MATLAB-first structure, event and selection contracts."""

from functools import partial

import numpy as np
import pytest

from tests.eeglab_tests import assert_matlab_equal, expanded_matlab_test
from tests.eeglab_tests.assertions import matlab_field_concat
from tests.test_eeg_store import _source_dataset_row
from tests.test_study_grouped_measure_plots_eeglab_tests import _cell_row, _records


SOURCE = "tests/matlab/expanded/test_eegprep_structures_expanded.m"
SOURCE_SHA256 = "b48c8aac12f08b72827c90839a9b216581e1a24bc640128513ed6ce508f85438"


def _native(test):
    return expanded_matlab_test(SOURCE, test, SOURCE_SHA256)


def _assert_double(actual, expected, *, python=False):
    actual = np.asarray(actual)
    if python and actual.ndim == 0:
        # Public Python metadata uses scalar int/float values and 1-D vectors;
        # MATLAB outputs retain their original array shape and declared class.
        actual = np.array([[float(actual)]])
    elif python and actual.ndim == 1:
        actual = actual[None, :]
    np.testing.assert_array_equal(actual, np.atleast_2d(np.asarray(expected, dtype=float)), strict=True)


@pytest.fixture
def assert_double(request):
    return partial(_assert_double, python=request.config.getoption("--eeglab-backend") == "python")


def _assert_cells(actual, expected, assert_double):
    assert actual.dtype == object
    assert actual.shape == expected.shape
    for index in np.ndindex(expected.shape):
        if isinstance(expected[index], str):
            assert isinstance(actual[index], str)
            assert actual[index] == expected[index]
        else:
            assert_double(actual[index], expected[index])


def _sample_eeg(backend, trials):
    eeg = backend("eeg_emptyset")
    shape = (4, 8, trials) if trials > 1 else (4, 8)
    eeg.update(
        data=np.arange(1, 4 * 8 * trials + 1, dtype=np.float32).reshape(shape, order="F"),
        nbchan=4.0,
        pnts=8.0,
        trials=float(trials),
        srate=4.0,
        xmin=0.0,
        xmax=7.0 / 4.0,
        times=np.arange(8.0)[None, :] * 250.0,
        chanlocs=_source_dataset_row(
            *[
                dict(labels=label, type=kind, X=float(index), Y=0.0, Z=0.0)
                for index, (label, kind) in enumerate(
                    [("Fz", "EEG"), ("VEOG", "EOG"), ("Pz", "EEG"), ("EMG", "EMG")], start=1
                )
            ]
        ),
    )
    return eeg


def _continuous_events(backend):
    eeg = _sample_eeg(backend, 1)
    events = [
        dict(type=kind, latency=np.array([[latency]]), urevent=np.array([[float(index)]]))
        for index, (kind, latency) in enumerate(
            zip(["first", "left", "middle", "right", "last"], [1.0, 2.25, 3.0, 6.75, 8.0], strict=True), start=1
        )
    ]
    eeg["event"] = _source_dataset_row(*events)
    eeg["urevent"] = _source_dataset_row(*[{key: event[key] for key in ("type", "latency")} for event in events])
    return eeg


def _epoched_events(backend, trials):
    eeg = _sample_eeg(backend, trials)
    eeg.update(xmin=-0.5, xmax=1.25, times=np.arange(-2.0, 6.0)[None, :] * 250.0)
    events = [
        dict(type=kind, latency=8.0 * trial + latency, epoch=float(trial + 1), urevent=float(2 * trial + within))
        for trial in range(trials)
        for within, (kind, latency) in enumerate([("stim", 3.0), ("response", 5.5)], start=1)
    ]
    eeg["event"] = _source_dataset_row(*events)
    eeg["urevent"] = _source_dataset_row(*[{key: event[key] for key in ("type", "latency")} for event in events])
    return backend("eeg_checkset", eeg, "eventconsistency")


def _event_epochs():
    return _source_dataset_row(
        dict(
            event=np.array([[1.0, 2.0]]),
            eventlatency=_cell_row(-250.0, 0.0),
            eventtype=_cell_row("cue", "stim"),
            eventcode=_cell_row(7.0, 11.0),
        ),
        dict(
            event=np.array([[3.0, 4.0]]),
            eventlatency=_cell_row(0.0, 250.0),
            eventtype=_cell_row("stim", "other"),
            eventcode=_cell_row(13.0, 17.0),
        ),
    )


@_native("test_epochformat_numeric_fields_and_event_indices")
def test_epochformat_numeric_fields_and_event_indices(eeglab_backend, assert_double):
    epochs, fields = eeglab_backend(
        "eeg_epochformat",
        np.array([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]]),
        "struct",
        _cell_row("var1"),
        np.array([[4.0, 9.0]]),
        nargout=2,
    )
    assert_matlab_equal(fields, _cell_row("var1", "var2", "var3"))
    for name, expected in [
        ("var1", [11.0, 21.0]),
        ("var2", [12.0, 22.0]),
        ("var3", [13.0, 23.0]),
        ("event", [4.0, 9.0]),
    ]:
        assert_double(matlab_field_concat(epochs, name), [expected])


@_native("test_epochformat_cells_preserve_labels_and_event_lists")
def test_epochformat_cells_preserve_labels_and_event_lists(eeglab_backend, assert_double):
    epochs, fields = eeglab_backend(
        "eeg_epochformat",
        np.array([["target", 125.0], ["standard", 250.0]], dtype=object),
        "struct",
        _cell_row("condition", "rt"),
        _cell_row(np.array([[2.0, 3.0]]), np.array([[7.0, 8.0, 9.0]])),
        nargout=2,
    )
    assert_matlab_equal(fields, _cell_row("condition", "rt"))
    assert [record["condition"] for record in _records(epochs)] == ["target", "standard"]
    assert_double(matlab_field_concat(epochs, "rt"), [[125.0, 250.0]])
    assert_double(_records(epochs)[0]["event"], [[2.0, 3.0]])
    assert_double(_records(epochs)[1]["event"], [[7.0, 8.0, 9.0]])


@_native("test_epochformat_array_selects_time_locking_events")
def test_epochformat_array_selects_time_locking_events(eeglab_backend, assert_double):
    values, fields = eeglab_backend("eeg_epochformat", _event_epochs(), "array", nargout=2)
    assert_matlab_equal(fields, _cell_row("event", "eventlatency", "eventtype", "eventcode").T)
    expected = np.array([[2.0, 0.0, "stim", 11.0], [3.0, 0.0, "stim", 13.0]], dtype=object)
    _assert_cells(values, expected, assert_double)


@_native("test_epochformat_selected_type_keeps_first_match_and_missing_trial")
def test_epochformat_selected_type_keeps_first_match_and_missing_trial(eeglab_backend, assert_double):
    epochs = _event_epochs()
    epochs["eventtype"][0, 0] = _cell_row("response", "response")
    values, fields = eeglab_backend("eeg_epochformat", epochs, "array", _cell_row("response"), nargout=2)
    assert_matlab_equal(fields, _cell_row("event", "eventlatency", "eventtype", "eventcode").T)
    expected = np.array([[1.0, -250.0, "response", 7.0], [np.nan] * 4], dtype=object)
    _assert_cells(values, expected, assert_double)


@_native("test_select_disjoint_samples_retains_fractional_events_and_urevents")
def test_select_disjoint_samples_retains_fractional_events_and_urevents(eeglab_backend, assert_double):
    eeg = _continuous_events(eeglab_backend)
    out = eeglab_backend("pop_select", eeg, "point", np.array([[1.0, 2.0], [7.0, 8.0]]))
    np.testing.assert_array_equal(out["data"], eeg["data"][:, [0, 1, 6, 7]], strict=True)
    assert_double(out["pnts"], 4)
    events = _records(out["event"])
    assert [event["type"] for event in events] == ["first", "left", "boundary", "right", "last"]
    assert_double(matlab_field_concat(out["event"], "latency"), [[1.0, 2.25, 2.5, 2.75, 4.0]])
    assert_double(events[2]["duration"], 4)
    assert_double(np.hstack([events[index]["urevent"] for index in [0, 1, 3, 4]]), [[1, 2, 4, 5]])
    assert_matlab_equal(out["urevent"], eeg["urevent"])


@_native("test_select_crop_keeps_edge_discontinuities_and_exact_samples")
def test_select_crop_keeps_edge_discontinuities_and_exact_samples(eeglab_backend, assert_double):
    eeg = _continuous_events(eeglab_backend)
    out = eeglab_backend("pop_select", eeg, "point", np.array([[2.0, 7.0]]))
    np.testing.assert_array_equal(out["data"], eeg["data"][:, 1:7], strict=True)
    assert_double(matlab_field_concat(out["event"], "latency"), [[0.5, 1.25, 2.0, 5.75, 6.5]])
    events = _records(out["event"])
    assert [event["type"] for event in events] == ["boundary", "left", "middle", "right", "boundary"]
    assert_double(np.hstack([events[index]["duration"] for index in [0, 4]]), [[1, 1]])
    assert_double(np.hstack([event["urevent"] for event in events[1:4]]), [[2, 3, 4]])
    assert_matlab_equal(out["urevent"], eeg["urevent"])


@_native("test_select_channel_types_keeps_locations_and_removed_metadata")
def test_select_channel_types_keeps_locations_and_removed_metadata(eeglab_backend, assert_double):
    eeg = _sample_eeg(eeglab_backend, 1)
    out = eeglab_backend("pop_select", eeg, "chantype", _cell_row("EEG", "EOG"), "rmchantype", "EOG")
    np.testing.assert_array_equal(out["data"], eeg["data"][[0, 2], :], strict=True)
    assert_double(out["nbchan"], 2)
    assert [channel["labels"] for channel in _records(out["chanlocs"])] == ["Fz", "Pz"]
    assert_double(matlab_field_concat(out["chanlocs"], "X"), [[1.0, 3.0]])
    removed = out["chaninfo"]["removedchans"]
    assert [channel["labels"] for channel in _records(removed)] == ["VEOG", "EMG"]
    assert_double(matlab_field_concat(removed, "X"), [[2.0, 4.0]])


@_native("test_select_channel_labels_use_original_order_and_exclusion")
def test_select_channel_labels_use_original_order_and_exclusion(eeglab_backend):
    eeg = _sample_eeg(eeglab_backend, 1)
    out = eeglab_backend("pop_select", eeg, "channel", _cell_row("EMG", "Fz", "Pz"), "rmchannel", _cell_row("Pz"))
    np.testing.assert_array_equal(out["data"], eeg["data"][[0, 3], :], strict=True)
    assert [channel["labels"] for channel in _records(out["chanlocs"])] == ["Fz", "EMG"]
    assert [channel["labels"] for channel in _records(out["chaninfo"]["removedchans"])] == ["VEOG", "Pz"]


@_native("test_select_reordered_trials_reindexes_events_without_reordering_urevents")
def test_select_reordered_trials_reindexes_events_without_reordering_urevents(eeglab_backend, subtests, assert_double):
    eeg = _epoched_events(eeglab_backend, 3)
    out = eeglab_backend("pop_select", eeg, "trial", np.array([[3.0, 1.0]]), "sorttrial", "off")
    np.testing.assert_array_equal(out["data"], eeg["data"][:, :, [2, 0]], strict=True)
    assert_double(out["trials"], 2)
    for trial, epoch in enumerate(_records(out["epoch"]), start=1):
        indices = np.asarray(epoch["event"]).astype(int).ravel() - 1
        linked = _source_dataset_row(*[_records(out["event"])[index] for index in indices])
        assert_double(matlab_field_concat(linked, "epoch"), [[trial, trial]])
        assert_double(matlab_field_concat(linked, "latency"), [[3.0 + 8 * (trial - 1), 5.5 + 8 * (trial - 1)]])
        assert_double(matlab_field_concat(linked, "urevent"), [[5, 6] if trial == 1 else [1, 2]])
    # Retain the native ascending epoch/latency regression, without xfail or skip.
    for field, expected in [("latency", [3.0, 5.5, 11.0, 13.5]), ("epoch", [1, 1, 2, 2]), ("urevent", [5, 6, 1, 2])]:
        with subtests.test(field=field):
            assert_double(matlab_field_concat(out["event"], field), [expected])
    epochs = _records(out["epoch"])
    for index, expected in enumerate([[[1.0, 2.0]], [[3.0, 4.0]]]):
        with subtests.test(epoch=index + 1):
            assert_double(epochs[index]["event"], expected)
    assert_double(np.hstack(epochs[0]["eventlatency"][0]), [[0.0, 625.0]])
    assert_matlab_equal(out["urevent"], eeg["urevent"])


@_native("test_select_duplicate_trials_keep_last_occurrences_in_input_order")
def test_select_duplicate_trials_keep_last_occurrences_in_input_order(eeglab_backend, assert_double):
    eeg = _epoched_events(eeglab_backend, 3)
    out = eeglab_backend("pop_select", eeg, "trial", np.array([[3.0, 1.0, 3.0]]), "sorttrial", "off")
    # Pinned unique_bc uses legacy last-occurrence semantics, not unique(...,'stable').
    np.testing.assert_array_equal(out["data"], eeg["data"][:, :, [0, 2]], strict=True)
    assert_double(out["trials"], 2)
    for field, expected in [("latency", [3.0, 5.5, 11.0, 13.5]), ("epoch", [1, 1, 2, 2]), ("urevent", [1, 2, 5, 6])]:
        assert_double(matlab_field_concat(out["event"], field), [expected])
    epochs = _records(out["epoch"])
    assert_double(epochs[0]["event"], [[1.0, 2.0]])
    assert_double(epochs[1]["event"], [[3.0, 4.0]])
    assert_matlab_equal(out["urevent"], eeg["urevent"])


@_native("test_select_one_trial_keeps_sample_axis_and_urevent_origin")
def test_select_one_trial_keeps_sample_axis_and_urevent_origin(eeglab_backend, assert_double):
    eeg = _epoched_events(eeglab_backend, 3)
    out = eeglab_backend("pop_select", eeg, "rmtrial", np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(out["data"], eeg["data"][:, :, 2], strict=True)
    assert_double(np.hstack([out[field] for field in ("nbchan", "pnts", "trials")]), [[4, 8, 1]])
    assert_double(matlab_field_concat(out["event"], "latency"), [[3.0, 5.5]])
    assert_double(matlab_field_concat(out["event"], "urevent"), [[5.0, 6.0]])
    assert all("epoch" not in event for event in _records(out["event"]))
    assert np.asarray(out["epoch"]).size == 0
    assert_matlab_equal(out["urevent"], eeg["urevent"])


@_native("test_select_explicit_empty_trial_selection")
def test_select_explicit_empty_trial_selection(eeglab_backend, assert_double):
    eeg = _epoched_events(eeglab_backend, 3)
    out = eeglab_backend("pop_select", eeg, "rmtrial", np.array([[1.0, 2.0, 3.0]]), "erroronempty", "off")
    for field in ("data", "event", "epoch"):
        assert np.asarray(out[field]).size == 0
    assert_double(out["trials"], 0)


@_native("test_mergeset_eventless_second_recording_adds_half_sample_boundary")
def test_mergeset_eventless_second_recording_adds_half_sample_boundary(eeglab_backend, assert_double):
    first = _continuous_events(eeglab_backend)
    second = _sample_eeg(eeglab_backend, 1)
    second["data"] += 100.0
    out = eeglab_backend("pop_mergeset", first, second, 0.0)
    np.testing.assert_array_equal(out["data"], np.hstack([first["data"], second["data"]]), strict=True)
    assert_double(np.hstack([out[field] for field in ("nbchan", "pnts", "trials")]), [[4, 16, 1]])
    events = _records(out["event"])
    assert [event["type"] for event in events] == ["first", "left", "middle", "right", "last", "boundary"]
    assert_double(matlab_field_concat(out["event"], "latency"), [[1.0, 2.25, 3.0, 6.75, 8.0, 8.5]])
    assert_double(np.hstack([event["urevent"] for event in events[:5]]), [[1, 2, 3, 4, 5]])
    assert _records(out["urevent"])[-1]["type"] == "boundary"
    assert_double(_records(out["urevent"])[-1]["latency"], 8.5)
    assert [channel["labels"] for channel in _records(out["chanlocs"])] == [
        channel["labels"] for channel in _records(first["chanlocs"])
    ]


@_native("test_mergeset_epoched_then_singleton_reconstructs_epoch_references")
def test_mergeset_epoched_then_singleton_reconstructs_epoch_references(eeglab_backend, assert_double):
    first = _epoched_events(eeglab_backend, 2)
    second = _sample_eeg(eeglab_backend, 1)
    second["data"] += 100.0
    second["event"] = dict(type="response", latency=2.5, urevent=1.0)
    second["urevent"] = dict(type="response", latency=2.5)
    del second["epoch"]
    out = eeglab_backend("pop_mergeset", first, second, 0.0)
    np.testing.assert_array_equal(
        out["data"], np.concatenate([first["data"], second["data"][:, :, None]], axis=2), strict=True
    )
    assert_double(np.hstack([out[field] for field in ("pnts", "trials", "xmin", "xmax")]), [[8, 3, -0.5, 1.25]])
    for field, expected in [
        ("latency", [3.0, 5.5, 11.0, 13.5, 18.5]),
        ("epoch", [1, 1, 2, 2, 3]),
        ("urevent", [1, 2, 3, 4, 6]),
    ]:
        assert_double(matlab_field_concat(out["event"], field), [expected])
    assert_double(_records(out["epoch"])[2]["event"], 5)
    _assert_cells(_records(out["epoch"])[2]["eventlatency"], _cell_row(np.array([[-125.0]])), assert_double)


@_native("test_mergeset_singleton_then_epoched_offsets_all_events")
def test_mergeset_singleton_then_epoched_offsets_all_events(eeglab_backend, assert_double):
    first = _sample_eeg(eeglab_backend, 1)
    first["event"] = dict(type="start", latency=1.5, urevent=1.0)
    first["urevent"] = dict(type="start", latency=1.5)
    second = _epoched_events(eeglab_backend, 2)
    second["data"] += 100.0
    out = eeglab_backend("pop_mergeset", first, second, 0.0)
    np.testing.assert_array_equal(
        out["data"], np.concatenate([first["data"][:, :, None], second["data"]], axis=2), strict=True
    )
    assert_double(np.hstack([out[field] for field in ("pnts", "trials", "xmin", "xmax")]), [[8, 3, 0, 1.75]])
    for field, expected in [
        ("latency", [1.5, 11.0, 13.5, 19.0, 21.5]),
        ("epoch", [1, 2, 2, 3, 3]),
        ("urevent", [1, 3, 4, 5, 6]),
    ]:
        assert_double(matlab_field_concat(out["event"], field), [expected])
    _assert_cells(_records(out["epoch"])[0]["eventlatency"], _cell_row(np.array([[125.0]])), assert_double)


@_native("test_checkur_sort_preserves_event_to_original_event_identity")
def test_checkur_sort_preserves_event_to_original_event_identity(eeglab_backend, assert_double):
    eeg = _sample_eeg(eeglab_backend, 1)
    eeg["event"] = _source_dataset_row(
        dict(type="late", latency=7.0, urevent=1.0), dict(type="early", latency=1.0, urevent=2.0)
    )
    eeg["urevent"] = _source_dataset_row(dict(type="late", latency=7.0), dict(type="early", latency=1.0))
    out = eeglab_backend("eeg_checkset", eeg, "checkur")
    assert [event["type"] for event in _records(out["urevent"])] == ["early", "late"]
    assert_double(matlab_field_concat(out["urevent"], "latency"), [[1.0, 7.0]])
    assert [event["type"] for event in _records(out["event"])] == ["late", "early"]
    assert_double(matlab_field_concat(out["event"], "urevent"), [[2.0, 1.0]])
    out = eeglab_backend("eeg_checkset", out, "eventconsistency")
    assert [event["type"] for event in _records(out["event"])] == ["early", "late"]
    assert_double(matlab_field_concat(out["event"], "latency"), [[1.0, 7.0]])
    assert_double(matlab_field_concat(out["event"], "urevent"), [[1.0, 2.0]])
    np.testing.assert_array_equal(out["data"], eeg["data"], strict=True)


@_native("test_eventconsistency_normalizes_initial_boundary_and_removes_nan_event")
def test_eventconsistency_normalizes_initial_boundary_and_removes_nan_event(eeglab_backend, assert_double):
    eeg = _sample_eeg(eeglab_backend, 1)
    eeg["event"] = _source_dataset_row(
        *[
            dict(type=kind, latency=latency, duration=duration)
            for kind, latency, duration in zip(
                ["boundary", "first", "invalid", "fractional", "last"],
                [0.75, 1.0, np.nan, 4.5, 8.0],
                [2.0, 0.0, 0.0, 0.0, 0.0],
                strict=True,
            )
        ]
    )
    out = eeglab_backend("eeg_checkset", eeg, "eventconsistency")
    assert [event["type"] for event in _records(out["event"])] == ["boundary", "first", "fractional", "last"]
    assert_double(matlab_field_concat(out["event"], "latency"), [[0.5, 1.0, 4.5, 8.0]])
    assert_double(matlab_field_concat(out["event"], "duration"), [[2.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(out["data"], eeg["data"], strict=True)
