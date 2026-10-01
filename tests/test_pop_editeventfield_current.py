"""Current eeglab_tests coverage for event-field editing."""

from pathlib import Path
import shutil

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import assert_matlab_equal, eeglab_test


UPSTREAM = "unittesting_popfunc/pop_editeventfield/popfunc_pop_editeventfield_wrapperTest.m"


@pytest.fixture
def reference_eventfield_files(eeglab_suite_root, eeglab_working_directory):
    source = eeglab_suite_root / "unittesting_popfunc/pop_editeventfield"
    for filename in ("test.set", "test_no_events.set", "addfieldindices.txt", "delim.txt", "skipline.txt"):
        shutil.copyfile(source / filename, eeglab_working_directory / filename)


def _reference_values(events, field):
    if isinstance(events, dict):
        return events[field]
    # MATLAB horizontal concatenation ignores empty field values such as the
    # boundary event's empty position in the original recording.
    return np.concatenate([value for value in events[field].ravel() if value.size], axis=1)


@eeglab_test(UPSTREAM, "test_pass_add_field_file")
def test_reference_add_field_file(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "test", "addfieldindices.txt", nargout=2)
    assert_matlab_equal(_reference_values(output["event"], "test"), np.array([[0.0, -5.33, 14.7, 10.0]]))


@eeglab_test(UPSTREAM, "test_pass_add_field_file_indices")
def test_reference_add_field_file_indices(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend(
        "pop_editeventfield",
        eeg,
        "test",
        "addfieldindices.txt",
        "indices",
        np.array([[2.0, 3.0, 6.0, 10.0]]),
        nargout=2,
    )
    for index, value in zip((1, 2, 5, 9), (0.0, -5.33, 14.7, 10.0), strict=True):
        assert_matlab_equal(output["event"]["test"][0, index], np.array([[value]]))


@eeglab_test(UPSTREAM, "test_pass_add_field_vector_indices")
def test_reference_add_field_vector(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    values = np.array([[-7.0, 2.0, 19.5]])
    output, _ = eeglab_backend("pop_editeventfield", eeg, "test", values, nargout=2)
    assert_matlab_equal(_reference_values(output["event"], "test"), values)


@eeglab_test(UPSTREAM, "test_pass_delim")
def test_reference_delimiters(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "test", "delim.txt", "delim", "\t ,", nargout=2)
    assert_matlab_equal(_reference_values(output["event"], "test"), np.array([[0.0, -5.33, 14.7, 10.0]]))


@eeglab_test(UPSTREAM, "test_pass_delold_invalid")
def test_reference_delold_invalid(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "delold", "abcd", "test", 54321.0, nargout=2)
    assert_matlab_equal(output["event"], eeg["event"])


@eeglab_test(UPSTREAM, "test_pass_delold_yes")
def test_reference_delold_yes(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "delold", "yes", "test", 54321.0, nargout=2)
    assert {"test", "urevent"} <= output["event"].keys()
    assert_matlab_equal(output["event"]["test"], np.array([[54321.0]]))


@eeglab_test(UPSTREAM, "test_pass_info")
def test_reference_field_description(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "typeinfo", "new comment", nargout=2)
    assert_matlab_equal(output["eventdescription"][:, :1], np.array([["new comment"]], dtype=object))


@eeglab_test(UPSTREAM, "test_pass_info_field_not_exist")
def test_reference_description_missing_field(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "abcdinfo", "new comment", nargout=2)
    assert_matlab_equal(output["eventdescription"], eeg["eventdescription"])


@eeglab_test(UPSTREAM, "test_pass_latency_timeunit")
def test_reference_latency_timeunit(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    values = np.array([[54.0, 321.0, 123.0, 45.0]])
    output, _ = eeglab_backend(
        "pop_editeventfield",
        eeg,
        "type",
        np.ones((1, 4)),
        "latency",
        np.array([[100.0, 130.0, 123.0, 400.0]]),
        "timeunit",
        1e-3,
        "test",
        values,
        nargout=2,
    )
    assert {"test", "latency"} <= set(output["event"].dtype.names)
    # The source checks the permutation of the new values, not numeric latencies.
    assert_matlab_equal(np.sort(_reference_values(output["event"], "test")), np.sort(values))


@eeglab_test(UPSTREAM, "test_pass_modify_field")
def test_reference_modify_field(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "type", 54321.0, nargout=2)
    assert_matlab_equal(output["event"]["type"], np.array([[54321.0]]))


@eeglab_test(UPSTREAM, "test_pass_modify_field_indices")
def test_reference_modify_field_indices(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "type", 54321.0, "indices", 2.0, nargout=2)
    assert_matlab_equal(output["event"]["type"][0, 1], np.array([[54321.0]]))


@eeglab_test(UPSTREAM, "test_pass_remove_field_not_exist")
def test_reference_remove_missing_field(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "test", np.empty((0, 0)), nargout=2)
    assert_matlab_equal(output["event"], eeg["event"])


@eeglab_test(UPSTREAM, "test_pass_rename_field")
def test_reference_rename_field(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "rename", "position->place", nargout=2)
    assert "position" not in output["event"].dtype.names
    assert_matlab_equal(_reference_values(output["event"], "place"), _reference_values(eeg["event"], "position"))


@eeglab_test(UPSTREAM, "test_pass_rename_field_invalid_format")
def test_reference_invalid_rename(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "rename", "position=>place", nargout=2)
    assert "position" in output["event"].dtype.names
    assert "place" not in output["event"].dtype.names
    assert_matlab_equal(output["event"], eeg["event"])


@eeglab_test(UPSTREAM, "test_pass_rename_field_not_exist")
def test_reference_rename_missing_field(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "rename", "abcd->place", nargout=2)
    assert "place" not in output["event"].dtype.names
    assert_matlab_equal(output["event"], eeg["event"])


@eeglab_test(UPSTREAM, "test_pass_skipline")
def test_reference_skip_lines(eeglab_backend, reference_eventfield_files):
    eeg = eeglab_backend("pop_loadset", "test_no_events.set", "")
    output, _ = eeglab_backend("pop_editeventfield", eeg, "test", "skipline.txt", "skipline", 3.0, nargout=2)
    assert_matlab_equal(_reference_values(output["event"], "test"), np.array([[0.0, -5.33, 14.7, 10.0]]))


@eeglab_test(UPSTREAM, "test_test_pop_editeventfield")
def test_reference_sample_descriptions(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    descriptions = np.array([" Position of the target", "Can be 1 or 2          "])
    eeglab_backend("pop_editeventfield", eeg, "indices", "1:154", "positioninfo", descriptions)


def _eeg(event_count: int = 10) -> dict:
    events = [
        {"type": "stim", "position": index % 2 + 1, "latency": float(index * 10 + 1), "urevent": index}
        for index in range(event_count)
    ]
    return {
        "data": np.zeros((1, 1000), dtype=np.float32),
        "nbchan": 1,
        "pnts": 1000,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 9.99,
        "times": np.arange(1000, dtype=float) * 10,
        "chanlocs": [{"labels": "Cz"}],
        "event": events,
        "urevent": [dict(event) for event in events],
        "epoch": [],
        "eventdescription": {"type": "kind", "position": "target position", "latency": "time"},
    }


def _write_values(path: Path, text: str = "0 -5.33 14.7 10") -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _events_list(events):
    return events.tolist() if isinstance(events, np.ndarray) else events


def test_current_add_field_from_file(tmp_path):
    values = _write_values(tmp_path / "addfieldindices.txt")

    output = pop_editeventfield(_eeg(4), "test", values)

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


def test_current_add_field_from_file_at_indices(tmp_path):
    values = _write_values(tmp_path / "addfieldindices.txt")
    indices = [2, 3, 6, 10]

    output = pop_editeventfield(_eeg(), "test", values, "indices", indices)

    np.testing.assert_allclose([output["event"][index - 1]["test"] for index in indices], [0, -5.33, 14.7, 10])
    assert all("test" not in output["event"][index] for index in {0, 3, 4, 6, 7, 8})


def test_current_add_field_from_numeric_vector():
    output = pop_editeventfield(_eeg(3), "test", [-7, 2, 19.5])

    np.testing.assert_allclose([event["test"] for event in output["event"]], [-7, 2, 19.5])


def test_current_file_import_honors_delimiters(tmp_path):
    values = _write_values(tmp_path / "delim.txt", "0,-5.33,14.7,10")

    output = pop_editeventfield(_eeg(4), "test", values, "delim", "\t ,")

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


def test_current_invalid_delold_leaves_events_unchanged():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "delold", "abcd", "test", 54321)

    assert _events_list(output["event"]) == eeg["event"]


def test_current_delold_replaces_events_and_rebuilds_urevents():
    output = pop_editeventfield(_eeg(), "delold", "yes", "test", 54321)

    assert output["event"] == [{"test": 54321, "urevent": 0}]
    assert output["urevent"] == [{"test": 54321}]


def test_current_adds_description_to_existing_field():
    output = pop_editeventfield(_eeg(), "typeinfo", "new comment")

    assert output["eventdescription"][0] == "new comment"


def test_current_ignores_description_for_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "abcdinfo", "new comment")

    assert output["eventdescription"] == ["kind", "target position", "time", ""]


def test_current_converts_latency_timeunit_and_keeps_field_rows_together():
    output = pop_editeventfield(
        _eeg(4),
        "type",
        [1, 1, 1, 1],
        "latency",
        [100, 130, 123, 400],
        "timeunit",
        1e-3,
        "test",
        [54, 321, 123, 45],
    )

    np.testing.assert_allclose([event["latency"] for event in output["event"]], [11, 13.3, 14, 41])
    assert sorted(event["test"] for event in output["event"]) == [45, 54, 123, 321]


def test_current_modifies_existing_field():
    output = pop_editeventfield(_eeg(1), "type", 54321)

    assert output["event"][0]["type"] == 54321


def test_current_modifies_existing_field_at_indices():
    output = pop_editeventfield(_eeg(), "type", 54321, "indices", [2])

    assert output["event"][1]["type"] == 54321
    assert output["event"][0]["type"] == "stim"


def test_current_ignores_removal_of_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "test", [])

    assert _events_list(output["event"]) == eeg["event"]


def test_current_renames_existing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "position->place")

    assert all("position" not in event for event in output["event"])
    assert [event["place"] for event in output["event"]] == [event["position"] for event in eeg["event"]]


def test_current_ignores_invalid_rename_syntax():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "position=>place")

    assert _events_list(output["event"]) == eeg["event"]


def test_current_ignores_rename_of_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "abcd->place")

    assert _events_list(output["event"]) == eeg["event"]


def test_current_file_import_honors_leading_lines(tmp_path):
    values = _write_values(tmp_path / "skipline.txt", "header one\nheader two\nheader three\n0 -5.33 14.7 10")

    output = pop_editeventfield(_eeg(4), "test", values, "skipline", 3)

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


def test_current_sample_description_accepts_matlab_colon_indices():
    eeg = pop_loadset("sample_data/eeglab_data.set")

    output = pop_editeventfield(
        eeg,
        "indices",
        "1:154",
        "positioninfo",
        ["Position of the target", "Can be 1 or 2"],
    )

    assert len(output["event"]) == 154
    assert "Position of the target" in output["eventdescription"][1]
