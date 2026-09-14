"""Current eeglab_tests coverage for event-field editing."""

from pathlib import Path

import numpy as np

from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import eeglab_test


UPSTREAM = "unittesting_popfunc/pop_editeventfield/popfunc_pop_editeventfield_wrapperTest.m"


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


@eeglab_test(UPSTREAM, "test_pass_add_field_file")
def test_current_add_field_from_file(tmp_path):
    values = _write_values(tmp_path / "addfieldindices.txt")

    output = pop_editeventfield(_eeg(4), "test", values)

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


@eeglab_test(UPSTREAM, "test_pass_add_field_file_indices")
def test_current_add_field_from_file_at_indices(tmp_path):
    values = _write_values(tmp_path / "addfieldindices.txt")
    indices = [2, 3, 6, 10]

    output = pop_editeventfield(_eeg(), "test", values, "indices", indices)

    np.testing.assert_allclose([output["event"][index - 1]["test"] for index in indices], [0, -5.33, 14.7, 10])
    assert all("test" not in output["event"][index] for index in {0, 3, 4, 6, 7, 8})


@eeglab_test(UPSTREAM, "test_pass_add_field_vector_indices")
def test_current_add_field_from_numeric_vector():
    output = pop_editeventfield(_eeg(3), "test", [-7, 2, 19.5])

    np.testing.assert_allclose([event["test"] for event in output["event"]], [-7, 2, 19.5])


@eeglab_test(UPSTREAM, "test_pass_delim")
def test_current_file_import_honors_delimiters(tmp_path):
    values = _write_values(tmp_path / "delim.txt", "0,-5.33,14.7,10")

    output = pop_editeventfield(_eeg(4), "test", values, "delim", "\t ,")

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


@eeglab_test(UPSTREAM, "test_pass_delold_invalid")
def test_current_invalid_delold_leaves_events_unchanged():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "delold", "abcd", "test", 54321)

    assert _events_list(output["event"]) == eeg["event"]


@eeglab_test(UPSTREAM, "test_pass_delold_yes")
def test_current_delold_replaces_events_and_rebuilds_urevents():
    output = pop_editeventfield(_eeg(), "delold", "yes", "test", 54321)

    assert output["event"] == [{"test": 54321, "urevent": 0}]
    assert output["urevent"] == [{"test": 54321}]


@eeglab_test(UPSTREAM, "test_pass_info")
def test_current_adds_description_to_existing_field():
    output = pop_editeventfield(_eeg(), "typeinfo", "new comment")

    assert output["eventdescription"][0] == "new comment"


@eeglab_test(UPSTREAM, "test_pass_info_field_not_exist")
def test_current_ignores_description_for_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "abcdinfo", "new comment")

    assert output["eventdescription"] == ["kind", "target position", "time", ""]


@eeglab_test(UPSTREAM, "test_pass_latency_timeunit")
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


@eeglab_test(UPSTREAM, "test_pass_modify_field")
def test_current_modifies_existing_field():
    output = pop_editeventfield(_eeg(1), "type", 54321)

    assert output["event"][0]["type"] == 54321


@eeglab_test(UPSTREAM, "test_pass_modify_field_indices")
def test_current_modifies_existing_field_at_indices():
    output = pop_editeventfield(_eeg(), "type", 54321, "indices", [2])

    assert output["event"][1]["type"] == 54321
    assert output["event"][0]["type"] == "stim"


@eeglab_test(UPSTREAM, "test_pass_remove_field_not_exist")
def test_current_ignores_removal_of_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "test", [])

    assert _events_list(output["event"]) == eeg["event"]


@eeglab_test(UPSTREAM, "test_pass_rename_field")
def test_current_renames_existing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "position->place")

    assert all("position" not in event for event in output["event"])
    assert [event["place"] for event in output["event"]] == [event["position"] for event in eeg["event"]]


@eeglab_test(UPSTREAM, "test_pass_rename_field_invalid_format")
def test_current_ignores_invalid_rename_syntax():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "position=>place")

    assert _events_list(output["event"]) == eeg["event"]


@eeglab_test(UPSTREAM, "test_pass_rename_field_not_exist")
def test_current_ignores_rename_of_missing_field():
    eeg = _eeg()

    output = pop_editeventfield(eeg, "rename", "abcd->place")

    assert _events_list(output["event"]) == eeg["event"]


@eeglab_test(UPSTREAM, "test_pass_skipline")
def test_current_file_import_honors_leading_lines(tmp_path):
    values = _write_values(tmp_path / "skipline.txt", "header one\nheader two\nheader three\n0 -5.33 14.7 10")

    output = pop_editeventfield(_eeg(4), "test", values, "skipline", 3)

    np.testing.assert_allclose([event["test"] for event in output["event"]], [0, -5.33, 14.7, 10])


@eeglab_test(UPSTREAM, "test_test_pop_editeventfield")
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
