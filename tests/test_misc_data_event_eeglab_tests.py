"""Current EEGLAB test ports for miscellaneous data and event utilities."""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep import (
    biosig2eeglabevent,
    eeg_regepochs,
    eeg_time2prev,
    eventalign,
    eventlock,
    getipsph,
    hist2,
    rmart,
    rmsave,
    shortread,
    unique_cell_string,
)
from eegprep.functions.sigprocfunc.floatread import floatread
from eegprep.functions.sigprocfunc.floatwrite import floatwrite
from tests.eeglab_tests import assert_matlab_near, eeglab_test


@eeglab_test("unittesting_sigprocfunc/eventalign/sigprocfunc_eventalign_wrapperTest.m", "test_fail_no_arg")
@eeglab_test("unittesting_sigprocfunc/eventalign/sigprocfunc_eventalign_wrapperTest.m", "test_pass_general")
@eeglab_test("unittesting_sigprocfunc/eventalign/sigprocfunc_eventalign_wrapperTest.m", "test_pass_matrix")
def test_reference_eventalign_non_enforcing_source(eeglab_backend, record_property):
    # These helpers catch errors and return a status that the source wrapper
    # ignores. Record that status; this is not a behavioral pass/fail contract.
    for name, arguments, expected in (
        ("fail_no_arg", (), None),
        ("pass_general", (2.0, np.arange(1.0, 6.0)[None, :], np.arange(6.0, 11.0)[None, :]), [[0]]),
        (
            "pass_matrix",
            (2.0, np.array([[1, 2, 3], [4, 5, 6]], dtype=float), np.array([[6, 7, 8], [9, 10, 11]], dtype=float)),
            [[1]],
        ),
    ):
        try:
            result = eeglab_backend("eventalign", *arguments)
            if expected is not None:
                assert_matlab_near(result, expected)
            status = "passed"
        except Exception as error:
            status = f"not passed: {type(error).__name__}: {error}"
        record_property(f"eventalign_{name}_ignored_source_status", status)


@eeglab_test("unittesting_miscfunc/getipsph/miscfunc_getipsph_wrapperTest.m", "test_test_getipsph")
def test_reference_getipsph(eeglab_backend):
    rng = np.random.default_rng(1)
    for dimensions in (8.0, 10.0, 1.0):
        data = rng.standard_normal((10, 1000))
        sphere = eeglab_backend("getipsph", data, dimensions)
        np.einsum("ij,jk->ik", sphere, data)


@eeglab_test("unittesting_miscfunc/rmart/miscfunc_rmart_wrapperTest.m", "test_test_rmart")
def test_reference_rmart(eeglab_backend, tmp_path):
    data = np.array(
        [
            [2, 5, 3, 6, 7, 2, 6, 8, 1, 2],
            [6, 1, 10, 234, 3, 5, 464, 3, 2, 5],
            [1, 1, 1, 1, 3, 5, 1, 1, 4, 5],
            [4, 23456, 2, 3, 1, 1, 34, 2, 3, 5],
            [20, 30, 10, 10, 34, 10, 30, 20, 30, 10],
        ],
        dtype=float,
    )
    source = tmp_path / "test_data.floats"
    destination = tmp_path / "test_result.floats"
    data.astype(np.float32).ravel(order="F").tofile(source)
    eeglab_backend("rmart", str(source), str(destination), 5.0, np.arange(1.0, 6.0)[None, :], 7.0, nargout=0)
    # The source deliberately reads file1, not the corrected destination.
    result = np.fromfile(source, dtype=np.float32).reshape(5, 10, order="F")
    assert_matlab_near([result.shape], [data.shape])


def _continuous_eeg(channels: int, points: int, sampling_rate: float) -> dict:
    data = np.arange(channels * points, dtype=float).reshape(channels, points)
    return {
        "data": data,
        "nbchan": channels,
        "pnts": points,
        "trials": 1,
        "srate": sampling_rate,
        "xmin": 0.0,
        "xmax": (points - 1) / sampling_rate,
        "times": np.arange(points) * 1000 / sampling_rate,
        "setname": "continuous",
        "event": [],
        "urevent": [],
        "epoch": [],
        "chanlocs": [],
    }


@eeglab_test(
    "unittesting_miscfunc/eeg_regepochs/miscfunc_eeg_regepochs_wrapperTest.m",
    "test_check_number_trials",
)
def test_eeg_regepochs_current_suite_preserves_all_regular_trials():
    eeg = _continuous_eeg(1, 40 * 500, 500)

    one_second = eeg_regepochs(
        eeg,
        "eventtype",
        "temp",
        "extractepochs",
        "on",
        "recurrence",
        1,
        "limits",
        [0, 1],
    )
    offset = eeg_regepochs(
        eeg,
        "eventtype",
        "temp",
        "extractepochs",
        "on",
        "recurrence",
        1,
        "limits",
        [0.5, 0.8],
    )

    assert one_second["trials"] == offset["trials"] == 40
    assert one_second["data"].shape == (1, 500, 40)
    assert offset["data"].shape == (1, 150, 40)
    np.testing.assert_array_equal(one_second["data"][0, :, 0], eeg["data"][0, :500])
    np.testing.assert_array_equal(offset["data"][0, :, 0], eeg["data"][0, 250:400])


@eeglab_test(
    "unittesting_miscfunc/eeg_regepochs/miscfunc_eeg_regepochs_wrapperTest.m",
    "test_pass_general",
)
def test_eeg_regepochs_current_suite_default_and_event_only_workflows():
    eeg = _continuous_eeg(2, 10, 3)
    eeg["xmax"] = 3
    eeg["data"] = np.array([[1, 1, 1, 1, 1, 1, 2, 2, 2, 3], [2, 1, 1, 2, 1, 1, 2, 1, 1, 2]])

    epoched = eeg_regepochs(eeg)
    event_only = eeg_regepochs(
        eeg,
        1,
        [0, 1],
        np.nan,
        extractepochs="off",
        eventdata={"condition": "rest"},
    )

    assert epoched["data"].shape == (2, 3, 3)
    np.testing.assert_array_equal(epoched["data"][:, :, 0], eeg["data"][:, :3])
    assert [event["latency"] for event in event_only["event"]] == [1, 4, 7]
    assert [event["urevent"] for event in event_only["event"]] == [0, 1, 2]
    assert {event["condition"] for event in event_only["urevent"]} == {"rest"}


@eeglab_test(
    "unittesting_miscfunc/eeg_time2prev/miscfunc_eeg_time2prev_wrapperTest.m",
    "test_pass_general",
)
def test_eeg_time2prev_current_suite_returns_positive_delays_and_zero_based_indices():
    urevents = [
        {"type": "square", "latency": 5},
        {"type": "square", "latency": 8},
        {"type": "rt", "latency": 8.5},
        {"type": "square", "latency": 12},
        {"type": "rt", "latency": 12.4},
    ]
    eeg = {
        "srate": 1000,
        "urevent": urevents,
        "event": [{**copy.deepcopy(event), "urevent": index} for index, event in enumerate(urevents)],
    }

    delays, targets, urtargets, urprevs = eeg_time2prev(eeg, ["rt"], ["square"])

    np.testing.assert_allclose(delays, [0.5, 0.4])
    np.testing.assert_array_equal(targets, [2, 4])
    np.testing.assert_array_equal(urtargets, [2, 4])
    np.testing.assert_array_equal(urprevs, [1, 3])

    no_previous = eeg_time2prev(eeg, ["square"], ["rt"])
    assert no_previous[0][0] == 0
    assert no_previous[3][0] == -1


def test_python_regression_getipsph_full_and_reduced_sphering():
    data = np.random.default_rng(812).standard_normal((10, 1000))
    centered = data - np.mean(data, axis=1, keepdims=True)

    for dimensions in (8, 10, 1):
        sphere = getipsph(data, dimensions)
        transformed = np.einsum("ij,jk->ik", sphere, centered)
        covariance = np.einsum("ij,kj->ik", transformed, transformed) / transformed.shape[1]
        assert sphere.shape == (dimensions, 10)
        np.testing.assert_allclose(covariance, np.eye(dimensions), atol=1e-11)


@eeglab_test(
    "unittesting_miscfunc/hist2/miscfunc_hist2_wrapperTest.m",
    "test_test_hist2",
)
def test_hist2_current_suite_uses_shared_bin_centers_and_complete_counts():
    figure, axes = plt.subplots()
    try:
        result = hist2([-2, -0.25, 0.25, 2], [-1, -0.75, 0.75, 1], [-1, 0, 1], ax=axes)
        first_counts = [patch.get_height() for patch in axes.containers[0]]
        second_counts = [patch.get_height() for patch in axes.containers[1]]
        assert result is axes
        assert first_counts == [1, 2, 1]
        assert second_counts == [2, 0, 2]
        assert axes.get_ylabel() == "Number of values"
        assert axes.get_xlim() == pytest.approx((-1, 1))
    finally:
        plt.close(figure)


def test_python_regression_rmart_writes_selected_channels_and_reduces_triggered_eog(tmp_path):
    rng = np.random.default_rng(193)
    data = rng.standard_normal((3, 320)).astype(np.float32)
    data[2] = 0
    data[2, 140:180] = 100 * np.sin(np.linspace(0, np.pi, 40))
    data[0] += 2 * data[2]
    source = tmp_path / "test_data.floats"
    destination = tmp_path / "test_result.floats"
    floatwrite(data, source)

    corrected = rmart(source, destination, 3, [1, 2], 3, 80)
    stored = floatread(destination, [2, np.inf])

    assert corrected.shape == stored.shape == (2, 320)
    np.testing.assert_allclose(stored, corrected, rtol=1e-6, atol=1e-6)
    centered = data[0] - np.mean(data[0])
    assert np.linalg.norm(corrected[0, 100:180]) < np.linalg.norm(centered[100:180]) * 0.25
    np.testing.assert_allclose(corrected[0, :80], centered[:80], rtol=1e-6, atol=2e-6)


@eeglab_test(
    "unittesting_miscfunc/rmsave/miscfunc_rmsave_wrapperTest.m",
    "test_test_rmsave",
)
def test_rmsave_current_suite_block_shapes_and_values():
    data = np.array(
        [
            [2, 5, 3, 6, 7, 2, 6, 8, 1, 2],
            [6, 1, 10, 234, 3, 5, 464, 3, 2, 5],
            [1, 1, 1, 1, 3, 5, 1, 1, 4, 5],
            [4, 23456, 2, 3, 1, 1, 34, 2, 3, 5],
            [20, 30, 10, 10, 34, 10, 30, 20, 30, 10],
        ]
    )
    assert rmsave(data, 10).shape == (5, 1)
    assert rmsave(data, 5).shape == (5, 2)
    random_data = np.random.default_rng(441).random((32, 100))
    assert rmsave(random_data, 25).shape == (32, 4)
    np.testing.assert_allclose(
        rmsave(data, 5)[0],
        [np.sqrt(np.mean(data[0, :5] ** 2)), np.sqrt(np.mean(data[0, 5:] ** 2))],
    )


@eeglab_test(
    "unittesting_miscfunc/shortread/miscfunc_shortread_wrapperTest.m",
    "test_test_shortread",
)
def test_shortread_current_suite_finite_inferred_and_offset_shapes(tmp_path):
    data = np.arange(2000, dtype=np.int16).reshape(20, 100, order="F")
    path = tmp_path / "test_short"
    np.ravel(data, order="F").tofile(path)

    np.testing.assert_array_equal(shortread(path, [20, 100]), data)
    np.testing.assert_array_equal(shortread(path, [20, np.inf]), data)
    np.testing.assert_array_equal(shortread(path, [10, 10]), np.arange(100).reshape(10, 10, order="F"))
    np.testing.assert_array_equal(shortread(path, [10, 1], offset=1990), np.arange(1990, 2000).reshape(10, 1))
    assert shortread(path, [5, np.inf], offset=20).shape == (5, 396)


@eeglab_test(
    "unittesting_miscfunc/uniqe_cell_string/miscfunc_uniqe_cell_string_wrapperTest.m",
    "test_test_uniqe_cell_string",
)
def test_unique_cell_string_current_suite_ignores_non_strings_and_preserves_order():
    values = [{"name": "haha", "func": "lazy"}, "We ", np.arange(6).reshape(3, 2), 235.3443]
    values.extend(["are ", "strings.", [" Not", "including me!"], "are "])
    assert unique_cell_string(values) == ["We ", "are ", "strings."]


@eeglab_test(
    "unittesting_sigprocfunc/biosig2eeglabevent/sigprocfunc_biosig2eeglabevent_wrapperTest.m",
    "test_pass_all_set",
)
def test_biosig2eeglabevent_current_suite_all_fields():
    source = {
        "TYP": ["a", "b", "a", "a", "b"],
        "POS": [1.0, 2.1, 6.294, 10.2, 42.943],
        "DUR": [0.03, 1.02, 3, 0.45, 1.9],
        "CHN": ["5", "2", "9", "5", "1"],
    }
    events = biosig2eeglabevent(source)
    assert [event["type"] for event in events] == source["TYP"]
    assert [event["latency"] for event in events] == source["POS"]
    assert [event["duration"] for event in events] == source["DUR"]
    assert [event["chanindex"] for event in events] == source["CHN"]


@eeglab_test(
    "unittesting_sigprocfunc/biosig2eeglabevent/sigprocfunc_biosig2eeglabevent_wrapperTest.m",
    "test_pass_some_set",
)
def test_biosig2eeglabevent_current_suite_partial_fields_and_one_based_interval():
    source = {"POS": [1, 5, 10, 15], "DUR": [0, 8, 4, 2]}
    all_events = biosig2eeglabevent(source)
    interval_events = biosig2eeglabevent(source, [5, 12])

    assert [event["latency"] for event in all_events] == source["POS"]
    assert [event["duration"] for event in all_events] == source["DUR"]
    assert interval_events == [{"latency": 1.0, "duration": 7.0}, {"latency": 6.0, "duration": 2.0}]


def test_python_regression_eventalign_requires_alignment_inputs():
    with pytest.raises(TypeError):
        eventalign()


def test_python_regression_eventalign_scalar_factor_defaults_to_median():
    assert eventalign(2, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == 0
    assert eventalign([2, 4], [1, 2, 3], [6, 8, 10]) == 0


def test_python_regression_eventalign_matrix_row_minima():
    first = np.array([[1, 2, 3], [4, 5, 6]])
    second = np.array([[6, 7, 8], [9, 10, 11]])
    assert eventalign(2, first, second) == 1
    assert eventalign(2, first, second, "mean") == 1


def test_python_regression_eventlock_multichannel_infers_frames_and_shifts_trials():
    data = np.arange(1, 25).reshape(2, 12)
    output, median_value, shifts = eventlock(data, 0, [1, 2, 3, 4], 2)
    expected = np.array(
        [
            [np.nan, 1, 2, 4, 5, 6, 8, 9, np.nan, 11, 12, np.nan],
            [np.nan, 13, 14, 16, 17, 18, 20, 21, np.nan, 23, 24, np.nan],
        ]
    )
    np.testing.assert_allclose(output, expected, equal_nan=True)
    assert median_value == 2
    np.testing.assert_array_equal(shifts, [1, 0, -1, -1])


def test_python_regression_eventlock_compact_time_axis_and_endpoint_clamping():
    data = np.arange(1, 25).reshape(4, 6)
    output, median_value, shifts = eventlock(data, [-1000, 4, 1], [-2000, -1000, 0, 1000, 2000, 3000])
    expected = np.array(
        [
            [np.nan, np.nan, 3, 10, 17, 18],
            [1, 2, 9, 16, 23, 24],
            [7, 8, 15, 22, np.nan, np.nan],
            [13, 14, 21, np.nan, np.nan, np.nan],
        ]
    )
    np.testing.assert_allclose(output, expected, equal_nan=True)
    assert median_value == 0
    np.testing.assert_array_equal(shifts, [1, 1, 0, -1, -2, -2])
