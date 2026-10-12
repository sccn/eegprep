from __future__ import annotations

import copy
import os
import unittest

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase, pop_rmbase_dialog_spec
from eegprep.functions.sigprocfunc.rmbase import rmbase
from tests.eeglab_tests import assert_matlab_near, eeglab_test

try:
    from .fixtures import SAMPLE_DATASET_PATH, create_test_eeg
except (ImportError, ValueError):
    from fixtures import SAMPLE_DATASET_PATH, create_test_eeg


@eeglab_test("unittesting_sigprocfunc/rmbase/sigprocfunc_rmbase_wrapperTest.m", "test_test_rmbase")
def test_reference_rmbase(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    output, _mean = eeglab_backend("rmbase", eeg["data"], nargout=2)
    assert_matlab_near([output.shape], [eeg["data"].shape])
    for arguments in ((384.0,), (192.0,), (384.0, np.arange(1.0, 129.0)[None, :])):
        eeglab_backend("rmbase", eeg["data"], *arguments, nargout=2)
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    eeglab_backend("rmbase", eeg["data"], nargout=2)
    eeglab_backend("rmbase", eeg["data"], 30504.0, nargout=2)
    for arguments in ((3813.0,), (3813.0, np.arange(1.0, 1001.0)[None, :])):
        _output, mean = eeglab_backend("rmbase", eeg["data"], *arguments, nargout=2)
        assert_matlab_near([mean.shape], [[32, 8]])


def _legacy_rmbase(
    data: np.ndarray,
    frames: int,
    basevector: list[int] | int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch loop used before rmbase was vectorized; kept as the numerical reference."""
    array = np.asarray(data)
    original_shape = array.shape
    matrix = array.transpose(0, 2, 1).reshape(array.shape[0], -1) if array.ndim == 3 else array
    channels, total_frames = matrix.shape
    epochs = total_frames // frames
    baseline = None if basevector == 0 else np.asarray(basevector, dtype=int) - 1

    output = matrix.astype(np.float64, copy=True) if not np.issubdtype(matrix.dtype, np.floating) else matrix.copy()
    means = np.zeros((channels, epochs), dtype=np.result_type(matrix.dtype, np.float64))
    for epoch in range(epochs):
        start = epoch * frames
        stop = start + frames
        if baseline is None:
            mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True, dtype=np.float64)
        else:
            mean = np.nanmean(matrix[:, start + baseline], axis=1, keepdims=True, dtype=np.float64)
        means[:, epoch : epoch + 1] = mean
        output[:, start:stop] = output[:, start:stop] - mean

    if array.ndim == 3:
        output = output.reshape(original_shape[0], original_shape[2], original_shape[1]).transpose(0, 2, 1)
    return output.reshape(original_shape), means


@pytest.mark.parametrize("shape", [(3, 85), (3, 17, 5)])
@pytest.mark.parametrize("basevector", [0, [1, 4, 7, 11]])
def test_rmbase_float32_matches_legacy_rounding(shape: tuple[int, ...], basevector: list[int] | int):
    rng = np.random.default_rng(268)
    data = (rng.standard_normal(shape) * 1_000).astype(np.float32)
    original = data.copy()

    expected, expected_means = _legacy_rmbase(data, frames=17, basevector=basevector)
    actual, actual_means = rmbase(data, frames=17, basevector=basevector, return_mean=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
    np.testing.assert_array_equal(data, original)
    assert actual.dtype == np.float32
    assert actual_means.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float64, np.int16])
def test_rmbase_matches_legacy_for_other_dtypes(dtype: type[np.generic]):
    rng = np.random.default_rng(269)
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(-2_000, 2_000, size=(4, 13, 3), dtype=dtype)
    else:
        data = (rng.standard_normal((4, 13, 3)) * 1_000).astype(dtype)

    expected, expected_means = _legacy_rmbase(data, frames=13, basevector=[2, 5, 9])
    actual, actual_means = rmbase(data, frames=13, basevector=[2, 5, 9], return_mean=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
    assert actual.dtype == (np.dtype(np.float64) if np.issubdtype(dtype, np.integer) else dtype)


def test_rmbase_preserves_nan_results_and_warning_behavior():
    data = np.arange(16, dtype=np.float32).reshape(2, 8)
    data[0, :2] = np.nan
    data[1, 4:6] = np.nan

    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        actual, actual_means = rmbase(data, frames=4, basevector=[1, 2], return_mean=True)
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        expected, expected_means = _legacy_rmbase(data, frames=4, basevector=[1, 2])

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)


def test_pop_rmbase_chanlist_subset_uses_one_based_channels():
    eeg = create_test_eeg(n_channels=5, n_samples=100, srate=100.0, n_trials=2)
    data_before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(1, 31), chanlist=[2, 4])

    np.testing.assert_allclose(np.mean(out["data"][[1, 3], 0:30, :], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(out["data"][[0, 2, 4]], data_before[[0, 2, 4]])


def test_pop_rmbase_continuous_with_boundaries_is_segmentwise():
    eeg = create_test_eeg(n_channels=2, n_samples=200, srate=200.0, n_trials=1)
    eeg["data"] = np.vstack(
        [
            np.r_[np.arange(50) + 10.0, np.arange(100) + 100.0, np.arange(50) - 20.0],
            np.r_[np.arange(50) - 5.0, np.arange(100) + 20.0, np.arange(50) + 70.0],
        ]
    )
    eeg["event"] = [
        {"type": "boundary", "latency": 50.5},
        {"type": "boundary", "latency": 150.5},
    ]

    out = pop_rmbase(eeg, pointrange=range(1, 201))

    for start, stop in [(0, 50), (50, 150), (150, 200)]:
        np.testing.assert_allclose(np.mean(out["data"][:, start:stop], axis=1), 0, atol=1e-10)


def test_pop_rmbase_continuous_boundary_partial_pointrange_only_changes_selected_samples():
    eeg = create_test_eeg(n_channels=2, n_samples=100, srate=100.0, n_trials=1)
    eeg["data"] = np.vstack([np.arange(100, dtype=float), np.arange(100, dtype=float) + 100.0])
    eeg["event"] = [{"type": "boundary", "latency": 50.5}]
    before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(1, 31))

    np.testing.assert_allclose(np.mean(out["data"][:, :30], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(out["data"][:, 30:], before[:, 30:])


def test_pop_rmbase_continuous_boundary_pointrange_can_span_boundary():
    eeg = create_test_eeg(n_channels=1, n_samples=60, srate=100.0, n_trials=1)
    eeg["data"] = np.arange(60, dtype=float)[np.newaxis, :]
    eeg["event"] = [{"type": "boundary", "latency": 30.5}]
    before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(21, 41))

    np.testing.assert_allclose(np.mean(out["data"][:, 20:30], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(np.mean(out["data"][:, 30:40], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(out["data"][:, :20], before[:, :20])
    np.testing.assert_allclose(out["data"][:, 40:], before[:, 40:])


def test_pop_rmbase_continuous_boundary_segment_without_baseline_is_unchanged():
    eeg = create_test_eeg(n_channels=1, n_samples=90, srate=100.0, n_trials=1)
    eeg["data"] = np.arange(90, dtype=float)[np.newaxis, :]
    eeg["event"] = [
        {"type": "boundary", "latency": 30.5},
        {"type": "boundary", "latency": 60.5},
    ]
    before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(1, 21))

    np.testing.assert_allclose(np.mean(out["data"][:, :20], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(out["data"][:, 20:], before[:, 20:])


def test_pop_rmbase_clears_icaact_and_preserves_decomposition_fields():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=2)
    eeg["icaact"] = np.random.randn(2, 50, 2)
    eeg["icaweights"] = np.eye(2)
    eeg["icasphere"] = np.eye(2)
    eeg["icawinv"] = np.eye(2)
    eeg["icachansind"] = np.arange(2)

    out = pop_rmbase(eeg, pointrange=range(1, 11))

    assert out["icaact"].size == 0
    np.testing.assert_allclose(out["icaweights"], np.eye(2))
    np.testing.assert_allclose(out["icasphere"], np.eye(2))
    np.testing.assert_allclose(out["icawinv"], np.eye(2))
    np.testing.assert_array_equal(out["icachansind"], np.arange(2))


def test_pop_rmbase_return_com_is_replayable_python_console_input():
    eeg = create_test_eeg(n_channels=3, n_samples=40, srate=100.0, n_trials=1)

    _out, command = pop_rmbase(eeg, pointrange=range(1, 6), chanlist=[1, 2], return_com=True)

    assert command == "EEG = pop_rmbase( EEG, [], [1 2 3 4 5], [1 2]);"
    converted = _console_python_command(command)
    assert converted == "EEG = pop_rmbase(EEG, timerange=[], pointrange=[1, 2, 3, 4, 5], chanlist=[1, 2])"


def test_pop_rmbase_dialog_accepts_numpy_chanlocs():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=2)
    eeg["chanlocs"] = np.asarray(
        [
            {"labels": "Cz", "type": "EEG"},
            {"labels": "EOG", "type": "EOG"},
        ],
        dtype=object,
    )

    spec = pop_rmbase_dialog_spec(eeg)
    controls = {control.tag: control for control in spec.controls if control.tag}

    assert controls["chantypes_button"].callback.params["channels"] == ["EEG", "EOG"]
    assert controls["channels_button"].callback.params["channels"] == ["Cz", "EOG"]


@eeglab_test("unittesting_popfunc/pop_rmbase/popfunc_pop_rmbase_wrapperTest.m", "test_test_pop_rmbase")
def test_reference_pop_rmbase_original_baselines_and_complete_commands(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    data = eeg["data"][0, :, 1]
    commands = []
    for arguments, baseline_stop in (
        ((np.array([[-1000.0, 0.0]]),), 129),
        ((np.empty((0, 0)), np.arange(1.0, 51.0)[None, :]), 50),
        ((np.empty((0, 0)), np.empty((0, 0))), data.size),
        ((np.array([[-1000.0, 0.0]]), np.arange(1.0, 52.0)[None, :]), 129),
    ):
        output, command = eeglab_backend("pop_rmbase", eeg, *arguments, nargout=2)
        # Preserve the source's double mean through subtraction, rounding only
        # the result to the single-precision recording's type.
        mean = np.mean(data[:baseline_stop], dtype=np.float64)
        expected = (data.astype(np.float64) - mean).astype(data.dtype)
        np.testing.assert_array_equal(output["data"][0, :, 1], expected)
        commands.append(command)

    assert tuple(commands) in (
        (
            "EEG = pop_rmbase( EEG, [-1000 0] ,[],[]);",
            "EEG = pop_rmbase( EEG, [],[1:50] ,[]);",
            "EEG = pop_rmbase( EEG, [],[],[]);",
            "EEG = pop_rmbase( EEG, [-1000 0] ,[],[]);",
        ),
        (
            "EEG = pop_rmbase( EEG, [-1000 0] ,[]);",
            "EEG = pop_rmbase( EEG, [],[1:50] );",
            "EEG = pop_rmbase( EEG, [],[]);",
            "EEG = pop_rmbase( EEG, [-1000 0] ,[]);",
        ),
    )


def test_pop_rmbase_current_suite_time_point_and_whole_epoch_baselines():
    eeg = pop_loadset("sample_data/eeglab_data_epochs_ica.set")

    by_time, time_command = pop_rmbase(eeg, [-1000, 0], return_com=True)
    by_point, point_command = pop_rmbase(eeg, [], range(1, 51), return_com=True)
    whole_epoch, whole_command = pop_rmbase(eeg, [], [], return_com=True)
    time_wins = pop_rmbase(eeg, [-1000, 0], range(1, 52))

    times = np.asarray(eeg["times"])
    baseline = np.flatnonzero((times >= -1000) & (times <= 0))
    data = eeg["data"][0, :, 1]
    expected_time = (data.astype(np.float64) - np.mean(data[baseline], dtype=np.float64)).astype(data.dtype)
    expected_point = (data.astype(np.float64) - np.mean(data[:50], dtype=np.float64)).astype(data.dtype)
    expected_whole = (data.astype(np.float64) - np.mean(data, dtype=np.float64)).astype(data.dtype)
    np.testing.assert_array_equal(by_time["data"][0, :, 1], expected_time)
    np.testing.assert_array_equal(by_point["data"][0, :, 1], expected_point)
    np.testing.assert_array_equal(whole_epoch["data"][0, :, 1], expected_whole)
    np.testing.assert_array_equal(time_wins["data"], by_time["data"])
    assert "[-1000 0]" in time_command
    assert "[1 2 3" in point_command
    assert "[], []" in whole_command


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopRmbaseParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")
        self.eeg = pop_loadset(SAMPLE_DATASET_PATH)

    def test_parity_chanlist_subset(self):
        pointrange = list(range(1, 31))
        chanlist = [1, 2, 3]

        py_eeg = pop_rmbase(copy.deepcopy(self.eeg), pointrange=pointrange, chanlist=chanlist)
        ml_eeg = self.eeglab.pop_rmbase(copy.deepcopy(self.eeg), [], pointrange, chanlist)

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=1e-6, rtol=1e-6)

    def test_parity_continuous_boundary_partial_pointrange(self):
        eeg = create_test_eeg(n_channels=2, n_samples=100, srate=100.0, n_trials=1)
        eeg["data"] = np.vstack([np.arange(100, dtype=float), np.arange(100, dtype=float) + 100.0])
        eeg["event"] = [{"type": "boundary", "latency": 50.5}]
        pointrange = list(range(1, 31))

        py_eeg = pop_rmbase(copy.deepcopy(eeg), pointrange=pointrange)
        ml_eeg = self.eeglab.pop_rmbase(copy.deepcopy(eeg), [], pointrange, [])

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
