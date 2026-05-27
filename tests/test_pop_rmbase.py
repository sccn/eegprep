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

try:
    from .fixtures import SAMPLE_DATASET_PATH, create_test_eeg
except (ImportError, ValueError):
    from fixtures import SAMPLE_DATASET_PATH, create_test_eeg


def test_rmbase_removes_epoch_baseline_and_returns_means():
    data = np.array(
        [
            [[1.0, 5.0], [3.0, 7.0], [5.0, 9.0]],
            [[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]],
        ]
    )

    out, means = rmbase(data, frames=3, basevector=[1, 2], return_mean=True)

    np.testing.assert_allclose(out[:, :2, :].mean(axis=1), 0)
    np.testing.assert_allclose(means, [[2.0, 6.0], [3.0, 9.0]])
    assert out.shape == data.shape


def test_pop_rmbase_epoched_pointrange_all_channels():
    eeg = create_test_eeg(n_channels=4, n_samples=200, srate=200.0, n_trials=3)
    data_before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(1, 51))

    assert out["data"].shape == (4, 200, 3)
    np.testing.assert_allclose(np.mean(out["data"][:, 0:50, :], axis=1), 0, atol=1e-10)
    assert not np.allclose(out["data"], data_before)
    np.testing.assert_allclose(eeg["data"], data_before)


def test_pop_rmbase_chanlist_subset_uses_one_based_channels():
    eeg = create_test_eeg(n_channels=5, n_samples=100, srate=100.0, n_trials=2)
    data_before = eeg["data"].copy()

    out = pop_rmbase(eeg, pointrange=range(1, 31), chanlist=[2, 4])

    np.testing.assert_allclose(np.mean(out["data"][[1, 3], 0:30, :], axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(out["data"][[0, 2, 4]], data_before[[0, 2, 4]])


def test_pop_rmbase_rejects_zero_based_channel_indices():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=1)

    with pytest.raises(ValueError, match="1-based"):
        pop_rmbase(eeg, pointrange=range(1, 11), chanlist=[0])


def test_pop_rmbase_continuous_no_boundaries():
    eeg = create_test_eeg(n_channels=3, n_samples=300, srate=150.0, n_trials=1)

    out = pop_rmbase(eeg, pointrange=range(51, 251))

    assert out["data"].ndim == 2
    np.testing.assert_allclose(np.mean(out["data"][:, 50:250], axis=1), 0, atol=1e-10)


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


def test_pop_rmbase_timerange_uses_eeg_times_units():
    eeg = create_test_eeg(n_channels=2, n_samples=100, srate=100.0, n_trials=1)
    eeg["times"] = np.arange(100, dtype=float)

    out = pop_rmbase(eeg, timerange=[10, 49])

    np.testing.assert_allclose(np.mean(out["data"][:, 10:50], axis=1), 0, atol=1e-10)


def test_pop_rmbase_bad_timerange_raises():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=1)
    eeg["times"] = np.arange(50, dtype=float)

    with pytest.raises(ValueError, match="Bad time range"):
        pop_rmbase(eeg, timerange=[-1.0, 999.0])


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


def test_pop_rmbase_return_com_preserves_channel_order():
    eeg = create_test_eeg(n_channels=3, n_samples=40, srate=100.0, n_trials=1)

    _out, command = pop_rmbase(eeg, pointrange=range(1, 6), chanlist=[3, 1], return_com=True)

    assert command == "EEG = pop_rmbase( EEG, [], [1 2 3 4 5], [3 1]);"
    converted = _console_python_command(command)
    assert converted == "EEG = pop_rmbase(EEG, timerange=[], pointrange=[1, 2, 3, 4, 5], chanlist=[3, 1])"


def test_pop_rmbase_return_com_timerange_uses_eeglab_history_shape():
    eeg = create_test_eeg(n_channels=2, n_samples=40, srate=100.0, n_trials=1)
    eeg["times"] = np.arange(40, dtype=float)

    _out, command = pop_rmbase(eeg, timerange=[10, 19], return_com=True)

    assert command == "EEG = pop_rmbase( EEG, [10 19], []);"
    converted = _console_python_command(command)
    assert converted == "EEG = pop_rmbase(EEG, timerange=[10, 19], pointrange=[])"


def test_pop_rmbase_gui_cancel_returns_original_dataset():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=1)

    out, command = pop_rmbase(eeg, gui=True, renderer=_CancelRenderer(), return_com=True)

    assert out is eeg
    assert command == ""


def test_pop_rmbase_dialog_disables_channel_controls_for_multiple_datasets():
    eeg = create_test_eeg(n_channels=2, n_samples=50, srate=50.0, n_trials=2)

    spec = pop_rmbase_dialog_spec(eeg, multiple=True)
    controls = {control.tag: control for control in spec.controls if control.tag}

    assert controls["chantypes"].enabled is False
    assert controls["channels"].enabled is False
    assert controls["chantypes_button"].enabled is False
    assert controls["channels_button"].enabled is False


def test_pop_rmbase_sample_data_zeroes_selected_baseline_channels_without_warnings():
    eeg = pop_loadset(SAMPLE_DATASET_PATH)

    out, command = pop_rmbase(eeg, pointrange=range(1, 21), chanlist=[1, 2], return_com=True)

    assert out["data"].shape == eeg["data"].shape
    np.testing.assert_allclose(np.nanmean(out["data"][0, :20]), 0, atol=1e-5)
    np.testing.assert_allclose(np.nanmean(out["data"][1, :20]), 0, atol=1e-5)
    np.testing.assert_allclose(out["data"][2], eeg["data"][2])
    assert command == ("EEG = pop_rmbase( EEG, [], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20], [1 2]);")


class _CancelRenderer:
    def run(self, _spec, initial_values=None):
        del initial_values
        return None


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopRmbaseParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")
        self.eeg = pop_loadset(SAMPLE_DATASET_PATH)

    def test_parity_pointrange_all_channels(self):
        pointrange = list(range(1, 51))

        py_eeg = pop_rmbase(copy.deepcopy(self.eeg), pointrange=pointrange)
        ml_eeg = self.eeglab.pop_rmbase(copy.deepcopy(self.eeg), [], pointrange, [])

        self.assertEqual(py_eeg["data"].shape, ml_eeg["data"].shape)
        np.testing.assert_allclose(py_eeg["data"], ml_eeg["data"], atol=1e-6, rtol=1e-6)

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
