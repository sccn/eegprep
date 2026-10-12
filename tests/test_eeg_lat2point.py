# test_eeg_lat2point.py
import numpy as np

from tests.eeglab_tests import assert_matlab_near, eeglab_test


def _assert_lat2point(eeglab_backend, latencies, epochs, expected, *, time_limits=(-1000, 2000), time_unit=1 / 1000):
    # Source arrays are MATLAB double rows, including the numeric contents of cells.
    latencies = np.asarray(latencies) if isinstance(latencies, np.ndarray) else np.asarray(latencies, dtype=float)
    epochs = np.asarray(epochs) if isinstance(epochs, np.ndarray) else np.asarray(epochs, dtype=float)
    arguments = [np.atleast_2d(latencies), np.atleast_2d(epochs), 1.0, np.array([time_limits], dtype=float)]
    if time_unit is not None:
        arguments.append(time_unit)
    actual = eeglab_backend("eeg_lat2point", *arguments)
    expected = np.array([expected], dtype=float)
    assert_matlab_near(actual, expected)


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_general")
def test_eeg_lat2point_current_suite_general_case(eeglab_backend):
    _assert_lat2point(
        eeglab_backend, [0, 300, 0, 0, 400, 0, 250], [1, 1, 2, 3, 3, 4, 4], [2, 2.3, 6, 10, 10.4, 14, 14.25]
    )


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_cell_epoch")
def test_eeg_lat2point_current_suite_cell_epoch_case(eeglab_backend):
    epochs = np.array([1, 1, 2, 3, 3, 4, 4], dtype=float).astype(object)
    _assert_lat2point(eeglab_backend, [0, 300, 0, 0, 400, 0, 250], epochs, [2, 2.3, 6, 10, 10.4, 14, 14.25])


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_cell_lat")
def test_eeg_lat2point_current_suite_cell_latency_case(eeglab_backend):
    latencies = np.array([0, 300, 0, 0, 400, 0, 250], dtype=float).astype(object)
    _assert_lat2point(eeglab_backend, latencies, [1, 1, 2, 3, 3, 4, 4], [2, 2.3, 6, 10, 10.4, 14, 14.25])


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_one_epoch")
def test_eeg_lat2point_current_suite_single_epoch_broadcast(eeglab_backend):
    _assert_lat2point(eeglab_backend, [0, 300, 0, 0, 400, 0, 250], [1], [2, 2.3, 2, 2, 2.4, 2, 2.25])


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_outbound")
def test_eeg_lat2point_current_suite_clamps_out_of_range_latency(eeglab_backend):
    _assert_lat2point(
        eeglab_backend,
        [0, 300, 0, 0, 400, 0, 250, 3500],
        [1, 1, 2, 3, 3, 4, 4, 4],
        [2, 2.3, 6, 10, 10.4, 14, 14.25, 16],
    )


@eeglab_test("unittesting_popfunc/eeg_lat2point/popfunc_eeg_lat2point_wrapperTest.m", "test_pass_std_timeunit")
def test_eeg_lat2point_current_suite_default_time_unit(eeglab_backend):
    _assert_lat2point(
        eeglab_backend,
        [0, 0.3, 0, 0, 0.4, 0, 0.25],
        [1, 1, 2, 3, 3, 4, 4],
        [2, 2.3, 6, 10, 10.4, 14, 14.25],
        time_limits=(-1, 2),
        time_unit=None,
    )
