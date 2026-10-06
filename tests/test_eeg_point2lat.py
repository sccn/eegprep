import numpy as np
from tests.eeglab_tests import assert_matlab_near, eeglab_test


def _matlab_row(x):
    """Convert a 1D numpy array or list into a MATLAB 1xN double."""
    x = np.asarray(x, dtype=float).ravel().tolist()
    return [[v for v in x]]


def _assert_point2lat(eeglab_backend, points, epochs, expected, *, time_limits=(-1000, 2000), time_unit=1 / 1000):
    points = np.asarray(points) if isinstance(points, np.ndarray) else np.asarray(points, dtype=float)
    epochs = np.asarray(epochs) if isinstance(epochs, np.ndarray) else np.asarray(epochs, dtype=float)
    arguments = [np.atleast_2d(points), np.atleast_2d(epochs), 1.0, np.array([time_limits], dtype=float)]
    if time_unit is not None:
        arguments.append(time_unit)
    actual = eeglab_backend("eeg_point2lat", *arguments)
    expected = np.array([expected], dtype=float)
    assert_matlab_near(actual, expected)


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_general")
def test_eeg_point2lat_current_suite_general_case(eeglab_backend):
    _assert_point2lat(
        eeglab_backend, [2, 2.3, 6, 10, 10.4, 14, 14.25], [1, 1, 2, 3, 3, 4, 4], [0, 300, 0, 0, 400, 0, 250]
    )


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_cell_epoch")
def test_eeg_point2lat_current_suite_cell_epoch_case(eeglab_backend):
    epochs = np.array([1, 1, 2, 3, 3, 4, 4], dtype=float).astype(object)
    _assert_point2lat(eeglab_backend, [2, 2.3, 6, 10, 10.4, 14, 14.25], epochs, [0, 300, 0, 0, 400, 0, 250])


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_cell_lat")
def test_eeg_point2lat_current_suite_cell_latency_case(eeglab_backend):
    points = np.array([2, 2.3, 6, 10, 10.4, 14, 14.25], dtype=float).astype(object)
    _assert_point2lat(eeglab_backend, points, [1, 1, 2, 3, 3, 4, 4], [0, 300, 0, 0, 400, 0, 250])


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_no_epoch")
def test_eeg_point2lat_current_suite_empty_epoch_defaults_to_one(eeglab_backend):
    expected = [0, 300, 4000, 8000, 8400, 12000, 12250]
    _assert_point2lat(eeglab_backend, [2, 2.3, 6, 10, 10.4, 14, 14.25], [], expected)


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_one_epoch")
def test_eeg_point2lat_current_suite_single_epoch_broadcast(eeglab_backend):
    expected = [0, 300, 4000, 8000, 8400, 12000, 12250]
    _assert_point2lat(eeglab_backend, [2, 2.3, 6, 10, 10.4, 14, 14.25], [1], expected)


@eeglab_test("unittesting_popfunc/eeg_point2lat/popfunc_eeg_point2lat_wrapperTest.m", "test_pass_std_timeunit")
def test_eeg_point2lat_current_suite_default_time_unit(eeglab_backend):
    _assert_point2lat(
        eeglab_backend,
        [2, 2.3, 6, 10, 10.4, 14, 14.25],
        [1, 1, 2, 3, 3, 4, 4],
        [0, 0.3, 0, 0, 0.4, 0, 0.25],
        time_limits=(-1, 2),
        time_unit=None,
    )
