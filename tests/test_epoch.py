# test_epoch.py
import os
import numpy as np
import unittest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.sigprocfunc.epoch import epoch  # Python translation under test
from tests.eeglab_tests import eeglab_test


def _ml_list_of_arrays_to_0_based(list_of_arrays):
    # MATLAB returns 1-based indices; convert to 0-based for parity checks
    out = []
    # Handle the nested structure from MATLAB
    if isinstance(list_of_arrays, np.ndarray) and list_of_arrays.dtype == object:
        # Flatten the outer structure and process each epoch's events
        for epoch_events in list_of_arrays.flatten():
            if isinstance(epoch_events, np.ndarray):
                # Convert each array in the epoch to 0-based
                converted = np.asarray(epoch_events).flatten().astype(int) - 1
                out.append(converted)
            else:
                out.append(np.array([]))
    else:
        # Original logic for simpler structures
        for arr in list_of_arrays:
            a = np.asarray(arr).astype(int) - 1
            out.append(a)
    return out


def _upstream_epoch_data():
    return np.arange(1, 41, dtype=float).reshape(2, 20)


@eeglab_test(
    "unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m",
    "test_pass_general",
)
@eeglab_test(
    "unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m",
    "test_pass_no_srate",
)
def test_epoch_upstream_default_and_explicit_sampling_rate_match_exact_slices():
    data = _upstream_epoch_data()
    events = [2, 5, 6, 9, 12, 18]
    expected = np.stack([data[:, event - 2 : event + 2] for event in events], axis=2)

    explicit = epoch(data, events, [-1, 3], srate=1, verbose="off")
    default = epoch(data, events, [-1, 3], verbose="off")

    for result in (explicit, default):
        epoched, newtime, indices, allevents, latencies, reallim = result
        np.testing.assert_array_equal(epoched, expected)
        np.testing.assert_array_equal(newtime, [-1, 2])
        np.testing.assert_array_equal(indices, np.arange(6))
        assert allevents == []
        assert latencies == []
        np.testing.assert_array_equal(reallim, [-1, 2])


@eeglab_test(
    "unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m",
    "test_pass_boundary",
)
def test_epoch_upstream_boundary_case_drops_only_out_of_bounds_window():
    data = _upstream_epoch_data()

    epoched, newtime, indices, *_ = epoch(data, [2, 5, 6, 9, 12, 19], [-1, 3], srate=1, verbose="off")

    np.testing.assert_array_equal(
        epoched, np.stack([data[:, event - 2 : event + 2] for event in [2, 5, 6, 9, 12]], axis=2)
    )
    np.testing.assert_array_equal(newtime, [-1, 2])
    np.testing.assert_array_equal(indices, np.arange(5))


@eeglab_test(
    "unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m",
    "test_pass_valuelim",
)
def test_epoch_upstream_value_limits_filter_on_all_channels():
    data = _upstream_epoch_data()

    epoched, newtime, indices, *_ = epoch(
        data,
        [2, 5, 6, 9, 12, 18],
        [-1, 3],
        srate=1,
        valuelim=[7, 35],
        verbose="off",
    )

    expected = np.stack([data[:, 7:11], data[:, 10:14]], axis=2)
    np.testing.assert_array_equal(epoched, expected)
    np.testing.assert_array_equal(newtime, [-1, 2])
    np.testing.assert_array_equal(indices, [3, 4])


@eeglab_test(
    "unittesting_sigprocfunc/epoch/sigprocfunc_epoch_wrapperTest.m",
    "test_pass_allevents",
)
def test_epoch_upstream_rereferences_all_events_with_zero_based_indices():
    data = _upstream_epoch_data()

    epoched, newtime, indices, event_indices, event_latencies, _ = epoch(
        data,
        [2, 5, 6, 9, 12, 18],
        [-1, 3],
        srate=1,
        allevents=[2, 5, 6, 9, 10, 12, 17, 18],
        verbose="off",
    )

    np.testing.assert_array_equal(
        epoched, np.stack([data[:, event - 2 : event + 2] for event in [2, 5, 6, 9, 12, 18]], axis=2)
    )
    np.testing.assert_array_equal(newtime, [-1, 2])
    np.testing.assert_array_equal(indices, np.arange(6))
    expected_indices = [[0], [1, 2], [1, 2], [3, 4], [5], [6, 7]]
    expected_latencies = [[0], [0, 1], [-1, 0], [0, 1], [0], [-1, 0]]
    for actual, expected in zip(event_indices, expected_indices):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(event_latencies, expected_latencies):
        np.testing.assert_array_equal(actual, expected)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEpochParity(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.eeglab = get_eeglab('MAT')

    def test_parity_continuous_basic(self):
        # 2 channels, 1000 samples, 100 Hz
        srate = 100.0
        n_ch, n_samp = 2, 1000
        data = np.random.randn(n_ch, n_samp)
        # Events in seconds
        events = np.array([2.0, 5.0, 7.5], dtype=float)
        lim = np.array([-0.2, 0.5], dtype=float)  # seconds

        py = epoch(data, events, lim, srate=srate, verbose='off')
        ml = self.eeglab.epoch(data, events, lim, 'srate', srate, 'verbose', 'off')

        py_epochdat, py_newtime, py_indexes, py_alleventout, py_alllatencyout, py_reallim = py
        ml_epochdat, ml_newtime, ml_indexes, ml_alleventout, ml_alllatencyout, ml_reallim = ml

        # MATLAB indexes are 1-based; convert to 0-based for comparison
        ml_indexes0 = np.asarray(ml_indexes).astype(int).flatten() - 1  # flatten to 1D

        self.assertTrue(np.allclose(py_epochdat, ml_epochdat, atol=1e-12))
        self.assertTrue(np.allclose(py_newtime, ml_newtime, atol=1e-12))
        self.assertTrue(np.array_equal(py_indexes, ml_indexes0))
        self.assertTrue(np.allclose(py_reallim, ml_reallim, atol=1e-12))

        # Rereferencing not requested here
        self.assertEqual(len(py_alleventout), 0)
        self.assertEqual(len(py_alllatencyout), 0)

    def test_parity_valuelim_filter(self):
        srate = 100.0
        n_ch, n_samp = 1, 1000
        data = np.zeros((n_ch, n_samp))
        # Make the second epoch violate valuelim by inserting a large artifact
        data[0, 600:650] = 1e3
        events = np.array([2.0, 6.2], dtype=float)  # seconds
        lim = np.array([-0.1, 0.4], dtype=float)

        valuelim = np.array([-50.0, 50.0], dtype=float)

        py = epoch(data, events, lim, srate=srate, valuelim=valuelim, verbose='off')
        ml = self.eeglab.epoch(data, events, lim, 'srate', srate, 'valuelim', valuelim, 'verbose', 'off')

        py_epochdat, py_newtime, py_indexes, _, _, py_reallim = py
        ml_epochdat, ml_newtime, ml_indexes, _, _, ml_reallim = ml

        ml_indexes0 = np.asarray(ml_indexes).astype(int).flatten() - 1  # flatten to 1D
        self.assertTrue(np.allclose(py_epochdat, ml_epochdat, atol=1e-12))
        self.assertTrue(np.allclose(py_newtime, ml_newtime, atol=1e-12))
        self.assertTrue(np.array_equal(py_indexes, ml_indexes0))
        self.assertTrue(np.allclose(py_reallim, ml_reallim, atol=1e-12))

        # Expect only the first event to survive
        self.assertTrue(np.array_equal(py_indexes, np.array([0])))

    def test_parity_rereference_allevents(self):
        srate = 100.0
        n_ch, n_samp = 2, 2000
        data = np.random.randn(n_ch, n_samp)
        events = np.array([5.0, 10.0], dtype=float)
        lim = np.array([-0.2, 0.8], dtype=float)

        # Define all events in seconds on the same scale as 'events'
        allevents = np.array([4.7, 4.9, 5.1, 5.6, 9.1, 9.85, 10.05], dtype=float)
        alleventrange = np.array([-0.1, 0.3], dtype=float)

        py = epoch(data, events, lim, srate=srate, allevents=allevents, alleventrange=alleventrange, verbose='off')
        ml = self.eeglab.epoch(
            data, events, lim, 'srate', srate, 'allevents', allevents, 'alleventrange', alleventrange, 'verbose', 'off'
        )

        py_epochdat, py_newtime, py_indexes, py_alleventout, py_alllatencyout, py_reallim = py
        ml_epochdat, ml_newtime, ml_indexes, ml_alleventout, ml_alllatencyout, ml_reallim = ml

        ml_indexes0 = np.asarray(ml_indexes).astype(int).flatten() - 1  # flatten to 1D
        self.assertTrue(np.allclose(py_epochdat, ml_epochdat, atol=1e-12))
        self.assertTrue(np.allclose(py_newtime, ml_newtime, atol=1e-12))
        self.assertTrue(np.array_equal(py_indexes, ml_indexes0))
        self.assertTrue(np.allclose(py_reallim, ml_reallim, atol=1e-12))

        # Compare rereferenced indices and latencies
        # Convert MATLAB's 1-based event indices to 0-based
        ml_alleventout0 = _ml_list_of_arrays_to_0_based(ml_alleventout)

        # Handle MATLAB's nested structure for alllatencyout (similar to alleventout)
        ml_alllatencyout_flat = []
        if isinstance(ml_alllatencyout, np.ndarray) and ml_alllatencyout.dtype == object:
            for epoch_latencies in ml_alllatencyout.flatten():
                if isinstance(epoch_latencies, np.ndarray):
                    flattened = np.asarray(epoch_latencies).flatten()
                    ml_alllatencyout_flat.append(flattened)
                else:
                    ml_alllatencyout_flat.append(np.array([]))
        else:
            ml_alllatencyout_flat = list(ml_alllatencyout)

        self.assertEqual(len(py_alleventout), len(ml_alleventout0))
        self.assertEqual(len(py_alllatencyout), len(ml_alllatencyout_flat))

        for i in range(len(py_alleventout)):
            self.assertTrue(np.array_equal(py_alleventout[i], np.asarray(ml_alleventout0[i])))
            self.assertTrue(np.allclose(py_alllatencyout[i], np.asarray(ml_alllatencyout_flat[i]), atol=1e-12))

    def test_parity_boundary_exclusion(self):
        # Place an event whose window crosses dataset boundary
        srate = 100.0
        n_ch, n_samp = 1, 500
        data = np.random.randn(n_ch, n_samp)
        events = np.array([0.1, 5.0], dtype=float)  # second event is near end
        lim = np.array([-0.2, 0.5], dtype=float)

        py = epoch(data, events, lim, srate=srate, verbose='off')
        ml = self.eeglab.epoch(data, events, lim, 'srate', srate, 'verbose', 'off')

        _, _, py_indexes, _, _, _ = py
        _, _, ml_indexes, _, _, _ = ml
        ml_indexes0 = np.asarray(ml_indexes).astype(int).flatten() - 1  # flatten to 1D

        self.assertTrue(np.array_equal(py_indexes, ml_indexes0))


class TestEpochFunctional(unittest.TestCase):
    def test_functional_3d_epoched_input_same_epoch_constraint(self):
        # data shaped (chan, frames, epochs) = (1, 100, 3)
        # time window must remain within the same pre-existing epoch
        srate = 100.0
        data = np.zeros((1, 100, 3))
        # Put a distinctive ramp in epoch 2 so we can detect correct slicing
        data[0, :, 1] = np.linspace(0, 1, 100)

        # Event at 1.5 s relative to the concatenated stream:
        # global sample for event center = floor(1.5 * 100) = 150
        # Epoch boundaries every 100 points, so the window stays inside epoch 2.
        events = np.array([1.5], dtype=float)
        lim = np.array([-0.2, 0.3], dtype=float)  # [-20, +29] samples window inside epoch 2

        ep, newtime, idx, _, _, _ = epoch(data, events, lim, srate=srate, verbose='off')

        # Expect one accepted epoch
        self.assertTrue(np.array_equal(idx, np.array([0])))
        # The extracted data should match the correct slice from linearized data.
        # MATLAB epoch.m uses data(:,posinit:posend) (1-based); Python slices [posinit-1:posend].
        pos0 = int(np.floor(events[0] * srate))  # 150
        reallim0 = int(np.round(lim[0] * srate))  # -20
        reallim1 = int(np.round(lim[1] * srate - 1))  # 29
        posinit = pos0 + reallim0  # 130 (MATLAB 1-based column)
        posend = pos0 + reallim1  # 179 (MATLAB 1-based column)

        # MATLAB slicing: posinit:posend (1-based) becomes [posinit-1:posend] (0-based)
        start_global = posinit - 1  # 129 (Python 0-based)
        end_global = posend  # 179 (Python 0-based exclusive)

        # Extract the expected slice from linearized data (Fortran order)
        data_linearized = data.reshape(1, -1, order='F')
        expected = data_linearized[0, start_global:end_global]
        self.assertTrue(np.allclose(ep[0, :, 0], expected, atol=1e-12))
        # newtime should reflect limits divided by srate with the -1 sample convention
        self.assertTrue(np.allclose(newtime, np.array([lim[0], np.round(lim[1] * srate - 1) / srate]), atol=1e-12))

        # A window that straddles the boundary between two source epochs (event at
        # 1.2 s spans columns 99..148, crossing the 100-sample epoch boundary) must
        # be rejected, matching EEGLAB's floor((posinit-1)/dataframes) test.
        _, _, idx_cross, _, _, _ = epoch(data, np.array([1.2]), lim, srate=srate, verbose='off')
        self.assertEqual(idx_cross.size, 0)

    def test_functional_valuelim_pass_all(self):
        srate = 200.0
        n_ch, n_samp = 3, 4000
        data = 1e-3 * np.random.randn(n_ch, n_samp)  # small amplitude noise
        events = np.array([2.0, 10.0, 15.0], dtype=float)
        lim = np.array([-0.25, 0.25], dtype=float)
        valuelim = np.array([-1e-2, 1e-2], dtype=float)

        ep, newtime, idx, _, _, _ = epoch(data, events, lim, srate=srate, valuelim=valuelim, verbose='off')
        self.assertTrue(np.array_equal(idx, np.arange(len(events))))

    def test_functional_no_allevents_outputs_empty_lists(self):
        srate = 100.0
        data = np.random.randn(2, 1000)
        events = np.array([2.0], dtype=float)
        lim = np.array([-0.1, 0.2], dtype=float)

        _, _, _, alleventout, alllatencyout, _ = epoch(data, events, lim, srate=srate, verbose='off')
        self.assertEqual(len(alleventout), 0)
        self.assertEqual(len(alllatencyout), 0)

    def test_boundary_first_last_sample(self):
        # data values equal their 0-based column index so we can verify exact slices
        srate = 100.0
        data = np.arange(500, dtype=float).reshape(1, 500)
        lim = np.array([-0.2, 0.5], dtype=float)  # reallim = [-20, 49] samples

        # Event whose window would start one sample before the data (posinit==0,
        # 0-based) must be rejected like EEGLAB, not produce a negative-index slice.
        _, _, idx_before, _, _, _ = epoch(data, np.array([0.2]), lim, srate=srate, verbose='off')
        self.assertEqual(idx_before.size, 0)

        # Event whose window starts exactly at the first sample (posinit==1, 1-based)
        # must be accepted and yield the true leading samples.
        ep_first, _, idx_first, _, _, _ = epoch(data, np.array([0.21]), lim, srate=srate, verbose='off')
        self.assertTrue(np.array_equal(idx_first, np.array([0])))
        self.assertTrue(np.allclose(ep_first[0, :3, 0], np.array([0.0, 1.0, 2.0])))

        # Event whose window ends exactly at the last sample (posend==datawidth)
        # must be accepted; the previous off-by-one rejected it.
        ep_last, _, idx_last, _, _, _ = epoch(data, np.array([4.51]), lim, srate=srate, verbose='off')
        self.assertTrue(np.array_equal(idx_last, np.array([0])))
        self.assertTrue(np.allclose(ep_last[0, -3:, 0], np.array([497.0, 498.0, 499.0])))


if __name__ == '__main__':
    unittest.main()
