# test_epoch.py
import numpy as np
import unittest

from eegprep.eeglabcompat import get_eeglab
from eegprep.epoch import epoch  # Python translation under test


def _ml_list_of_arrays_to_0_based(list_of_arrays):
    # MATLAB returns 1-based indices; convert to 0-based for parity checks
    out = []
    for arr in list_of_arrays:
        a = np.asarray(arr).astype(int) - 1
        out.append(a)
    return out


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

#     def test_parity_rereference_allevents(self):
#         srate = 100.0
#         n_ch, n_samp = 2, 2000
#         data = np.random.randn(n_ch, n_samp)
#         events = np.array([5.0, 10.0], dtype=float)
#         lim = np.array([-0.2, 0.8], dtype=float)

#         # Define all events in seconds on the same scale as 'events'
#         allevents = np.array([4.7, 4.9, 5.1, 5.6, 9.1, 9.85, 10.05], dtype=float)
#         alleventrange = np.array([-0.1, 0.3], dtype=float)

#         py = epoch(
#             data, events, lim,
#             srate=srate,
#             allevents=allevents,
#             alleventrange=alleventrange,
#             verbose='off'
#         )
#         ml = self.eeglab.epoch(
#             data, events, lim,
#             'srate', srate,
#             'allevents', allevents,
#             'alleventrange', alleventrange,
#             'verbose', 'off'
#         )

#         py_epochdat, py_newtime, py_indexes, py_alleventout, py_alllatencyout, py_reallim = py
#         ml_epochdat, ml_newtime, ml_indexes, ml_alleventout, ml_alllatencyout, ml_reallim = ml

#         ml_indexes0 = np.asarray(ml_indexes).astype(int) - 1
#         self.assertTrue(np.allclose(py_epochdat, ml_epochdat, atol=1e-12))
#         self.assertTrue(np.allclose(py_newtime, ml_newtime, atol=1e-12))
#         self.assertTrue(np.array_equal(py_indexes, ml_indexes0))
#         self.assertTrue(np.allclose(py_reallim, ml_reallim, atol=1e-12))

#         # Compare rereferenced indices and latencies
#         # Convert MATLAB's 1-based event indices to 0-based
#         ml_alleventout0 = _ml_list_of_arrays_to_0_based(ml_alleventout)

#         self.assertEqual(len(py_alleventout), len(ml_alleventout0))
#         self.assertEqual(len(py_alllatencyout), len(ml_alllatencyout))

#         for i in range(len(py_alleventout)):
#             self.assertTrue(np.array_equal(py_alleventout[i], np.asarray(ml_alleventout0[i])))
#             self.assertTrue(np.allclose(py_alllatencyout[i], np.asarray(ml_alllatencyout[i]), atol=1e-12))

#     def test_parity_boundary_exclusion(self):
#         # Place an event whose window crosses dataset boundary
#         srate = 100.0
#         n_ch, n_samp = 1, 500
#         data = np.random.randn(n_ch, n_samp)
#         events = np.array([0.1, 5.0], dtype=float)  # second event is near end
#         lim = np.array([-0.2, 0.5], dtype=float)

#         py = epoch(data, events, lim, srate=srate, verbose='off')
#         ml = self.eeglab.epoch(data, events, lim, 'srate', srate, 'verbose', 'off')

#         _, _, py_indexes, _, _, _ = py
#         _, _, ml_indexes, _, _, _ = ml
#         ml_indexes0 = np.asarray(ml_indexes).astype(int) - 1

#         self.assertTrue(np.array_equal(py_indexes, ml_indexes0))


class TestEpochFunctional(unittest.TestCase):

    def test_functional_3d_epoched_input_same_epoch_constraint(self):
        # data shaped (chan, frames, epochs) = (1, 100, 3)
        # time window must remain within the same pre-existing epoch
        srate = 100.0
        data = np.zeros((1, 100, 3))
        # Put a distinctive ramp in epoch 2 so we can detect correct slicing
        data[0, :, 1] = np.linspace(0, 1, 100)

        # Event at 1.2 s relative to concatenated stream:
        # global sample for event center = floor(1.2 * 100) = 120
        # Epoch boundaries every 100 points
        events = np.array([1.2], dtype=float)
        lim = np.array([-0.2, 0.3], dtype=float)  # [-20, +29] samples window inside epoch 2

        ep, newtime, idx, _, _, _ = epoch(data, events, lim, srate=srate, verbose='off')

        # Expect one accepted epoch
        self.assertTrue(np.array_equal(idx, np.array([0])))
        # The extracted data should match the correct slice from linearized data
        # With MATLAB-compatible indexing: event at 1.2s (sample 120) with window [-0.2, 0.3] 
        # becomes MATLAB indices [101, 149] (1-based), which in Python becomes [99:149] (0-based)
        pos0 = int(np.floor(events[0] * srate))  # 120 (0-based)
        reallim0 = int(np.round(lim[0] * srate))  # -20
        reallim1 = int(np.round(lim[1] * srate - 1))  # 29
        posinit = pos0 + reallim0  # 100 (0-based)
        posend = pos0 + reallim1   # 149 (0-based)
        
        # MATLAB slicing: posinit:posend (1-based) becomes [posinit-1:posend] (0-based)
        start_global = posinit - 1  # 99 (Python 0-based)
        end_global = posend         # 149 (Python 0-based exclusive)
        
        # Extract the expected slice from linearized data (Fortran order)
        data_linearized = data.reshape(1, -1, order='F')
        expected = data_linearized[0, start_global:end_global]
        self.assertTrue(np.allclose(ep[0, :, 0], expected, atol=1e-12))
        # newtime should reflect limits divided by srate with the -1 sample convention
        self.assertTrue(np.allclose(newtime, np.array([lim[0], np.round(lim[1]*srate-1)/srate]), atol=1e-12))

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


if __name__ == '__main__':
    unittest.main()