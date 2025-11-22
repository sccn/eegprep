# test_eeg_lat2point.py
import os
import numpy as np
import unittest

from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_lat2point import eeg_lat2point


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEegLat2PointParity(unittest.TestCase):

    def setUp(self):
        self.eeglab = get_eeglab('MAT')

    def test_parity_continuous(self):
        srate = 100.0
        timewin = [0.0, 1.0]
        lat_array = np.array([0.0, 0.5, 1.0])  # seconds
        epoch_array = 1  # continuous

        py_out, py_flag = eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)
        ml_out = self.eeglab.eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_parity_epoched(self):
        srate = 100.0
        timewin = [-0.2, 0.8]  # seconds
        lat_array = np.array([-0.2, 0.3, 0.8, -0.2], dtype=float)
        epoch_array = np.array([1, 1, 1, 2], dtype=float)

        py_out, py_flag = eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)
        ml_out = self.eeglab.eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_parity_milliseconds(self):
        srate = 1000.0
        timewin = [0.0, 1000.0]  # milliseconds (same units as lat_array)
        lat_array = np.array([0.0, 1000.0], dtype=float)  # milliseconds
        epoch_array = 1  # continuous

        py_out, py_flag = eeg_lat2point(lat_array, epoch_array, srate, timewin, 1e-3)
        ml_out = self.eeglab.eeg_lat2point(lat_array, epoch_array, srate, timewin, 1e-3)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_parity_outrange_clamp(self):
        # One latency slightly beyond xmax to trigger clamp
        srate = 100.0
        timewin = [0.0, 1.0]
        lat_array = np.array([0.0, 1.0000001])  # seconds
        epoch_array = 1

        py_out, py_flag = eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)  # default outrange=1
        ml_out = self.eeglab.eeg_lat2point(lat_array, epoch_array, srate, timewin, 1.0)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))


class TestEegLat2PointFunctional(unittest.TestCase):

    def test_eeg_lat2point_continuous(self):
        # 100 Hz, epoch 0–1 s → 101 points; 0,0.5,1.0 s map to 1,51,101 (1-based)
        srate = 100
        timewin = [0, 1]
        lat_array = [0.0, 0.5, 1.0]
        out, flag = eeg_lat2point(lat_array, [], srate, timewin, 1)
        expected = np.array([1, 51, 101], dtype=float)
        self.assertTrue(np.allclose(out, expected, atol=1e-9))
        self.assertEqual(int(flag), 0)

    def test_eeg_lat2point_epoched(self):
        # 100 Hz, [-0.2, 0.8] s → 101 points per epoch
        # Epoch 1: -0.2, 0.3, 0.8 → 1, 51, 101 (1-based)
        # Epoch 2: -0.2 → 1 + 101 = 102 in concatenated points
        srate = 100
        timewin = [-0.2, 0.8]
        lat_array = [-0.2, 0.3, 0.8, -0.2]
        epoch_array = [1, 1, 1, 2]
        out, flag = eeg_lat2point(lat_array, epoch_array, srate, timewin, 1)
        expected = np.array([1, 51, 101, 102], dtype=float)
        self.assertTrue(np.allclose(out, expected, atol=1e-9))
        self.assertEqual(int(flag), 0)

    def test_eeg_lat2point_milliseconds(self):
        # ms output: 0 ms and 1000 ms → points 1 and 1001 at 1000 Hz (1-based)
        srate = 1000
        timewin = [0, 1000]  # milliseconds (same units as lat_array)
        lat_array = [0.0, 1000.0]  # ms
        out, flag = eeg_lat2point(lat_array, [], srate, timewin, 1e-3)
        expected = np.array([1, 1001], dtype=float)
        self.assertTrue(np.allclose(out, expected, atol=1e-9))
        self.assertEqual(int(flag), 0)

    def test_eeg_lat2point_outrange_clamp(self):
        # With default outrange=1, clamp to max valid point and set flag=1
        srate = 100
        timewin = [0, 1]
        lat_array = [1.2]  # seconds, clearly beyond xmax
        out, flag = eeg_lat2point(lat_array, 1, srate, timewin, 1.0)  # default outrange=1
        self.assertEqual(int(out[0]), 101)  # max valid point (1-based)
        self.assertEqual(int(flag), 1)

    def test_eeg_lat2point_outrange_error(self):
        # With outrange=0, raise error for out-of-range points
        srate = 100
        timewin = [0, 1]
        lat_array = [1.2]
        with self.assertRaises(Exception):
            eeg_lat2point(lat_array, 1, srate, timewin, 1.0, outrange=0)


if __name__ == '__main__':
    # test only test_parity_continuous
    unittest.main()
