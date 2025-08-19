import numpy as np
from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_point2lat import eeg_point2lat
import unittest

def _matlab_row(x):
    """Convert a 1D numpy array or list into a MATLAB 1xN double."""
    x = np.asarray(x, dtype=float).ravel().tolist()
    return [[v for v in x]]


class TestEegPoint2LatParity(unittest.TestCase):

    def setUp(self):
        self.eeglab = get_eeglab('MAT')
        
    def test_parity_continuous(self):
        srate = 100.0
        timewin = [0.0, 1.0]
        lat_array = np.array([1, 51, 101])
        epoch_array = []

        py_out = eeg_point2lat(lat_array, epoch_array, srate, timewin, 1.0)

        ml_out = self.eeglab.eeg_point2lat(lat_array, epoch_array, srate, timewin, 1.0)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_parity_epoched(self):
        srate = 100.0
        timewin = [-0.2, 0.8]
        lat_array = np.array([1, 51, 101, 102], dtype=float)
        epoch_array = np.array([1, 1, 1, 2], dtype=float)

        py_out = eeg_point2lat(lat_array, epoch_array, srate, timewin, 1.0)

        ml_out = self.eeglab.eeg_point2lat(lat_array, epoch_array, srate, timewin, 1.0)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_parity_milliseconds(self):
        srate = 1000.0
        timewin = [0.0, 1.0]
        lat_array = np.array([1, 1001], dtype=float)
        epoch_array = []

        py_out = eeg_point2lat(lat_array, epoch_array, srate, timewin, 1e-3)

        ml_out = self.eeglab.eeg_point2lat(lat_array, epoch_array, srate, timewin, 1e-3)

        self.assertTrue(np.allclose(py_out, ml_out, atol=1e-9))

    def test_eeg_point2lat_continuous(self):    
        # Continuous data: 100 Hz, 0–1 s
        srate = 100
        timewin = [0, 1]   # seconds
        lat_array = [1, 51, 101]  # 1-based samples
        # Expected latencies in seconds: [0.0, 0.5, 1.0]
        out = eeg_point2lat(lat_array, [], srate, timewin, 1)
        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(out, expected, atol=1e-9)

    def test_eeg_point2lat_epoched(self):
        # Epoched: 100 Hz, [-0.2, 0.8] s per epoch, 101 points
        srate = 100
        timewin = [-0.2, 0.8]
        lat_array = [1, 51, 101, 102]  # across 2 epochs concatenated
        epoch_array = [1, 1, 1, 2]
        out = eeg_point2lat(lat_array, epoch_array, srate, timewin, 1)
        # First epoch: sample 1 → -0.2 s, sample 51 → 0.3 s, sample 101 → 0.8 s
        # Second epoch: sample 102 is the first point of epoch 2 → -0.2 s
        expected = np.array([-0.2, 0.3, 0.8, -0.2])
        assert np.allclose(out, expected, atol=1e-9)

    def test_eeg_point2lat_milliseconds(self):
        # Output in ms instead of seconds
        srate = 1000
        timewin = [0, 1]   # sec
        lat_array = [1, 1001]
        out = eeg_point2lat(lat_array, [], srate, timewin, 1e-3)  # ms
        expected = np.array([0.0, 1000.0])  # ms
        assert np.allclose(out, expected, atol=1e-9)
        
if __name__ == '__main__':
    unittest.main()