import os
import numpy as np
from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_point2lat import eeg_point2lat
import unittest

def _matlab_row(x):
    """Convert a 1D numpy array or list into a MATLAB 1xN double."""
    x = np.asarray(x, dtype=float).ravel().tolist()
    return [[v for v in x]]


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
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
        
class TestEegPoint2LatEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for eeg_point2lat."""
    
    def test_eeg_point2lat_missing_srate(self):
        """Test eeg_point2lat with missing srate."""
        with self.assertRaises(ValueError):
            eeg_point2lat([100], None, None)  # No srate provided
    
    def test_eeg_point2lat_empty_points(self):
        """Test eeg_point2lat with empty points array."""
        result = eeg_point2lat([], None, 250.0)
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, np.ndarray)
    
    def test_eeg_point2lat_single_point(self):
        """Test eeg_point2lat with single point."""
        # Test with single point
        result = eeg_point2lat([500], None, 250.0)
        self.assertEqual(len(result), 1)
        expected_lat = (500 - 1) / 250.0  # Convert from 1-based to 0-based, then to seconds
        self.assertAlmostEqual(result[0], expected_lat, places=6)
    
    def test_eeg_point2lat_multiple_points(self):
        """Test eeg_point2lat with multiple points."""
        points = [1, 250, 500, 1000]
        result = eeg_point2lat(points, None, 250.0)
        
        self.assertEqual(len(result), len(points))
        
        # Check each conversion
        for i, point in enumerate(points):
            expected_lat = (point - 1) / 250.0
            self.assertAlmostEqual(result[i], expected_lat, places=6)
    
    def test_eeg_point2lat_different_time_units(self):
        """Test eeg_point2lat with different time units."""
        points = [250, 500, 750]  # Points at approximately 1s, 2s, 3s for 250Hz
        
        # Test seconds (default)
        result_sec = eeg_point2lat(points, None, 250.0, None, 1.0)
        expected_sec = [(p - 1) / 250.0 for p in points]
        for i, expected in enumerate(expected_sec):
            self.assertAlmostEqual(result_sec[i], expected, places=6)
        
        # Test milliseconds
        result_ms = eeg_point2lat(points, None, 250.0, None, 1e-3)
        expected_ms = [val * 1000 for val in expected_sec]
        for i, expected in enumerate(expected_ms):
            self.assertAlmostEqual(result_ms[i], expected, places=3)
        
        # Test microseconds
        result_us = eeg_point2lat(points, None, 250.0, None, 1e-6)
        expected_us = [val * 1000000 for val in expected_sec]
        for i, expected in enumerate(expected_us):
            self.assertAlmostEqual(result_us[i], expected, places=0)
    
    def test_eeg_point2lat_epoched_data(self):
        """Test eeg_point2lat with epoched data."""
        points = [1, 50, 101]
        epochs = [1, 1, 2]  # First two points in epoch 1, last in epoch 2
        timewin = [-0.2, 0.8]  # 1 second epochs
        srate = 100.0
        
        result = eeg_point2lat(points, epochs, srate, timewin)
        
        self.assertEqual(len(result), len(points))
        
        # For epoched data, formula is more complex
        # Check that results are reasonable
        self.assertTrue(all(isinstance(r, (float, np.floating)) for r in result))
        self.assertTrue(all(np.isfinite(r) for r in result))
    
    def test_eeg_point2lat_boundary_conditions(self):
        """Test eeg_point2lat at boundary conditions."""
        # First point (1-based indexing)
        result = eeg_point2lat([1], None, 250.0)
        self.assertAlmostEqual(result[0], 0.0, places=6)
        
        # Large point value
        result = eeg_point2lat([2000], None, 250.0)
        expected_lat = (2000 - 1) / 250.0
        self.assertAlmostEqual(result[0], expected_lat, places=6)
    
    def test_eeg_point2lat_numpy_array_input(self):
        """Test eeg_point2lat with numpy array input."""
        points_array = np.array([100, 200, 300, 400])
        result = eeg_point2lat(points_array, None, 250.0)
        
        self.assertEqual(len(result), len(points_array))
        
        # Check conversion
        for i, point in enumerate(points_array):
            expected_lat = (point - 1) / 250.0
            self.assertAlmostEqual(result[i], expected_lat, places=6)
    
    def test_eeg_point2lat_float_points(self):
        """Test eeg_point2lat with floating point input."""
        # Float points should be handled
        float_points = [100.5, 200.7, 300.2]
        result = eeg_point2lat(float_points, None, 250.0)
        
        self.assertEqual(len(result), len(float_points))
        
        # Check that float inputs are properly converted
        for i, point in enumerate(float_points):
            expected_lat = (point - 1) / 250.0
            self.assertAlmostEqual(result[i], expected_lat, places=6)
    
    def test_eeg_point2lat_high_sampling_rate(self):
        """Test eeg_point2lat with high sampling rate."""
        points = [1000, 2000]  # 1s and 2s at 1kHz
        result = eeg_point2lat(points, None, 1000.0)
        
        expected = [0.999, 1.999]  # (point-1)/srate
        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(result[i], expected_val, places=6)
    
    def test_eeg_point2lat_low_sampling_rate(self):
        """Test eeg_point2lat with low sampling rate."""
        points = [25, 50]  # 0.48s and 0.98s at 50Hz
        result = eeg_point2lat(points, None, 50.0)
        
        expected = [0.48, 0.98]  # (point-1)/srate
        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(result[i], expected_val, places=6)
    
    def test_eeg_point2lat_timewin_parameter(self):
        """Test eeg_point2lat with timewin parameter."""
        points = [1, 51, 101]
        epochs = [1, 1, 1]
        timewin = [-0.2, 0.8]  # 1 second window
        srate = 100.0
        
        result = eeg_point2lat(points, epochs, srate, timewin)
        
        # Should return reasonable time values within the epoch
        self.assertEqual(len(result), len(points))
        for r in result:
            self.assertTrue(timewin[0] <= r <= timewin[1])
    
    def test_eeg_point2lat_precision(self):
        """Test eeg_point2lat numerical precision."""
        # Test points that should give exact results
        points = [257, 513, 769]  # 1s, 2s, 3s at 256Hz
        result = eeg_point2lat(points, None, 256.0)
        
        expected = [1.0, 2.0, 3.0]
        for i, expected_val in enumerate(expected):
            self.assertAlmostEqual(result[i], expected_val, places=10)


class TestEegPoint2LatIntegration(unittest.TestCase):
    """Integration tests for eeg_point2lat with realistic EEG data scenarios."""
    
    def test_eeg_point2lat_with_events(self):
        """Test eeg_point2lat in context of event processing."""
        # Simulate event latencies
        event_points = [500, 750, 1500]
        srate = 500.0
        
        # Convert latencies
        converted_lats = eeg_point2lat(event_points, None, srate)
        
        # Verify conversions
        expected_times = [(p - 1) / srate for p in event_points]
        for i, expected in enumerate(expected_times):
            self.assertAlmostEqual(converted_lats[i], expected, places=6)
    
    def test_eeg_point2lat_epoch_boundaries(self):
        """Test eeg_point2lat at epoch boundaries."""
        # Test epoch start, middle, and end
        boundary_points = [1, 100, 200]  # Start, middle, end
        epochs = [1, 1, 1]  # All in same epoch
        timewin = [-1.0, 0.99]  # 1.99 second epoch
        srate = 100.0
        
        result = eeg_point2lat(boundary_points, epochs, srate, timewin)
        
        # Should return values within the epoch time window
        for r in result:
            self.assertTrue(timewin[0] <= r <= timewin[1])
    
    def test_eeg_point2lat_performance_large_dataset(self):
        """Test eeg_point2lat performance with large point arrays."""
        # Large number of points
        large_points = list(range(1, 1001))  # 1000 points
        srate = 250.0
        
        # Should complete without issues
        result = eeg_point2lat(large_points, None, srate)
        
        self.assertEqual(len(result), 1000)
        
        # Spot check a few values
        self.assertAlmostEqual(result[0], 0.0, places=6)  # First point
        self.assertAlmostEqual(result[499], 499 / srate, places=6)  # 500th point
        self.assertAlmostEqual(result[999], 999 / srate, places=6)  # 1000th point


if __name__ == '__main__':
    unittest.main()