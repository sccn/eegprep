import unittest
import numpy as np
from unittest.mock import patch

from eegprep.plugins.clean_rawdata.asr_calibrate import asr_calibrate
from eegprep.plugins.clean_rawdata.asr_process import asr_process
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr


class TestAsrCalibrate(unittest.TestCase):
    """Test the asr_calibrate function."""

    def setUp(self):
        """Set up test fixtures with synthetic EEG data."""
        np.random.seed(42)  # For reproducible tests
        self.n_channels = 8
        self.n_samples = 1000
        self.srate = 250.0

        # Create synthetic clean EEG data (zero-mean)
        self.clean_data = np.random.randn(self.n_channels, self.n_samples) * 0.5

        # Add some realistic structure (autocorrelation)
        for i in range(self.n_channels):
            # Simple AR(1) process for more realistic EEG-like data
            for j in range(1, self.n_samples):
                self.clean_data[i, j] += 0.8 * self.clean_data[i, j - 1]

    def test_basic_calibration(self):
        """Test basic ASR calibration functionality."""
        state = asr_calibrate(self.clean_data, self.srate)

        # Check that all required state variables are present
        required_keys = ['M', 'T', 'B', 'A', 'sos', 'iir_state']
        for key in required_keys:
            self.assertIn(key, state)

        # Check matrix shapes
        self.assertEqual(state['M'].shape, (self.n_channels, self.n_channels))
        self.assertEqual(state['T'].shape, (self.n_channels, self.n_channels))

        # Check that M is symmetric and positive definite (within tolerance)
        M = state['M']
        np.testing.assert_allclose(M, M.T, atol=1e-10)
        eigenvals = np.linalg.eigvals(M)
        self.assertTrue(np.all(eigenvals > -1e-10))  # Allow small numerical errors

        # Check that filter coefficients are finite
        self.assertTrue(np.all(np.isfinite(state['B'])))
        self.assertTrue(np.all(np.isfinite(state['A'])))

    def test_different_sampling_rates(self):
        """Test calibration with different sampling rates and precomputed filters."""
        test_srates = [100, 128, 200, 256, 300, 500, 512]

        for srate in test_srates:
            with self.subTest(srate=srate):
                # Create appropriate data size for the sampling rate
                n_samples = max(1000, int(srate * 2))  # At least 2 seconds
                data = np.random.randn(4, n_samples) * 0.3

                state = asr_calibrate(data, srate)

                # Check that state is valid
                self.assertIsInstance(state, dict)
                self.assertIn('M', state)
                self.assertIn('T', state)

                # Check filter coefficients are reasonable
                self.assertTrue(len(state['B']) > 1)
                self.assertTrue(len(state['A']) > 0)

    def test_unsupported_sampling_rate_raises(self):
        """Unsupported sampling rates must fail loudly, not silently degrade.

        Common rates like 999/1000/1024 Hz have no pre-computed spectral filter.
        Substituting a trivial difference filter would silently miscalibrate ASR
        thresholds, so asr_calibrate must raise rather than warn-and-continue.
        """
        data = np.random.randn(4, 1000) * 0.3

        for unsupported_srate in (999.0, 1000.0, 1024.0):
            with self.subTest(srate=unsupported_srate):
                with self.assertRaises(ValueError) as cm:
                    asr_calibrate(data, unsupported_srate)
                self.assertIn('No pre-computed ASR spectral filter', str(cm.exception))

    def test_unsupported_sampling_rate_allows_explicit_filter(self):
        """An explicit B/A bypasses the precomputed-filter lookup for any srate."""
        data = np.random.randn(4, 1000) * 0.3
        B = np.array([1.0, -0.5])
        A = np.array([1.0])

        state = asr_calibrate(data, 999.0, B=B, A=A)
        self.assertIn('M', state)
        np.testing.assert_array_equal(state['B'], B)
        np.testing.assert_array_equal(state['A'], A)

    def test_parameter_validation(self):
        """Test parameter validation and edge cases."""
        # Test with 1D data (should raise error)
        with self.assertRaises(ValueError):
            asr_calibrate(np.random.randn(100), self.srate)

        # Test with 3D data (should raise error)
        with self.assertRaises(ValueError):
            asr_calibrate(np.random.randn(4, 100, 10), self.srate)

        # Test with too little data
        short_data = np.random.randn(self.n_channels, 50)
        with self.assertRaises(ValueError):
            asr_calibrate(short_data, self.srate)

    def test_custom_parameters(self):
        """Test calibration with custom parameters."""
        state = asr_calibrate(
            self.clean_data,
            self.srate,
            cutoff=3.0,
            blocksize=20,
            window_len=0.25,
            window_overlap=0.5,
            max_dropout_fraction=0.2,
            min_clean_fraction=0.3,
            maxmem=32,
        )

        # Should complete without errors
        self.assertIsInstance(state, dict)
        self.assertIn('M', state)
        self.assertIn('T', state)

    def test_custom_filter_coefficients(self):
        """Test calibration with custom filter coefficients."""
        # Simple custom filter
        B = np.array([1.0, -0.5])
        A = np.array([1.0, -0.3])

        state = asr_calibrate(self.clean_data, self.srate, B=B, A=A)

        # Check that custom coefficients are stored
        np.testing.assert_array_equal(state['B'], B)
        np.testing.assert_array_equal(state['A'], A)

    def test_riemannian_calibration(self):
        """Test Riemannian ASR calibration variant."""
        # Mock the cov_mean function to avoid complex dependencies
        with patch('eegprep.plugins.clean_rawdata.asr_calibrate.cov_mean') as mock_cov_mean:
            mock_cov_mean.return_value = np.eye(self.n_channels) * 0.5

            asr_calibrate(self.clean_data, self.srate, useriemannian='calib')

            # Should have called cov_mean with robust=True
            mock_cov_mean.assert_called_once()
            call_args = mock_cov_mean.call_args
            self.assertTrue(call_args[1]['robust'])

    def test_nan_handling(self):
        """Test handling of NaN values in input data."""
        data_with_nans = self.clean_data.copy()
        data_with_nans[2, 100:110] = np.nan
        data_with_nans[5, 500] = np.inf

        # Should not raise error - NaNs should be replaced with zeros
        state = asr_calibrate(data_with_nans, self.srate)

        self.assertIsInstance(state, dict)
        self.assertTrue(np.all(np.isfinite(state['M'])))
        self.assertTrue(np.all(np.isfinite(state['T'])))

    def test_filter_divergence_error(self):
        """Test error handling when IIR filter diverges."""
        # Mock scipy.signal.sosfilt to return NaN values (simulating filter divergence)
        with patch('scipy.signal.sosfilt') as mock_sosfilt:
            # Return data with NaN values to simulate filter divergence
            mock_sosfilt.return_value = (
                np.full((4, 1000), np.nan),
                np.zeros((2, 4, 2)),  # Mock iir_state
            )

            with self.assertRaises(RuntimeError) as cm:
                asr_calibrate(self.clean_data, self.srate)

            self.assertIn('IIR filter diverged', str(cm.exception))

    def test_threshold_calculation_robustness(self):
        """Test robustness of threshold calculation with edge cases."""
        # Create data with some extreme values
        data = self.clean_data.copy()
        data[:, :100] *= 10  # Add some "artifacts"

        with patch('eegprep.plugins.clean_rawdata.asr_calibrate.fit_eeg_distribution') as mock_fit:
            # Mock successful fitting for most components
            mock_fit.return_value = (1.0, 0.5, None, None)

            state = asr_calibrate(data, self.srate)

            # Should complete successfully
            self.assertIsInstance(state, dict)
            self.assertTrue(np.all(np.isfinite(state['T'])))

    def test_blocksize_calculation(self):
        """Test automatic blocksize calculation based on memory constraints."""
        # Test with very low memory limit
        state = asr_calibrate(self.clean_data, self.srate, maxmem=1)  # 1 MB

        # Should still work, just with larger blocksize
        self.assertIsInstance(state, dict)

    def test_geometric_median_fallback(self):
        """Test fallback to geometric median when Riemannian method fails."""
        with patch('eegprep.plugins.clean_rawdata.asr_calibrate.cov_mean') as mock_cov_mean:
            # Make cov_mean return NaNs to trigger fallback
            mock_cov_mean.return_value = np.full((self.n_channels, self.n_channels), np.nan)

            with patch('eegprep.plugins.clean_rawdata.asr_calibrate.geometric_median') as mock_geom_median:
                mock_geom_median.return_value = np.eye(self.n_channels).flatten()

                with self.assertLogs('eegprep.plugins.clean_rawdata.asr_calibrate', level='WARNING') as log:
                    asr_calibrate(self.clean_data, self.srate, useriemannian='calib')

                # Check that warning was logged and fallback was used
                self.assertTrue(any('NaNs' in msg for msg in log.output))
                mock_geom_median.assert_called_once()


class TestAsrProcess(unittest.TestCase):
    """Test the asr_process function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(123)
        self.n_channels = 6
        self.n_samples = 500
        self.srate = 200.0

        # Create calibration data and state
        calib_data = np.random.randn(self.n_channels, 1000) * 0.3
        self.state = asr_calibrate(calib_data, self.srate)

        # Create test data with some artifacts
        self.test_data = np.random.randn(self.n_channels, self.n_samples) * 0.4
        # Add some artifacts to specific channels/times
        self.test_data[2, 100:150] += np.random.randn(50) * 2.0  # Large artifacts

    def test_basic_processing(self):
        """Test basic ASR processing functionality."""
        cleaned_data, new_state = asr_process(self.test_data, self.srate, self.state)

        # Check output shapes
        self.assertEqual(cleaned_data.shape, self.test_data.shape)

        # Check that state is updated
        self.assertIsInstance(new_state, dict)
        self.assertIn('M', new_state)
        self.assertIn('T', new_state)
        self.assertIn('carry', new_state)
        self.assertIn('cov', new_state)

        # Check that cleaned data is finite
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_empty_data_handling(self):
        """Test processing with empty data."""
        empty_data = np.empty((self.n_channels, 0))
        cleaned_data, new_state = asr_process(empty_data, self.srate, self.state)

        # Should return empty data unchanged
        self.assertEqual(cleaned_data.shape, (self.n_channels, 0))
        self.assertEqual(new_state['M'].shape, self.state['M'].shape)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with 1D data
        with self.assertRaises(ValueError):
            asr_process(np.random.randn(100), self.srate, self.state)

        # Test with 3D data
        with self.assertRaises(ValueError):
            asr_process(np.random.randn(4, 100, 10), self.srate, self.state)

    def test_custom_processing_parameters(self):
        """Test processing with custom parameters."""
        cleaned_data, new_state = asr_process(
            self.test_data, self.srate, self.state, window_len=0.25, lookahead=0.1, step_size=16, max_dims=0.5
        )

        # Should complete without errors
        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_max_dims_as_integer(self):
        """Test processing with max_dims specified as integer."""
        cleaned_data, new_state = asr_process(
            self.test_data,
            self.srate,
            self.state,
            max_dims=3,  # Integer instead of fraction
        )

        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_nan_handling_in_processing(self):
        """Test handling of NaN values during processing."""
        data_with_nans = self.test_data.copy()
        data_with_nans[1, 50:60] = np.nan
        data_with_nans[3, 200] = np.inf

        cleaned_data, new_state = asr_process(data_with_nans, self.srate, self.state)

        # Should handle NaNs gracefully
        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_memory_management(self):
        """Test memory management with large data chunks."""
        # Create larger data that might trigger splitting
        large_data = np.random.randn(self.n_channels, 5000) * 0.5

        with patch('psutil.virtual_memory') as mock_vm:
            # Mock low available memory to trigger splitting
            mock_vm.return_value.free = 50 * 1024**2  # 50 MB

            with self.assertLogs('eegprep.plugins.clean_rawdata.asr_process', level='INFO') as log:
                cleaned_data, new_state = asr_process(
                    large_data,
                    self.srate,
                    self.state,
                    max_mem=10,  # Low memory limit
                )

            # Check that splitting was logged
            self.assertTrue(any('blocks' in msg for msg in log.output))

        # Check output
        self.assertEqual(cleaned_data.shape, large_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_memory_error_handling(self):
        """Test error handling when memory is insufficient."""
        with patch('psutil.virtual_memory') as mock_vm:
            # Mock extremely low memory
            mock_vm.return_value.free = 1024  # 1 KB

            with self.assertRaises(RuntimeError) as cm:
                asr_process(self.test_data, self.srate, self.state, max_mem=0.001)

            self.assertIn('Not enough memory', str(cm.exception))

    def test_rank_deficient_covariance_produces_sane_output(self):
        """Process genuinely rank-deficient data (singular covariance).

        Duplicate and zeroed channels make the per-window covariance singular,
        exercising the eigendecomposition and pseudo-inverse paths with a real
        degenerate input rather than monkeypatching numpy to raise. The cleaned
        output must stay finite and keep its shape.
        """
        degenerate = self.test_data.copy()
        degenerate[3, :] = degenerate[0, :]  # duplicate channel -> singular covariance
        degenerate[5, :] = 0.0  # flat channel -> singular covariance

        cleaned_data, new_state = asr_process(degenerate, self.srate, self.state)

        self.assertEqual(cleaned_data.shape, degenerate.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_extreme_artifact_amplitudes_produce_sane_output(self):
        """Process data with extreme-amplitude artifacts.

        Huge transient amplitudes drive the reconstruction matrix toward
        ill-conditioning, exercising the same numeric path. The cleaned output
        must remain finite, keep its shape, and attenuate the injected spike.
        """
        extreme = self.test_data.copy()
        spike_peak = float(np.max(np.abs(extreme))) * 1e4
        extreme[2, 100:150] += spike_peak

        cleaned_data, new_state = asr_process(extreme, self.srate, self.state)

        self.assertEqual(cleaned_data.shape, extreme.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))
        self.assertLess(float(np.max(np.abs(cleaned_data))), spike_peak)

    def test_component_selection_shape_error_propagates(self):
        """A shape/contract bug during component selection must surface, not be
        silently swallowed into a no-op (keep-all) for the affected window.
        """
        bad_state = dict(self.state)
        # T is C x C in a valid state; an incompatible threshold matrix makes
        # finite_matmul(T, V) fail. This must raise rather than disable cleaning.
        bad_state['T'] = np.ones((self.n_channels + 1, self.n_channels + 1))

        with self.assertRaises(ValueError):
            asr_process(self.test_data, self.srate, bad_state)

    def test_state_persistence_across_calls(self):
        """Test that state is properly maintained across multiple processing calls."""
        # First call
        chunk1 = self.test_data[:, :250]
        cleaned1, state1 = asr_process(chunk1, self.srate, self.state)

        # Second call with updated state
        chunk2 = self.test_data[:, 250:]
        cleaned2, state2 = asr_process(chunk2, self.srate, state1)

        # Check that carry buffer was maintained
        self.assertIsNotNone(state1['carry'])
        self.assertIsNotNone(state2['carry'])

        # Check output shapes
        self.assertEqual(cleaned1.shape, chunk1.shape)
        self.assertEqual(cleaned2.shape, chunk2.shape)

    def test_window_length_adjustment(self):
        """Test automatic window length adjustment for small datasets."""
        # Create data that would require window length adjustment
        small_data = np.random.randn(self.n_channels, 50) * 0.5

        cleaned_data, new_state = asr_process(
            small_data,
            self.srate,
            self.state,
            window_len=0.1,  # Very small window
        )

        # Should complete without errors despite small data
        self.assertEqual(cleaned_data.shape, small_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_component_selection_error_handling(self):
        """An unexpected error during threshold computation must propagate.

        Previously such errors were swallowed and the window silently kept all
        components (no artifact removal). They must now surface so genuine bugs are
        visible rather than producing quietly under-cleaned data.
        """
        with patch('numpy.sum', side_effect=Exception("Threshold error")):
            with self.assertRaises(Exception) as cm:
                asr_process(self.test_data, self.srate, self.state)
            self.assertIn('Threshold error', str(cm.exception))


class TestAsrIntegration(unittest.TestCase):
    """Integration tests for ASR calibration and processing."""

    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(456)
        self.n_channels = 4
        self.srate = 128.0

        # Create realistic calibration data
        self.calib_data = self.create_realistic_eeg(self.n_channels, int(self.srate * 60))

        # Create test data with artifacts
        self.test_data = self.create_realistic_eeg(self.n_channels, int(self.srate * 10))
        self.add_artifacts(self.test_data)

    def create_realistic_eeg(self, n_channels, n_samples):
        """Create more realistic EEG-like data."""
        data = np.random.randn(n_channels, n_samples) * 0.2

        # Add some correlated structure between channels
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if np.random.rand() > 0.7:  # 30% chance of correlation
                    correlation = 0.3 * np.random.randn()
                    data[j] += correlation * data[i]

        # Add some temporal autocorrelation
        for i in range(n_channels):
            for j in range(1, n_samples):
                data[i, j] += 0.7 * data[i, j - 1] * np.random.rand()

        return data

    def add_artifacts(self, data):
        """Add realistic artifacts to EEG data."""
        n_channels, n_samples = data.shape

        # Add muscle artifacts (high frequency, high amplitude)
        artifact_start = n_samples // 4
        artifact_end = artifact_start + n_samples // 10
        data[1, artifact_start:artifact_end] += np.random.randn(artifact_end - artifact_start) * 3.0

        # Add eye blink artifacts (affects frontal channels)
        blink_times = [n_samples // 2, 3 * n_samples // 4]
        for blink_time in blink_times:
            blink_duration = 20
            if blink_time + blink_duration < n_samples:
                blink_artifact = 5.0 * np.exp(-np.arange(blink_duration) / 5.0)
                data[0, blink_time : blink_time + blink_duration] += blink_artifact

    def test_full_calibration_and_processing_pipeline(self):
        """Test complete ASR pipeline from calibration to processing."""
        # Calibrate ASR
        state = asr_calibrate(self.calib_data, self.srate)

        # Process test data
        cleaned_data, final_state = asr_process(self.test_data, self.srate, state)

        # Basic checks
        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

        # Check that artifacts were reduced (RMS should be lower in artifact regions)
        artifact_region = slice(
            self.test_data.shape[1] // 4, self.test_data.shape[1] // 4 + self.test_data.shape[1] // 10
        )

        original_rms = np.sqrt(np.mean(self.test_data[1, artifact_region] ** 2))
        cleaned_rms = np.sqrt(np.mean(cleaned_data[1, artifact_region] ** 2))

        # Cleaned data should have lower RMS in artifact region
        self.assertLess(cleaned_rms, original_rms)

    def test_streaming_processing_simulation(self):
        """Test ASR processing in streaming mode with multiple chunks."""
        # Calibrate
        state = asr_calibrate(self.calib_data, self.srate)

        # Process in chunks to simulate streaming
        chunk_size = int(self.srate * 2)  # 2-second chunks
        n_chunks = self.test_data.shape[1] // chunk_size

        cleaned_chunks = []
        current_state = state

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.test_data.shape[1])
            chunk = self.test_data[:, start_idx:end_idx]

            cleaned_chunk, current_state = asr_process(chunk, self.srate, current_state)
            cleaned_chunks.append(cleaned_chunk)

        # Concatenate results
        full_cleaned = np.concatenate(cleaned_chunks, axis=1)

        # Check results
        expected_length = n_chunks * chunk_size
        self.assertEqual(full_cleaned.shape[1], expected_length)
        self.assertTrue(np.all(np.isfinite(full_cleaned)))

    def test_different_calibration_and_processing_parameters(self):
        """Test ASR with various parameter combinations."""
        parameter_sets = [
            {'cutoff': 3.0, 'window_len': 0.25, 'max_dims': 0.5},
            {'cutoff': 7.0, 'window_len': 1.0, 'max_dims': 2},
            {'cutoff': 4.0, 'blocksize': 20, 'window_overlap': 0.8},
        ]

        for params in parameter_sets:
            with self.subTest(params=params):
                # Split parameters between calibration and processing
                calib_params = {k: v for k, v in params.items() if k in ['cutoff', 'blocksize', 'window_overlap']}
                process_params = {k: v for k, v in params.items() if k in ['window_len', 'max_dims']}

                # Test pipeline
                state = asr_calibrate(self.calib_data, self.srate, **calib_params)
                cleaned_data, _ = asr_process(self.test_data, self.srate, state, **process_params)

                # Should complete without errors
                self.assertEqual(cleaned_data.shape, self.test_data.shape)
                self.assertTrue(np.all(np.isfinite(cleaned_data)))

    def test_robustness_with_challenging_data(self):
        """Test ASR robustness with challenging data conditions."""
        # Test with very noisy calibration data
        noisy_calib = self.calib_data + np.random.randn(*self.calib_data.shape) * 0.5

        # Should still calibrate successfully
        state = asr_calibrate(noisy_calib, self.srate)
        cleaned_data, _ = asr_process(self.test_data, self.srate, state)

        self.assertEqual(cleaned_data.shape, self.test_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

        # Test with data containing extreme outliers
        outlier_data = self.test_data.copy()
        outlier_data[2, 100:105] = 1000.0  # Extreme outliers

        cleaned_data, _ = asr_process(outlier_data, self.srate, state)

        # Should handle outliers gracefully
        self.assertEqual(cleaned_data.shape, outlier_data.shape)
        self.assertTrue(np.all(np.isfinite(cleaned_data)))

        # Extreme values should be reduced
        max_cleaned = np.max(np.abs(cleaned_data[2, 100:105]))
        max_original = np.max(np.abs(outlier_data[2, 100:105]))
        self.assertLess(max_cleaned, max_original)


class TestAsrEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for ASR functions."""

    def test_minimum_viable_data_sizes(self):
        """Test ASR with minimum viable data sizes."""
        n_channels = 2
        srate = 100.0

        # Minimum calibration data (need enough for multiple windows)
        # window_len=0.5s, overlap=0.66 means step=0.17s, need at least 2 windows
        min_samples = int(srate * 1.2)  # 1.2 seconds to allow multiple windows
        min_data = np.random.randn(n_channels, min_samples) * 0.3

        state = asr_calibrate(min_data, srate, window_len=0.5)

        # Should work with minimal data
        self.assertIsInstance(state, dict)
        self.assertIn('M', state)

        # Test processing with minimal data
        test_data = np.random.randn(n_channels, 20) * 0.4
        cleaned_data, _ = asr_process(test_data, srate, state)

        self.assertEqual(cleaned_data.shape, test_data.shape)

    def test_single_channel_data(self):
        """Test ASR with single channel data."""
        n_channels = 1
        srate = 200.0
        calib_data = np.random.randn(n_channels, 1000) * 0.3

        state = asr_calibrate(calib_data, srate)

        # Should work with single channel
        self.assertEqual(state['M'].shape, (1, 1))
        self.assertEqual(state['T'].shape, (1, 1))

        # Test processing
        test_data = np.random.randn(n_channels, 100) * 0.4
        cleaned_data, _ = asr_process(test_data, srate, state)

        self.assertEqual(cleaned_data.shape, test_data.shape)

    def test_very_high_sampling_rate(self):
        """Very high (unsupported) sampling rate must fail loudly in calibration.

        2000 Hz has no pre-computed spectral filter; calibration must raise rather
        than silently substitute a degenerate difference filter.
        """
        n_channels = 4
        srate = 2000.0  # High, unsupported sampling rate

        n_samples = int(srate * 2)  # 2 seconds
        calib_data = np.random.randn(n_channels, n_samples) * 0.3

        with self.assertRaises(ValueError) as cm:
            asr_calibrate(calib_data, srate)
        self.assertIn('No pre-computed ASR spectral filter', str(cm.exception))

        # With explicit filter coefficients the high rate is processable end-to-end.
        B = np.array([1.0, -0.5])
        A = np.array([1.0])
        state = asr_calibrate(calib_data, srate, B=B, A=A)
        test_data = np.random.randn(n_channels, 200) * 0.4
        cleaned_data, _ = asr_process(test_data, srate, state)
        self.assertEqual(cleaned_data.shape, test_data.shape)

    def test_zero_variance_data(self):
        """Test ASR with zero variance data."""
        n_channels = 3
        srate = 250.0

        # Create data with zero variance in some channels
        calib_data = np.random.randn(n_channels, 1000) * 0.3
        calib_data[1, :] = 1.0  # Constant channel

        # Should handle zero variance gracefully
        state = asr_calibrate(calib_data, srate)

        self.assertIsInstance(state, dict)
        self.assertTrue(np.all(np.isfinite(state['M'])))

    def test_memory_usage_calculation_accuracy(self):
        """Test that memory usage calculations are reasonable."""
        n_channels = 8
        srate = 250.0
        n_samples = 10000  # Large dataset

        data = np.random.randn(n_channels, n_samples) * 0.3

        # Test with different memory limits
        for max_mem in [1, 10, 100]:  # MB
            with self.subTest(max_mem=max_mem):
                state = asr_calibrate(data, srate, maxmem=max_mem)

                # Should complete regardless of memory limit
                self.assertIsInstance(state, dict)

                # Test processing with memory limits
                cleaned_data, _ = asr_process(data[:, :1000], srate, state, max_mem=max_mem)
                self.assertEqual(cleaned_data.shape, (n_channels, 1000))


class TestCleanAsrNoMutation(unittest.TestCase):
    """Regression tests that clean_asr never mutates the caller's EEG."""

    def setUp(self):
        np.random.seed(7)
        n_channels = 8
        n_samples = 2500
        srate = 250.0
        data = np.random.randn(n_channels, n_samples) * 0.5
        for i in range(n_channels):
            for j in range(1, n_samples):
                data[i, j] += 0.8 * data[i, j - 1]
        # Inject a non-finite sample to exercise the in-place NaN-zeroing path
        # that asr_calibrate applies to whatever array it receives.
        data[0, 100] = np.nan
        self.EEG = {
            'data': data,
            'srate': srate,
            'nbchan': n_channels,
            'pnts': n_samples,
            'etc': {},
        }

    def test_does_not_mutate_input_data(self):
        """clean_asr must leave the caller's EEG['data'] (incl. NaNs) unchanged."""
        EEG_in = self.EEG
        original_data = EEG_in['data'].copy()

        EEG_out = clean_asr(EEG_in, ref_maxbadchannels='off')

        # The caller's data is byte-for-byte unchanged, including the NaN that
        # asr_calibrate would otherwise have zeroed in place.
        self.assertTrue(np.array_equal(original_data, EEG_in['data'], equal_nan=True))
        # Output is a distinct object with distinct data.
        self.assertIsNot(EEG_out, EEG_in)
        self.assertIsNot(EEG_out['data'], EEG_in['data'])


if __name__ == '__main__':
    unittest.main()
