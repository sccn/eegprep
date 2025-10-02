import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from eegprep.utils.ransac import rand_sample, calc_projector


class TestRandSample(unittest.TestCase):
    """Test the rand_sample function for random sampling without replacement."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    def test_basic_sampling(self):
        """Test basic random sampling functionality."""
        n, m = 10, 5
        result = rand_sample(n, m, self.rng)
        
        # Check output shape and type
        self.assertEqual(result.shape, (m,))
        self.assertEqual(result.dtype, int)
        
        # Check all values are within valid range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < n))
        
        # Check no duplicates (sampling without replacement)
        self.assertEqual(len(np.unique(result)), m)
    
    def test_deterministic_with_fixed_seed(self):
        """Test that results are deterministic with fixed random state."""
        n, m = 8, 4
        rng1 = np.random.RandomState(123)
        rng2 = np.random.RandomState(123)
        
        result1 = rand_sample(n, m, rng1)
        result2 = rand_sample(n, m, rng2)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_seeds_give_different_results(self):
        """Test that different seeds produce different results."""
        n, m = 10, 5
        rng1 = np.random.RandomState(111)
        rng2 = np.random.RandomState(222)
        
        result1 = rand_sample(n, m, rng1)
        result2 = rand_sample(n, m, rng2)
        
        # Results should be different (very high probability)
        self.assertFalse(np.array_equal(result1, result2))
    
    def test_sample_all_elements(self):
        """Test sampling all available elements."""
        n = 5
        m = n  # Sample all elements
        result = rand_sample(n, m, self.rng)
        
        # Should get all indices in some order
        sorted_result = np.sort(result)
        expected = np.arange(n)
        np.testing.assert_array_equal(sorted_result, expected)
    
    def test_sample_one_element(self):
        """Test sampling a single element."""
        n, m = 10, 1
        result = rand_sample(n, m, self.rng)
        
        self.assertEqual(result.shape, (1,))
        self.assertTrue(0 <= result[0] < n)
    
    def test_edge_case_small_pool(self):
        """Test with very small pool size."""
        n, m = 2, 1
        result = rand_sample(n, m, self.rng)
        
        self.assertEqual(result.shape, (1,))
        self.assertIn(result[0], [0, 1])
    
    def test_sampling_algorithm_coverage(self):
        """Test that the sampling algorithm covers the pool properly."""
        n, m = 6, 3
        
        # Run multiple times to check distribution
        results = []
        for seed in range(100):
            rng = np.random.RandomState(seed)
            result = rand_sample(n, m, rng)
            results.extend(result.tolist())
        
        # Each index should appear at least once across many runs
        unique_indices = set(results)
        self.assertEqual(len(unique_indices), n)  # All indices should appear


class TestCalcProjector(unittest.TestCase):
    """Test the calc_projector function for RANSAC reconstruction matrices."""
    
    def setUp(self):
        """Set up test fixtures with synthetic channel locations."""
        # Create synthetic 3D channel locations (spherical coordinates)
        self.n_channels = 8
        theta = np.linspace(0, 2*np.pi, self.n_channels, endpoint=False)
        phi = np.pi/4  # Fixed elevation
        
        self.locs = np.column_stack([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi) * np.ones(self.n_channels)
        ])
        
        # Test parameters
        self.num_samples = 5
        self.subset_size = 4
        self.rng = np.random.RandomState(12345)
    
    def test_basic_projector_calculation(self):
        """Test basic projector matrix calculation."""
        # Mock the sphericalSplineInterpolate function
        # Input: src_locs (3, subset_size), dest_locs (3, n_channels)
        # Output: W (n_channels, subset_size)
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size (columns in transposed input)
            n_dest = dest_locs.shape[1]  # n_channels (columns in transposed input)
            return np.random.randn(n_dest, n_src), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            result = calc_projector(
                self.locs, 
                self.num_samples, 
                self.subset_size, 
                stream=self.rng
            )
        
        # Check output shape
        expected_shape = (self.n_channels, self.n_channels * self.num_samples)
        self.assertEqual(result.shape, expected_shape)
        
        # Verify interpolation function was called correct number of times
        self.assertEqual(mock_interp.call_count, self.num_samples)
    
    def test_deterministic_with_fixed_stream(self):
        """Test that results are deterministic with fixed random stream."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            return np.ones((n_dest, n_src)), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            rng1 = np.random.RandomState(999)
            rng2 = np.random.RandomState(999)
            
            result1 = calc_projector(self.locs, 3, 2, stream=rng1)
            result2 = calc_projector(self.locs, 3, 2, stream=rng2)
            
            np.testing.assert_array_equal(result1, result2)
    
    def test_default_random_stream(self):
        """Test that default random stream is used when none provided."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            # Return deterministic values for reproducible test
            return np.ones((n_dest, n_src)), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Call without providing stream - should use default seed
            result1 = calc_projector(self.locs, 2, 2)
            result2 = calc_projector(self.locs, 2, 2)
            
            # Should be identical due to fixed default seed
            np.testing.assert_array_equal(result1, result2)
    
    def test_matlab_subroutine(self):
        """Test using MATLAB subroutine."""
        mock_matlab = MagicMock()
        def mock_matlab_interp(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # MATLAB uses transposed input
            n_dest = dest_locs.shape[1]
            return np.random.randn(n_dest, n_src), None
        mock_matlab.sphericalSplineInterpolate.side_effect = mock_matlab_interp
        
        with patch('eegprep.utils.ransac.get_eeglab') as mock_get_eeglab:
            mock_get_eeglab.return_value = mock_matlab
            
            result = calc_projector(
                self.locs, 
                self.num_samples, 
                self.subset_size, 
                stream=self.rng,
                subroutine='matlab'
            )
        
        # Check that MATLAB was requested and used
        mock_get_eeglab.assert_called_once_with('MAT')
        self.assertEqual(mock_matlab.sphericalSplineInterpolate.call_count, self.num_samples)
        self.assertEqual(result.shape, (self.n_channels, self.n_channels * self.num_samples))
    
    def test_octave_subroutine(self):
        """Test using Octave subroutine."""
        mock_octave = MagicMock()
        def mock_octave_interp(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # Octave uses transposed input
            n_dest = dest_locs.shape[1]
            return np.random.randn(n_dest, n_src), None
        mock_octave.sphericalSplineInterpolate.side_effect = mock_octave_interp
        
        with patch('eegprep.utils.ransac.get_eeglab') as mock_get_eeglab:
            mock_get_eeglab.return_value = mock_octave
            
            result = calc_projector(
                self.locs, 
                self.num_samples, 
                self.subset_size, 
                stream=self.rng,
                subroutine='octave'
            )
        
        # Check that Octave was requested and used
        mock_get_eeglab.assert_called_once_with('OCT')
        self.assertEqual(mock_octave.sphericalSplineInterpolate.call_count, self.num_samples)
        self.assertEqual(result.shape, (self.n_channels, self.n_channels * self.num_samples))
    
    def test_invalid_subroutine_error(self):
        """Test error handling for invalid subroutine."""
        with self.assertRaises(ValueError) as cm:
            calc_projector(
                self.locs, 
                self.num_samples, 
                self.subset_size, 
                stream=self.rng,
                subroutine='invalid_subroutine'
            )
        
        self.assertIn('Unknown subroutine: invalid_subroutine', str(cm.exception))
    
    def test_different_sample_parameters(self):
        """Test with different sampling parameters."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            # Return identity-like matrix truncated to correct size
            result = np.zeros((n_dest, n_src))
            min_dim = min(n_dest, n_src)
            result[:min_dim, :min_dim] = np.eye(min_dim)
            return result, None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Test with different num_samples
            result1 = calc_projector(self.locs, 3, 2, stream=self.rng)
            result2 = calc_projector(self.locs, 6, 2, stream=self.rng)
            
            self.assertEqual(result1.shape, (self.n_channels, self.n_channels * 3))
            self.assertEqual(result2.shape, (self.n_channels, self.n_channels * 6))
            
            # Test with different subset_size
            result3 = calc_projector(self.locs, 4, 3, stream=self.rng)
            self.assertEqual(result3.shape, (self.n_channels, self.n_channels * 4))
    
    def test_complex_interpolation_result_handling(self):
        """Test handling of complex interpolation results."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            # Create complex interpolation result
            real_part = np.random.randn(n_dest, n_src)
            imag_part = np.random.randn(n_dest, n_src)
            return real_part + 1j * imag_part, None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            result = calc_projector(self.locs, 2, 2, stream=self.rng)
            
            # Result should be real (np.real applied)
            self.assertTrue(np.isrealobj(result))
            self.assertFalse(np.iscomplexobj(result))
    
    def test_sampling_coverage_across_channels(self):
        """Test that sampling covers different channel subsets."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            result = np.zeros((n_dest, n_src))
            min_dim = min(n_dest, n_src)
            result[:min_dim, :min_dim] = np.eye(min_dim)
            return result, None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Use many samples to ensure different subsets are chosen
            result = calc_projector(
                self.locs, 
                num_samples=20, 
                subset_size=3, 
                stream=np.random.RandomState(777)
            )
            
            # Check that result has expected shape
            expected_shape = (self.n_channels, self.n_channels * 20)
            self.assertEqual(result.shape, expected_shape)
            
            # Verify interpolation was called for each sample
            self.assertEqual(mock_interp.call_count, 20)
    
    def test_edge_case_single_sample(self):
        """Test with single sample."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            return np.random.randn(n_dest, n_src), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            result = calc_projector(self.locs, 1, 2, stream=self.rng)
            
            self.assertEqual(result.shape, (self.n_channels, self.n_channels))
            self.assertEqual(mock_interp.call_count, 1)
    
    def test_large_subset_size(self):
        """Test with subset size close to total number of channels."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            return np.random.randn(n_dest, n_src), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Use subset_size = n_channels - 1
            result = calc_projector(
                self.locs, 
                3, 
                self.n_channels - 1, 
                stream=self.rng
            )
            
            self.assertEqual(result.shape, (self.n_channels, self.n_channels * 3))
            self.assertEqual(mock_interp.call_count, 3)


class TestRansacIntegration(unittest.TestCase):
    """Integration tests for RANSAC functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create more realistic channel locations
        self.n_channels = 16
        
        # Create locations on unit sphere (typical EEG setup)
        np.random.seed(42)
        self.locs = np.random.randn(self.n_channels, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(self.locs, axis=1, keepdims=True)
        self.locs = self.locs / norms
    
    def test_no_fail_path_with_realistic_data(self):
        """Test that RANSAC functions don't fail with realistic data."""
        # Mock the interpolation to avoid dependency on complex spatial functions
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            return np.random.randn(n_dest, n_src), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # This should not raise any exceptions
            result = calc_projector(
                self.locs, 
                num_samples=10, 
                subset_size=8, 
                stream=np.random.RandomState(555)
            )
            
            # Basic sanity checks
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (self.n_channels, self.n_channels * 10))
            self.assertFalse(np.any(np.isnan(result)))
            self.assertFalse(np.any(np.isinf(result)))
    
    def test_synthetic_noisy_channel_detection_simulation(self):
        """Simulate bad channel detection scenario."""
        # Create synthetic data where some channels are "bad"
        n_good_channels = 12
        n_bad_channels = 4
        total_channels = n_good_channels + n_bad_channels
        
        # Create locations
        locs = np.random.randn(total_channels, 3)
        locs = locs / np.linalg.norm(locs, axis=1, keepdims=True)
        
        # Mock interpolation that simulates good reconstruction for good channels
        def mock_interp_func(src_locs, dest_locs):
            # Create a reconstruction matrix that works well for "good" channels
            n_src = src_locs.shape[1]  # Transposed input
            n_dest = dest_locs.shape[1]
            
            # Simulate good reconstruction (identity-like for good channels)
            result = np.random.randn(n_dest, n_src) * 0.1
            # Add some structure to simulate realistic interpolation
            if n_src == n_dest:
                result += np.eye(n_dest, n_src)
            
            return result, None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func):
            # Calculate projector matrix
            projector = calc_projector(
                locs, 
                num_samples=15, 
                subset_size=10, 
                stream=np.random.RandomState(888)
            )
            
            # Verify basic properties
            self.assertEqual(projector.shape, (total_channels, total_channels * 15))
            
            # In a real RANSAC application, this projector would be used to:
            # 1. Reconstruct each channel from subsets
            # 2. Compute reconstruction errors
            # 3. Identify channels with consistently high errors as "bad"
            
            # For testing purposes, just verify the projector is reasonable
            self.assertFalse(np.any(np.isnan(projector)))
            self.assertFalse(np.any(np.isinf(projector)))
            
            # Check that projector has some non-zero structure
            self.assertTrue(np.any(projector != 0))
    
    def test_deterministic_behavior_for_reproducibility(self):
        """Test that RANSAC behavior is reproducible for debugging."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            # Use fixed seed for reproducible results
            rng = np.random.RandomState(123)
            return rng.randn(n_dest, n_src), None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Multiple runs with same seed should give identical results
            results = []
            for _ in range(3):
                result = calc_projector(
                    self.locs, 
                    num_samples=5, 
                    subset_size=6,
                    stream=np.random.RandomState(999)  # Same seed each time
                )
                results.append(result.copy())
            
            # All results should be identical
            for i in range(1, len(results)):
                np.testing.assert_array_equal(results[0], results[i])
    
    def test_parameter_validation_implicit(self):
        """Test that functions handle edge cases gracefully."""
        def mock_interp_func(src_locs, dest_locs):
            n_src = src_locs.shape[1]  # subset_size
            n_dest = dest_locs.shape[1]  # n_channels
            result = np.zeros((n_dest, n_src))
            min_dim = min(n_dest, n_src)
            result[:min_dim, :min_dim] = np.eye(min_dim)
            return result, None
        
        with patch('eegprep.utils.ransac.sphericalSplineInterpolate', side_effect=mock_interp_func) as mock_interp:
            
            # Test minimum viable parameters
            result = calc_projector(
                self.locs, 
                num_samples=1, 
                subset_size=1, 
                stream=np.random.RandomState(111)
            )
            
            self.assertEqual(result.shape, (self.n_channels, self.n_channels))
            
            # Test with subset_size equal to number of channels
            result = calc_projector(
                self.locs, 
                num_samples=2, 
                subset_size=self.n_channels, 
                stream=np.random.RandomState(222)
            )
            
            self.assertEqual(result.shape, (self.n_channels, self.n_channels * 2))


if __name__ == '__main__':
    unittest.main()
